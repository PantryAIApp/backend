from typing import Annotated, Optional, List
from fastapi import FastAPI, Depends, HTTPException, status,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials, firestore
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
import os
from dotenv import dotenv_values
from app.models import UserEmailAndPassword, Recipe
import requests
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import base64
import io
from PIL import Image


app = FastAPI()

origins = ["*"] # this is added to allow all origins so our app can access this backend. 
print(origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cred = credentials.Certificate("/app/service-account.json") # for prod
# cred = credentials.Certificate("./service-account.json") # for local development

# This part might be needed
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account.json" # for local dev
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service-account.json" # for prod


# config = dotenv_values(".env") # for local
config = dotenv_values("/app/.env") # for prod

FIREBASE_API_KEY = config.get("FIREBASE_API_KEY", "")
# GOOGLE_API_KEY = config.get("GOOGLE_API_KEY", "")

default_app = firebase_admin.initialize_app(cred)

db = firestore.client()



# https://medium.com/@gabriel.cournelle/firebase-authentication-in-the-backend-with-fastapi-4ff3d5db55ca

# use of a simple bearer scheme as auth is handled by firebase and not fastapi
# we set auto_error to False because fastapi incorrectly returns a 403 intead 
# of a 401
# see: https://github.com/tiangolo/fastapi/pull/2120
bearer_scheme = HTTPBearer(auto_error=False)

def get_firebase_user_from_token(
    token: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
) -> Optional[dict]:
    """Uses bearer token to identify firebase user id

    Args:
        token : the bearer token. Can be None as we set auto_error to False

    Returns:
        dict: the firebase user on success
    Raises:
        HTTPException 401 if user does not exist or token is invalid
    """
    try:
        if not token:
            # raise and catch to return 401, only needed because fastapi returns 403
            # by default instead of 401 so we set auto_error to False
            # 401 is the correct error in this situation
            raise ValueError("No token")
        user = auth.verify_id_token(token.credentials)
        return user

    except Exception:
        # https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not logged in or Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
async def root():
    """
    Base test function
    """
    return {"status": "OK"}

@app.get("/requires-auth")
async def requires_auth_example(user: Annotated[dict, Depends(get_firebase_user_from_token)]):
    """
    Example function to show the authguard. 

    :param user: dict Information about the user.
    """
    return {"user": user}

@app.post('/create-user')
async def create_user(user: UserEmailAndPassword):
    """
    This function creates the user using the email and password. Afterwards the /login method can be used to get the idtoken

    :param user: UserEmailAndPassword The user object containing the email and password
    """
    try:
        user = auth.create_user(email=user.email, password=user.password)
        return {"user": user}
    except Exception as e:
        raise HTTPException(status_code=500, 
                            detail="Server error " + str(e))

@app.post('/login')
async def login(user: UserEmailAndPassword): 
    """
    This function logs in the user using the email and password. The "idtoken" can be used for development purposes. 

    :param user: UserEmailAndPassword The user object containing the email and password
    """
    try:
        user_obj = {"email": user.email, "password": user.password}
        output = requests.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=' + FIREBASE_API_KEY, json={**user_obj,'returnSecureToken': True})
        return {"output": output.json()}
    except Exception as e:
        raise HTTPException(status_code=500, 
                            detail="Server error " + str(e))
    
@app.post("/recipes", response_model=Recipe)
async def create_recipe(
    recipe: Recipe, 
    user: dict = Depends(get_firebase_user_from_token)
):

    recipe.created_by = user.get("uid")
    
    doc_ref = db.collection("recipes").document()
    recipe.id = doc_ref.id  
    doc_ref.set(recipe.model_dump())
    return recipe

@app.get("/recipes/{recipe_id}", response_model=Recipe)
async def get_recipe(
    recipe_id: str, 
    user: dict = Depends(get_firebase_user_from_token)
):
    doc_ref = db.collection("recipes").document(recipe_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Recipe not found"
        )
    recipe_data = doc.to_dict()

    if recipe_data.get("created_by") != user.get("uid"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Not authorized to view this recipe"
        )
    
    return Recipe(**recipe_data)

@app.put("/recipes/{recipe_id}", response_model=Recipe)
async def update_recipe(
    recipe_id: str, 
    recipe: Recipe,
    user: dict = Depends(get_firebase_user_from_token)
):
    doc_ref = db.collection("recipes").document(recipe_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Recipe not found"
        )
    
    existing_recipe = doc.to_dict()
    if existing_recipe.get("created_by") != user.get("uid"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Not authorized to update this recipe"
        )
    
    update_data = recipe.model_dump(exclude_unset=True)
    update_data.pop("created_by", None)
    
    doc_ref.update(update_data)

    updated_doc = doc_ref.get().to_dict()
    return Recipe(**updated_doc)

@app.delete("/recipes/{recipe_id}")
async def delete_recipe(
    recipe_id: str,
    user: dict = Depends(get_firebase_user_from_token)
):
    doc_ref = db.collection("recipes").document(recipe_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Recipe not found"
        )
    recipe_data = doc.to_dict()
    if recipe_data.get("created_by") != user.get("uid"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Not authorized to delete this recipe"
        )
    doc_ref.delete()
    return {"detail": "Recipe deleted successfully"}

@app.get("/recipes", response_model=List[Recipe])
async def list_recipes(user: dict = Depends(get_firebase_user_from_token)):
    recipes_ref = db.collection("recipes").where("created_by", "==", user.get("uid")).stream()
    recipes = []
    for doc in recipes_ref:
        recipes.append(Recipe(**doc.to_dict()))
    return recipes


# Add this to your environment variables section
# GOOGLE_API_KEY = config.get("GOOGLE_API_KEY", "")

# Define the output structure using Pydantic
class IngredientList(BaseModel):
    ingredients: list[str] = Field(description="List of ingredients identified in the image")

# Create the output parser
parser = PydanticOutputParser(pydantic_object=IngredientList)

# Define few-shot examples
FEW_SHOT_EXAMPLES = """
Example 1:
Image: A bowl of pasta with tomatoes, basil, and cheese
Output: {"ingredients": ["pasta", "tomatoes", "basil", "cheese"]}

Example 2:
Image: A plate with grilled chicken, rice, and steamed vegetables
Output: {"ingredients": ["chicken", "rice", "broccoli", "carrots"]}

Example 3:
Image: A fruit salad with apples, bananas, and grapes
Output: {"ingredients": ["apples", "bananas", "grapes"]}
"""

# Create the prompt template with few-shot examples
INGREDIENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert chef analyzing food images. Your task is to identify all ingredients in the image.
    Follow these rules:
    1. List only ingredients you can see with high confidence
    2. Use common names for ingredients
    3. Don't include quantities or measurements
    4. Don't include preparation instructions
    5. Return the output in JSON format as shown in the examples
    
    Examples:
    {few_shot_examples}
    
    {format_instructions}"""),
    (
        "human",
        "Identify the ingredients in the image."  # This message will populate the required text field.
    ),
    ("human", [
        {
            "type": "image_url",
            "image_url": {"url": "data:{image_type};base64,{image_data}"}
        }
    ])
])

@app.post("/extract-ingredients")
async def extract_ingredients(
    image: UploadFile = File(...),
    user: dict = Depends(get_firebase_user_from_token)
):
    """
    Extract ingredients from an uploaded image using LangChain and Gemini
    """
    try:
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents))
        
        buffered = io.BytesIO()
        image_pil.save(buffered, format=image_pil.format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        model = ChatVertexAI(
            model="gemini-2.0-flash"
        ) 
        
        chain = INGREDIENT_PROMPT | model | parser
        
        result = chain.invoke({
            "few_shot_examples": FEW_SHOT_EXAMPLES,
            "format_instructions": parser.get_format_instructions(),
            "image_type": image.content_type,
            "image_data": img_str
        })
        
        return result.dict()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )


# Define the output structure using Pydantic
class RecipeOutput(BaseModel):
    ingredients: List[str] = Field(description="List of ingredients for the recipe")
    steps: List[str] = Field(description="List of steps to prepare the recipe")

# Create the output parser
recipe_parser = PydanticOutputParser(pydantic_object=RecipeOutput)

RECIPE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a master chef. Your task is to create an original, easy-to-follow recipe based on the provided ingredients.
        Follow these rules:
        1. Use all the ingredients provided.
        2. Provide clear and concise steps.
        3. Ensure the recipe is easy to follow.
        4. Return the output in JSON format with 'ingredients' and 'steps' keys.
        
        
        Please generate a recipe in the following JSON format:
        {
            "ingredients": ["list of ingredients"],
            "steps": ["step 1", "step 2", "step 3"]
        }

        Example 1:
        Ingredients: ["2 strawberries", "3 pomegranates", "2 cups water"]
        Output: {
            "ingredients": ["2 strawberries", "3 pomegranates", "2 cups water"],
            "steps": ["Mix the strawberries and pomegranates", "Add water and stir"]
        }

        Example 2:
        Ingredients: ["1 apple", "1 banana", "1 cup yogurt"]
        Output: {
            "ingredients": ["1 apple", "1 banana", "1 cup yogurt"],
            "steps": ["Chop the apple and banana", "Mix with yogurt"]
        }
        {format_instructions}"""
    ),
    (
        "human",
        """
        Now, please generate a recipe for the given ingredients.
        Ingredients: {ingredients}"""
    )
])

# Define a Pydantic model for the request body
class IngredientsRequest(BaseModel):
    ingredients: List[str]

# Configure logging

@app.post("/generate-recipe")
async def generate_recipe(request: IngredientsRequest):
    """
    Generate a recipe from a list of ingredients using LangChain and Gemini
    """
    try:
        # Log the ingredients received

        if not request.ingredients:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ingredients list cannot be empty"
            )

        # Initialize the Gemini model
        model = ChatVertexAI(
            model="gemini-2.0-flash",
            # model="gemini-2.0-flash",
            # text="Identify the ingredients in the image",
            # temperature=0.1,
            # google_api_key=GOOGLE_API_KEY,
            # convert_system_message_to_human=True
        ) 
        
        # Create the chain
        chain = RECIPE_PROMPT | model | recipe_parser
        print(request.ingredients)
        # Prepare the input for the LLM
        input_data = {
            "format_instructions": recipe_parser.get_format_instructions(),
            "ingredients": ", ".join(request.ingredients) # Ensure this matches the expected variable name
              # Add format instructions
        }
        
        # Log the input data
        print(input_data)
        # Run the chain
        raw_result = chain.invoke(input_data)

        # Log the raw result
        # Check if the result is empty
        if not raw_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Received empty response from LLM"
            )

        # Parse the result
        return raw_result.dict()
        
    except Exception as e:
        # Log the error with traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recipe: {str(e)}"
        )