from typing import Annotated, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials, firestore
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
import os
from dotenv import dotenv_values
from app.models import UserEmailAndPassword
import requests



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

# cred = credentials.Certificate("/app/service-account.json") # for prod
cred = credentials.Certificate("./service-account.json") # for local development

# This part might be needed
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account.json" # for local dev
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service-account.json" # for prod


config = dotenv_values(".env") # for local
# config = dotenv_values("/app/.env") # for prod

FIREBASE_API_KEY = config.get("FIREBASE_API_KEY", "")

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
