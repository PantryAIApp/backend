from pydantic import BaseModel
from typing import List, Optional

class UserEmailAndPassword(BaseModel):
    email: str
    password: str


class Recipe(BaseModel):
    id: Optional[str] = None
    name: str
    ingredients: List[str]
    instructions: List[str]
    image_url: Optional[str] = None
    time_required: Optional[int]  
    difficulty: Optional[str]  
    created_by: Optional[str]