from pydantic import BaseModel

class UserEmailAndPassword(BaseModel):
    email: str
    password: str
