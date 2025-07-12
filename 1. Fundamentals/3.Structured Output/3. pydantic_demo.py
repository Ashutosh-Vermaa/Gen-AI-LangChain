from pydantic import BaseModel
from typing import Optional, EmailStr

class Student(BaseModel):
    name: str = 'Ash' # setting a default value of name var/attribute
    age: Optional[int]= None # Optional makes the parameter optional and is assigned None by default
    email: EmailStr

new_student={'age':'32'} # pydantic does type conversion and if gets the value doesn't throw error

student=Student(**new_student)

print(student)