from pydantic import BaseModel
from datetime import date
from typing import List, Optional

class Login(BaseModel):
    email: str
    password: str


class StudentCreate(BaseModel):
    name: str
    roll_no: str
    email: str
    password: str

class TeacherCreate(BaseModel):
    name: str
    email: str
    password: str

class AttendanceConfirm(BaseModel):
    student_id: int
    date: date
    status: str


class Token(BaseModel):
    token: str


class AttendanceScriptTrigger(BaseModel):
    script_path: Optional[str] = None
    arguments: Optional[List[str]] = None


class LightToggle(BaseModel):
    status: bool
    script_path: Optional[str] = None
    arguments: Optional[List[str]] = None
