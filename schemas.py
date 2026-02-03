from datetime import date, datetime
from pydantic import BaseModel
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
    student_id: Optional[int] = None
    roll_no: Optional[str] = None
    date: Optional[date] = None
    status: str = "present"
    teacher_email: str
    confidence: float
    timestamp: Optional[datetime] = None


class Token(BaseModel):
    token: str


class AttendanceScriptTrigger(BaseModel):
    script_path: Optional[str] = None
    arguments: Optional[List[str]] = None


class AttendanceProcessRequest(BaseModel):
    teacher_email: str
    status: str = "present"


class LightToggle(BaseModel):
    status: bool
    script_path: Optional[str] = None
    arguments: Optional[List[str]] = None


class AdminCreate(BaseModel):
    name: str
    email: str
    password: str


class AdminLogin(BaseModel):
    email: str
    password: str


class AllowedTeacherEmailBase(BaseModel):
    email: str


class AllowedTeacherEmailCreate(AllowedTeacherEmailBase):
    pass


class AllowedTeacherEmail(AllowedTeacherEmailBase):
    id: int

    class Config:
        orm_mode = True
