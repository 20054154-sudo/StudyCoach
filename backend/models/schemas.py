"""
backend/models/schemas.py — Schemi Pydantic per l'API Study Coach

"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ===========================================================================
# AUTH
# ===========================================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, examples=["mario_rossi"])
    email: str = Field(..., max_length=320, examples=["mario@example.com"])
    password: str = Field(..., min_length=6, examples=["secret123"])
    password_confirm: str = Field(..., min_length=6, exclude=True)

    @field_validator("password", "password_confirm")
    @classmethod
    def password_fits_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("La password non può superare 72 byte UTF-8.")
        return value

    @field_validator("username")
    @classmethod
    def username_strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        email = value.strip().casefold()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise ValueError("Inserisci un indirizzo email valido.")
        return email

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.password_confirm:
            raise ValueError("Le password non coincidono.")
        return self


class LoginRequest(BaseModel):
    username: str = Field(..., examples=["mario_rossi"])
    password: str = Field(..., examples=["secret123"])


class AuthResponse(BaseModel):
    student_id: int
    username: str
    message: str


# ===========================================================================
# EXAMS
# ===========================================================================

class ExamCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["Programmazione AI a.a. 2025/26"])

    @field_validator("name")
    @classmethod
    def name_strip(cls, v: str) -> str:
        return v.strip()


class ExamResponse(BaseModel):
    id: int
    student_id: int
    name: str
    created_at: Optional[datetime] = None
    n_sessions: int = 0
    avg_score: Optional[float] = None

    model_config = {"from_attributes": True}


class ExamListResponse(BaseModel):
    exams: list[ExamResponse]
    total: int


class IngestResponse(BaseModel):
    file: str
    pages: int
    chunks: int
    collection: str
    message: str = "PDF indicizzato con successo"


# ===========================================================================
# QUIZ
# ===========================================================================

class QuizGenerateRequest(BaseModel):
    topic_query: str = Field(
        default="concetti principali del corso",
        examples=["meccanismi di attention nei Transformer"],
    )
    n_questions: int = Field(default=5, ge=1, le=20)
    source_name: str = Field(default="materiale di studio", examples=["Slide Lezione 3"])
    material_ids: Optional[list[int]] = Field(default=None, description="ID delle fonti da usare")


class Difficulty(str, Enum):
    base = "base"
    intermedia = "intermedia"
    avanzata = "avanzata"


class QuestionOut(BaseModel):
    id: int
    text: str
    difficulty: Difficulty
    topic_hint: str = Field(..., max_length=200)


class QuizGenerateResponse(BaseModel):
    exam_id: int
    student_id: int
    source_name: str
    n_questions: int
    prompt_version: str
    questions: list[QuestionOut]


# ===========================================================================
# SESSIONS
# ===========================================================================

class SessionCreate(BaseModel):
    """Apre una nuova sessione per un esame."""
    exam_id: int
    material_ids: Optional[list[int]] = None


class SessionResponse(BaseModel):
    id: int
    exam_id: int
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    score_total: Optional[float] = None
    report_path: Optional[str] = None
    report_available: bool = False
    has_attempts: bool = False
    has_quiz: bool = False
    status: str = "in_corso"

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


# ===========================================================================
# SOCRATIC TUTOR — step-by-step loop
# ===========================================================================

class TutorStartRequest(BaseModel):
    """Avvia il loop socratico per una singola domanda."""
    session_id: int
    question: QuestionOut


class TutorStartResponse(BaseModel):
    question_text: str
    difficulty: str
    topic_hint: str
    message: str = "Rispondi alla domanda."


class Answer1Request(BaseModel):
    session_id: int
    question_id: int
    answer: str = Field(..., min_length=1, max_length=10000)


class Answer1Response(BaseModel):
    guiding_question: str
    positive_feedback: str
    focus_area: str
    message: str = "Rifletti sulla domanda guida e rispondi di nuovo."


class Answer2Request(BaseModel):
    session_id: int
    question_id: int
    answer: str = Field(..., min_length=1, max_length=10000)


class Answer2Response(BaseModel):
    score: int = Field(..., ge=0, le=10)
    motivation: str
    correct_points: list[str]
    missing_points: list[str]
    improvement_after_guidance: str
    reliable: bool = True
    can_contest: bool = True


class ContestRequest(BaseModel):
    session_id: int
    question_id: int
    argumentation: str = Field(..., min_length=10, max_length=5000)


class ContestResponse(BaseModel):
    final_score: int = Field(..., ge=0, le=10)
    score_changed: bool
    score_delta: int
    contestation_assessment: str
    final_motivation: str
    response_to_student: str


class CloseQuestionRequest(BaseModel):
    session_id: int
    question_id: int


class CloseQuestionResponse(BaseModel):
    final_score: int
    topic_hint: str
    message: str = "Domanda chiusa. Prossima domanda disponibile."


# ===========================================================================
# ADAPTIVE ENGINE
# ===========================================================================

class NextQuestionRequest(BaseModel):
    session_id: int
    remaining_questions: list[QuestionOut]


class NextQuestionResponse(BaseModel):
    question: Optional[QuestionOut] = None
    is_last: bool = False
    message: str = ""


# ===========================================================================
# REPORT
# ===========================================================================

class SessionCloseRequest(BaseModel):
    session_id: int
    exam_name: str = "Esame"


class TopicBreakdown(BaseModel):
    topic: str
    avg_score: float
    n_attempts: int
    is_weak: bool


class SessionReportResponse(BaseModel):
    session_id: int
    student_id: int
    exam_id: int
    overall_avg: float
    n_questions_answered: int
    weak_topics: list[str]
    strong_topics: list[str]
    topic_breakdown: list[TopicBreakdown]
    summary: str
    report_md_path: Optional[str] = None
    report_pdf_path: Optional[str] = None


# ===========================================================================
# ERRORI
# ===========================================================================

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
