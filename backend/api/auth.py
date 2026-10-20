
from fastapi import APIRouter, HTTPException, Response, Request, status
from sqlalchemy.exc import IntegrityError

from backend.db.models import SessionLocal, Student, init_db
from backend.models.schemas import AuthResponse, LoginRequest, RegisterRequest
from backend.security import hash_password, verify_password
from backend.login_limiter import retry_after, record_failure, clear_account
from backend.session_cookie import COOKIE_NAME, COOKIE_MAX_AGE, get_or_create_token, invalidate_token, verify_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=AuthResponse, summary="Sessione corrente")
def current_user(request: Request) -> AuthResponse:
    student = verify_token(request.cookies.get(COOKIE_NAME, ""))
    if student is None:
        raise HTTPException(status_code=401, detail="Accesso richiesto.")
    return AuthResponse(student_id=student.id, username=student.username, message="Sessione attiva.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_student_by_username(username: str) -> Student | None:
    db = SessionLocal()
    try:
        return db.query(Student).filter(Student.username == username).first()
    finally:
        db.close()


def _get_student_by_email(email: str) -> Student | None:
    db = SessionLocal()
    try:
        return db.query(Student).filter(Student.email == email).first()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registra un nuovo studente",
)
def register(payload: RegisterRequest, response: Response) -> AuthResponse:
    """
    Crea un nuovo account studente.
    - Username deve essere unico e lungo almeno 3 caratteri.
    - Password deve essere lunga almeno 6 caratteri.
    """
    init_db()
    existing = _get_student_by_username(payload.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{payload.username}' già in uso.",
        )
    if _get_student_by_email(payload.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Questa email è già registrata.")

    db = SessionLocal()
    try:
        student = Student(
            username=payload.username,
            email=payload.email,
            password_hash=hash_password(payload.password),
        )
        db.add(student)
        db.commit()
        db.refresh(student)
        token = get_or_create_token(student.id)
        response.set_cookie(COOKIE_NAME, token, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
        return AuthResponse(
            student_id=student.id,
            username=student.username,
            message="Account creato con successo.",
        )
    except IntegrityError as exc:
        db.rollback()
        if "email" in str(exc).lower():
            detail = "Questa email è già registrata."
        else:
            detail = f"Username '{payload.username}' già in uso."
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        ) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Errore database durante la registrazione.",
        ) from exc
    finally:
        db.close()


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login studente",
)
def login(payload: LoginRequest, response: Response, request: Request) -> AuthResponse:
    """
    Verifica le credenziali e crea un cookie di sessione HttpOnly.
    """
    username = payload.username.strip()
    origin = request.client.host if request.client else "unknown"
    wait = retry_after(username, origin)
    if wait:
        raise HTTPException(status_code=429, detail="Troppi tentativi di accesso. Riprova più tardi.",
                            headers={"Retry-After": str(wait)})
    student = _get_student_by_username(username)
    if not student or not verify_password(payload.password, student.password_hash):
        record_failure(username, origin)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide.",
        )
    clear_account(username, origin)
    token = get_or_create_token(student.id)
    response.set_cookie(COOKIE_NAME, token, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    return AuthResponse(
        student_id=student.id,
        username=student.username,
        message="Login effettuato con successo.",
    )


@router.post(
    "/logout",
    summary="Revoca la sessione API e cancella il cookie",
)
def logout(request: Request, response: Response) -> dict:
    authorization = request.headers.get("authorization", "")
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else request.cookies.get(COOKIE_NAME, "")
    student = verify_token(token)
    if student:
        invalidate_token(student.id)
    response.delete_cookie(COOKIE_NAME)
    return {"message": "Logout effettuato. Pulisci il token lato client."}
