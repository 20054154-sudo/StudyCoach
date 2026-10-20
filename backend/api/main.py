

from contextlib import asynccontextmanager
import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.api.auth import router as auth_router
from backend.api.exams import router as exams_router
from backend.api.sessions import router as sessions_router
from backend.api.frontend_extras import router as frontend_extras_router
from backend.api.session_extras import router as session_extras_router
from backend.db.models import init_db
from frontend.web import router as web_router
from backend.session_cookie import COOKIE_NAME, verify_token
from backend.observability import start_trace, end_trace, trace_event

# ---------------------------------------------------------------------------
# Inizializzazione app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Inizializza risorse all'avvio dell'applicazione."""
    init_db()
    print("[StudyCoach API] Database inizializzato. Server pronto.")
    yield


app = FastAPI(
    title="Study Coach API",
    description=(
        "API per il tutor AI universitario basato su RAG. "
        "Supporta autenticazione studenti, gestione esami, "
        "indicizzazione PDF, generazione quiz e loop socratico adattivo."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Origini aggiuntive per eventuali client API separati.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    trace_id, token, started = start_trace()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        trace_event("http", method=request.method, path=request.url.path,
                    status=response.status_code, elapsed_ms=round((time.perf_counter() - started) * 1000))
        return response
    except Exception:
        trace_event("http_error", method=request.method, path=request.url.path,
                    elapsed_ms=round((time.perf_counter() - started) * 1000))
        raise
    finally:
        end_trace(token)


@app.middleware("http")
async def protect_web_forms(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/web/"):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return PlainTextResponse("Origine della richiesta non valida.", status_code=403)
    return await call_next(request)



# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(exams_router)
app.include_router(sessions_router)
app.include_router(frontend_extras_router)
app.include_router(session_extras_router)
app.include_router(web_router)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parents[2] / "frontend" / "static")), name="static")


@app.get("/", include_in_schema=False)
def home(request: Request):
    target = "/web/exams" if verify_token(request.cookies.get(COOKIE_NAME, "")) else "/web/login"
    return RedirectResponse(target, status_code=303)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/logo.svg", status_code=307)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"], summary="Health check")
def health() -> dict:
    return {"status": "ok", "service": "Study Coach API"}
