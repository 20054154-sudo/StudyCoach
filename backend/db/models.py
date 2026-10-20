from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Index, event
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

_DB_DIR = Path(__file__).parent.parent.parent / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)

def now_utc():
    return datetime.now(timezone.utc)

class Student(Base):
    __tablename__ = "students"
    id            = Column(Integer, primary_key=True)
    username      = Column(String(50), unique=True, nullable=False)
    email         = Column(String(320), nullable=True)
    password_hash = Column(String(128), nullable=False)
    session_token = Column(String(64), unique=True, nullable=True)
    session_token_hash = Column(String(100), nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at    = Column(DateTime(timezone=True), default=now_utc)
    exams         = relationship("Exam", back_populates="student", cascade="all, delete")


class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    scope_key = Column(String(64), primary_key=True)
    failures = Column(Integer, nullable=False)
    window_started = Column(Float, nullable=False)
    blocked_until = Column(Float, nullable=False, default=0.0)

class Exam(Base):
    __tablename__ = "exams"
    id         = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    name       = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), default=now_utc)
    student    = relationship("Student", back_populates="exams")
    sessions   = relationship("Session", back_populates="exam", cascade="all, delete")
    materials  = relationship("Material", back_populates="exam", cascade="all, delete")
    doubts     = relationship("DoubtConversation", cascade="all, delete")
    doubt_threads = relationship("DoubtThread", cascade="all, delete")

class Material(Base):
    __tablename__ = "materials"
    id           = Column(Integer, primary_key=True)
    exam_id      = Column(Integer, ForeignKey("exams.id"), nullable=False)
    filename     = Column(String(255), nullable=False)
    stored_path  = Column(String(500), nullable=False)
    chunks_count = Column(Integer, default=0)
    pages_count  = Column(Integer, default=0)
    pages_with_text = Column(Integer, default=0)
    pages_skipped = Column(Integer, default=0)
    file_hash    = Column(String(64), nullable=True, index=True)
    uploaded_at  = Column(DateTime(timezone=True), default=now_utc)
    exam         = relationship("Exam", back_populates="materials")

class Session(Base):
    __tablename__ = "sessions"
    id          = Column(Integer, primary_key=True)
    exam_id     = Column(Integer, ForeignKey("exams.id"), nullable=False)
    student_id  = Column(Integer, nullable=True)
    started_at  = Column(DateTime(timezone=True), default=now_utc)
    ended_at    = Column(DateTime(timezone=True), nullable=True)
    active_started_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    score_total = Column(Float, nullable=True)
    qa_log      = Column(Text, nullable=True)
    quiz_json   = Column(Text, nullable=True)
    current_turn_json = Column(Text, nullable=True)
    report_path = Column(String(200), nullable=True)
    status      = Column(String(20), nullable=False, default="in_corso")
    exam        = relationship("Exam", back_populates="sessions")

class DoubtThread(Base):
    __tablename__ = "doubt_threads"
    id = Column(String(36), primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    title = Column(String(160), nullable=False, default="Nuova chat")
    created_at = Column(DateTime(timezone=True), default=now_utc)


class DoubtConversation(Base):
    __tablename__ = "doubt_conversations"
    id = Column(String(36), primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    thread_id = Column(String(36), ForeignKey("doubt_threads.id"), nullable=True)
    doubt_text = Column(Text, nullable=False)
    guiding_question = Column(Text, nullable=False)
    topic_hint = Column(String(200), nullable=False, default="")
    answered = Column(Integer, nullable=False, default=0)
    student_answer = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)

Index("ix_exams_student_id", Exam.student_id)
Index("ix_students_email_unique", Student.email, unique=True)
Index("ix_materials_exam_id", Material.exam_id)
Index("ix_sessions_exam_id", Session.exam_id)
engine = create_engine(f"sqlite:///{_DB_DIR / 'study_coach.db'}", echo=False)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

def init_db():
    Base.metadata.create_all(engine)
    _ensure_student_email_column()
    _ensure_session_token_column()
    _ensure_session_status_column()
    _ensure_material_file_hash_column()
    _ensure_material_page_stats_columns()
    _ensure_session_persistence_columns()
    _ensure_session_clock_columns()
    _ensure_student_token_security_columns()
    _ensure_doubt_history_columns()


def _ensure_student_email_column():
    from sqlalchemy import inspect, text
    columns = {column["name"] for column in inspect(engine).get_columns("students")}
    with engine.begin() as conn:
        if "email" not in columns:
            conn.execute(text("ALTER TABLE students ADD COLUMN email VARCHAR(320)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_students_email_unique "
                          "ON students (email)"))


def _ensure_doubt_history_columns():
    import uuid
    from sqlalchemy import inspect, text
    columns = {c["name"] for c in inspect(engine).get_columns("doubt_conversations")}
    with engine.begin() as conn:
        if "student_answer" not in columns:
            conn.execute(text("ALTER TABLE doubt_conversations ADD COLUMN student_answer TEXT"))
        if "result_json" not in columns:
            conn.execute(text("ALTER TABLE doubt_conversations ADD COLUMN result_json TEXT"))
        if "created_at" not in columns:
            conn.execute(text("ALTER TABLE doubt_conversations ADD COLUMN created_at DATETIME"))
        if "thread_id" not in columns:
            conn.execute(text("ALTER TABLE doubt_conversations ADD COLUMN thread_id VARCHAR(36)"))
        legacy = conn.execute(text("SELECT student_id, exam_id, MIN(doubt_text) FROM doubt_conversations WHERE thread_id IS NULL GROUP BY student_id, exam_id")).all()
        for student_id, exam_id, title in legacy:
            thread_id = str(uuid.uuid4())
            conn.execute(text("INSERT INTO doubt_threads (id, student_id, exam_id, title, created_at) VALUES (:id, :student_id, :exam_id, :title, CURRENT_TIMESTAMP)"),
                         {"id": thread_id, "student_id": student_id, "exam_id": exam_id, "title": (title or "Chat precedente")[:160]})
            conn.execute(text("UPDATE doubt_conversations SET thread_id=:thread_id WHERE student_id=:student_id AND exam_id=:exam_id AND thread_id IS NULL"),
                         {"thread_id": thread_id, "student_id": student_id, "exam_id": exam_id})


def _ensure_student_token_security_columns():
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "students" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("students")]
    with engine.begin() as conn:
        if "session_token_hash" not in columns:
            conn.execute(text("ALTER TABLE students ADD COLUMN session_token_hash VARCHAR(100)"))
        if "token_expires_at" not in columns:
            conn.execute(text("ALTER TABLE students ADD COLUMN token_expires_at DATETIME"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_students_session_token_hash ON students (session_token_hash)"))


def _ensure_session_persistence_columns():
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "sessions" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("sessions")]
    with engine.begin() as conn:
        if "quiz_json" not in columns:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN quiz_json TEXT"))
        if "current_turn_json" not in columns:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN current_turn_json TEXT"))
        if "student_id" not in columns:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN student_id INTEGER"))


def _ensure_session_clock_columns():
    from sqlalchemy import inspect, text
    columns = {c["name"] for c in inspect(engine).get_columns("sessions")}
    with engine.begin() as conn:
        if "duration_seconds" not in columns:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN duration_seconds FLOAT NOT NULL DEFAULT 0"))
            conn.execute(text("UPDATE sessions SET duration_seconds = MAX(0, "
                              "(julianday(ended_at) - julianday(started_at)) * 86400) "
                              "WHERE ended_at IS NOT NULL AND started_at IS NOT NULL"))
        if "active_started_at" not in columns:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN active_started_at DATETIME"))
            conn.execute(text("UPDATE sessions SET active_started_at = started_at "
                              "WHERE status = 'in_corso' AND started_at IS NOT NULL"))


def _ensure_session_token_column():
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "students" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("students")]
    if "session_token" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE students ADD COLUMN session_token VARCHAR(64)"))


def _ensure_session_status_column():
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "sessions" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("sessions")]
    if "status" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'in_corso'"))


def _ensure_material_file_hash_column():
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "materials" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("materials")]
    if "file_hash" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE materials ADD COLUMN file_hash VARCHAR(64)"))


def _ensure_material_page_stats_columns():
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "materials" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("materials")]
    with engine.begin() as conn:
        if "pages_with_text" not in columns:
            conn.execute(text("ALTER TABLE materials ADD COLUMN pages_with_text INTEGER DEFAULT 0"))
        if "pages_skipped" not in columns:
            conn.execute(text("ALTER TABLE materials ADD COLUMN pages_skipped INTEGER DEFAULT 0"))

SessionLocal = sessionmaker(bind=engine)
