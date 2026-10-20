"""Unit/integration test — backend/db/models.py."""

from backend.db.models import DoubtConversation, DoubtThread, Exam, Material, SessionLocal, Session as DbSession, Student, init_db


def test_init_db_is_idempotent():
    init_db()
    init_db()


def test_session_clock_columns_migrate_existing_sessions(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, inspect, text
    from backend.db import models
    old_engine = create_engine(f"sqlite:///{tmp_path / 'old_sessions.db'}")
    with old_engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id INTEGER PRIMARY KEY, started_at DATETIME, "
                          "ended_at DATETIME, status VARCHAR(20))"))
        conn.execute(text("INSERT INTO sessions VALUES "
                          "(1, '2026-10-13 10:00:00', '2026-10-13 10:01:30', 'completata'), "
                          "(2, '2026-10-13 10:00:00', NULL, 'in_corso')"))
    monkeypatch.setattr(models, "engine", old_engine)
    models._ensure_session_clock_columns()
    models._ensure_session_clock_columns()
    assert {"duration_seconds", "active_started_at"} <= {
        column["name"] for column in inspect(old_engine).get_columns("sessions")}
    with old_engine.connect() as conn:
        rows = conn.execute(text("SELECT duration_seconds, active_started_at "
                                 "FROM sessions ORDER BY id")).all()
    assert round(rows[0][0]) == 90
    assert rows[1][1] == "2026-10-13 10:00:00"


def test_token_index_is_added_after_legacy_student_columns(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, inspect, text
    from backend.db import models
    old_engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with old_engine.begin() as conn:
        conn.execute(text("CREATE TABLE students (id INTEGER PRIMARY KEY, username VARCHAR(50), password_hash VARCHAR(128))"))
        conn.execute(text("INSERT INTO students (username, password_hash) VALUES ('legacy', 'hash')"))
    monkeypatch.setattr(models, "engine", old_engine)
    models.init_db()
    columns = {column["name"] for column in inspect(old_engine).get_columns("students")}
    indexes = {index["name"] for index in inspect(old_engine).get_indexes("students")}
    assert {"session_token_hash", "token_expires_at", "email"} <= columns
    assert "ix_students_session_token_hash" in indexes
    assert "ix_students_email_unique" in indexes
    with old_engine.connect() as conn:
        assert conn.execute(text("SELECT username, email FROM students")).one() == ("legacy", None)


def test_existing_doubts_are_grouped_without_losing_messages():
    with SessionLocal() as db:
        student = Student(username="storico", password_hash="hash")
        db.add(student)
        db.flush()
        exam = Exam(student_id=student.id, name="Fisica")
        db.add(exam)
        db.flush()
        db.add_all([
            DoubtConversation(id="legacy-1", student_id=student.id, exam_id=exam.id,
                              doubt_text="Prima domanda", guiding_question="Prima guida"),
            DoubtConversation(id="legacy-2", student_id=student.id, exam_id=exam.id,
                              doubt_text="Seconda domanda", guiding_question="Seconda guida"),
        ])
        db.commit()
    init_db()
    init_db()
    with SessionLocal() as db:
        messages = db.query(DoubtConversation).order_by(DoubtConversation.id).all()
        assert len(messages) == 2
        assert messages[0].thread_id == messages[1].thread_id
        assert db.query(DoubtThread).filter_by(id=messages[0].thread_id).count() == 1


def test_create_student_and_exam_relationship():
    db = SessionLocal()
    try:
        student = Student(username="mario", password_hash="hash")
        db.add(student)
        db.commit()
        db.refresh(student)

        exam = Exam(student_id=student.id, name="Analisi 1")
        db.add(exam)
        db.commit()
        db.refresh(student)

        assert len(student.exams) == 1
        assert student.exams[0].name == "Analisi 1"
    finally:
        db.close()


def test_deleting_student_cascades_to_exams():
    db = SessionLocal()
    try:
        student = Student(username="mario", password_hash="hash")
        db.add(student)
        db.commit()
        db.refresh(student)

        exam = Exam(student_id=student.id, name="Analisi 1")
        db.add(exam)
        db.commit()
        exam_id = exam.id

        db.delete(student)
        db.commit()

        assert db.query(Exam).filter(Exam.id == exam_id).first() is None
    finally:
        db.close()


def test_deleting_exam_cascades_to_sessions_and_materials():
    db = SessionLocal()
    try:
        student = Student(username="mario", password_hash="hash")
        db.add(student)
        db.commit()
        db.refresh(student)

        exam = Exam(student_id=student.id, name="Analisi 1")
        db.add(exam)
        db.commit()
        db.refresh(exam)

        sess = DbSession(exam_id=exam.id)
        material = Material(exam_id=exam.id, filename="a.pdf", stored_path="/tmp/a.pdf")
        db.add_all([sess, material])
        db.commit()
        sess_id, material_id = sess.id, material.id

        db.delete(exam)
        db.commit()

        assert db.query(DbSession).filter(DbSession.id == sess_id).first() is None
        assert db.query(Material).filter(Material.id == material_id).first() is None
    finally:
        db.close()


def test_student_username_unique_constraint():
    db = SessionLocal()
    try:
        db.add(Student(username="mario", password_hash="h1"))
        db.commit()

        db.add(Student(username="mario", password_hash="h2"))
        try:
            db.commit()
            assert False, "doveva sollevare un errore di integrita'"
        except Exception:
            db.rollback()
    finally:
        db.close()


def test_session_defaults_and_optional_fields():
    db = SessionLocal()
    try:
        student = Student(username="mario", password_hash="hash")
        db.add(student)
        db.commit()
        db.refresh(student)

        exam = Exam(student_id=student.id, name="Analisi 1")
        db.add(exam)
        db.commit()
        db.refresh(exam)

        sess = DbSession(exam_id=exam.id)
        db.add(sess)
        db.commit()
        db.refresh(sess)

        assert sess.started_at is not None
        assert sess.ended_at is None
        assert sess.score_total is None
        assert sess.qa_log is None
    finally:
        db.close()
