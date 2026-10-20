"""Integration test — backend/api/exams.py (via FastAPI TestClient, DB e Chroma isolati)."""

from pathlib import Path


def test_data_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    from backend.agents import ingestor
    from backend.api import exams

    project_root = Path(exams.__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    assert Path(ingestor.CHROMA_PATH) == project_root / "data" / "chroma"
    assert exams.DEFAULT_UPLOAD_DIR == project_root / "data" / "uploads"


def _register(api_client, username="mario_rossi") -> int:
    resp = api_client.post("/auth/register", json={"username": username, "email": f"{username}@example.com", "password": "secret123", "password_confirm": "secret123"})
    return resp.json()["student_id"]


def test_create_and_list_exam(api_client):
    student_id = _register(api_client)

    resp = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"})
    assert resp.status_code == 201
    exam = resp.json()
    assert exam["name"] == "Analisi 1"
    assert exam["n_sessions"] == 0
    assert exam["avg_score"] is None

    resp = api_client.get("/exams", params={"student_id": student_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["exams"][0]["name"] == "Analisi 1"


def test_exams_require_authentication(api_client):
    assert api_client.get("/exams", params={"student_id": 1}).status_code == 401


def test_exam_owner_cannot_be_spoofed_by_student_id(api_client):
    owner_id = _register(api_client, "owner_user")
    exam_id = api_client.post("/exams", params={"student_id": owner_id}, json={"name": "Privato"}).json()["id"]
    other_id = _register(api_client, "other_user")
    assert api_client.get("/exams", params={"student_id": owner_id}).status_code == 404
    assert api_client.delete(f"/exams/{exam_id}", params={"student_id": other_id}).status_code == 404
    assert api_client.post(
        f"/exams/{exam_id}/ingest", params={"student_id": other_id},
        files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
    ).status_code == 404


def test_list_exams_isolated_per_student(api_client):
    student_a = _register(api_client, "mario")
    student_b = _register(api_client, "giulia")

    api_client.post("/auth/login", json={"username": "mario", "email": "mario@example.com", "password": "secret123", "password_confirm": "secret123"})
    api_client.post("/exams", params={"student_id": student_a}, json={"name": "Esame A"})

    api_client.post("/auth/login", json={"username": "giulia", "email": "giulia@example.com", "password": "secret123", "password_confirm": "secret123"})
    resp_b = api_client.get("/exams", params={"student_id": student_b})
    assert resp_b.json()["total"] == 0


def test_create_exam_rejects_empty_name(api_client):
    student_id = _register(api_client)
    resp = api_client.post("/exams", params={"student_id": student_id}, json={"name": ""})
    assert resp.status_code == 422


def test_delete_exam_removes_it(api_client):
    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}).json()["id"]

    resp = api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})
    assert resp.status_code == 204

    resp = api_client.get("/exams", params={"student_id": student_id})
    assert resp.json()["total"] == 0


def test_delete_exam_removes_its_report_files(api_client, isolated_reports_dir):
    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    report = isolated_reports_dir / f"{student_id}_{exam_id}_7.md"
    report.write_text("report")
    unrelated = isolated_reports_dir / f"{student_id}_{exam_id + 1}_7.md"
    unrelated.write_text("other")
    assert api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id}).status_code == 204
    assert not report.exists()
    assert unrelated.exists()


def test_delete_exam_removes_uploaded_pdf(api_client, sample_pdf_path):
    from backend.db.models import Material, SessionLocal

    student_id = _register(api_client, "delete_upload_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    with sample_pdf_path.open("rb") as file:
        response = api_client.post(
            f"/exams/{exam_id}/ingest", params={"student_id": student_id},
            files={"file": ("dispensa.pdf", file, "application/pdf")},
        )
    assert response.status_code == 200
    with SessionLocal() as db:
        stored = Path(db.query(Material).filter(Material.exam_id == exam_id).one().stored_path)
    assert stored.exists()
    assert api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id}).status_code == 204
    assert not stored.exists()


def test_delete_exam_chroma_failure_preserves_db_and_files(api_client, isolated_reports_dir, monkeypatch):
    from backend.db.models import Exam, SessionLocal
    from backend import exam_deletion

    student_id = _register(api_client, "delete_failure_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    report = isolated_reports_dir / f"{student_id}_{exam_id}_1.md"
    report.write_text("existing report")
    exam_deletion.ingestor.get_collection(student_id, exam_id)

    def fail_delete(*args):
        raise OSError("SECRET storage path")

    monkeypatch.setattr(exam_deletion.ingestor, "delete_exam_collection", fail_delete)
    response = api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})
    assert response.status_code == 500
    assert "SECRET" not in response.text
    with SessionLocal() as db:
        assert db.get(Exam, exam_id) is not None
    assert report.read_text() == "existing report"


def test_delete_exam_restores_index_if_chroma_fails_after_delete(api_client, monkeypatch):
    from backend import exam_deletion
    from backend.db.models import Exam, SessionLocal

    student_id = _register(api_client, "delete_late_chroma_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    exam_deletion.ingestor.get_collection(student_id, exam_id).add(
        ids=["passage"], documents=["contenuto utile"], metadatas=[{"source_file": "test.pdf"}]
    )
    real_delete = exam_deletion.ingestor.delete_exam_collection

    def fail_after_delete(*args):
        real_delete(*args)
        raise OSError("late SECRET")

    monkeypatch.setattr(exam_deletion.ingestor, "delete_exam_collection", fail_after_delete)
    response = api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})
    assert response.status_code == 500
    with SessionLocal() as db:
        assert db.get(Exam, exam_id) is not None
    assert exam_deletion.ingestor.get_collection(student_id, exam_id).get(ids=["passage"])["ids"] == ["passage"]


def test_delete_exam_reports_incomplete_recovery(api_client, monkeypatch):
    from backend import exam_deletion

    student_id = _register(api_client, "delete_recovery_failure_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    exam_deletion.ingestor.get_collection(student_id, exam_id)
    real_delete = exam_deletion.ingestor.delete_exam_collection

    def fail_after_delete(*args):
        real_delete(*args)
        raise OSError("SECRET late failure")

    monkeypatch.setattr(exam_deletion.ingestor, "delete_exam_collection", fail_after_delete)
    monkeypatch.setattr(exam_deletion.ingestor.client, "get_or_create_collection", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("SECRET recovery")))
    response = api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})
    assert response.status_code == 500
    assert "ripristino incompleto" in response.json()["detail"]
    assert "SECRET" not in response.text


def test_delete_exam_database_failure_restores_index_and_files(api_client, isolated_reports_dir, monkeypatch):
    from backend import exam_deletion
    from backend.db.models import Exam, SessionLocal

    student_id = _register(api_client, "delete_db_failure_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    report = isolated_reports_dir / f"{student_id}_{exam_id}_1.md"
    report.write_text("existing report")
    collection = exam_deletion.ingestor.get_collection(student_id, exam_id)
    collection.add(ids=["passage"], documents=["contenuto utile"], metadatas=[{"source_file": "test.pdf"}])
    real_session_local = exam_deletion.SessionLocal

    def failing_session():
        db = real_session_local()
        def fail_commit():
            raise OSError("database SECRET")
        monkeypatch.setattr(db, "commit", fail_commit)
        return db

    monkeypatch.setattr(exam_deletion, "SessionLocal", failing_session)
    response = api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})
    assert response.status_code == 500
    assert "SECRET" not in response.text
    with SessionLocal() as db:
        assert db.get(Exam, exam_id) is not None
    assert report.read_text() == "existing report"
    assert exam_deletion.ingestor.get_collection(student_id, exam_id).get(ids=["passage"])["ids"] == ["passage"]


def test_delete_exam_reports_partial_cleanup_truthfully(api_client, isolated_reports_dir, monkeypatch):
    from backend.db.models import Exam, SessionLocal

    student_id = _register(api_client, "delete_cleanup_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    report = isolated_reports_dir / f"{student_id}_{exam_id}_1.md"
    report.write_text("report")
    real_unlink = Path.unlink

    def fail_staged_unlink(path, *args, **kwargs):
        if ".delete-" in path.name:
            raise OSError("SECRET cleanup")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_staged_unlink)
    response = api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})
    assert response.status_code == 500
    assert "Esame eliminato" in response.json()["detail"]
    assert "SECRET" not in response.text
    with SessionLocal() as db:
        assert db.get(Exam, exam_id) is None


def test_delete_unknown_exam_returns_404(api_client):
    student_id = _register(api_client)
    resp = api_client.delete("/exams/999999", params={"student_id": student_id})
    assert resp.status_code == 404


def test_ingest_pdf_indexes_material(api_client, sample_pdf_path):
    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}).json()["id"]

    with open(sample_pdf_path, "rb") as f:
        resp = api_client.post(
            f"/exams/{exam_id}/ingest",
            params={"student_id": student_id},
            files={"file": (sample_pdf_path.name, f, "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"] > 0
    assert body["pages"] >= 1
    assert body["collection"] == f"student_{student_id}_exam_{exam_id}"


def test_ingest_database_failure_removes_file_and_new_vectors(api_client, sample_pdf_path, isolated_upload_dir, monkeypatch):
    from backend.api import exams as exams_api
    from backend.agents.ingestor import get_collection
    from backend.db.models import Material, SessionLocal

    student_id = _register(api_client, "ingest_db_failure_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    original_session_local = exams_api.SessionLocal

    def failing_session():
        db = original_session_local()
        def fail_commit():
            raise OSError("SECRET database")
        monkeypatch.setattr(db, "commit", fail_commit)
        return db

    monkeypatch.setattr(exams_api, "SessionLocal", failing_session)
    with sample_pdf_path.open("rb") as file:
        response = api_client.post(
            f"/exams/{exam_id}/ingest", params={"student_id": student_id},
            files={"file": ("fonte.pdf", file, "application/pdf")},
        )
    assert response.status_code == 500
    assert "SECRET" not in response.text
    with SessionLocal() as db:
        assert db.query(Material).filter(Material.exam_id == exam_id).count() == 0
    assert not list(isolated_upload_dir.glob("*.pdf"))
    assert get_collection(student_id, exam_id).count() == 0


def test_ingest_rejects_non_pdf_file(api_client):
    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}).json()["id"]

    resp = api_client.post(
        f"/exams/{exam_id}/ingest",
        params={"student_id": student_id},
        files={"file": ("note.txt", b"non e' un pdf", "text/plain")},
    )
    assert resp.status_code == 400


def test_ingest_embedding_failure_gives_recovery_hint(api_client, monkeypatch):
    from backend.api import exams as exams_api
    from backend.agents.ingestor import EmbeddingModelUnavailable

    student_id = _register(api_client, "embedding_failure_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    def fail_ingest(*args, **kwargs):
        raise EmbeddingModelUnavailable("Modello di embedding non disponibile. Controlla il download.")
    monkeypatch.setattr(exams_api, "ingest_pdf", fail_ingest)
    response = api_client.post(
        f"/exams/{exam_id}/ingest", params={"student_id": student_id},
        files={"file": ("dispensa.pdf", b"%PDF-fake", "application/pdf")},
    )
    assert response.status_code == 500
    assert "download" in response.json()["detail"]


def test_ingest_rejects_pdf_extension_with_non_pdf_content(api_client):
    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}).json()["id"]
    response = api_client.post(
        f"/exams/{exam_id}/ingest", params={"student_id": student_id},
        files={"file": ("falso.pdf", b"plain text", "application/pdf")},
    )
    assert response.status_code == 400


def test_ingest_rejects_oversized_pdf_before_writing(api_client, monkeypatch):
    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}).json()["id"]
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "8")
    response = api_client.post(
        f"/exams/{exam_id}/ingest", params={"student_id": student_id},
        files={"file": ("big.pdf", b"%PDF-123456789", "application/pdf")},
    )
    assert response.status_code == 413


def test_ingest_unknown_exam_returns_404(api_client, sample_pdf_path):
    student_id = _register(api_client)
    with open(sample_pdf_path, "rb") as f:
        resp = api_client.post(
            "/exams/999999/ingest",
            params={"student_id": student_id},
            files={"file": (sample_pdf_path.name, f, "application/pdf")},
        )
    assert resp.status_code == 404


def test_delete_exam_also_removes_chroma_collection(api_client, sample_pdf_path):
    from backend.agents.ingestor import retrieve_chunks

    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi 1"}).json()["id"]

    with open(sample_pdf_path, "rb") as f:
        api_client.post(
            f"/exams/{exam_id}/ingest",
            params={"student_id": student_id},
            files={"file": (sample_pdf_path.name, f, "application/pdf")},
        )

    api_client.delete(f"/exams/{exam_id}", params={"student_id": student_id})

    chunks = retrieve_chunks("machine learning", student_id=student_id, exam_id=exam_id, n_results=3)
    assert chunks == []


def test_delete_selected_material_keeps_other_sources(api_client, sample_pdf_path, isolated_upload_dir):
    from fpdf import FPDF
    from backend.agents.ingestor import get_collection, verify_exam_material
    from backend.db.models import Material, SessionLocal

    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    second_pdf = isolated_upload_dir / "secondo.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "La statistica studia i dati e le distribuzioni. " * 20)
    pdf.output(str(second_pdf))
    for path in (sample_pdf_path, second_pdf):
        with path.open("rb") as file:
            response = api_client.post(
                f"/exams/{exam_id}/ingest", params={"student_id": student_id},
                files={"file": ("materiale.pdf", file, "application/pdf")},
            )
        assert response.status_code == 200

    with SessionLocal() as db:
        materials = db.query(Material).filter(Material.exam_id == exam_id).order_by(Material.id).all()
        removed_id, removed_hash, removed_path = materials[0].id, materials[0].file_hash, Path(materials[0].stored_path)
        kept_id, kept_hash, kept_path = materials[1].id, materials[1].file_hash, Path(materials[1].stored_path)

    response = api_client.delete(
        f"/exams/{exam_id}/materials/{removed_id}", params={"student_id": student_id}
    )
    assert response.status_code == 204
    with SessionLocal() as db:
        assert db.get(Material, removed_id) is None
        assert db.get(Material, kept_id) is not None
    assert not removed_path.exists()
    assert kept_path.exists()
    collection = get_collection(student_id, exam_id)
    assert not collection.get(where={"file_hash": removed_hash})["ids"]
    assert collection.get(where={"file_hash": kept_hash})["ids"]
    assert verify_exam_material(student_id, exam_id)["ok"]


def test_delete_material_commit_failure_restores_file_and_chunks(api_client, sample_pdf_path, monkeypatch):
    from backend import materials as material_ops
    from backend.agents.ingestor import get_collection
    from backend.db.models import Material, SessionLocal

    student_id = _register(api_client, "material_rollback_owner")
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    with sample_pdf_path.open("rb") as file:
        assert api_client.post(
            f"/exams/{exam_id}/ingest", params={"student_id": student_id},
            files={"file": ("fonte.pdf", file, "application/pdf")},
        ).status_code == 200
    with SessionLocal() as db:
        material = db.query(Material).filter(Material.exam_id == exam_id).one()
        material_id, file_hash, path = material.id, material.file_hash, Path(material.stored_path)
    original_session_local = material_ops.SessionLocal

    def failing_session():
        db = original_session_local()
        def fail_commit():
            raise OSError("SECRET db")
        monkeypatch.setattr(db, "commit", fail_commit)
        return db

    monkeypatch.setattr(material_ops, "SessionLocal", failing_session)
    response = api_client.delete(
        f"/exams/{exam_id}/materials/{material_id}", params={"student_id": student_id}
    )
    assert response.status_code == 500
    assert "SECRET" not in response.text
    with SessionLocal() as db:
        assert db.get(Material, material_id) is not None
    assert path.exists()
    assert get_collection(student_id, exam_id).get(where={"file_hash": file_hash})["ids"]


def test_delete_material_requires_owner_and_existing_source(api_client, sample_pdf_path):
    from backend.db.models import Material, SessionLocal

    owner_id = _register(api_client, "material_owner")
    exam_id = api_client.post("/exams", params={"student_id": owner_id}, json={"name": "Analisi"}).json()["id"]
    with sample_pdf_path.open("rb") as file:
        assert api_client.post(
            f"/exams/{exam_id}/ingest", params={"student_id": owner_id},
            files={"file": (sample_pdf_path.name, file, "application/pdf")},
        ).status_code == 200
    with SessionLocal() as db:
        material_id = db.query(Material).filter(Material.exam_id == exam_id).one().id

    other_id = _register(api_client, "other_student")
    assert api_client.delete(
        f"/exams/{exam_id}/materials/{material_id}", params={"student_id": other_id}
    ).status_code == 404
    assert api_client.delete(
        f"/exams/{exam_id}/materials/999999", params={"student_id": owner_id}
    ).status_code == 404
    with SessionLocal() as db:
        assert db.get(Material, material_id) is not None


def test_delete_duplicate_material_preserves_shared_index(api_client, sample_pdf_path, isolated_upload_dir):
    from backend.agents.ingestor import get_collection
    from backend.db.models import Material, SessionLocal

    student_id = _register(api_client)
    exam_id = api_client.post("/exams", params={"student_id": student_id}, json={"name": "Analisi"}).json()["id"]
    with sample_pdf_path.open("rb") as file:
        assert api_client.post(
            f"/exams/{exam_id}/ingest", params={"student_id": student_id},
            files={"file": (sample_pdf_path.name, file, "application/pdf")},
        ).status_code == 200
    with SessionLocal() as db:
        original = db.query(Material).filter(Material.exam_id == exam_id).one()
        duplicate_path = isolated_upload_dir / "duplicate.pdf"
        duplicate_path.write_bytes(Path(original.stored_path).read_bytes())
        duplicate = Material(
            exam_id=exam_id, filename="duplicate.pdf", stored_path=str(duplicate_path),
            chunks_count=original.chunks_count, pages_count=original.pages_count,
            file_hash=original.file_hash,
        )
        db.add(duplicate)
        db.commit()
        original_id, duplicate_id, file_hash = original.id, duplicate.id, original.file_hash

    assert api_client.delete(
        f"/exams/{exam_id}/materials/{original_id}", params={"student_id": student_id}
    ).status_code == 204
    assert get_collection(student_id, exam_id).get(where={"file_hash": file_hash})["ids"]
    assert duplicate_path.exists()
    assert api_client.delete(
        f"/exams/{exam_id}/materials/{duplicate_id}", params={"student_id": student_id}
    ).status_code == 204
    assert not get_collection(student_id, exam_id).get(where={"file_hash": file_hash})["ids"]
    assert not duplicate_path.exists()
