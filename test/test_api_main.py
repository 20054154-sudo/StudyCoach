"""Integration test — backend/api/main.py (app FastAPI: routing, health, CORS)."""


def test_health_check(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "Study Coach API"}
    assert len(resp.headers["X-Request-ID"]) == 32


def test_docs_available(api_client):
    resp = api_client.get("/docs")
    assert resp.status_code == 200


def test_openapi_lists_all_routers(api_client):
    resp = api_client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/auth/register" in paths
    assert "/exams" in paths
    assert "/sessions" in paths
    assert "/sessions/{session_id}/tutor/start" in paths
    assert "/auth/me" in paths
    assert "/exams/{exam_id}/materials" in paths
    assert "/exams/{exam_id}/doubts" in paths
    assert "/sessions/{session_id}/detail" in paths


def test_no_cross_origin_access_by_default(api_client):
    resp = api_client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") is None


def test_unknown_route_returns_404(api_client):
    resp = api_client.get("/does-not-exist")
    assert resp.status_code == 404
