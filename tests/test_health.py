"""Tests for health, frontend, and application-level error handling."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_api_health_reports_service_name(client, settings):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": settings.app_name}


def test_homepage_renders_app_name(client, settings):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert settings.app_name in response.text


def test_static_assets_are_served(client):
    response = client.get("/static/js/app.js")
    assert response.status_code == 200


def test_startup_creates_audit_database(client, settings):
    assert settings.audit_db_path.exists()


def test_unhandled_errors_return_generic_500():
    def boom():
        raise RuntimeError("secret internal detail")

    app.add_api_route("/__test_boom", boom)
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.get("/__test_boom")
    finally:
        app.router.routes[:] = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) != "/__test_boom"
        ]

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
