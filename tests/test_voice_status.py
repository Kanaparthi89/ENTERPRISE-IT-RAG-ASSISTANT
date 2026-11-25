"""Voice V1b: the VOICE_ENABLED flag and GET /api/voice/status (no network)."""

import pytest
from pydantic import ValidationError

from app.api import routes, streaming
from app.core import config
from app.services.audit import get_recent_audits
from tests.conftest import IsolatedSettings
from tests.test_chat_api import workflow_result

STATUS_URL = "/api/voice/status"
NOT_FOUND = {"detail": "Not Found"}


@pytest.fixture
def voice_on(monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_enabled", True)
    return settings


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

def test_voice_is_disabled_by_default(settings):
    assert config.Settings.model_fields["voice_enabled"].default is False
    assert IsolatedSettings().voice_enabled is False
    assert settings.voice_enabled is False  # the test suite's own settings


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("False", False),
        ("  true\n", True),
    ],
)
def test_voice_enabled_accepts_only_true_or_false(value, expected):
    # Strings are what the environment and .env provide.
    assert IsolatedSettings(voice_enabled=value).voice_enabled is expected


@pytest.mark.parametrize(
    "value",
    ["1", "0", "yes", "no", "on", "off", "t", "f", "y", "n", "", " ", "enabled", "truee", 1, 0, None, 1.0],
)
def test_voice_enabled_rejects_ambiguous_values(value):
    # Pydantic's default boolean parsing would accept several of these.
    with pytest.raises(ValidationError, match="must be true or false"):
        IsolatedSettings(voice_enabled=value)


def test_the_other_flags_keep_their_existing_parsing():
    # Strictness is limited to VOICE_ENABLED; existing flags are unchanged.
    assert IsolatedSettings(stream_enabled="1", sessions_enabled="yes").stream_enabled is True


# ---------------------------------------------------------------------------
# GET /api/voice/status
# ---------------------------------------------------------------------------

def test_status_is_404_while_disabled(client, settings):
    response = client.get(STATUS_URL)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_disabled_status_is_indistinguishable_from_an_unknown_route(client):
    status = client.get(STATUS_URL)
    unknown = client.get("/api/voice/no-such-route")

    assert (status.status_code, status.json()) == (unknown.status_code, unknown.json())
    assert status.headers["content-type"] == unknown.headers["content-type"]


def test_status_is_404_when_explicitly_disabled(client, monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_enabled", False)

    assert client.get(STATUS_URL).status_code == 404


def test_status_reports_tts_when_enabled(client, voice_on):
    response = client.get(STATUS_URL)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"tts": True}  # exactly this: no settings, versions, or paths
    assert response.json()["tts"] is True  # a boolean, not the string "true"


@pytest.mark.parametrize("enabled", [False, True])
def test_other_methods_get_the_standard_405(client, monkeypatch, settings, enabled):
    # FastAPI matches the path before the handler runs, so this is 405 even
    # while voice is disabled (documented in the README).
    monkeypatch.setattr(settings, "voice_enabled", enabled)

    for method in ("POST", "PUT", "DELETE"):
        response = client.request(method, STATUS_URL)
        assert response.status_code == 405, method
        assert response.headers["allow"] == "GET"


def test_status_reflects_the_flag_on_every_request(client, monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_enabled", True)
    assert client.get(STATUS_URL).status_code == 200
    monkeypatch.setattr(settings, "voice_enabled", False)
    assert client.get(STATUS_URL).status_code == 404


@pytest.mark.parametrize("enabled", [False, True])
def test_status_has_no_side_effects(client, monkeypatch, settings, blocked_network, enabled):
    monkeypatch.setattr(settings, "voice_enabled", enabled)

    def forbidden(*args, **kwargs):
        raise AssertionError("the voice status must not call a provider or the workflow")

    # No speech provider, model, or workflow is touched.
    from openai.resources.audio.speech import Speech
    from openai.resources.audio.transcriptions import Transcriptions

    monkeypatch.setattr(Speech, "create", forbidden)
    monkeypatch.setattr(Transcriptions, "create", forbidden)
    monkeypatch.setattr(routes, "ask", forbidden)
    audits_before = len(get_recent_audits(500))

    response = client.get(STATUS_URL)

    assert response.status_code == (200 if enabled else 404)
    assert blocked_network() == []  # no outbound connection was attempted
    assert len(get_recent_audits(500)) == audits_before  # nothing audited
    assert not settings.session_db_path.exists()  # no session storage created
    assert streaming.peek_stream_service() is None  # no stream service started


# ---------------------------------------------------------------------------
# Existing routes are unchanged by the flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [False, True])
def test_existing_routes_behave_the_same_with_voice_on_or_off(client, monkeypatch, settings, enabled):
    monkeypatch.setattr(settings, "voice_enabled", enabled)
    monkeypatch.setattr(routes, "ask", lambda question: workflow_result())

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": settings.app_name}

    chat = client.post("/api/chat", json={"question": "How do I connect to the VPN?"})
    assert chat.status_code == 200
    assert chat.json()["answer"] == "Use the GlobalProtect client."

    # Streaming and sessions keep their own flags (both off here).
    assert client.post("/api/chat/stream", json={"question": "VPN?"}).json() == NOT_FOUND
    assert client.post("/api/sessions", json={}).status_code == 404

    # Validation is unchanged.
    assert client.post("/api/chat", json={"question": "x"}).status_code == 422
