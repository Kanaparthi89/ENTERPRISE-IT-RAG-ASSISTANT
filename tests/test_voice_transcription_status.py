"""Voice V3b: GET /api/voice/transcription/status (no network).

Reports whether the page may offer dictation. It follows the conventions of
GET /api/voice/status (V1b), but reads VOICE_TRANSCRIPTION_ENABLED, and the
V1 status contract stays exactly as it was.
"""

import pytest

from app.api import routes, streaming
from app.core import config
from app.services import transcription
from app.services.audit import get_recent_audits
from tests.conftest import IsolatedSettings

URL = "/api/voice/transcription/status"
TTS_URL = "/api/voice/status"
TRANSCRIBE_URL = "/api/voice/transcribe"
NOT_FOUND = {"detail": "Not Found"}


@pytest.fixture
def stt_on(monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_transcription_enabled", True)
    return settings


def test_dictation_status_is_disabled_by_default(client, settings):
    assert config.Settings.model_fields["voice_transcription_enabled"].default is False
    assert IsolatedSettings().voice_transcription_enabled is False
    assert settings.voice_transcription_enabled is False
    assert client.get(URL).status_code == 404


def test_disabled_status_is_indistinguishable_from_an_unknown_route(client):
    status = client.get(URL)
    unknown = client.get("/api/voice/transcription/no-such-route")

    assert status.json() == NOT_FOUND
    assert (status.status_code, status.json()) == (unknown.status_code, unknown.json())
    assert status.headers["content-type"] == unknown.headers["content-type"]


def test_status_is_404_when_explicitly_disabled(client, monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_transcription_enabled", False)

    assert client.get(URL).status_code == 404


def test_status_reports_stt_when_enabled(client, stt_on):
    response = client.get(URL)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"stt": True}  # exactly this: no settings, model, limits or paths
    assert response.json()["stt"] is True  # a boolean, not the string "true"
    assert "set-cookie" not in response.headers


def test_status_reflects_the_flag_on_every_request(client, monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_transcription_enabled", True)
    assert client.get(URL).status_code == 200
    monkeypatch.setattr(settings, "voice_transcription_enabled", False)
    assert client.get(URL).status_code == 404


@pytest.mark.parametrize("tts", [False, True])
@pytest.mark.parametrize("stt", [False, True])
def test_the_two_voice_statuses_are_independent(client, monkeypatch, settings, tts, stt):
    monkeypatch.setattr(settings, "voice_enabled", tts)
    monkeypatch.setattr(settings, "voice_transcription_enabled", stt)

    tts_status = client.get(TTS_URL)
    stt_status = client.get(URL)

    # The V1 contract is unchanged: 404, or exactly {"tts": true}.
    assert (tts_status.status_code, tts_status.json()) == ((200, {"tts": True}) if tts else (404, NOT_FOUND))
    assert (stt_status.status_code, stt_status.json()) == ((200, {"stt": True}) if stt else (404, NOT_FOUND))


def test_status_reports_only_the_flag_not_the_provider_key(client, monkeypatch, stt_on):
    # Like the V1 status, it reads one setting. Whether a key is configured is
    # never revealed; a missing key is reported by POST /api/voice/transcribe (503).
    from pydantic import SecretStr

    monkeypatch.setattr(stt_on, "openai_api_key", SecretStr(""))
    assert client.get(URL).json() == {"stt": True}
    monkeypatch.setattr(stt_on, "openai_api_key", SecretStr("sk-test-not-a-real-key"))
    assert client.get(URL).json() == {"stt": True}


@pytest.mark.parametrize("enabled", [False, True])
def test_other_methods_get_the_standard_405(client, monkeypatch, settings, enabled):
    # FastAPI matches the path before the handler runs, so this is 405 even
    # while dictation is disabled, as for the V1 status.
    monkeypatch.setattr(settings, "voice_transcription_enabled", enabled)

    for method in ("POST", "PUT", "DELETE"):
        response = client.request(method, URL)
        assert response.status_code == 405, method
        assert response.headers["allow"] == "GET"


@pytest.mark.parametrize("enabled", [False, True])
def test_status_has_no_side_effects(client, monkeypatch, settings, blocked_network, enabled):
    monkeypatch.setattr(settings, "voice_transcription_enabled", enabled)

    def forbidden(*args, **kwargs):
        raise AssertionError("the dictation status must not call a provider, the workflow or the upload path")

    from openai.resources.audio.speech import Speech
    from openai.resources.audio.transcriptions import Transcriptions

    monkeypatch.setattr(Speech, "create", forbidden)
    monkeypatch.setattr(Transcriptions, "create", forbidden)
    monkeypatch.setattr(transcription, "AsyncOpenAI", forbidden)
    monkeypatch.setattr(transcription, "read_bounded_body", forbidden)
    monkeypatch.setattr(transcription, "transcribe", forbidden)
    monkeypatch.setattr(routes, "ask", forbidden)
    audits_before = len(get_recent_audits(500))

    response = client.get(URL)

    assert response.status_code == (200 if enabled else 404)
    assert blocked_network() == []  # no outbound connection was attempted
    assert len(get_recent_audits(500)) == audits_before  # nothing audited
    assert not settings.session_db_path.exists()  # no session storage created
    assert streaming.peek_stream_service() is None  # no stream service started


@pytest.mark.parametrize("enabled", [False, True])
def test_the_transcribe_endpoint_is_unchanged(client, monkeypatch, settings, enabled):
    # The status route does not shadow or alter POST /api/voice/transcribe.
    monkeypatch.setattr(settings, "voice_transcription_enabled", enabled)

    response = client.get(TRANSCRIBE_URL)
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
    if not enabled:
        assert client.post(TRANSCRIBE_URL, content=b"", headers={"content-type": "text/plain"}).json() == NOT_FOUND


@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
def test_both_entry_points_serve_the_status(entry):
    import importlib

    app = importlib.import_module(entry).app
    paths = {
        (route.path, tuple(sorted(route.methods)))
        for route in app.routes
        if hasattr(route, "methods")
    }
    assert ("/api/voice/transcription/status", ("GET",)) in paths
