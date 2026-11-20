
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: repository root, based on this file's location.
BASE_DIR = Path(__file__).resolve().parents[2]

# FINALIZE_GRACE (5 s) in app/api/streaming.py plus a 10 s margin (design §6).
_STREAM_FINALIZE_MARGIN_SECONDS = 15.0


class Settings(BaseSettings):
    """Centralized, validated application configuration."""

    # Application
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # LLM provider
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # OpenAI client limits for chat and embeddings. The timeout applies to
    # each HTTP operation (connect, read, write, pool), not to a request's
    # total duration. Proposed defaults; not validated in production.
    openai_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    openai_max_retries: int = Field(default=2, ge=0, le=10)

    # Retrieval and search
    tavily_api_key: SecretStr = SecretStr("")
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"
    top_k: int = Field(default=4, ge=1, le=50)
    max_retries: int = Field(default=2, ge=0, le=10)

    # Finite timeouts for Pinecone and Tavily calls (H1), separate from the
    # OpenAI limits above. Pinecone: per HTTP attempt of a query or upsert,
    # and of the index list, describe and create calls in ensure_index();
    # the SDK retries a failed attempt up to 5 times.
    # Tavily: Requests connect and read timeouts for one search request
    # (no retries).
    pinecone_timeout_seconds: float = Field(default=10.0, gt=0, le=15)
    tavily_timeout_seconds: float = Field(default=15.0, gt=0, le=30)

    # H1-5 (A): the longest wait the Pinecone SDK honors for one
    # ``Retry-After`` header (HTTP 413, 429, 503). Unset (the default) keeps
    # the SDK's policy unchanged: up to 21,600 s (6 hours) per retry. When
    # set, every other retry setting stays the SDK default; a longer
    # Retry-After is shortened to this value (0: the normal backoff is used
    # instead). Not a deadline: retries and per-attempt timeouts still add up.
    pinecone_retry_after_max_seconds: int | None = Field(default=None, ge=0, le=21600)

    # H1-5 (B): optional deadline for a stateless POST /api/chat question,
    # counted from when its handler starts. Unset (the default): no deadline,
    # as before. When set, a question past it starts no further external
    # call (OpenAI, Pinecone, Tavily) and fails with the generic 500 and
    # audit error type TimeoutError. A call already in progress is not
    # interrupted, so a question can run past this value. Session questions
    # use SESSION_TURN_MAX_SECONDS instead.
    chat_deadline_seconds: float | None = Field(default=None, gt=0, le=3600)

    # Streaming chat (POST /api/chat/stream). Disabled by default: a
    # controlled prototype, not production-ready. Pinecone and Tavily calls
    # now have per-attempt timeouts (above), but no call has a total
    # deadline, and other production blockers remain open. The values
    # below are proposals, not production-validated.
    stream_enabled: bool = False
    stream_heartbeat_seconds: float = Field(default=15.0, gt=0, le=300)
    stream_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    max_concurrent_streams: int = Field(default=8, ge=1, le=64)
    stream_retry_after_seconds: int = Field(default=5, ge=1, le=300)

    # Conversation sessions (A2). Disabled by default. Session memory lives
    # in its own SQLite file, separate from the audit database.
    sessions_enabled: bool = False
    session_db_path: Path = BASE_DIR / "data" / "sessions.db"
    session_idle_timeout_seconds: float = Field(default=1800.0, ge=60, le=86400)
    session_absolute_timeout_seconds: float = Field(default=28800.0, ge=60, le=604800)
    max_active_sessions: int = Field(default=5000, ge=1, le=100000)
    session_db_busy_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    session_cleanup_batch: int = Field(default=500, ge=1, le=10000)
    session_cleanup_interval_seconds: float = Field(default=300.0, ge=10, le=3600)
    # Turn leases: one question at a time per session, renewed while running.
    session_lease_ttl_seconds: float = Field(default=30.0, ge=3, le=300)
    session_lease_margin_seconds: float = Field(default=2.0, ge=0, le=60)
    session_turn_max_seconds: float = Field(default=180.0, ge=10, le=3600)
    session_max_turns: int = Field(default=20, ge=1, le=200)
    session_history_turns: int = Field(default=4, ge=0, le=50)
    session_stored_answer_chars: int = Field(default=2000, ge=100, le=20000)
    # Bounded shutdown (design §7.5): grace + keeper join + release budget.
    session_shutdown_grace_seconds: float = Field(default=3.0, ge=0, le=30)
    session_keeper_join_seconds: float = Field(default=2.0, ge=0, le=30)
    session_release_budget_seconds: float = Field(default=2.0, gt=0, le=30)

    # Voice (V1). Disabled by default. Only reports, through
    # GET /api/voice/status, that spoken answers may be offered; speech runs
    # in the browser with on-device voices, so the server calls no speech
    # provider. When on, the page offers an opt-in "Read answers aloud"
    # toggle. Strict: only true or false.
    voice_enabled: bool = False

    # Voice V2: POST /api/voice/transcribe (speech-to-text with OpenAI).
    # Disabled by default and separate from VOICE_ENABLED. Audio is held in
    # memory for one request (at most 2 MiB, a fixed limit in code) and never
    # stored; the transcript is returned, never stored or sent to the chat.
    # The provider timeout applies to each attempt; retries are bounded
    # (default none). Strict: only true or false.
    voice_transcription_enabled: bool = False
    voice_transcription_model: str = "whisper-1"
    voice_transcription_timeout_seconds: float = Field(default=30.0, ge=1, le=120)
    voice_transcription_max_retries: int = Field(default=0, ge=0, le=2)

    # Application authentication
    admin_api_key: SecretStr = SecretStr("")

    # Security S1-3: access codes for live AI features (Architecture C).
    # Disabled by default. Security S1-5: every paid AI route (chat,
    # streaming, sessions, transcription) requires a verified access cookie,
    # so with access disabled those routes answer 503 (read-only mode) and
    # only the page, static files and status routes work. When enabled, every setting below must be
    # valid or the application refuses to start (app/services/access.py).
    # Codes are stored as SHA-256 digests only: "id:digest" or
    # "id:digest:YYYY-MM-DD", comma-separated. The cookie secret signs access
    # cookies (HMAC-SHA256); previous secrets only verify cookies until they
    # expire. Throttling needs a trusted client identity: ACCESS_CLIENT_IP_HEADER
    # (a header verified to be set, not passed through, by the platform) or
    # ACCESS_TRUST_SOCKET_PEER=true (clients connect directly). In production
    # one of them is required; elsewhere the socket peer is the default.
    # ACCESS_COOKIE_SECURE=false is allowed only with APP_ENV=development.
    access_enabled: bool = False
    access_code_hashes: str = ""
    access_cookie_secret: SecretStr = SecretStr("")
    access_cookie_previous_secrets: SecretStr = SecretStr("")
    access_cookie_max_age_seconds: int = Field(default=7 * 24 * 3600, ge=300, le=30 * 24 * 3600)
    access_cookie_secure: bool = True
    access_client_ip_header: str = ""
    access_trust_socket_peer: bool = False
    access_max_failed_attempts: int = Field(default=5, ge=1, le=100)
    access_failure_window_seconds: float = Field(default=900.0, ge=60, le=86400)
    access_max_global_failed_attempts: int = Field(default=200, ge=1, le=100000)
    # Security S1-5: the exact browser origins (scheme://host[:port],
    # comma-separated, at most 10, https in production) that may call the
    # paid AI routes. Cross-site checks compare against this list only, never
    # against the request's Host or forwarded headers. Empty (the default):
    # paid AI routes answer 503. A malformed value refuses to start.
    public_origins: str = ""

    # Security S1-4b: chat quotas (app/services/quota.py). Disabled by default.
    # When enabled, every chat route (stateless and session /api/chat and
    # /api/chat/stream) shares one in-memory quota pool per process: per access
    # code (from a verified access cookie; everyone else shares one anonymous
    # bucket) and globally. Limits are read once, when the engine is created;
    # changing them needs a restart. Enabling quotas also requires bounded
    # deadlines (validated below): CHAT_DEADLINE_SECONDS and
    # STREAM_TIMEOUT_SECONDS at most 90, OPENAI_TIMEOUT_SECONDS at most 30,
    # OPENAI_MAX_RETRIES at most 1, SESSION_TURN_MAX_SECONDS at most 105 and
    # PINECONE_RETRY_AFTER_MAX_SECONDS set, from 0 to 5. A deadline stops a question before its next
    # external call; a provider call already in progress still finishes (with
    # its own timeout and retries). With transcription enabled too:
    # VOICE_TRANSCRIPTION_TIMEOUT_SECONDS at most 30 and
    # VOICE_TRANSCRIPTION_MAX_RETRIES=0. In production, transcription is refused
    # until S1-5 enforces verified access on it (ACCESS_ENABLED alone does not;
    # validated below, with no bypass setting). Strict: only true or false.
    quota_enabled: bool = False
    quota_chat_per_code_hourly: int = Field(default=10, ge=1, le=100000)
    quota_chat_per_code_daily: int = Field(default=25, ge=1, le=100000)
    quota_chat_global_hourly: int = Field(default=60, ge=1, le=100000)
    quota_chat_global_daily: int = Field(default=150, ge=1, le=100000)
    quota_chat_per_code_concurrent: int = Field(default=1, ge=1, le=100)
    quota_chat_global_concurrent: int = Field(default=2, ge=1, le=100)
    quota_max_identities: int = Field(default=1000, ge=1, le=100000)
    quota_busy_retry_after_seconds: int = Field(default=5, ge=1, le=300)
    # Security S1-4c: transcription quotas (POST /api/voice/transcribe), in
    # the same engine. Audio seconds are measured exactly from the uploaded
    # canonical WAV before the provider is called; a recording is at most 60 s.
    quota_transcription_per_code_daily: int = Field(default=10, ge=1, le=1000)
    quota_transcription_per_code_daily_seconds: int = Field(default=300, ge=60, le=86400)
    quota_transcription_global_daily_seconds: int = Field(default=1800, ge=60, le=864000)
    quota_transcription_per_code_concurrent: int = Field(default=1, ge=1, le=10)
    quota_transcription_global_concurrent: int = Field(default=1, ge=1, le=10)

    # Storage paths
    audit_db_path: Path = BASE_DIR / "data" / "audit.db"
    upload_dir: Path = BASE_DIR / "uploads"
    sample_kb_dir: Path = BASE_DIR / "data" / "sample_kb"

    # Upload limits
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_upload_extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".pdf",
        ".docx",
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    @field_validator(
        "voice_enabled",
        "voice_transcription_enabled",
        "access_enabled",
        "access_cookie_secure",
        "access_trust_socket_peer",
        "quota_enabled",
        mode="before",
    )
    @classmethod
    def validate_strict_boolean(cls, value):
        """Accept only a real boolean or the words true/false (any case).

        Pydantic's default boolean parsing also accepts 1, yes, on, t and y.
        A feature flag that controls what users are offered must not turn on
        from an ambiguous value, so anything else is rejected at startup.
        Pydantic's StrictBool cannot be used: environment values are strings.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("This setting must be true or false.")

    @field_validator("pinecone_index_name", "pinecone_namespace", "voice_transcription_model")
    @classmethod
    def validate_non_empty_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This setting cannot be empty.")
        return value

    @model_validator(mode="after")
    def validate_session_timeouts(self) -> "Settings":
        if self.session_absolute_timeout_seconds < self.session_idle_timeout_seconds:
            raise ValueError(
                "session_absolute_timeout_seconds must be at least session_idle_timeout_seconds."
            )
        # A streamed turn must never lose its lease before the stream's own
        # deadline plus its finalize grace (5 s) plus a 10 s margin.
        minimum_turn = self.stream_timeout_seconds + _STREAM_FINALIZE_MARGIN_SECONDS
        if self.session_turn_max_seconds < minimum_turn:
            raise ValueError(
                f"session_turn_max_seconds must be at least stream_timeout_seconds + "
                f"{_STREAM_FINALIZE_MARGIN_SECONDS:g} ({minimum_turn:g})."
            )
        if self.session_lease_margin_seconds >= self.session_lease_ttl_seconds / 3:
            raise ValueError(
                "session_lease_margin_seconds must be smaller than one renewal interval "
                "(session_lease_ttl_seconds / 3)."
            )
        return self

    @model_validator(mode="after")
    def validate_access(self) -> "Settings":
        """Access codes must never switch on with missing, weak or malformed settings."""
        if self.access_enabled:
            from app.services.access import AccessConfig, AccessConfigError

            try:
                AccessConfig.from_settings(self)
            except AccessConfigError as exc:
                raise ValueError(str(exc)) from None
        return self

    @model_validator(mode="after")
    def validate_public_origins(self) -> "Settings":
        """A set PUBLIC_ORIGINS must be exact origins; empty leaves paid AI routes off (S1-5)."""
        from app.services.access_policy import PublicOriginsError, public_origins

        try:
            public_origins(self)
        except PublicOriginsError as exc:
            raise ValueError(str(exc)) from None
        return self

    @model_validator(mode="after")
    def validate_quota(self) -> "Settings":
        """Quotas must never switch on with invalid limits or unbounded deadlines."""
        if self.quota_enabled:
            from app.services.quota import QuotaConfigError
            from app.services.quota_runtime import check_settings

            try:
                check_settings(self)
            except QuotaConfigError as exc:
                raise ValueError(str(exc)) from None
        return self

    @model_validator(mode="after")
    def validate_transcription_deployment(self) -> "Settings":
        """In production, transcription is refused until verified access enforcement exists (S1-4c)."""
        from app.services.quota import QuotaConfigError
        from app.services.quota_runtime import check_transcription_deployment

        try:
            check_transcription_deployment(self)
        except QuotaConfigError as exc:
            raise ValueError(str(exc)) from None
        return self

    def ensure_directories(self) -> None:
        """Create required local directories when needed."""
        self.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.sample_kb_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
