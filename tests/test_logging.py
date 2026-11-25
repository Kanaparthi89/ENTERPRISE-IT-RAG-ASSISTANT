"""Tests for log-level resolution in configure_logging."""

import logging

import pytest

from app.core.logging import configure_logging


@pytest.fixture
def root_logger(monkeypatch):
    """Restore the root logger's level, handler levels and formatters."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_TO_FILE", raising=False)

    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = [
        (handler, handler.level, handler.formatter) for handler in root.handlers
    ]
    yield root
    root.setLevel(saved_level)
    for handler, level, formatter in saved_handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)


def test_level_defaults_to_settings(root_logger, settings, monkeypatch):
    monkeypatch.setattr(settings, "log_level", "WARNING")

    configure_logging()

    assert root_logger.level == logging.WARNING


def test_process_environment_takes_precedence(root_logger, settings, monkeypatch):
    monkeypatch.setattr(settings, "log_level", "WARNING")
    monkeypatch.setenv("LOG_LEVEL", "error")

    configure_logging()

    assert root_logger.level == logging.ERROR


def test_unknown_environment_level_falls_back_to_info(root_logger, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    configure_logging()

    assert root_logger.level == logging.INFO
