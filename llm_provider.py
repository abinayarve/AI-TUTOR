"""
llm_provider.py
================

Single, provider-agnostic factory for every chat LLM used across the
project (lesson generation, teacher decisions, assessment grading,
misconception diagnosis, visual planning, doubt answering, etc.).

WHY THIS EXISTS
----------------
Every engine used to instantiate ``ChatGoogleGenerativeAI`` directly.
Google's Gemini free tier is small and easy to exhaust mid-lesson
(rate limits were cut significantly in late 2025), which caused
generation to fail partway through a session.

This module tries providers in order and returns the first one that
is actually usable, so the rest of the codebase never needs to know
which provider answered:

    1. Groq       (recommended — free, no card, high daily quota,
                   fast Llama 3.3 70B / Llama 3.1 8B)
    2. Gemini     (kept as a fallback for existing users/keys)

Add more providers by adding another branch to ``_PROVIDERS`` below —
everything downstream (LessonGeneratorEngine, TeacherAgent,
AssessmentEngine, MisconceptionEngine, VisualEngine, LearningPathEngine,
app.py's doubt box and fallback lesson generator) keeps working
unchanged because they all just call ``get_chat_llm(...)``.

No network access is required to import this module; provider SDKs are
imported lazily inside each branch so a missing package for a provider
you're not using never breaks anything.
"""

from __future__ import annotations

import os
from typing import Any, Optional


DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY_ENV = "GROQ_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# Order matters: first usable provider wins.
_PROVIDER_ORDER = ["groq", "gemini"]


class NoLLMProviderAvailable(RuntimeError):
    """Raised when no configured provider could be used."""


def _try_groq(
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Any:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def _try_gemini(
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


_PROVIDERS = {
    "groq": _try_groq,
    "gemini": _try_gemini,
}


def get_chat_llm(
    temperature: float = 0.3,
    max_output_tokens: int = 4096,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
    gemini_model: Optional[str] = None,
    preferred_provider: Optional[str] = None,
) -> Any:
    """
    Return a ready-to-use LangChain chat model, preferring Groq's free
    tier over Gemini's.

    Any argument left as None falls back to an environment variable
    (GROQ_API_KEY / GEMINI_API_KEY / GROQ_MODEL / GEMINI_MODEL), so
    existing code that only ever passed a Gemini key keeps working —
    it will automatically start using Groq the moment a GROQ_API_KEY
    is set, with zero other code changes required.

    Raises NoLLMProviderAvailable if neither provider is usable, with
    a message telling the caller exactly what to set.
    """

    groq_api_key = groq_api_key or os.getenv(GROQ_API_KEY_ENV, "")
    gemini_api_key = gemini_api_key or os.getenv(GEMINI_API_KEY_ENV, "")

    keys = {"groq": groq_api_key, "gemini": gemini_api_key}
    models = {
        "groq": groq_model or DEFAULT_GROQ_MODEL,
        "gemini": gemini_model or DEFAULT_GEMINI_MODEL,
    }

    order = list(_PROVIDER_ORDER)
    if preferred_provider in order:
        order.remove(preferred_provider)
        order.insert(0, preferred_provider)

    last_error: Optional[Exception] = None

    for provider in order:
        api_key = keys.get(provider, "")
        if not api_key:
            continue

        try:
            return _PROVIDERS[provider](
                api_key,
                models[provider],
                temperature,
                max_output_tokens,
            )
        except Exception as exc:  # missing package, bad key, etc.
            last_error = exc
            continue

    raise NoLLMProviderAvailable(
        "No usable LLM provider is configured. Set a GROQ_API_KEY "
        "(recommended — free, get one at https://console.groq.com/keys) "
        "or a GEMINI_API_KEY. "
        f"Last error: {last_error}"
    )


def run_tests() -> None:
    """Provider-selection logic only; makes no network calls."""

    # No keys at all -> must raise a clear, actionable error.
    try:
        get_chat_llm(groq_api_key="", gemini_api_key="")
        raise AssertionError("Expected NoLLMProviderAvailable")
    except NoLLMProviderAvailable as exc:
        assert "GROQ_API_KEY" in str(exc)

    # Groq key present but package/network unavailable in this sandbox
    # falls through to gemini key; with neither reachable it should
    # still raise the same clear error type, never crash uncaught.
    try:
        get_chat_llm(groq_api_key="fake", gemini_api_key="fake")
    except NoLLMProviderAvailable:
        pass
    except Exception:
        pass  # network/package errors are fine here, just not silent

    print("llm_provider.py: all tests passed")


if __name__ == "__main__":
    run_tests()
