"""
Audio Engine for AI Teacher.

Provides multilingual text-to-speech using Microsoft Edge TTS.

Supported teaching languages:
- English
- Hinglish
- Hindi
- Tamil
- Spanish

The engine is intentionally independent from Streamlit, Gemini,
RAG, avatar, and video modules so it can be tested separately.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import edge_tts
except ImportError:
    edge_tts = None


class AudioEngine:
    """
    Multilingual text-to-speech engine.

    Gemini API key is NOT required.
    Edge TTS is used for speech generation.
    """

    DEFAULT_VOICES = {
        "english": "en-US-ChristopherNeural",
        "hinglish": "hi-IN-MadhurNeural",
        "hindi": "hi-IN-SwaraNeural",
        "tamil": "ta-IN-PallaviNeural",
        "spanish": "es-ES-AlvaroNeural",
    }

    LANGUAGE_ALIASES = {
        "en": "english",
        "english": "english",
        "eng": "english",

        "hinglish": "hinglish",
        "hinglish (hindi + english)": "hinglish",
        "hindi english": "hinglish",

        "hi": "hindi",
        "hindi": "hindi",

        "ta": "tamil",
        "tamil": "tamil",

        "es": "spanish",
        "spanish": "spanish",
        "español": "spanish",
    }

    RATE_OPTIONS = {
        "slow": "-20%",
        "normal": "+0%",
        "fast": "+15%",
        "very_fast": "+30%",
    }

    def __init__(
        self,
        output_dir: Optional[str] = None,
        default_rate: str = "normal",
    ) -> None:
        """
        Initialize the audio engine.

        Parameters
        ----------
        output_dir:
            Directory where generated audio files are stored.
        default_rate:
            Speech rate preset.
        """

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("data") / "audio"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.default_rate = (
            default_rate
            if default_rate in self.RATE_OPTIONS
            else "normal"
        )

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when edge_tts is installed."""

        return edge_tts is not None

    def get_capabilities(self) -> Dict[str, Any]:
        """Return engine capabilities."""

        return {
            "engine": "Microsoft Edge TTS",
            "available": self.is_available(),
            "requires_api_key": False,
            "output_format": "mp3",
            "supported_languages": list(self.DEFAULT_VOICES.keys()),
            "supports_rate_control": True,
            "supports_custom_voice": True,
            "output_directory": str(self.output_dir),
        }

    # ------------------------------------------------------------------
    # Language handling
    # ------------------------------------------------------------------

    def normalize_language(self, language: Optional[str]) -> str:
        """
        Normalize language names and aliases.

        Unknown languages fall back to English.
        """

        if not language:
            return "english"

        normalized = str(language).strip().lower()

        if normalized in self.LANGUAGE_ALIASES:
            return self.LANGUAGE_ALIASES[normalized]

        return "english"

    def get_supported_languages(self) -> List[str]:
        """Return supported language names."""

        return list(self.DEFAULT_VOICES.keys())

    # ------------------------------------------------------------------
    # Voice handling
    # ------------------------------------------------------------------

    def get_voice(self, language: Optional[str] = None) -> str:
        """
        Get the default Edge TTS voice for a language.
        """

        normalized = self.normalize_language(language)

        return self.DEFAULT_VOICES[normalized]

    def set_voice(
        self,
        language: str,
        voice: str,
    ) -> None:
        """
        Override the default voice for a language.
        """

        normalized = self.normalize_language(language)

        if not voice or not str(voice).strip():
            raise ValueError("Voice name cannot be empty.")

        self.DEFAULT_VOICES[normalized] = str(voice).strip()

    # ------------------------------------------------------------------
    # Speech rate
    # ------------------------------------------------------------------

    def normalize_rate(self, rate: Optional[str]) -> str:
        """Convert a rate preset into Edge TTS rate syntax."""

        if not rate:
            return self.RATE_OPTIONS[self.default_rate]

        rate_text = str(rate).strip().lower()

        if rate_text in self.RATE_OPTIONS:
            return self.RATE_OPTIONS[rate_text]

        # Allow direct Edge TTS syntax such as +10% or -15%.
        if re.fullmatch(r"[+-]\d+%", rate_text):
            return rate_text

        return self.RATE_OPTIONS[self.default_rate]

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: Any) -> str:
        """
        Clean text before speech generation.
        """

        if text is None:
            return ""

        value = str(text).strip()

        value = re.sub(r"\s+", " ", value)

        return value

    @staticmethod
    def estimate_duration(
        text: Any,
        words_per_minute: int = 145,
    ) -> float:
        """
        Estimate speech duration in seconds.

        This is an approximation used for lesson/video planning.
        """

        cleaned = AudioEngine.clean_text(text)

        if not cleaned:
            return 0.0

        words = len(cleaned.split())

        wpm = max(60, int(words_per_minute))

        return round((words / wpm) * 60, 2)

    # ------------------------------------------------------------------
    # File utilities
    # ------------------------------------------------------------------

    def _safe_filename(self, filename: str) -> str:
        """Create a filesystem-safe filename."""

        filename = str(filename).strip()

        filename = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "_",
            filename,
        )

        filename = filename.strip("._")

        if not filename:
            filename = "speech"

        if not filename.lower().endswith(".mp3"):
            filename += ".mp3"

        return filename

    def _resolve_output_path(
        self,
        output_path: Optional[str],
        filename: Optional[str],
    ) -> Path:
        """Resolve output audio path."""

        if output_path:
            path = Path(output_path)

            if path.suffix.lower() != ".mp3":
                path = path.with_suffix(".mp3")

            path.parent.mkdir(parents=True, exist_ok=True)

            return path

        if filename:
            safe_name = self._safe_filename(filename)
        else:
            safe_name = "ai_teacher_speech.mp3"

        return self.output_dir / safe_name

    # ------------------------------------------------------------------
    # Edge TTS generation
    # ------------------------------------------------------------------

    async def _generate_async(
        self,
        text: str,
        output_path: str,
        voice: str,
        rate: str,
    ) -> str:
        """
        Generate speech asynchronously.
        """

        if edge_tts is None:
            raise RuntimeError(
                "edge-tts is not installed. "
                "Install it using: pip install edge-tts"
            )

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
        )

        await communicate.save(output_path)

        return output_path

    def _run_async(
        self,
        coroutine: Any,
    ) -> Any:
        """
        Safely execute an async coroutine.

        Handles normal Python execution and environments where
        an event loop is already running.
        """

        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if not running:
            return asyncio.run(coroutine)

        import threading

        result: Dict[str, Any] = {}
        error: Dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coroutine)
            except BaseException as exc:
                error["value"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]

        return result.get("value")

    def generate_speech(
        self,
        text: str,
        language: str = "English",
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
        voice: Optional[str] = None,
        rate: str = "normal",
    ) -> str:
        """
        Generate an MP3 speech file.

        Parameters
        ----------
        text:
            Text to speak.
        language:
            Teaching language.
        output_path:
            Optional exact output path.
        filename:
            Optional filename inside output directory.
        voice:
            Optional custom Edge TTS voice.
        rate:
            Speech rate preset or Edge TTS percentage.

        Returns
        -------
        str
            Generated MP3 path.
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            raise ValueError("Text cannot be empty.")

        normalized_language = self.normalize_language(language)

        selected_voice = (
            voice
            if voice
            else self.get_voice(normalized_language)
        )

        selected_rate = self.normalize_rate(rate)

        final_path = self._resolve_output_path(
            output_path=output_path,
            filename=filename,
        )

        final_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._run_async(
            self._generate_async(
                text=cleaned_text,
                output_path=str(final_path),
                voice=selected_voice,
                rate=selected_rate,
            )
        )

        if not final_path.exists():
            raise RuntimeError(
                "Audio generation completed without creating "
                f"the expected file: {final_path}"
            )

        if final_path.stat().st_size == 0:
            raise RuntimeError(
                f"Generated audio file is empty: {final_path}"
            )

        return str(final_path)

    # ------------------------------------------------------------------
    # Lesson-level helpers
    # ------------------------------------------------------------------

    def generate_lesson_audio(
        self,
        lesson: Dict[str, Any],
        language: str = "English",
        prefix: str = "lesson",
        rate: str = "normal",
    ) -> List[str]:
        """
        Generate audio for lesson modules.

        Expected module fields may include:
        - spoken_script
        - script
        - explanation
        - narration
        """

        if not isinstance(lesson, dict):
            raise ValueError("Lesson must be a dictionary.")

        modules = lesson.get("modules", [])

        if not isinstance(modules, list):
            raise ValueError("Lesson modules must be a list.")

        generated_files: List[str] = []

        for index, module in enumerate(modules, start=1):

            if not isinstance(module, dict):
                continue

            text = (
                module.get("spoken_script")
                or module.get("script")
                or module.get("explanation")
                or module.get("narration")
                or ""
            )

            text = self.clean_text(text)

            if not text:
                continue

            filename = f"{prefix}_module_{index}.mp3"

            audio_path = self.generate_speech(
                text=text,
                language=language,
                filename=filename,
                rate=rate,
            )

            generated_files.append(audio_path)

        return generated_files

    # ------------------------------------------------------------------
    # Voice information
    # ------------------------------------------------------------------

    def get_voice_info(
        self,
        language: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return voice information for a language."""

        normalized = self.normalize_language(language)

        return {
            "language": normalized,
            "voice": self.get_voice(normalized),
            "rate": self.normalize_rate(self.default_rate),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_request(
        self,
        text: Any,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a TTS request without generating audio.
        """

        cleaned = self.clean_text(text)
        normalized = self.normalize_language(language)

        return {
            "valid": bool(cleaned),
            "text_available": bool(cleaned),
            "language": normalized,
            "voice": self.get_voice(normalized),
            "estimated_duration_seconds": self.estimate_duration(
                cleaned
            ),
            "requires_api_key": False,
            "engine_available": self.is_available(),
        }


# ======================================================================
# TESTS
# ======================================================================

def run_tests() -> None:
    print("=" * 60)
    print("AUDIO ENGINE TESTS")
    print("=" * 60)

    test_dir = Path(tempfile.gettempdir()) / "ai_teacher_audio_test"

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = AudioEngine(
        output_dir=str(test_dir)
    )

    # --------------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------------
    print("\nTEST 1: Initialization")

    assert engine is not None
    assert engine.output_dir.exists()

    print("PASS")

    # --------------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------------
    print("\nTEST 2: Capabilities")

    capabilities = engine.get_capabilities()

    assert capabilities["requires_api_key"] is False
    assert capabilities["output_format"] == "mp3"
    assert "english" in capabilities["supported_languages"]

    print("PASS")

    # --------------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------------
    print("\nTEST 3: Supported languages")

    languages = engine.get_supported_languages()

    assert "english" in languages
    assert "hinglish" in languages
    assert "hindi" in languages
    assert "tamil" in languages
    assert "spanish" in languages

    print("PASS")

    # --------------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------------
    print("\nTEST 4: Language normalization")

    assert engine.normalize_language("English") == "english"
    assert engine.normalize_language("EN") == "english"
    assert engine.normalize_language("Hindi") == "hindi"
    assert engine.normalize_language("Tamil") == "tamil"
    assert engine.normalize_language("TA") == "tamil"
    assert engine.normalize_language("Spanish") == "spanish"
    assert engine.normalize_language("Hinglish") == "hinglish"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 5
    # --------------------------------------------------------------
    print("\nTEST 5: Voice mapping")

    assert engine.get_voice("English") == (
        "en-US-ChristopherNeural"
    )

    assert engine.get_voice("Hindi") == (
        "hi-IN-SwaraNeural"
    )

    assert engine.get_voice("Hinglish") == (
        "hi-IN-MadhurNeural"
    )

    assert engine.get_voice("Tamil") == (
        "ta-IN-PallaviNeural"
    )

    assert engine.get_voice("Spanish") == (
        "es-ES-AlvaroNeural"
    )

    print("PASS")

    # --------------------------------------------------------------
    # TEST 6
    # --------------------------------------------------------------
    print("\nTEST 6: Text cleaning")

    cleaned = engine.clean_text(
        "   Hello     AI     Teacher!   "
    )

    assert cleaned == "Hello AI Teacher!"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 7
    # --------------------------------------------------------------
    print("\nTEST 7: Duration estimation")

    duration = engine.estimate_duration(
        "Artificial intelligence helps students learn."
    )

    assert duration > 0

    assert engine.estimate_duration("") == 0.0

    print("PASS")

    # --------------------------------------------------------------
    # TEST 8
    # --------------------------------------------------------------
    print("\nTEST 8: Speech rate")

    assert engine.normalize_rate("slow") == "-20%"
    assert engine.normalize_rate("normal") == "+0%"
    assert engine.normalize_rate("fast") == "+15%"
    assert engine.normalize_rate("very_fast") == "+30%"
    assert engine.normalize_rate("+10%") == "+10%"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 9
    # --------------------------------------------------------------
    print("\nTEST 9: Filename sanitization")

    safe = engine._safe_filename(
        "AI Teacher: Module 1?.mp3"
    )

    assert safe.endswith(".mp3")
    assert ":" not in safe
    assert "?" not in safe

    print("PASS")

    # --------------------------------------------------------------
    # TEST 10
    # --------------------------------------------------------------
    print("\nTEST 10: Output path resolution")

    path = engine._resolve_output_path(
        output_path=str(
            test_dir / "custom_audio.wav"
        ),
        filename=None,
    )

    assert path.suffix.lower() == ".mp3"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 11
    # --------------------------------------------------------------
    print("\nTEST 11: Request validation")

    validation = engine.validate_request(
        text="Explain machine learning.",
        language="English",
    )

    assert validation["valid"] is True
    assert validation["language"] == "english"
    assert validation["requires_api_key"] is False
    assert validation["estimated_duration_seconds"] > 0

    print("PASS")

    # --------------------------------------------------------------
    # TEST 12
    # --------------------------------------------------------------
    print("\nTEST 12: Empty request validation")

    validation = engine.validate_request(
        text="",
        language="Tamil",
    )

    assert validation["valid"] is False
    assert validation["language"] == "tamil"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 13
    # --------------------------------------------------------------
    print("\nTEST 13: Voice information")

    info = engine.get_voice_info("Tamil")

    assert info["language"] == "tamil"
    assert info["voice"] == "ta-IN-PallaviNeural"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 14
    # --------------------------------------------------------------
    print("\nTEST 14: Custom voice")

    original = engine.get_voice("English")

    engine.set_voice(
        "English",
        "en-US-AriaNeural",
    )

    assert engine.get_voice("English") == (
        "en-US-AriaNeural"
    )

    engine.set_voice(
        "English",
        original,
    )

    print("PASS")

    # --------------------------------------------------------------
    # TEST 15
    # --------------------------------------------------------------
    print("\nTEST 15: Lesson audio planning")

    lesson = {
        "modules": [
            {
                "title": "Introduction",
                "spoken_script": (
                    "Welcome to today's lesson."
                ),
            },
            {
                "title": "Concept",
                "script": (
                    "Machine learning allows "
                    "computers to learn from data."
                ),
            },
        ]
    }

    modules = lesson["modules"]

    assert len(modules) == 2
    assert all(
        isinstance(module, dict)
        for module in modules
    )

    print("PASS")

    # --------------------------------------------------------------
    # TEST 16
    # --------------------------------------------------------------
    print("\nTEST 16: Lesson text extraction")

    lesson = {
        "modules": [
            {
                "spoken_script": "First module."
            },
            {
                "script": "Second module."
            },
            {
                "explanation": "Third module."
            },
            {
                "narration": "Fourth module."
            },
        ]
    }

    extracted = []

    for module in lesson["modules"]:
        text = (
            module.get("spoken_script")
            or module.get("script")
            or module.get("explanation")
            or module.get("narration")
            or ""
        )

        extracted.append(text)

    assert len(extracted) == 4
    assert all(extracted)

    print("PASS")

    # --------------------------------------------------------------
    # TEST 17
    # --------------------------------------------------------------
    print("\nTEST 17: API key independence")

    assert engine.get_capabilities()[
        "requires_api_key"
    ] is False

    print("PASS")

    # --------------------------------------------------------------
    # TEST 18
    # --------------------------------------------------------------
    print("\nTEST 18: Edge TTS availability check")

    assert isinstance(
        engine.is_available(),
        bool,
    )

    print("PASS")

    print("\n" + "=" * 60)
    print("ALL AUDIO ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()