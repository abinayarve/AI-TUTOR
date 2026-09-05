"""
Avatar Component for AI Teacher.

Provides a lightweight interactive teacher avatar for Streamlit.

The component is intentionally independent from Gemini, RAG,
audio generation, and video rendering.

Gemini API key is NOT required.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional


class AvatarComponent:
    """
    Interactive AI teacher avatar.

    The avatar is rendered using HTML/CSS/JavaScript inside Streamlit.
    """

    SUPPORTED_EXPRESSIONS = {
        "neutral",
        "happy",
        "thinking",
        "explaining",
        "questioning",
        "encouraging",
        "concerned",
    }

    SUPPORTED_LANGUAGES = {
        "English",
        "Hinglish",
        "Hindi",
        "Tamil",
        "Spanish",
    }

    EXPRESSION_EMOJIS = {
        "neutral": "🙂",
        "happy": "😊",
        "thinking": "🤔",
        "explaining": "🧑‍🏫",
        "questioning": "❓",
        "encouraging": "🌟",
        "concerned": "💡",
    }

    def __init__(
        self,
        teacher_name: str = "Prof. AI",
        avatar_size: int = 260,
    ) -> None:
        """
        Initialize avatar component.
        """

        self.teacher_name = (
            str(teacher_name).strip()
            if teacher_name
            else "Prof. AI"
        )

        self.avatar_size = max(
            160,
            min(int(avatar_size), 500),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def normalize_expression(
        self,
        expression: Optional[str],
    ) -> str:
        """Normalize avatar expression."""

        if not expression:
            return "neutral"

        value = str(expression).strip().lower()

        if value not in self.SUPPORTED_EXPRESSIONS:
            return "neutral"

        return value

    def normalize_language(
        self,
        language: Optional[str],
    ) -> str:
        """Normalize display language."""

        if not language:
            return "English"

        value = str(language).strip()

        for supported in self.SUPPORTED_LANGUAGES:
            if value.lower() == supported.lower():
                return supported

        return "English"

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate component configuration."""

        return {
            "valid": bool(self.teacher_name),
            "teacher_name": self.teacher_name,
            "avatar_size": self.avatar_size,
            "supported_expressions": sorted(
                self.SUPPORTED_EXPRESSIONS
            ),
            "supported_languages": sorted(
                self.SUPPORTED_LANGUAGES
            ),
            "requires_api_key": False,
        }

    # ------------------------------------------------------------------
    # Avatar state
    # ------------------------------------------------------------------

    def get_state(
        self,
        is_speaking: bool = False,
        expression: str = "neutral",
        language: str = "English",
        topic: str = "",
        status: str = "",
    ) -> Dict[str, Any]:
        """
        Build avatar state.
        """

        normalized_expression = self.normalize_expression(
            expression
        )

        normalized_language = self.normalize_language(
            language
        )

        return {
            "teacher_name": self.teacher_name,
            "is_speaking": bool(is_speaking),
            "expression": normalized_expression,
            "expression_emoji": self.EXPRESSION_EMOJIS[
                normalized_expression
            ],
            "language": normalized_language,
            "topic": str(topic).strip(),
            "status": str(status).strip(),
        }

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def build_html(
        self,
        is_speaking: bool = False,
        expression: str = "neutral",
        language: str = "English",
        topic: str = "",
        status: str = "",
    ) -> str:
        """
        Build the complete avatar HTML.

        The generated HTML is intended for Streamlit's
        components.html().
        """

        state = self.get_state(
            is_speaking=is_speaking,
            expression=expression,
            language=language,
            topic=topic,
            status=status,
        )

        teacher_name = html.escape(
            state["teacher_name"]
        )

        language_text = html.escape(
            state["language"]
        )

        topic_text = html.escape(
            state["topic"]
        )

        status_text = html.escape(
            state["status"]
        )

        expression_class = html.escape(
            state["expression"]
        )

        speaking_class = (
            "speaking"
            if state["is_speaking"]
            else "silent"
        )

        size = self.avatar_size

        return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

.teacher-container {{
    width: 100%;
    min-height: 430px;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    padding: 18px;
}}

.teacher-card {{
    width: min(100%, 420px);

    padding: 18px 20px 22px;

    border-radius: 24px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.96),
            rgba(244,247,255,0.96)
        );

    box-shadow:
        0 12px 35px rgba(0,0,0,0.10);

    border: 1px solid rgba(100,100,100,0.12);

    text-align: center;
}}

.teacher-name {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
}}

.teacher-role {{
    font-size: 13px;
    opacity: 0.65;
    margin-bottom: 14px;
}}

.avatar {{
    position: relative;

    width: {size}px;
    height: {size}px;

    margin: 0 auto 15px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle at 50% 35%,
            #f7d4b5 0%,
            #edbd98 48%,
            #d7966f 100%
        );

    border: 7px solid #ffffff;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.18);

    overflow: hidden;
}}

.hair {{
    position: absolute;

    width: 76%;
    height: 45%;

    left: 12%;
    top: -7%;

    background: #30251f;

    border-radius:
        48% 48% 35% 35%;

    transform: rotate(-2deg);
}}

.hair::after {{
    content: "";

    position: absolute;

    width: 45%;
    height: 70%;

    left: 4%;
    top: 15%;

    background: #30251f;

    border-radius: 50%;
}}

.ear {{
    position: absolute;

    width: 12%;
    height: 20%;

    top: 40%;

    background: #e7ad88;

    border-radius: 50%;
}}

.ear.left {{
    left: -2%;
}}

.ear.right {{
    right: -2%;
}}

.eye {{
    position: absolute;

    width: 12%;
    height: 10%;

    top: 42%;

    background: #222;

    border-radius: 50%;

    animation:
        blink 5s infinite;
}}

.eye.left {{
    left: 27%;
}}

.eye.right {{
    right: 27%;
}}

.eye::after {{
    content: "";

    position: absolute;

    width: 30%;
    height: 30%;

    top: 12%;
    left: 17%;

    background: white;

    border-radius: 50%;
}}

.nose {{
    position: absolute;

    width: 8%;
    height: 13%;

    left: 46%;
    top: 48%;

    border-right: 2px solid rgba(100,60,40,0.45);
    border-bottom: 2px solid rgba(100,60,40,0.45);

    border-radius: 0 0 50% 0;
}}

.mouth {{
    position: absolute;

    width: 26%;
    height: 8%;

    left: 37%;
    top: 65%;

    background: #9e4e59;

    border-radius: 0 0 50% 50%;

    transition:
        all 0.15s ease;
}}

.avatar.speaking .mouth {{
    height: 15%;
    width: 20%;
    left: 40%;

    animation:
        talk 0.28s infinite alternate;
}}

.avatar.thinking .mouth {{
    width: 17%;
    height: 5%;
    left: 41.5%;

    border-radius: 50%;
}}

.avatar.questioning .mouth {{
    width: 18%;
    height: 6%;
    left: 41%;

    border-radius: 50%;
}}

.shoulders {{
    position: absolute;

    width: 100%;
    height: 38%;

    left: 0;
    bottom: -18%;

    background:
        linear-gradient(
            90deg,
            #536dfe,
            #7986cb
        );

    border-radius:
        50% 50% 0 0;
}}

.badge {{
    position: absolute;

    bottom: 11%;

    left: 50%;

    transform: translateX(-50%);

    padding: 5px 10px;

    background: rgba(255,255,255,0.92);

    border-radius: 20px;

    font-size: 11px;

    font-weight: 600;
}}

.status {{
    min-height: 24px;

    font-size: 14px;

    margin: 7px 0;
}}

.topic {{
    font-size: 14px;
    font-weight: 600;

    margin-top: 8px;
}}

.language {{
    display: inline-block;

    margin-top: 8px;

    padding: 5px 10px;

    border-radius: 15px;

    background: rgba(83,109,254,0.10);

    font-size: 12px;

    font-weight: 600;
}}

.speaking-indicator {{
    display: inline-flex;

    align-items: center;

    gap: 4px;

    margin-left: 6px;
}}

.dot {{
    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: currentColor;

    animation:
        pulse 0.8s infinite alternate;
}}

.dot:nth-child(2) {{
    animation-delay: 0.15s;
}}

.dot:nth-child(3) {{
    animation-delay: 0.30s;
}}

@keyframes blink {{
    0%, 45%, 49%, 100% {{
        transform: scaleY(1);
    }}

    47% {{
        transform: scaleY(0.08);
    }}
}}

@keyframes talk {{
    from {{
        transform: scaleY(0.65);
    }}

    to {{
        transform: scaleY(1.2);
    }}
}}

@keyframes pulse {{
    from {{
        opacity: 0.25;
        transform: scale(0.8);
    }}

    to {{
        opacity: 1;
        transform: scale(1.15);
    }}
}}

.avatar.speaking {{
    box-shadow:
        0 0 0 5px rgba(83,109,254,0.10),
        0 12px 35px rgba(0,0,0,0.18);
}}

</style>
</head>

<body>

<div class="teacher-container">

    <div class="teacher-card">

        <div class="teacher-name">
            {teacher_name}
        </div>

        <div class="teacher-role">
            AI Personal Teacher
        </div>

        <div class="avatar {speaking_class} {expression_class}">

            <div class="hair"></div>

            <div class="ear left"></div>
            <div class="ear right"></div>

            <div class="eye left"></div>
            <div class="eye right"></div>

            <div class="nose"></div>

            <div class="mouth"></div>

            <div class="shoulders"></div>

            <div class="badge">
                {state["expression_emoji"]}
            </div>

        </div>

        <div class="status">
            {status_text}

            {
                '''
                <span class="speaking-indicator">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </span>
                '''
                if state["is_speaking"]
                else ""
            }
        </div>

        <div class="topic">
            {topic_text}
        </div>

        <div class="language">
            {language_text}
        </div>

    </div>

</div>

</body>
</html>
"""

    # ------------------------------------------------------------------
    # Streamlit rendering
    # ------------------------------------------------------------------

    def render(
        self,
        is_speaking: bool = False,
        expression: str = "neutral",
        language: str = "English",
        topic: str = "",
        status: str = "",
        height: int = 470,
    ) -> None:
        """
        Render the avatar inside Streamlit.
        """

        try:
            import streamlit.components.v1 as components
        except ImportError as exc:
            raise RuntimeError(
                "Streamlit is required to render the avatar. "
                "Install it using: pip install streamlit"
            ) from exc

        components.html(
            self.build_html(
                is_speaking=is_speaking,
                expression=expression,
                language=language,
                topic=topic,
                status=status,
            ),
            height=height,
            scrolling=False,
        )


# ======================================================================
# BACKWARD-COMPATIBLE FUNCTION
# ======================================================================

def render_interactive_avatar(
    is_speaking: bool = False,
    teacher_name: str = "Prof. AI",
    expression: str = "neutral",
    language: str = "English",
    topic: str = "",
    status: str = "",
    height: int = 470,
) -> None:
    """
    Backward-compatible helper for existing app.py usage.
    """

    avatar = AvatarComponent(
        teacher_name=teacher_name
    )

    avatar.render(
        is_speaking=is_speaking,
        expression=expression,
        language=language,
        topic=topic,
        status=status,
        height=height,
    )


# ======================================================================
# TESTS
# ======================================================================

def run_tests() -> None:
    print("=" * 60)
    print("AVATAR COMPONENT TESTS")
    print("=" * 60)

    avatar = AvatarComponent()

    # --------------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------------
    print("\nTEST 1: Initialization")

    assert avatar is not None
    assert avatar.teacher_name == "Prof. AI"
    assert avatar.avatar_size >= 160

    print("PASS")

    # --------------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------------
    print("\nTEST 2: Configuration")

    config = avatar.validate_configuration()

    assert config["valid"] is True
    assert config["requires_api_key"] is False

    print("PASS")

    # --------------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------------
    print("\nTEST 3: Expression normalization")

    assert avatar.normalize_expression(
        "happy"
    ) == "happy"

    assert avatar.normalize_expression(
        "THINKING"
    ) == "thinking"

    assert avatar.normalize_expression(
        "unknown"
    ) == "neutral"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------------
    print("\nTEST 4: Language normalization")

    assert avatar.normalize_language(
        "English"
    ) == "English"

    assert avatar.normalize_language(
        "Tamil"
    ) == "Tamil"

    assert avatar.normalize_language(
        "Hindi"
    ) == "Hindi"

    assert avatar.normalize_language(
        "Spanish"
    ) == "Spanish"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 5
    # --------------------------------------------------------------
    print("\nTEST 5: Speaking state")

    state = avatar.get_state(
        is_speaking=True,
        expression="explaining",
        language="English",
        topic="Machine Learning",
        status="Explaining the concept...",
    )

    assert state["is_speaking"] is True
    assert state["expression"] == "explaining"
    assert state["topic"] == "Machine Learning"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 6
    # --------------------------------------------------------------
    print("\nTEST 6: Silent state")

    state = avatar.get_state(
        is_speaking=False,
        expression="thinking",
    )

    assert state["is_speaking"] is False
    assert state["expression"] == "thinking"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 7
    # --------------------------------------------------------------
    print("\nTEST 7: HTML generation")

    html_output = avatar.build_html(
        is_speaking=True,
        expression="explaining",
        language="English",
        topic="Neural Networks",
        status="Teaching...",
    )

    assert isinstance(html_output, str)
    assert "<html>" in html_output
    assert "Neural Networks" in html_output
    assert "Teaching..." in html_output
    assert "speaking" in html_output

    print("PASS")

    # --------------------------------------------------------------
    # TEST 8
    # --------------------------------------------------------------
    print("\nTEST 8: Tamil rendering state")

    state = avatar.get_state(
        is_speaking=True,
        expression="encouraging",
        language="Tamil",
        topic="Machine Learning",
    )

    assert state["language"] == "Tamil"
    assert state["expression"] == "encouraging"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 9
    # --------------------------------------------------------------
    print("\nTEST 9: Hindi rendering state")

    state = avatar.get_state(
        is_speaking=True,
        expression="questioning",
        language="Hindi",
    )

    assert state["language"] == "Hindi"
    assert state["expression"] == "questioning"

    print("PASS")

    # --------------------------------------------------------------
    # TEST 10
    # --------------------------------------------------------------
    print("\nTEST 10: HTML escaping")

    html_output = avatar.build_html(
        teacher_name if False else False,
        topic="<script>alert('x')</script>",
    )

    assert "<script>alert('x')</script>" not in html_output

    print("PASS")

    # --------------------------------------------------------------
    # TEST 11
    # --------------------------------------------------------------
    print("\nTEST 11: Custom teacher")

    custom = AvatarComponent(
        teacher_name="Dr. Nova",
        avatar_size=320,
    )

    assert custom.teacher_name == "Dr. Nova"
    assert custom.avatar_size == 320

    html_output = custom.build_html()

    assert "Dr. Nova" in html_output

    print("PASS")

    # --------------------------------------------------------------
    # TEST 12
    # --------------------------------------------------------------
    print("\nTEST 12: Size bounds")

    small = AvatarComponent(
        avatar_size=50
    )

    large = AvatarComponent(
        avatar_size=1000
    )

    assert small.avatar_size == 160
    assert large.avatar_size == 500

    print("PASS")

    # --------------------------------------------------------------
    # TEST 13
    # --------------------------------------------------------------
    print("\nTEST 13: Expression coverage")

    for expression in AvatarComponent.SUPPORTED_EXPRESSIONS:
        state = avatar.get_state(
            expression=expression
        )

        assert state["expression"] == expression

    print("PASS")

    # --------------------------------------------------------------
    # TEST 14
    # --------------------------------------------------------------
    print("\nTEST 14: Language coverage")

    for language in AvatarComponent.SUPPORTED_LANGUAGES:
        state = avatar.get_state(
            language=language
        )

        assert state["language"] == language

    print("PASS")

    # --------------------------------------------------------------
    # TEST 15
    # --------------------------------------------------------------
    print("\nTEST 15: Backward-compatible helper")

    assert callable(
        render_interactive_avatar
    )

    print("PASS")

    # --------------------------------------------------------------
    # TEST 16
    # --------------------------------------------------------------
    print("\nTEST 16: API-key independence")

    assert avatar.validate_configuration()[
        "requires_api_key"
    ] is False

    print("PASS")

    print("\n" + "=" * 60)
    print("ALL AVATAR COMPONENT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()