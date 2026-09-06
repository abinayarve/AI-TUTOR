import os
import time
import inspect
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ================================================================
# CORE CONFIGURATION
# ================================================================
from config import (
    DATA_DIR,
    DEFAULT_GEMINI_MODEL,
    QUESTION_TYPES,
)


# ================================================================
# CORE ENGINES
# ================================================================
from rag_engine import RAGEngine
RAGKnowledgeEngine = RAGEngine
from lesson_generator import LessonGeneratorEngine

# Use a current stable Gemini model for new API projects.
# Some API keys/projects may not have access to Gemini 2.5 models.
RUNTIME_GEMINI_MODEL = "gemini-3.5-flash"

def _set_runtime_model(module_name):
    try:
        import importlib
        module = importlib.import_module(module_name)
        if hasattr(module, "DEFAULT_GEMINI_MODEL"):
            module.DEFAULT_GEMINI_MODEL = RUNTIME_GEMINI_MODEL
    except Exception:
        pass

for _module_name in [
    "lesson_generator",
    "teacher_agent",
    "assessment_engine",
    "misconception_engine",
    "visual_engine",
    "video_engine",
]:
    _set_runtime_model(_module_name)

from teacher_agent import TeacherAgent
from assessment_engine import AssessmentEngine
from misconception_engine import MisconceptionEngine
from visual_engine import VisualEngine
from video_engine import VideoEngine
from audio_engine import AudioEngine
from avatar_component import AvatarComponent

from database import DatabaseManager
from learner_profile import LearnerProfileManager
from learning_memory import LearningMemory


# ================================================================
# UI COMPONENTS
# ================================================================
from components.progress import ProgressComponent
from components.chat import ChatComponent
from components.video_player import VideoPlayer


# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="PedagogyEngine AI - Interactive AI Teacher",
    page_icon="🎓",
    layout="wide",
)


# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown(
    """
    <style>

        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            opacity: 0.8;
            margin-bottom: 1.5rem;
        }

        .concept-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 0.7rem;
        }

        .teacher-box {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(99,102,241,0.25);
            background: rgba(99,102,241,0.05);
        }

        .success-box {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(34,197,94,0.35);
            background: rgba(34,197,94,0.08);
        }

        .warning-box {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(234,179,8,0.35);
            background: rgba(234,179,8,0.08);
        }

        .small-muted {
            font-size: 0.85rem;
            opacity: 0.7;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            text-align: center;
        }

        .student-hero{padding:1.5rem 1.7rem;border-radius:24px;background:linear-gradient(135deg,#eef2ff,#fdf2f8 55%,#ecfeff);border:1px solid rgba(99,102,241,.18);box-shadow:0 12px 35px rgba(79,70,229,.10);margin-bottom:1.2rem;color:#172033!important}.student-hero *{color:#172033!important}.student-hero h1{margin:0;font-size:2.45rem;font-weight:800}.student-hero p{margin:.45rem 0 0;font-size:1.02rem;opacity:.78}.feature-card{padding:1.05rem;border-radius:18px;min-height:130px;background:rgba(255,255,255,.92);border:1px solid rgba(148,163,184,.22);box-shadow:0 8px 24px rgba(15,23,42,.055);color:#172033!important}.feature-card *{color:#172033!important}.feature-icon{font-size:1.8rem}.feature-title{font-weight:750;margin-top:.35rem}.feature-text{font-size:.86rem;opacity:.72;margin-top:.2rem}.flow-pill{display:inline-block;padding:.42rem .72rem;margin:.2rem;border-radius:999px;background:rgba(99,102,241,.09);border:1px solid rgba(99,102,241,.15);font-size:.82rem;font-weight:650;color:#3730a3!important}.voice-card{padding:1rem;border-radius:18px;background:linear-gradient(135deg,rgba(14,165,233,.08),rgba(168,85,247,.08));border:1px solid rgba(14,165,233,.16);color:#172033!important}.voice-card *{color:#172033!important}.lesson-banner{padding:1rem 1.2rem;border-radius:18px;background:linear-gradient(90deg,rgba(99,102,241,.10),rgba(16,185,129,.08));border:1px solid rgba(99,102,241,.15);color:#172033!important}.lesson-banner *{color:#172033!important}.small-muted{color:#475569!important}.badge{display:inline-block;padding:.28rem .62rem;border-radius:999px;background:rgba(16,185,129,.10);color:#047857!important;font-size:.78rem;font-weight:700}.stButton>button{border-radius:12px!important;font-weight:650!important}[data-testid="stMetric"]{border-radius:14px}
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
def initialize_session_state():

    defaults = {
        "page": "home",

        "rag_engine": None,
        "lesson_plan": None,
        "current_module_idx": 0,

        "gemini_api_key": os.getenv(
            "GEMINI_API_KEY",
            os.getenv("GOOGLE_API_KEY", ""),
        ),

        "groq_api_key": os.getenv("GROQ_API_KEY", ""),

        "grounded_context": "",
        "source_name": "",
        "source_type": "topic",

        "audio_file": None,
        "audio_requested": False,
        "selected_voice": None,
        "voice_rate": "normal",
        "avatar_expression": "explaining",
        "current_visual_type": None,
        "current_video_file": None,

        "user_score": 0,
        "total_questions": 0,

        "misconceptions": [],
        "assessment_results": [],

        "current_question_answered": False,
        "current_question_correct": False,

        "remediation": None,
        "retest_mode": False,
        "retest_answered": False,

        "learner_id": "default_learner",
        "learner_name": "",

        "learner_level": "Beginner",
        "learner_goal": "Understand the topic",
        "learner_language": "English",

        "learning_style": "Visual",
        "preferred_depth": "Standard",

        "session_duration": 20,

        "chat_messages": [],

        "lesson_started": False,
        "lesson_completed": False,

        "video_plan": None,
        "video_manifest": None,

        "topic_mastery": 0.0,

        "setup_complete": False,
        "generation_timings": {},
        "video_plan_requested": False,

        # Chapter/topic focus requested after uploading material.
        "focus_topic": "",

        # Sidebar module-completion dashboard.
        "completed_modules": [],

        # Mid-lesson doubt box (uses components.chat.ChatComponent).
        "doubt_chat": None,

        # Real rendered MP4 pipeline.
        "module_audio_files": {},
        "module_audio_errors": {},
        "lesson_video_path": None,
        "lesson_video_status": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ================================================================
# ENGINE INITIALIZATION
# ================================================================
@st.cache_resource(show_spinner=False)
def get_rag_engine():
    # Heavy embedding/vector resources are created only when RAG is actually needed.
    return RAGKnowledgeEngine()


@st.cache_resource(show_spinner=False)
def get_database():
    db_path = os.path.join(
        DATA_DIR,
        "ai_teacher.db",
    )

    return DatabaseManager(
        db_path=db_path,
    )


# Streamlit reruns this script after interactions. Cache reusable engines so
# their underlying LLM/client resources are not reconstructed on every rerun.
@st.cache_resource(show_spinner=False)
def get_teacher_engine(api_key):
    return TeacherAgent(api_key=api_key or None)


@st.cache_resource(show_spinner=False)
def get_assessment_engine(api_key):
    return AssessmentEngine(api_key=api_key or None)


@st.cache_resource(show_spinner=False)
def get_misconception_engine(api_key):
    return MisconceptionEngine(api_key=api_key or None)


@st.cache_resource(show_spinner=False)
def get_visual_engine(api_key):
    return VisualEngine(api_key=api_key or None)


@st.cache_resource(show_spinner=False)
def get_video_engine(api_key):
    return VideoEngine(api_key=api_key or None)


@st.cache_resource(show_spinner=False)
def get_audio_engine():
    return AudioEngine()


@st.cache_resource(show_spinner=False)
def get_avatar_engine():
    return AvatarComponent()


@st.cache_resource(show_spinner=False)
def get_lesson_generator(api_key):
    return LessonGeneratorEngine(api_key=api_key or None)


def get_engines():
    api_key = st.session_state.gemini_api_key
    database = get_database()

    profile_manager = LearnerProfileManager(
        learner_id=st.session_state.learner_id,
        database=database,
    )

    return {
        "database": database,
        "profile": profile_manager,
        "teacher": get_teacher_engine(api_key),
        "assessment": get_assessment_engine(api_key),
        "misconception": get_misconception_engine(api_key),
        "visual": get_visual_engine(api_key),
        "video": get_video_engine(api_key),
        "audio": get_audio_engine(),
        "avatar": get_avatar_engine(),
    }


# ================================================================
# LEARNING MEMORY
# ================================================================
def get_learning_memory():

    database = get_database()

    profile_manager = LearnerProfileManager(
        learner_id=st.session_state.learner_id,
        database=database,
    )

    try:

        signature = inspect.signature(
            LearningMemory
        )

        candidate_values = {

            "database": database,

            "db": database,

            "db_manager": database,

            "learner_id":
                st.session_state.learner_id,

            "profile_manager":
                profile_manager,

            "learner_profile":
                profile_manager,
        }

        kwargs = {}

        for name in signature.parameters:

            if name == "self":
                continue

            if name in candidate_values:
                kwargs[name] = candidate_values[name]

        return LearningMemory(**kwargs)

    except Exception:

        try:
            return LearningMemory(database)

        except Exception:
            return None


# ================================================================
# SAFE METHOD CALLER
# ================================================================
def safe_method_call(
    obj,
    method_name,
    **kwargs,
):

    if obj is None:
        return None

    method = getattr(
        obj,
        method_name,
        None,
    )

    if method is None:
        return None

    try:

        signature = inspect.signature(method)

        accepted = {}

        for name in signature.parameters:

            if name in kwargs:
                accepted[name] = kwargs[name]

        return method(**accepted)

    except Exception:
        return None


# ================================================================
# PROFILE SETUP
# ================================================================
def create_or_update_learner_profile():

    engines = get_engines()

    profile_manager = engines["profile"]

    return safe_method_call(
        profile_manager,
        "create_or_update_profile",

        name=st.session_state.learner_name,

        level=st.session_state.learner_level,

        goal=st.session_state.learner_goal,

        language=st.session_state.learner_language,

        learning_style=st.session_state.learning_style,

        preferred_depth=st.session_state.preferred_depth,
    )


# ================================================================
# TOPIC FROM FILENAME
# ================================================================
def infer_topic_from_filename(filename):

    if not filename:
        return "Educational Topic"

    name = Path(filename).stem

    name = name.replace(
        "_",
        " ",
    )

    name = name.replace(
        "-",
        " ",
    )

    return name.strip().title()


# ================================================================
# VISUAL RENDERING
# ================================================================
def render_visual_content(
    visual_type,
    visual_content,
    topic,
    concept="",
    visual_data=None,
):

    visual_data = visual_data or {}

    visual_type = str(
        visual_type or "TEXT"
    ).upper()

    topic_lower = (
        f"{topic} {concept}"
    ).lower()


    # ============================================================
    # OHM'S LAW / CIRCUIT
    # ============================================================
    if (
        "ohm" in topic_lower
        or "circuit" in topic_lower
        or "electric" in topic_lower
        or visual_type in {
            "SIMULATION",
            "EQUATION",
        }
    ):

        st.markdown(
            "### ⚡ Interactive Circuit / Ohm's Law Visual"
        )

        voltage = st.slider(
            "Voltage (V)",
            min_value=1,
            max_value=24,
            value=12,
            key=f"voltage_{st.session_state.current_module_idx}",
        )

        resistance = st.slider(
            "Resistance (Ω)",
            min_value=1,
            max_value=50,
            value=6,
            key=f"resistance_{st.session_state.current_module_idx}",
        )

        current = voltage / resistance

        resistance_range = np.linspace(
            1,
            50,
            100,
        )

        current_range = (
            voltage / resistance_range
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=resistance_range,
                y=current_range,
                mode="lines",
                name="I = V / R",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[resistance],
                y=[current],
                mode="markers+text",
                text=[
                    f"I = {current:.2f} A"
                ],
                textposition="top right",
                marker=dict(
                    size=14,
                ),
            )
        )

        fig.update_layout(
            xaxis_title="Resistance (Ω)",
            yaxis_title="Current (A)",
            height=320,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.latex(
            r"I = \frac{V}{R}"
        )

        st.info(
            f"With V = {voltage} V and "
            f"R = {resistance} Ω, "
            f"the current is {current:.2f} A."
        )

        return


    # ============================================================
    # EQUATION
    # ============================================================
    if visual_type in {
        "EQUATION",
        "LATEX",
    }:

        st.markdown(
            "### 🧮 Mathematical Representation"
        )

        if visual_content:

            try:
                st.latex(
                    visual_content
                )

            except Exception:
                st.code(
                    visual_content
                )

        else:
            st.info(
                "Equation generated by the AI Teacher."
            )

        return


    # ============================================================
    # GRAPH
    # ============================================================
    if visual_type == "GRAPH":

        st.markdown(
            "### 📈 Concept Graph"
        )

        points = visual_data.get("chart_points") or []

        if points and len(points) >= 2:
            x = [p[0] for p in points]
            y = [p[1] for p in points]
        else:
            # Concept-derived fallback (not a fixed generic curve):
            # different concepts at least produce a different-looking
            # curve instead of always the same y = x^2.
            seed = max(1, len(concept or topic or "concept") % 9 + 1)
            x = list(range(1, 11))
            y = [seed * (11 - xi) for xi in x]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=concept or "Concept Relationship",
            )
        )

        fig.update_layout(
            xaxis_title=visual_data.get("x_label") or "Input",
            yaxis_title=visual_data.get("y_label") or "Output",
            height=320,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        if visual_content:
            st.caption(
                visual_content
            )

        return


    # ============================================================
    # CODE
    # ============================================================
    if visual_type == "CODE":

        st.markdown(
            "### 💻 Programming Example"
        )

        st.code(
            visual_content
            or "# Example code will appear here",
            language="python",
        )

        st.info(
            "Code is displayed safely as an educational "
            "example. Arbitrary code execution is disabled."
        )

        return


    # ============================================================
    # FLOWCHART
    # ============================================================
    if visual_type == "FLOWCHART":

        st.markdown(
            "### 🔄 Process Flow"
        )

        steps = visual_data.get("flow_steps") or []

        if not steps:
            # Derive concept-specific steps from visual_content instead
            # of a fixed generic "Input/Process/Decision/Output" set.
            raw = (visual_content or concept or "").strip()
            parts = [
                p.strip(" .")
                for p in raw.replace(";", ",").split(",")
                if p.strip()
            ]
            steps = [p for p in parts if 2 <= len(p) <= 40][:6]

        if not steps:
            steps = [
                f"Introduce {concept or topic}",
                "Explain the mechanism",
                "Show the result",
            ]

        cols = st.columns(
            len(steps)
        )

        for index, step in enumerate(
            steps
        ):

            with cols[index]:

                st.markdown(
                    f"""
                    <div class="concept-card">
                        <b>{index + 1}. {step}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        return


    # ============================================================
    # TIMELINE
    # ============================================================
    if visual_type == "TIMELINE":

        st.markdown(
            "### 🕒 Timeline"
        )

        timeline_points = visual_data.get("timeline_events") or []

        if not timeline_points:
            raw = (visual_content or concept or "").strip()
            sentences = [
                s.strip()
                for s in raw.replace("\n", ". ").split(".")
                if s.strip()
            ]
            timeline_points = sentences[:6]

        if not timeline_points:
            timeline_points = [
                f"Origin of {concept or topic}",
                "Key development",
                "Present-day significance",
            ]

        for index, point in enumerate(
            timeline_points
        ):

            st.markdown(
                f"**{index + 1}. {point}**"
            )

        return


    # ============================================================
    # TABLE
    # ============================================================
    if visual_type == "TABLE":

        st.markdown(
            "### 📋 Concept Comparison"
        )

        rows = visual_data.get("table_rows") or []

        if rows:
            aspects = [str(r[0]) if len(r) > 0 else "" for r in rows]
            explanations = [str(r[1]) if len(r) > 1 else "" for r in rows]
        else:
            aspects = ["Definition", "Key Idea", "Application"]
            explanations = [
                visual_content or f"What {concept or topic} means",
                f"The core idea behind {concept or topic}",
                f"Where {concept or topic} is used in practice",
            ]

        st.dataframe(
            {
                "Aspect": aspects,

                "Explanation": explanations,
            },

            use_container_width=True,
        )

        return


    # ============================================================
    # DEFAULT
    # ============================================================
    st.markdown(
        "### 📝 Visual Explanation"
    )

    if visual_content:

        st.markdown(
            visual_content
        )

    else:

        st.info(
            "The AI Teacher is preparing "
            "a visual explanation."
        )


# ================================================================
# PHASE 2 — MULTI-VOICE AUDIO + TEACHER MEDIA
# ================================================================
def _voice_options_for_language(language):
    """Return friendly voice labels mapped to Edge-TTS voice IDs."""
    language_key = str(language or "English")
    normalized = language_key.lower()
    aliases = {
        "english": "English",
        "hinglish": "Hinglish",
        "hindi": "Hindi",
        "tamil": "Tamil",
        "spanish": "Spanish",
    }
    canonical = aliases.get(normalized, language_key)
    voices = getattr(AudioEngine, 'DEFAULT_VOICES', {}).get(canonical, {})
    if isinstance(voices, str):
        voices = {"Default": voices}
    return voices or {"Default": "en-US-ChristopherNeural"}


def render_voice_controls():
    """Phase-2 voice studio. Changing controls never regenerates the lesson."""
    audio_engine = get_audio_engine()
    language = st.session_state.learner_language
    voices = _voice_options_for_language(language)
    labels = list(voices.keys())

    current_voice = st.session_state.get("selected_voice")
    if current_voice not in voices.values():
        current_voice = labels[0] if labels else "Default"
        st.session_state.selected_voice = voices[current_voice]

    current_label = next(
        (label for label, voice_id in voices.items() if voice_id == current_voice),
        labels[0] if labels else "Default",
    )

    with st.expander("🎙️ Teacher Voice Studio", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            selected_label = st.selectbox(
                "Teacher Voice",
                labels,
                index=labels.index(current_label) if current_label in labels else 0,
                key="voice_selector",
                help="Choose the AI teacher's voice for the current language.",
            )
            selected_voice = voices[selected_label]
            st.session_state.selected_voice = selected_voice
        with c2:
            rate = st.selectbox(
                "Speaking Speed",
                ["slow", "normal", "fast"],
                index=["slow", "normal", "fast"].index(
                    st.session_state.get("voice_rate", "normal")
                ),
                key="voice_rate_selector",
            )
            st.session_state.voice_rate = rate

        st.markdown(f'<div class="voice-card">🎙️ <b>{selected_label}</b> · {language} · {rate.title()} speed<br><span class="small-muted">Natural AI narration • Choose your preferred teacher voice</span></div>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔊 Generate / Replay Voice", use_container_width=True, key="generate_voice_phase2"):
                st.session_state.audio_file = None
                st.session_state.audio_requested = False
                st.rerun()
        with b2:
            if st.button("🔄 Reset Voice", use_container_width=True, key="reset_voice_phase2"):
                st.session_state.selected_voice = None
                st.session_state.voice_rate = "normal"
                st.session_state.audio_file = None
                st.session_state.audio_requested = False
                st.rerun()


def generate_audio(script, filename, expression="explaining"):
    """Generate natural speech using the selected Phase-2 voice."""
    output_dir = Path(DATA_DIR) / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    audio_engine = get_audio_engine()

    if not audio_engine.is_available():
        st.session_state.last_audio_error = (
            "The edge-tts package is not installed. Run: "
            "pip install edge-tts"
        )
        return None

    try:
        language = st.session_state.learner_language
        voice = st.session_state.get("selected_voice")
        rate = st.session_state.get("voice_rate", "normal")

        # Validate and normalize the requested voice through the engine.
        if voice:
            try:
                audio_engine.set_voice(language, voice)
            except Exception:
                pass

        result = audio_engine.generate_speech(
            text=script,
            language=language,
            voice=voice,
            rate=rate,
            output_path=str(output_path),
        )
        if result:
            st.session_state.avatar_expression = expression
            st.session_state.last_audio_error = None
            return result

        st.session_state.last_audio_error = (
            "Audio engine returned no output path."
        )
        return None

    except Exception as exc:
        # NOTE: there used to be a "backward-compatible fallback" here
        # that called `from audio_engine import generate_speech` — but
        # no such module-level function exists in audio_engine.py
        # (only the AudioEngine class method above does), so it always
        # raised ImportError and silently returned None, hiding the
        # real error from the user. Surface the real error instead.
        st.session_state.last_audio_error = str(exc)
        return None


def prepare_module_audio(module_index):
    plan = st.session_state.lesson_plan
    if not plan:
        return None
    modules = plan.get("modules", [])
    if module_index >= len(modules):
        return None

    module = modules[module_index]
    script = module.get("spoken_script", "")
    if not script:
        return None

    filename = (
        f"lesson_{st.session_state.learner_id}_"
        f"module_{module_index}_"
        f"{st.session_state.get('selected_voice', 'default').replace('-', '_')}_"
        f"{st.session_state.get('voice_rate', 'normal')}.mp3"
    )

    expression = "explaining"
    if st.session_state.get("retest_mode"):
        expression = "questioning"

    audio_path = generate_audio(script, filename, expression=expression)
    st.session_state.audio_file = audio_path
    st.session_state.audio_requested = bool(audio_path)
    return audio_path


def prepare_remediation_audio(script, module_index):
    filename = (
        f"lesson_{st.session_state.learner_id}_remediation_"
        f"{module_index}_{st.session_state.get('voice_rate', 'normal')}.mp3"
    )
    path = generate_audio(script, filename, expression="concerned")
    st.session_state.audio_file = path
    return path


def render_media_sync_banner(is_speaking):
    """Visual synchronization cue for the avatar + narration player."""
    if is_speaking:
        st.success("🔴 **Prof. AI is speaking** — follow the highlighted explanation below.")
    else:
        st.caption("🟢 Teacher is ready. Generate the narration to start the speaking state.")


# ================================================================
# PREPARE VIDEO PLAN
# ================================================================
def prepare_video_plan():

    if not st.session_state.lesson_plan:
        return None

    video_engine = get_engines()["video"]

    plan = (
        st.session_state.lesson_plan
    )


    # ------------------------------------------------------------
    # Preferred video-plan API
    # ------------------------------------------------------------
    try:

        result = video_engine.create_video_plan(
            lesson_plan=plan,
        )

        st.session_state.video_plan = result

        return result

    except Exception:
        pass


    # ------------------------------------------------------------
    # Scene-based fallback
    # ------------------------------------------------------------
    try:

        scenes = video_engine.create_lesson_scenes(
            lesson_plan=plan,
        )

        timeline = video_engine.build_timeline(
            scenes,
        )

        result = {
            "scenes": scenes,
            "timeline": timeline,
        }

        st.session_state.video_plan = result

        return result

    except Exception:
        return None


# ================================================================
# RECORD TEACHING EVENT
# ================================================================
def record_teaching_event(
    action,
    topic,
    concept,
    details=None,
):

    memory = get_learning_memory()

    safe_method_call(
        memory,

        "record_teaching_action",

        action=action,

        topic=topic,

        concept=concept,

        details=details or {},
    )


# ================================================================
# RECORD LESSON START
# ================================================================
def record_lesson_start():

    if not st.session_state.lesson_plan:
        return

    memory = get_learning_memory()

    safe_method_call(
        memory,

        "record_lesson_started",

        topic=
            st.session_state.lesson_plan.get(
                "topic_title",
                "Unknown Topic",
            ),
    )


# ================================================================
# RECORD ASSESSMENT
# ================================================================
def record_assessment_memory(
    question,
    answer,
    correct,
    concept,
    mastery,
):

    memory = get_learning_memory()

    topic = (
        st.session_state.lesson_plan.get(
            "topic_title",
            "Unknown Topic",
        )
    )

    safe_method_call(
        memory,

        "record_assessment_result",

        topic=topic,

        concept=concept,

        question=question,

        answer=answer,

        correct=correct,

        mastery=mastery,
    )

    safe_method_call(
        memory,

        "update_concept_from_assessment",

        topic=topic,

        concept=concept,

        mastery=mastery,

        correct=correct,
    )


# ================================================================
# NORMALIZE ANSWER
# ================================================================
def normalize_answer(
    answer,
):

    if answer is None:
        return ""

    return str(
        answer
    ).strip().lower()


# ================================================================
# EVALUATE ANSWER
# ================================================================
def evaluate_checkpoint(
    checkpoint,
    user_answer,
    concept,
):

    assessment = get_engines()[
        "assessment"
    ]

    correct_answer = checkpoint.get(
        "correct_answer",
        "",
    )

    deterministic_correct = (
        normalize_answer(
            user_answer
        )
        ==
        normalize_answer(
            correct_answer
        )
    )


    possible_methods = [
        "evaluate_answer",
        "evaluate_response",
        "assess_answer",
        "evaluate",
    ]


    for method_name in possible_methods:

        method = getattr(
            assessment,
            method_name,
            None,
        )

        if method is None:
            continue

        try:

            signature = inspect.signature(
                method
            )

            candidate = {

                "question":
                    checkpoint.get(
                        "question",
                        "",
                    ),

                "user_answer":
                    user_answer,

                "answer":
                    user_answer,

                "correct_answer":
                    correct_answer,

                "expected_answer":
                    correct_answer,

                "question_type":
                    checkpoint.get(
                        "question_type",
                        "MCQ",
                    ),

                "concept":
                    concept,

                "topic":
                    st.session_state.lesson_plan.get(
                        "topic_title",
                        "",
                    ),
            }

            kwargs = {
                key: value
                for key, value in candidate.items()
                if key in signature.parameters
            }

            result = method(
                **kwargs
            )


            if isinstance(
                result,
                bool,
            ):

                return {
                    "correct":
                        result,

                    "score":
                        1.0
                        if result
                        else 0.0,

                    "raw":
                        result,
                }


            if isinstance(
                result,
                dict,
            ):

                score = result.get(
                    "score",
                    result.get(
                        "mastery",
                        result.get(
                            "confidence",
                            None,
                        ),
                    ),
                )

                correct = result.get(
                    "correct",
                    result.get(
                        "is_correct",
                        None,
                    ),
                )


                if (
                    correct is None
                    and score is not None
                ):

                    try:

                        correct = (
                            float(score)
                            >= 0.70
                        )

                    except Exception:

                        correct = (
                            deterministic_correct
                        )


                if correct is None:

                    correct = (
                        deterministic_correct
                    )


                return {

                    "correct":
                        bool(correct),

                    "score":
                        float(score)
                        if score is not None
                        else (
                            1.0
                            if correct
                            else 0.0
                        ),

                    "raw":
                        result,
                }

        except Exception:
            continue


    # ------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------
    return {

        "correct":
            deterministic_correct,

        "score":
            1.0
            if deterministic_correct
            else 0.0,

        "raw":
            None,
    }


# ================================================================
# MISCONCEPTION + REMEDIATION
# ================================================================
def generate_remediation(
    question,
    wrong_answer,
    concept,
):

    misconception_engine = (
        get_engines()["misconception"]
    )

    topic = (
        st.session_state.lesson_plan.get(
            "topic_title",
            "",
        )
    )


    method_names = [
        "analyze_response",
        "detect_misconception",
        "analyze_misconception",
        "generate_remediation",
    ]


    for method_name in method_names:

        method = getattr(
            misconception_engine,
            method_name,
            None,
        )

        if method is None:
            continue

        try:

            signature = inspect.signature(
                method
            )

            candidate = {

                "topic":
                    topic,

                "concept":
                    concept,

                "question":
                    question,

                "user_answer":
                    wrong_answer,

                "wrong_answer":
                    wrong_answer,

                "answer":
                    wrong_answer,

                "language":
                    st.session_state.learner_language,
            }

            kwargs = {
                key: value
                for key, value in candidate.items()
                if key in signature.parameters
            }

            result = method(
                **kwargs
            )

            if isinstance(
                result,
                dict,
            ):
                return result

        except Exception:
            continue


    # ------------------------------------------------------------
    # LessonGenerator fallback
    # ------------------------------------------------------------
    try:

        generator = get_lesson_generator(
            st.session_state.gemini_api_key
        )

        result = generator.generate_remediation(
            topic=topic,

            question=question,

            wrong_answer=wrong_answer,

            language=
                st.session_state.learner_language,
        )

        if result:
            return result

    except Exception:
        pass


    # ------------------------------------------------------------
    # Final deterministic fallback
    # ------------------------------------------------------------
    return {

        "new_analogy":
            "Let's look at the same concept "
            "from a simpler real-world perspective.",

        "misconception_reason":
            "Your answer suggests that one part "
            "of the concept may need clarification.",

        "spoken_remediation_script":
            f"Let's revisit {concept} step by step. "
            "The key idea is to focus on the relationship "
            "between the main parts of the concept.",
    }


# ================================================================
# TEACHER AGENT DECISION
# ================================================================
def get_teacher_decision(
    concept,
    mastery,
    misconception=None,
    attempts=0,
):

    teacher = get_engines()[
        "teacher"
    ]

    try:

        return teacher.decide_from_mastery(
            mastery=mastery,

            misconception=
                misconception,

            attempts=
                attempts,
        )

    except Exception:

        return {

            "action":
                (
                    "CONTINUE"
                    if mastery >= 0.70
                    else "SIMPLIFY"
                ),

            "reason":
                "Default pedagogical decision",
        }


# ================================================================
# MID-LESSON DOUBT / FOLLOW-UP QUESTION
# ================================================================
def answer_doubt(question, module, topic, level, language):
    """
    Answer a free-form follow-up question from the student while
    keeping the current module's lesson context, without disrupting
    the main teaching flow. Uses get_chat_llm() (Groq preferred,
    Gemini fallback) so it works even when the structured
    LessonGeneratorEngine path is unavailable.
    """

    gemini_key = str(st.session_state.get("gemini_api_key", "") or "").strip()
    groq_key = str(st.session_state.get("groq_api_key", "") or os.getenv("GROQ_API_KEY", "") or "").strip()

    if not gemini_key and not groq_key:
        return "Please add a Groq (free) or Gemini API key in AI Configuration to ask a doubt."

    try:
        from llm_provider import get_chat_llm
    except Exception as exc:
        return f"AI is unavailable right now: {exc}"

    context_script = module.get("spoken_script", "") if isinstance(module, dict) else ""
    concept = module.get("concept", module.get("module_title", topic)) if isinstance(module, dict) else topic

    prompt = f"""
You are PedagogyEngine AI, a warm human-like teacher, currently in the
middle of teaching a lesson.

LESSON TOPIC: {topic}
CURRENT CONCEPT BEING TAUGHT: {concept}
LEARNER LEVEL: {level}
TEACHING LANGUAGE: {language}

WHAT YOU JUST EXPLAINED TO THE STUDENT:
{context_script}

The student has a follow-up doubt about what you just taught. Answer it
directly and conversationally, staying strictly consistent with what you
already explained above. Keep it concise (3-6 sentences), use the same
teaching language ({language}), and end by gently inviting them to
continue the lesson when ready. Do not repeat the entire explanation.

STUDENT'S QUESTION:
{question}
"""

    try:
        llm = get_chat_llm(
            temperature=0.4,
            max_output_tokens=1024,
            groq_api_key=groq_key,
            gemini_api_key=gemini_key,
        )
        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response)).strip()
        if answer:
            return answer
    except Exception:
        pass

    return "Sorry, I couldn't answer that right now — please try again."


def get_doubt_chat():
    """Lazily build the ChatComponent used for the mid-lesson doubt box."""

    if st.session_state.get("doubt_chat") is None:
        try:
            from components.chat import ChatComponent
            st.session_state.doubt_chat = ChatComponent()
        except Exception:
            st.session_state.doubt_chat = False

    return st.session_state.doubt_chat


def render_doubt_box(current_module, topic, level, language):
    """Mid-lesson 'ask a doubt' box (Task 2: follow-up questions while
    maintaining lesson context), built on the previously-unused
    ChatComponent."""

    with st.expander("🙋 Ask a Doubt", expanded=False):

        chat = get_doubt_chat()

        if not chat:
            st.info(
                "Doubt box is unavailable right now — "
                "the chat component could not be loaded."
            )
            return

        chat.set_topic(topic)
        chat.set_language(language)

        if chat.message_count() == 0:
            chat.add_teacher_message(
                "Ask me anything about what we just covered — "
                "I'll answer without losing our place in the lesson."
            )

        chat.render()

        doubt_text = st.text_input(
            "Your question",
            key=f"doubt_input_{st.session_state.current_module_idx}",
            placeholder="e.g. Why does current decrease when resistance increases?",
        )

        if st.button(
            "Ask",
            key=f"ask_doubt_{st.session_state.current_module_idx}",
            use_container_width=True,
        ):
            if doubt_text.strip():
                chat.add_student_message(doubt_text.strip())
                with st.spinner("🤔 Thinking..."):
                    answer = answer_doubt(
                        question=doubt_text.strip(),
                        module=current_module,
                        topic=topic,
                        level=level,
                        language=language,
                    )
                chat.add_teacher_message(answer)
                st.rerun()


# ================================================================
# FULL LESSON VIDEO (REAL MP4 RENDERING)
# ================================================================
def generate_all_module_audio():
    """Ensure every module in the current lesson has narration audio,
    generating any that are missing. Returns {index: path}.

    Also populates st.session_state.module_audio_errors with a clear,
    per-module reason for any failures, so the UI can tell the user
    exactly why (e.g. edge-tts missing, empty script, network error)
    instead of a generic 'no audio' dead end.
    """

    plan = st.session_state.lesson_plan
    if not plan:
        return {}

    modules = plan.get("modules", [])
    audio_map = dict(st.session_state.get("module_audio_files", {}))
    errors = {}

    audio_engine = get_audio_engine()
    if not audio_engine.is_available():
        st.session_state.module_audio_errors = {
            "_all": (
                "The edge-tts package is not installed in this "
                "Python environment. Install it with:\n\n"
                "pip install edge-tts"
            )
        }
        return audio_map

    for index, module in enumerate(modules):
        if index in audio_map and audio_map[index] and os.path.exists(str(audio_map[index])):
            continue

        script = module.get("spoken_script", "")
        if not script:
            errors[index] = "This module has no spoken script to narrate."
            continue

        filename = (
            f"video_{st.session_state.learner_id}_module_{index}.mp3"
        )
        path = generate_audio(script, filename, expression="explaining")
        if path:
            audio_map[index] = path
        else:
            errors[index] = (
                st.session_state.get("last_audio_error")
                or "Unknown audio generation failure."
            )

    st.session_state.module_audio_files = audio_map
    st.session_state.module_audio_errors = errors
    return audio_map


def render_full_lesson_video_section():
    """Real 'AI Teaching Video' generation and playback, replacing the
    old scene-planning-only preview."""

    with st.expander("🎬 AI Teaching Video", expanded=False):

        st.write(
            "Render the complete lesson — avatar, narration and the "
            "subject-aware visuals for every module — into one real "
            "MP4 video you can watch or download."
        )

        plan = st.session_state.lesson_plan
        module_count = len(plan.get("modules", [])) if plan else 0

        if st.button(
            "🎥 Generate Full Lesson Video",
            use_container_width=True,
            key="generate_full_lesson_video",
            type="primary",
        ):
            with st.spinner(
                f"🎙️ Generating narration for {module_count} module(s)..."
            ):
                audio_map = generate_all_module_audio()

            if not audio_map:
                errors = st.session_state.get("module_audio_errors", {})
                if "_all" in errors:
                    st.error(errors["_all"])
                elif errors:
                    st.error("Narration could not be generated:")
                    for idx, msg in errors.items():
                        st.caption(f"Module {idx + 1}: {msg}")
                else:
                    st.error(
                        "Could not generate narration audio for any "
                        "module."
                    )
            else:
                with st.spinner(
                    "🎬 Rendering avatar + visuals + narration into an MP4 "
                    "(this can take a minute)..."
                ):
                    video_engine = get_engines()["video"]
                    result = video_engine.render_lesson_video(
                        lesson_plan=plan,
                        module_audio_paths={
                            i: str(p) for i, p in audio_map.items()
                        },
                        teacher_name="Prof. AI",
                        output_filename=f"lesson_{st.session_state.learner_id}.mp4",
                    )

                st.session_state.lesson_video_status = result

                if result.get("success"):
                    st.session_state.lesson_video_path = result.get("output_path")
                    st.success(
                        f"✅ Video ready — {result.get('scenes_rendered')} "
                        f"of {result.get('modules_total')} modules rendered."
                    )
                    # Partial render: some modules were skipped or failed
                    # even though the overall MP4 was produced. Surface
                    # this instead of hiding it.
                    skipped = result.get("missing_audio_modules") or []
                    failed = result.get("failed_render_modules") or {}
                    if skipped:
                        st.caption(
                            "⚠️ Skipped (no narration audio): module(s) "
                            + ", ".join(str(i + 1) for i in skipped)
                        )
                    if failed:
                        st.caption(
                            "⚠️ Rendering issues on: "
                            + ", ".join(str(i + 1) for i in failed)
                        )
                else:
                    st.session_state.lesson_video_path = None
                    status = result.get("status", "unknown")

                    if status == "ffmpeg_unavailable":
                        st.warning(
                            "FFmpeg is not installed on this machine, so a "
                            "real MP4 cannot be produced here. Install "
                            "FFmpeg and try again — the rest of the lesson "
                            "remains fully usable."
                        )
                    elif status == "render_failed":
                        # Narration existed — the video pipeline itself
                        # broke. Show the real per-module error(s) instead
                        # of a generic "no narration" message.
                        st.error(
                            "Narration audio was generated successfully, "
                            "but building the MP4 itself failed. This is a "
                            "rendering bug, not a missing-narration issue."
                        )
                        st.caption(result.get("message", ""))
                        failed = result.get("failed_render_modules") or {}
                        for idx, msg in failed.items():
                            st.markdown(f"### Module {idx + 1}")
                            st.code(msg)
                        st.info(
                            "Common causes: Pillow/matplotlib not installed "
                            "in this environment, or FFmpeg failing on a "
                            "specific frame. Check the details above."
                        )
                    elif status == "no_narrated_modules":
                        st.warning(
                            "No module has narration audio yet. Click "
                            "'Generate Full Lesson Video' again — this "
                            "first generates narration for every module, "
                            "then renders the video. If narration keeps "
                            "failing, expand the section below."
                        )
                        errors = st.session_state.get("module_audio_errors", {})
                        if errors:
                            st.markdown("### Why narration failed")
                            for idx, msg in errors.items():
                                st.caption(f"Module {idx + 1}: {msg}")
                    else:
                        st.warning(
                            f"Video rendering did not complete ({status}). "
                            f"{result.get('message', '')}"
                        )

        video_path = st.session_state.get("lesson_video_path")

        if video_path and os.path.exists(str(video_path)):
            try:
                from components.video_player import VideoPlayer

                player = VideoPlayer(
                    title="Your AI Teaching Video",
                    topic=plan.get("topic_title", "") if plan else "",
                    language=plan.get("language", "") if plan else "",
                )
                player.set_video(str(video_path))
                player.render()
            except Exception:
                st.video(str(video_path))

            with open(video_path, "rb") as handle:
                st.download_button(
                    "⬇️ Download lesson video",
                    data=handle.read(),
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                    use_container_width=True,
                )


# ================================================================
# MOVE TO NEXT MODULE
# ================================================================
def move_to_next_module():

    plan = (
        st.session_state.lesson_plan
    )

    if not plan:
        return

    completed = list(st.session_state.get("completed_modules", []))
    if st.session_state.current_module_idx not in completed:
        completed.append(st.session_state.current_module_idx)
    st.session_state.completed_modules = completed

    next_index = (
        st.session_state.current_module_idx
        + 1
    )

    if next_index >= len(
        plan.get(
            "modules",
            [],
        )
    ):

        complete_lesson()

        return


    st.session_state.current_module_idx = (
        next_index
    )

    st.session_state.current_question_answered = (
        False
    )

    st.session_state.current_question_correct = (
        False
    )

    st.session_state.remediation = None

    st.session_state.retest_mode = False

    st.session_state.retest_answered = False


    # Audio is generated on demand from the teaching page.
    st.session_state.audio_file = None


    next_module = plan[
        "modules"
    ][next_index]

    next_concept = next_module.get(
        "concept",
        next_module.get(
            "module_title",
            "",
        ),
    )


    record_teaching_event(
        action="CONTINUE",

        topic=plan.get(
            "topic_title",
            "",
        ),

        concept=next_concept,
    )


    st.rerun()


# ================================================================
# COMPLETE LESSON
# ================================================================
def complete_lesson():

    st.session_state.lesson_completed = True

    plan = (
        st.session_state.lesson_plan
    )

    if not plan:
        return

    total_modules = len(plan.get("modules", []))
    st.session_state.completed_modules = list(range(total_modules))

    topic = plan.get(
        "topic_title",
        "Unknown Topic",
    )

    memory = get_learning_memory()

    safe_method_call(
        memory,

        "record_lesson_completed",

        topic=topic,

        score=
            st.session_state.user_score,
    )

    st.session_state.page = (
        "report"
    )

    st.rerun()


# ================================================================
# HOME PAGE
# ================================================================
def _estimate_spoken_words(plan):
    if not isinstance(plan, dict): return 0
    return sum(len(str(m.get("spoken_script", "") or "").split()) for m in (plan.get("modules", []) or []) if isinstance(m, dict))

def _duration_targets(minutes):
    return {5:(575,725),20:(2300,2850),60:(6900,8400)}.get(minutes,(max(300,int(minutes*115)),int(minutes*145)))

def _merge_lesson_plans(plans,total_minutes):
    valid=[p for p in plans if isinstance(p,dict) and p.get("modules")]
    if not valid:return None
    if len(valid)==1:
        result=dict(valid[0]);result["session_duration"]=total_minutes;result["target_spoken_minutes"]=total_minutes;result["spoken_word_count"]=_estimate_spoken_words(result);return result
    merged=dict(valid[0]);modules=[];objectives=[];concepts=[]
    for seg,plan in enumerate(valid,1):
        for item in plan.get("modules",[]) or []:
            if isinstance(item,dict): item=dict(item);item["segment"]=seg;modules.append(item)
        for key in ("objectives","learning_objectives"):
            value=plan.get(key,[])
            if isinstance(value,list):objectives.extend(str(x) for x in value if x)
        value=plan.get("key_concepts",[])
        if isinstance(value,list):concepts.extend(str(x) for x in value if x)
    for i,m in enumerate(modules,1):m["module_id"]=i;m["order"]=i
    merged["modules"]=modules;merged["objectives"]=list(dict.fromkeys(objectives));merged["learning_objectives"]=list(dict.fromkeys(objectives));merged["key_concepts"]=list(dict.fromkeys(concepts));merged["session_duration"]=total_minutes;merged["target_spoken_minutes"]=total_minutes;merged["generation_segments"]=len(valid);merged["spoken_word_count"]=_estimate_spoken_words(merged);return merged


def _extract_json_object(raw_text):
    """Extract a JSON object even when Gemini wraps it in markdown fences."""
    if raw_text is None:
        return None
    if isinstance(raw_text, dict):
        return raw_text

    text = str(raw_text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _direct_gemini_lesson_fallback(topic, context, level, language, minutes, segment_instruction=""):
    """
    Independent lesson-generation fallback.

    The tested LessonGeneratorEngine remains the first path. If it returns None
    because its internal JSON/parser path fails, this function calls Gemini
    directly and normalizes the response into the exact structure expected by
    the existing teaching/assessment UI.
    """
    api_key = str(st.session_state.get("gemini_api_key", "") or "").strip()
    groq_key = str(st.session_state.get("groq_api_key", "") or os.getenv("GROQ_API_KEY", "") or "").strip()
    if not api_key:
        return None, "No Gemini API key was supplied."

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:
        return None, f"Gemini LangChain package is unavailable: {exc}"

    word_targets = {
        5: "about 600–700 spoken words",
        20: "about 2200–2600 spoken words",
        60: "about 2200–2600 spoken words for this 20-minute segment",
    }
    target = word_targets.get(minutes, f"about {max(600, minutes * 115)}–{max(750, minutes * 140)} spoken words")

    prompt = f"""
You are PedagogyEngine AI, a human-like adaptive teacher.

Create a complete, student-friendly lesson plan for:
TOPIC: {topic}
LEARNER LEVEL: {level}
TEACHING LANGUAGE: {language}
SESSION LENGTH: {minutes} minutes
TARGET SPOKEN LENGTH: {target}

GROUNDED SOURCE CONTEXT:
{context}

{segment_instruction}

IMPORTANT:
- Teach, do not merely summarize.
- Use simple explanations appropriate for the learner level.
- Include intuition, examples, demonstrations and a useful checkpoint.
- The spoken_script must contain the actual words the AI teacher will say.
- Make the spoken_script substantially match the requested duration.
- Do not put markdown fences around the JSON.
- Return ONLY one valid JSON object.
- Do not add commentary before or after the JSON.

Required JSON structure:
{{
  "topic_title": "{topic}",
  "target_level": "{level}",
  "language": "{language}",
  "session_duration": {minutes},
  "objectives": ["objective 1", "objective 2", "objective 3"],
  "learning_objectives": ["objective 1", "objective 2", "objective 3"],
  "key_concepts": ["concept 1", "concept 2", "concept 3"],
  "modules": [
    {{
      "module_title": "Clear module title",
      "concept": "Concept name",
      "learning_objective": "What the learner should understand",
      "explanation": "Concise visual-friendly explanation",
      "spoken_script": "Full natural teacher narration for this module",
      "visual_type": "DIAGRAM",
      "visual_content": "Detailed description of the visual to show",
      "visual_data": {{
        "x_label": "",
        "y_label": "",
        "chart_points": [],
        "flow_steps": [],
        "timeline_events": [],
        "table_rows": []
      }},
      "checkpoint_question": {{
        "question": "A meaningful checkpoint question",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "Option A",
        "explanation": "Why the answer is correct",
        "question_type": "MCQ"
      }}
    }}
  ]
}}

For 5 minutes, create 1 focused module.
For 20 minutes, create 3 connected modules.
For a 20-minute segment of a 60-minute lesson, create 3 connected modules.
Each module must have a real spoken_script and a checkpoint.

For visual_data: populate ONLY the fields relevant to visual_type, with
real concept-specific data (real [x, y] points for a GRAPH, the real
ordered steps for a FLOWCHART/PROCESS, the real chronological events for
a TIMELINE, real comparison rows for a TABLE). Never leave a chart/flow/
timeline/table visual_type with an empty visual_data — invent reasonable,
concept-accurate numbers/steps if the source material does not give exact
ones, but they must be specific to this concept, not generic placeholders.
"""

    from llm_provider import get_chat_llm

    # Groq (free, high quota) is tried once first; Gemini models are
    # tried next in case only a Gemini key is configured.
    attempts = []
    if groq_key:
        attempts.append(("groq", None))
    for model_name in ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"]:
        if api_key:
            attempts.append(("gemini", model_name))

    errors = []

    for provider, model_name in attempts:
        try:
            # Do not pass temperature/top_p/top_k for Gemini 3 models:
            # current Gemini 3 models recommend avoiding sampling params.
            llm = get_chat_llm(
                max_output_tokens=8192,
                groq_api_key=groq_key if provider == "groq" else "",
                gemini_api_key=api_key if provider == "gemini" else "",
                gemini_model=model_name,
                preferred_provider=provider,
            )
            response = llm.invoke(prompt)
            raw = getattr(response, "content", response)
            parsed = _extract_json_object(raw)

            if isinstance(parsed, dict) and isinstance(parsed.get("modules"), list) and parsed["modules"]:
                parsed["session_duration"] = minutes
                parsed["target_spoken_minutes"] = minutes
                parsed["generation_model"] = model_name or "groq"
                parsed["fallback_generation"] = True
                return parsed, None

            errors.append(f"{provider}/{model_name}: returned no usable JSON/modules.")
        except Exception as exc:
            errors.append(f"{provider}/{model_name}: {type(exc).__name__}: {exc}")

    return None, " | ".join(errors)


def _generate_one_segment(generator, topic, context, level, language, minutes, instruction=""):
    """Try the tested engine first, then direct Gemini fallback."""
    try:
        plan = generator.generate_lesson_plan(
            topic=topic if not instruction else f"{topic}\n\n{instruction}",
            grounded_context=context,
            level=level,
            language=language,
            available_time=minutes,
        )
        if isinstance(plan, dict) and plan.get("modules"):
            return plan, None
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
    else:
        primary_error = "LessonGeneratorEngine returned None or an invalid plan."

    fallback_plan, fallback_error = _direct_gemini_lesson_fallback(
        topic=topic,
        context=context,
        level=level,
        language=language,
        minutes=minutes,
        segment_instruction=instruction,
    )

    if fallback_plan:
        return fallback_plan, None

    return None, f"Primary generator: {primary_error}; Direct Gemini fallback: {fallback_error}"


def generate_duration_adaptive_lesson(generator,topic,context,level,language,minutes):
    if minutes == 60:
        instructions = [
            "PART 1 OF 3 — Foundations: teach intuition, definitions, prerequisites and first examples. Create a complete 20-minute spoken segment.",
            "PART 2 OF 3 — Deepening: build on the topic with mechanisms, worked examples, demonstrations and common mistakes. Create a complete 20-minute spoken segment.",
            "PART 3 OF 3 — Application: teach applications, advanced reasoning, misconceptions, recap and exam-style thinking. Create a complete 20-minute spoken segment.",
        ]
        plans = []
        errors = []
        for instruction in instructions:
            plan, error = _generate_one_segment(
                generator, topic, context, level, language, 20, instruction
            )
            if plan:
                plans.append(plan)
            elif error:
                errors.append(error)

        merged = _merge_lesson_plans(plans, 60)
        if merged:
            return merged

        if errors:
            st.session_state["lesson_generation_error"] = " || ".join(errors)
        return None

    plan, error = _generate_one_segment(
        generator, topic, context, level, language, minutes
    )
    if plan:
        return _merge_lesson_plans([plan], minutes)

    if error:
        st.session_state["lesson_generation_error"] = error
    return None


def render_home():

    st.markdown(
        '<div class="main-title">'
        '🎓 PedagogyEngine AI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Human-like AI Teacher that understands, teaches,
        questions, evaluates, detects misconceptions and adapts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")


    # ============================================================
    # LEARNER PROFILE
    # ============================================================
    st.markdown("""<div class="student-hero"><div style="font-size:2.1rem">🎓 ✨ 🧠</div><h1>PedagogyEngine AI</h1><p>Your personal AI teacher — learns your level, teaches your way, asks questions, catches misconceptions, and adapts.</p><div style="margin-top:.8rem"><span class="flow-pill">📚 Understand</span><span class="flow-pill">🗺️ Plan</span><span class="flow-pill">🗣️ Explain</span><span class="flow-pill">🎨 Demonstrate</span><span class="flow-pill">❓ Question</span><span class="flow-pill">🧩 Adapt</span></div></div>""",unsafe_allow_html=True)
    f1,f2,f3,f4=st.columns(4)
    for col,(icon,title,text) in zip((f1,f2,f3,f4),[("📖","Grounded Learning","Teach from your PDF, notes, PPT or a topic."),("🎙️","Your Teacher Voice","Choose the voice and speaking speed you like."),("🧠","Adaptive Teaching","Wrong answer? The teacher explains differently and retests you."),("📈","Progress Memory","Scores, weak concepts and learning history stay connected.")]):
        with col: st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{text}</div></div>',unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>",unsafe_allow_html=True)

    st.markdown(
        "## 👤 Learner Profile"
    )

    profile_col1, profile_col2 = (
        st.columns(2)
    )


    with profile_col1:

        learner_name = st.text_input(
            "Your Name",

            value=
                st.session_state.learner_name,

            placeholder=
                "Enter your name",
        )


        levels = [
            "Beginner",
            "Intermediate",
            "Advanced",
        ]

        learner_level = st.selectbox(
            "Learning Level",

            levels,

            index=
                levels.index(
                    st.session_state.learner_level
                ),
        )


        goals = [
            "Understand the topic",
            "Prepare for an exam",
            "Build practical knowledge",
            "Revise quickly",
            "Master the topic",
        ]

        current_goal = (
            st.session_state.learner_goal
        )

        learner_goal = st.selectbox(
            "Learning Objective",

            goals,

            index=
                goals.index(
                    current_goal
                )
                if current_goal in goals
                else 0,
        )


    with profile_col2:

        styles = [
            "Visual",
            "Conceptual",
            "Example-based",
            "Problem-solving",
        ]

        current_style = (
            st.session_state.learning_style
        )

        learning_style = st.selectbox(
            "Preferred Learning Style",

            styles,

            index=
                styles.index(
                    current_style
                )
                if current_style in styles
                else 0,
        )


        depths = [
            "Concise",
            "Standard",
            "Deep",
        ]

        current_depth = (
            st.session_state.preferred_depth
        )

        preferred_depth = st.selectbox(
            "Preferred Depth",

            depths,

            index=
                depths.index(
                    current_depth
                )
                if current_depth in depths
                else 1,
        )


        languages = [
            "English",
            "Hinglish",
            "Hindi",
            "Tamil",
            "Spanish",
        ]

        teaching_lang = st.selectbox(
            "Teaching Language",

            languages,

            index=
                languages.index(
                    st.session_state.learner_language
                ),
        )


    st.markdown("---")


    # ============================================================
    # LEARNING MATERIAL
    # ============================================================
    st.markdown(
        "## 📚 Learning Material"
    )

    input_type = st.radio(
        "Choose learning source",

        [
            "Enter Topic Directly",
            "Upload Material",
        ],

        horizontal=True,
    )


    topic_input = ""

    uploaded_file = None


    if (
        input_type
        == "Enter Topic Directly"
    ):

        topic_input = st.text_input(
            "Topic",

            value=
                "Ohm's Law & Circuit Behavior",

            placeholder=
                "e.g. Neural Networks, "
                "Photosynthesis, SQL Joins",
        )


    else:

        uploaded_file = st.file_uploader(

            "Upload textbook / notes / course material",

            type=[
                "pdf",
                "txt",
                "docx",
                "doc",
            ],

            help=(
                "The uploaded material will be indexed "
                "and used as grounded context."
            ),
        )


        if uploaded_file:

            st.success(
                f"📄 {uploaded_file.name}"
            )

            st.session_state.focus_topic = st.text_input(
                "Focus on a specific topic/chapter (optional)",

                value=st.session_state.get("focus_topic", ""),

                placeholder=(
                    "e.g. 'Chapter 4 only', 'Only the Ohm's Law "
                    "section', leave blank to teach the whole document"
                ),

                help=(
                    "If you only want the AI Teacher to focus on part "
                    "of the uploaded material, describe it here. "
                    "Retrieval and the lesson plan will be scoped to "
                    "that part instead of the entire document."
                ),
            )


    # ============================================================
    # SESSION SETTINGS
    # ============================================================
    st.markdown(
        "## ⏱️ Session Settings"
    )

    time_col1, time_col2 = (
        st.columns(2)
    )


    with time_col1:

        # Streamlit requires the slider value to exist in options.
        # Older sessions may still contain 10 minutes, so normalize it.
        valid_durations = [5, 20, 60]
        current_duration = st.session_state.get("session_duration", 20)
        if current_duration not in valid_durations:
            current_duration = 20
            st.session_state.session_duration = current_duration

        available_time = st.select_slider(

            "Available Learning Time",

            options=valid_durations,

            value=current_duration,

            format_func=lambda minutes: {
                5: "⚡ 5 min · Quick",
                20: "📚 20 min · Structured",
                60: "🧠 60 min · Deep",
            }[minutes],
        )


    with time_col2:

        st.info(
            "⚡ 5 min → quick concept\n\n"
            "📚 20 min → structured lesson\n\n"
            "🧠 60 min → deep multi-part learning\n\n"
            "Spoken content scales with the selected time."
        )


    # ============================================================
    # AI PROVIDER KEYS
    # ============================================================
    st.markdown(
        "## 🔐 AI Configuration"
    )

    st.caption(
        "Groq is free with a much higher daily quota than Gemini's "
        "free tier and is used automatically when a key is set. "
        "Get a free key at https://console.groq.com/keys — no card "
        "required. Gemini is used only if Groq is not configured."
    )

    groq_api_key = st.text_input(

        "Groq API Key (recommended — free)",

        type="password",

        value=
            st.session_state.groq_api_key,

        key="groq_api_key_input",

        help=(
            "Free, no credit card, ~14,400 requests/day. "
            "Used first whenever set. Paste only the key itself "
            "(a short string, not a file or a long block of text)."
        ),
    )

    gemini_api_key = st.text_input(

        "Gemini API Key (fallback, optional if Groq key is set)",

        type="password",

        value=
            st.session_state.gemini_api_key,

        key="gemini_api_key_input",

        help=(
            "Only needed if you are not using Groq. Gemini's free "
            "tier has a much lower daily quota."
        ),
    )

    # --------------------------------------------------------------
    # Defensive sanitation.
    #
    # A real API key is a short string (well under a few hundred
    # characters). If something far larger ends up here — an
    # accidental paste of a whole file, a doubled/duplicated value
    # from a rerun, etc. — os.environ[...] raises ValueError on
    # Windows (its env vars have a hard 32,767-character limit) and
    # can crash the entire app on every rerun. Guard against that
    # unconditionally rather than trusting the input's length.
    # --------------------------------------------------------------
    MAX_KEY_LENGTH = 300

    groq_api_key = (groq_api_key or "").strip()
    gemini_api_key = (gemini_api_key or "").strip()

    if len(groq_api_key) > MAX_KEY_LENGTH:
        st.error(
            "The Groq API key looks far too long for a real key "
            f"({len(groq_api_key)} characters). Please clear the "
            "field and paste only the key from "
            "https://console.groq.com/keys."
        )
        groq_api_key = ""

    if len(gemini_api_key) > MAX_KEY_LENGTH:
        st.error(
            "The Gemini API key looks far too long for a real key "
            f"({len(gemini_api_key)} characters). Please clear the "
            "field and paste only the key itself."
        )
        gemini_api_key = ""

    st.session_state.groq_api_key = groq_api_key

    if groq_api_key:
        try:
            os.environ["GROQ_API_KEY"] = groq_api_key
        except (ValueError, TypeError) as exc:
            st.error(f"Could not set Groq API key: {exc}")
            st.session_state.groq_api_key = ""


    st.markdown("---")


    # ============================================================
    # START
    # ============================================================
    if st.button(

        "🚀 Start Personalized AI Teacher",

        type="primary",

        use_container_width=True,
    ):

        if not gemini_api_key and not groq_api_key:

            st.error(
                "Please provide a Groq API key (recommended, free) "
                "or a Gemini API key."
            )

            return


        if (
            input_type
            == "Enter Topic Directly"
            and not topic_input.strip()
        ):

            st.error(
                "Please enter a topic."
            )

            return


        if (
            input_type
            == "Upload Material"
            and uploaded_file is None
        ):

            st.error(
                "Please upload educational material."
            )

            return


        # --------------------------------------------------------
        # Save profile settings
        # --------------------------------------------------------
        st.session_state.gemini_api_key = (
            gemini_api_key
        )

        st.session_state.groq_api_key = (
            groq_api_key
        )

        st.session_state.learner_name = (
            learner_name.strip()
            or "Learner"
        )

        st.session_state.learner_level = (
            learner_level
        )

        st.session_state.learner_goal = (
            learner_goal
        )

        st.session_state.learner_language = (
            teaching_lang
        )

        st.session_state.learning_style = (
            learning_style
        )

        st.session_state.preferred_depth = (
            preferred_depth
        )

        st.session_state.session_duration = (
            available_time
        )


        # --------------------------------------------------------
        # Profile
        # --------------------------------------------------------
        try:

            create_or_update_learner_profile()

        except Exception as exc:

            st.warning(
                f"Could not update learner profile: {exc}"
            )


        # ========================================================
        # PERFORMANCE TIMING
        # ========================================================
        generation_started_at = time.perf_counter()
        st.session_state.generation_timings = {}

        # ========================================================
        # PROCESS SOURCE
        # ========================================================
        with st.spinner(
            "🧠 Understanding learning material..."
        ):

            rag = get_rag_engine()

            context = (
                "Use standard curriculum knowledge "
                "for this topic."
            )

            source_name = (
                "Direct Topic"
            )

            source_type = (
                "topic"
            )


            if uploaded_file is not None:

                source_name = (
                    uploaded_file.name
                )

                source_type = (
                    "uploaded_material"
                )


                try:

                    indexing_result = (
                        rag.process_and_index_file(
                            uploaded_file
                        )
                    )


                    if (
                        indexing_result.get(
                            "status"
                        )
                        == "success"
                        or indexing_result.get("success")
                    ):

                        search_term = (
                            topic_input
                            if topic_input
                            else infer_topic_from_filename(
                                uploaded_file.name
                            )
                        )

                        focus_topic = str(
                            st.session_state.get("focus_topic", "") or ""
                        ).strip()

                        context = (
                            rag.retrieve_context(
                                search_term,
                                top_k=5,
                                focus=focus_topic or None,
                            )
                        )

                        if focus_topic:
                            st.caption(
                                f"🔎 Focusing this lesson on: {focus_topic}"
                            )

                    else:

                        st.warning(
                            "Material could not be indexed. "
                            "The teacher will use general knowledge."
                        )

                except Exception as exc:

                    st.warning(
                        f"RAG indexing issue: {exc}"
                    )


            st.session_state.grounded_context = (
                context
            )

            st.session_state.source_name = (
                source_name
            )

            st.session_state.source_type = (
                source_type
            )

            st.session_state.generation_timings["source"] = round(
                time.perf_counter() - generation_started_at, 2
            )


        # ========================================================
        # TARGET TOPIC
        # ========================================================
        if topic_input.strip():

            target_topic = (
                topic_input.strip()
            )

        else:

            target_topic = (
                infer_topic_from_filename(
                    uploaded_file.name
                )
            )

        focus_topic = str(
            st.session_state.get("focus_topic", "") or ""
        ).strip()

        if focus_topic:
            target_topic = (
                f"{target_topic} — focus specifically on: {focus_topic}. "
                "Do not teach unrelated parts of the material."
            )


        # ========================================================
        # LESSON GENERATION
        # ========================================================
        with st.spinner(
            "👨‍🏫 Planning your personalized lesson..."
        ):

            try:

                generator = get_lesson_generator(
                    st.session_state.gemini_api_key
                )

                plan = generate_duration_adaptive_lesson(
                    generator=generator, topic=target_topic, context=context,
                    level=learner_level, language=teaching_lang, minutes=available_time,
                )

            except Exception as exc:

                st.error(
                    f"Lesson generation failed: {exc}"
                )

                return

            st.session_state.generation_timings["lesson_generation"] = round(
                time.perf_counter() - generation_started_at, 2
            )


        # Never call .get() on an exception returned by an older generator.
        if isinstance(plan, BaseException):
            st.error("Lesson generation failed.")
            st.code(f"{type(plan).__name__}: {plan}")
            st.info(
                "The app is configured to use Gemini 3.5 Flash. "
                "If this still fails, verify that your Gemini API key has Gemini API access."
            )
            return

        if not isinstance(plan, dict):
            st.error("The AI Teacher could not create a valid lesson plan.")
            generation_error = st.session_state.get(
                "lesson_generation_error",
                "The lesson generator returned None without a detailed error.",
            )
            st.warning("Generation details:")
            st.code(str(generation_error))
            st.info(
                "The app first tries the tested LessonGeneratorEngine and then "
                "automatically retries through Gemini directly using supported "
                "current Gemini Flash models."
            )
            return

        if not plan.get("modules"):
            st.error("The AI Teacher could not create lesson modules.")
            st.info("Try the topic again and verify your Gemini API key.")
            return


        # ========================================================
        # SAVE LESSON STATE
        # ========================================================
        st.session_state.lesson_plan = (
            plan
        )

        st.session_state.current_module_idx = (
            0
        )

        st.session_state.user_score = (
            0
        )

        st.session_state.total_questions = (
            len(
                plan.get(
                    "modules",
                    [],
                )
            )
        )

        st.session_state.misconceptions = (
            []
        )

        st.session_state.assessment_results = (
            []
        )

        st.session_state.current_question_answered = (
            False
        )

        st.session_state.current_question_correct = (
            False
        )

        st.session_state.remediation = (
            None
        )

        st.session_state.retest_mode = (
            False
        )

        st.session_state.retest_answered = (
            False
        )

        st.session_state.lesson_started = (
            True
        )

        st.session_state.lesson_completed = (
            False
        )

        st.session_state.topic_mastery = (
            0.0
        )

        st.session_state.completed_modules = []
        st.session_state.doubt_chat = None
        st.session_state.module_audio_files = {}
        st.session_state.lesson_video_path = None
        st.session_state.lesson_video_status = None


        # ========================================================
        # PERFORMANCE: DEFER AUDIO + VIDEO
        # ========================================================
        # These are secondary media operations. Do not run them during lesson
        # startup; the learner can request them from the teaching page.
        st.session_state.audio_file = None
        st.session_state.video_plan = None


        # ========================================================
        # MEMORY
        # ========================================================
        record_lesson_start()


        modules = plan.get(
            "modules",
            [],
        )

        first_concept = ""

        if modules:

            first_module = modules[0]

            first_concept = (
                first_module.get(
                    "concept",
                    first_module.get(
                        "module_title",
                        "",
                    ),
                )
            )


        record_teaching_event(

            action="INTRODUCE",

            topic=plan.get(
                "topic_title",
                target_topic,
            ),

            concept=first_concept,
        )


        st.session_state.generation_timings["total_to_ready"] = round(
            time.perf_counter() - generation_started_at, 2
        )

        st.session_state.page = (
            "teaching"
        )

        st.rerun()


# ================================================================
# TEACHING PAGE
# ================================================================
def render_teaching():

    plan = (
        st.session_state.lesson_plan
    )

    if not plan:

        st.session_state.page = (
            "home"
        )

        st.rerun()

        return


    duration=int(plan.get("session_duration",st.session_state.get("session_duration",20)) or 20)
    spoken_words=_estimate_spoken_words(plan)
    low,high=_duration_targets(duration)
    pace="On target" if low<=spoken_words<=high else ("Shorter than target" if spoken_words<low else "Longer than target")
    st.markdown(f'<div class="lesson-banner"><span class="badge">LIVE AI LESSON</span> &nbsp; ⏱️ <b>{duration} min</b> &nbsp; · &nbsp; 🗣️ <b>{spoken_words:,} spoken words</b> &nbsp; · &nbsp; {pace}</div>',unsafe_allow_html=True)
    st.markdown("<div style='height:.65rem'></div>",unsafe_allow_html=True)

    modules = plan.get(
        "modules",
        [],
    )

    if not modules:

        st.error(
            "No lesson modules were generated."
        )

        return


    idx = min(

        st.session_state.current_module_idx,

        len(modules) - 1,
    )

    current_module = modules[
        idx
    ]


    topic = plan.get(
        "topic_title",
        "AI Lesson",
    )

    module_title = current_module.get(
        "module_title",
        f"Module {idx + 1}",
    )

    concept = current_module.get(
        "concept",
        module_title,
    )


    # ============================================================
    # HEADER
    # ============================================================
    st.title(
        f"📖 {topic}"
    )

    st.caption(

        f"Module {idx + 1} of {len(modules)} • "
        f"{module_title} • "
        f"{plan.get('target_level', st.session_state.learner_level)} • "
        f"{plan.get('language', st.session_state.learner_language)}"
    )

    st.progress(
        (idx + 1)
        / len(modules)
    )

    timings = st.session_state.get("generation_timings", {})
    if timings.get("total_to_ready") is not None:
        st.caption(
            f"⚡ Lesson ready in {timings['total_to_ready']:.1f}s "
            "• Audio/video are generated only when requested"
        )


    # ============================================================
    # STATUS
    # ============================================================
    status_col1, status_col2, status_col3 = (
        st.columns(3)
    )


    with status_col1:

        st.metric(
            "Questions",
            st.session_state.total_questions,
        )


    with status_col2:

        st.metric(
            "Correct",
            st.session_state.user_score,
        )


    with status_col3:

        mastery = (
            st.session_state.user_score
            / max(
                st.session_state.total_questions,
                1,
            )
        )

        st.metric(
            "Current Mastery",
            f"{mastery * 100:.0f}%",
        )


    st.markdown("---")


    left, right = st.columns(
        [1, 1],
        gap="large",
    )


    # ============================================================
    # AI TEACHER
    # ============================================================
    with left:

        st.subheader(
            "🗣️ AI Teacher"
        )

        render_voice_controls()

        audio_path = (
            st.session_state.audio_file
        )

        if st.session_state.get("retest_mode"):
            st.session_state.avatar_expression = "questioning"
        elif st.session_state.get("current_question_answered"):
            st.session_state.avatar_expression = (
                "happy" if st.session_state.get("current_question_correct") else "concerned"
            )
        else:
            st.session_state.avatar_expression = "explaining"

        is_speaking = bool(

            audio_path

            and os.path.exists(
                str(audio_path)
            )
        )


        try:

            avatar = get_engines()[
                "avatar"
            ]

            avatar.render(

                is_speaking=is_speaking,

                teacher_name="Prof. AI",

                expression=st.session_state.get("avatar_expression", "explaining"),
            )

        except Exception:

            try:

                from avatar_component import (
                    render_interactive_avatar,
                )

                render_interactive_avatar(

                    is_speaking=
                        is_speaking,

                    teacher_name=
                        "Prof. AI",
                )

            except Exception:

                st.info(
                    "👨‍🏫 Prof. AI is teaching..."
                )


        if not is_speaking:
            # The Phase-2 studio controls the voice; generation stays on-demand
            # so the proven Phase-1 lesson startup time is preserved.
            if st.button(
                "▶️ Speak This Module",
                use_container_width=True,
                key=f"speak_module_{idx}",
                type="primary",
            ):
                with st.spinner("🎙️ AI Teacher is generating natural narration..."):
                    requested_audio = prepare_module_audio(idx)
                if requested_audio:
                    st.rerun()
                else:
                    st.warning(
                        "Teacher voice could not be generated. Check your internet connection and Edge-TTS installation."
                    )

        if is_speaking:
            st.audio(str(audio_path), format="audio/mp3")

        render_media_sync_banner(is_speaking)


        st.markdown(
            '<div class="teacher-box">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 🎙️ Spoken Explanation"
        )

        if is_speaking:
            st.markdown("**🗣️ Now speaking:**")

        st.write(
            current_module.get(
                "spoken_script",
                "Let's learn this concept step by step.",
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


        # --------------------------------------------------------
        # TEACHER STRATEGY
        # --------------------------------------------------------
        st.markdown(
            "#### 🧠 Teaching Strategy"
        )


        decision = get_teacher_decision(

            concept=concept,

            mastery=(
                st.session_state.user_score
                / max(
                    st.session_state.total_questions,
                    1,
                )
            ),
        )


        if isinstance(
            decision,
            dict,
        ):

            action = decision.get(
                "action",
                "EXPLAIN",
            )

            reason = decision.get(
                "reason",
                "",
            )

        else:

            action = "EXPLAIN"

            reason = ""


        st.info(
            f"Current teacher action: **{action}**"
        )


        if reason:

            st.caption(
                reason
            )


    # ============================================================
    # VISUAL
    # ============================================================
    with right:

        st.subheader(
            "🎨 Subject-Aware Visual"
        )


        visual_type = current_module.get(

            "visual_type",

            current_module.get(
                "visual",
                "TEXT",
            ),
        )


        visual_content = current_module.get(
            "visual_content",
            "",
        )


        render_visual_content(

            visual_type=
                visual_type,

            visual_content=
                visual_content,

            topic=
                topic,

            concept=
                concept,

            visual_data=
                current_module.get("visual_data", {}),
        )


    st.markdown("---")

    # ============================================================
    # AI TEACHING VIDEO (real MP4 rendering)
    # ============================================================
    render_full_lesson_video_section()

    # ============================================================
    # MID-LESSON DOUBT BOX
    # ============================================================
    render_doubt_box(
        current_module=current_module,
        topic=topic,
        level=plan.get("target_level", st.session_state.learner_level),
        language=plan.get("language", st.session_state.learner_language),
    )

    st.markdown("---")

    # ============================================================
    # CHECKPOINT
    # ============================================================
    st.subheader(
        "❓ Interactive Checkpoint"
    )


    checkpoint = current_module.get(
        "checkpoint_question",
        {},
    )


    question = checkpoint.get(
        "question",
        "",
    )

    options = checkpoint.get(
        "options",
        [],
    )


    if not question:

        st.warning(
            "This module does not contain a checkpoint."
        )

        if st.button(
            "Continue →",
            use_container_width=True,
        ):

            move_to_next_module()

        return


    # ============================================================
    # RETEST
    # ============================================================
    if st.session_state.retest_mode:

        render_retest(

            checkpoint=
                checkpoint,

            concept=
                concept,
        )

        return


    # ============================================================
    # NORMAL QUESTION
    # ============================================================
    st.markdown(
        f"### {question}"
    )


    if options:

        user_answer = st.radio(

            "Select your answer:",

            options,

            key=f"answer_{idx}",
        )

    else:

        user_answer = st.text_input(

            "Your answer:",

            key=f"text_answer_{idx}",
        )


    if st.button(

        "✅ Submit Answer",

        type="primary",

        use_container_width=True,

        key=f"submit_{idx}",
    ):


        if not str(
            user_answer
        ).strip():

            st.warning(
                "Please provide an answer before submitting."
            )

            return


        result = evaluate_checkpoint(

            checkpoint=
                checkpoint,

            user_answer=
                user_answer,

            concept=
                concept,
        )


        correct = bool(
            result.get(
                "correct",
                False,
            )
        )


        st.session_state.current_question_answered = (
            True
        )

        st.session_state.current_question_correct = (
            correct
        )


        st.session_state.assessment_results.append(
            {

                "module":
                    module_title,

                "concept":
                    concept,

                "question":
                    question,

                "answer":
                    user_answer,

                "correct":
                    correct,

                "score":
                    result.get(
                        "score",
                        1.0
                        if correct
                        else 0.0,
                    ),
            }
        )


        record_assessment_memory(

            question=
                question,

            answer=
                user_answer,

            correct=
                correct,

            concept=
                concept,

            mastery=
                result.get(
                    "score",
                    1.0
                    if correct
                    else 0.0,
                ),
        )


        # ========================================================
        # CORRECT
        # ========================================================
        if correct:

            st.session_state.user_score += (
                1
            )


            st.success(
                "🎉 Correct! Excellent understanding."
            )


            record_teaching_event(

                action="EVALUATE",

                topic=topic,

                concept=concept,

                details={
                    "result":
                        "correct",
                },
            )


            mastery = (

                st.session_state.user_score

                / max(
                    st.session_state.total_questions,
                    1,
                )
            )


            if mastery >= 0.85:

                st.info(
                    "🔥 Strong mastery detected. "
                    "The teacher can increase difficulty."
                )

            elif mastery >= 0.70:

                st.info(
                    "👍 Good mastery. Continuing to the next concept."
                )

            else:

                st.info(
                    "📚 Let's continue building your understanding."
                )

            move_to_next_module()


        # ========================================================
        # WRONG
        # ========================================================
        else:

            st.error(
                "❌ The teacher detected a possible misconception."
            )


            with st.spinner(
                "👨‍🏫 Analyzing your misunderstanding..."
            ):

                remediation = (
                    generate_remediation(

                        question=
                            question,

                        wrong_answer=
                            user_answer,

                        concept=
                            concept,
                    )
                )


            st.session_state.remediation = (
                remediation
            )


            misconception_entry = {

                "module":
                    module_title,

                "concept":
                    concept,

                "question":
                    question,

                "wrong_answer":
                    user_answer,

                "explanation":
                    remediation.get(
                        "misconception_reason",
                        "Concept needs further clarification.",
                    ),

                "analogy":
                    remediation.get(
                        "new_analogy",
                        "",
                    ),
            }


            st.session_state.misconceptions.append(
                misconception_entry
            )


            # ----------------------------------------------------
            # MEMORY
            # ----------------------------------------------------
            memory = get_learning_memory()


            safe_method_call(

                memory,

                "record_misconception",

                topic=
                    topic,

                concept=
                    concept,

                misconception=
                    remediation.get(
                        "misconception_reason",
                        "",
                    ),
            )


            record_teaching_event(

                action="REMEDIATE",

                topic=topic,

                concept=concept,

                details={
                    "wrong_answer":
                        user_answer,
                },
            )


            # ----------------------------------------------------
            # REMEDIATION DISPLAY
            # ----------------------------------------------------
            st.markdown(

                '<div class="warning-box">',

                unsafe_allow_html=True,
            )


            st.markdown(
                "### 💡 Let's Try a Different Explanation"
            )


            analogy = remediation.get(
                "new_analogy",
                "",
            )


            if analogy:

                st.markdown(
                    f"**New Analogy:** {analogy}"
                )


            reason = remediation.get(
                "misconception_reason",
                "",
            )


            if reason:

                st.markdown(
                    f"**Why your answer was incorrect:** {reason}"
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


            # ----------------------------------------------------
            # REMEDIATION AUDIO
            # ----------------------------------------------------
            remediation_script = (
                remediation.get(
                    "spoken_remediation_script",
                    "",
                )
            )


            if remediation_script:

                remedy_filename = (

                    f"lesson_"
                    f"{st.session_state.learner_id}_"
                    f"remedy_"
                    f"{idx}.mp3"
                )


                remedy_audio = prepare_remediation_audio(
                    remediation_script,
                    idx,
                )


                if remedy_audio:

                    st.session_state.audio_file = (
                        remedy_audio
                    )

                    st.audio(

                        str(remedy_audio),

                        format="audio/mp3",
                    )


                st.markdown(
                    "### 🎙️ Teacher Re-explanation"
                )

                st.write(
                    remediation_script
                )


            # ----------------------------------------------------
            # RETEST
            # ----------------------------------------------------
            st.markdown("---")


            st.session_state.retest_mode = (
                True
            )


            st.info(

                "The AI Teacher will now re-test you "
                "using the same concept."
            )


            st.rerun()


# ================================================================
# RETEST
# ================================================================
def render_retest(
    checkpoint,
    concept,
):

    st.markdown(

        '<div class="success-box">',

        unsafe_allow_html=True,
    )


    st.markdown(
        "### 🔄 Re-Test: Show Me Your Understanding"
    )


    st.write(
        "Try the concept again after the teacher's explanation."
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    options = checkpoint.get(
        "options",
        [],
    )


    if options:

        retest_answer = st.radio(

            "Choose your answer:",

            options,

            key=
                f"retest_{st.session_state.current_module_idx}",
        )

    else:

        retest_answer = st.text_input(

            "Your answer:",

            key=
                f"retest_text_{st.session_state.current_module_idx}",
        )


    if st.button(

        "🔄 Submit Re-Test",

        type="primary",

        use_container_width=True,
    ):


        if not str(
            retest_answer
        ).strip():

            st.warning(
                "Please provide an answer."
            )

            return


        result = evaluate_checkpoint(

            checkpoint=
                checkpoint,

            user_answer=
                retest_answer,

            concept=
                concept,
        )


        topic = (
            st.session_state.lesson_plan.get(
                "topic_title",
                "",
            )
        )


        memory = get_learning_memory()


        # ========================================================
        # RETEST SUCCESS
        # ========================================================
        if result.get(
            "correct",
            False,
        ):

            st.success(
                "🎉 Excellent! You corrected the misunderstanding."
            )


            safe_method_call(

                memory,

                "record_retest_result",

                topic=
                    topic,

                concept=
                    concept,

                correct=
                    True,

                mastery=
                    result.get(
                        "score",
                        1.0,
                    ),
            )


            safe_method_call(

                memory,

                "record_remediation_completed",

                topic=
                    topic,

                concept=
                    concept,
            )


            record_teaching_event(

                action="CONTINUE",

                topic=
                    topic,

                concept=
                    concept,

                details={
                    "retest":
                        "resolved",
                },
            )


            st.session_state.retest_mode = (
                False
            )

            st.session_state.remediation = (
                None
            )


            st.info(
                "🧠 Misconception resolved. "
                "The learning path can now continue."
            )


            move_to_next_module()


        # ========================================================
        # RETEST FAILURE
        # ========================================================
        else:

            st.error(
                "The concept still needs reinforcement."
            )


            record_teaching_event(

                action="SIMPLIFY",

                topic=
                    topic,

                concept=
                    concept,

                details={
                    "retest":
                        "incorrect",
                },
            )


            with st.spinner(
                "👨‍🏫 Simplifying the explanation further..."
            ):

                remediation = (
                    generate_remediation(

                        question=
                            checkpoint.get(
                                "question",
                                "",
                            ),

                        wrong_answer=
                            retest_answer,

                        concept=
                            concept,
                    )
                )


            st.session_state.remediation = (
                remediation
            )


            st.warning(

                remediation.get(

                    "new_analogy",

                    "Let's simplify this concept further.",
                )
            )


            st.info(

                remediation.get(

                    "misconception_reason",

                    "Let's revisit the fundamental idea.",
                )
            )


            script = remediation.get(

                "spoken_remediation_script",

                "",
            )


            if script:

                st.write(
                    script
                )


                audio_path = generate_audio(

                    script,

                    (

                        f"lesson_"
                        f"{st.session_state.learner_id}_"
                        f"second_remedy_"
                        f"{st.session_state.current_module_idx}.mp3"
                    ),
                )


                if audio_path:

                    st.session_state.audio_file = (
                        audio_path
                    )


                    st.audio(

                        str(audio_path),

                        format="audio/mp3",
                    )


            st.info(

                "Take another look at the explanation, "
                "then try the re-test again."
            )


# ================================================================
# REPORT
# ================================================================
def render_report():

    plan = (
        st.session_state.lesson_plan
    )

    if not plan:

        st.session_state.page = (
            "home"
        )

        st.rerun()

        return


    topic = plan.get(
        "topic_title",
        "Learning Session",
    )


    st.title(
        "📊 Learning Assessment & Progress Report"
    )


    st.caption(

        f"Topic: {topic} • "
        f"Learner: {st.session_state.learner_name}"
    )


    st.markdown("---")


    # ============================================================
    # RESULTS
    # ============================================================
    total = max(

        st.session_state.total_questions,

        1,
    )

    correct = (
        st.session_state.user_score
    )

    mastery = (
        correct / total
    )


    st.session_state.topic_mastery = (
        mastery
    )


    # ============================================================
    # METRICS
    # ============================================================
    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Score",
            f"{correct}/{total}",
        )


    with c2:

        st.metric(
            "Mastery",
            f"{mastery * 100:.0f}%",
        )


    with c3:

        st.metric(

            "Misconceptions",

            len(
                st.session_state.misconceptions
            ),
        )


    with c4:

        if mastery >= 0.85:
            status = "Strong"

        elif mastery >= 0.70:
            status = "Good"

        elif mastery >= 0.50:
            status = "Developing"

        else:
            status = "Needs Support"


        st.metric(
            "Learning Status",
            status,
        )


    st.markdown("---")


    # ============================================================
    # PERSISTED LEARNER SUMMARY
    # ============================================================
    profile_manager = None

    try:

        profile_manager = (
            get_engines()["profile"]
        )

    except Exception:
        profile_manager = None


    summary = safe_method_call(
        profile_manager,
        "get_summary",
    )


    topic_progress = safe_method_call(

        profile_manager,

        "get_topic_progress",

        topic=topic,
    )


    strong_from_memory = safe_method_call(

        profile_manager,

        "get_strong_concepts",
    )


    developing_from_memory = safe_method_call(

        profile_manager,

        "get_developing_concepts",
    )


    weak_from_memory = safe_method_call(

        profile_manager,

        "get_weak_concepts",
    )


    active_misconceptions = safe_method_call(

        profile_manager,

        "get_active_misconceptions",
    )


    # ============================================================
    # PERSISTED PROGRESS
    # ============================================================
    st.subheader(
        "📈 Learner Progress"
    )


    progress_col1, progress_col2 = (
        st.columns(2)
    )


    with progress_col1:

        st.markdown(
            "### Current Session"
        )

        st.progress(
            mastery
        )

        st.write(
            f"**Session mastery:** "
            f"{mastery * 100:.1f}%"
        )

        st.write(
            f"**Questions attempted:** "
            f"{len(st.session_state.assessment_results)}"
        )


    with progress_col2:

        st.markdown(
            "### Learner Profile"
        )

        st.write(
            f"**Level:** "
            f"{st.session_state.learner_level}"
        )

        st.write(
            f"**Goal:** "
            f"{st.session_state.learner_goal}"
        )

        st.write(
            f"**Learning style:** "
            f"{st.session_state.learning_style}"
        )

        st.write(
            f"**Language:** "
            f"{st.session_state.learner_language}"
        )


    if isinstance(
        topic_progress,
        dict,
    ):

        st.markdown(
            "### 💾 Persisted Topic Progress"
        )

        persisted_mastery = (
            topic_progress.get(
                "mastery",
                topic_progress.get(
                    "mastery_score",
                    None,
                ),
            )
        )

        if persisted_mastery is not None:

            try:

                persisted_mastery = float(
                    persisted_mastery
                )

                st.write(
                    f"Database topic mastery: "
                    f"{persisted_mastery * 100:.1f}%"
                )

            except Exception:
                pass


    st.markdown("---")


    # ============================================================
    # STRONG / WEAK
    # ============================================================
    col1, col2 = (
        st.columns(2)
    )


    with col1:

        st.subheader(
            "💪 Strong Concepts"
        )


        strong_concepts = []


        if isinstance(
            strong_from_memory,
            list,
        ):

            strong_concepts.extend(
                strong_from_memory
            )


        for result in (
            st.session_state.assessment_results
        ):

            if result.get(
                "correct"
            ):

                strong_concepts.append(

                    result.get(
                        "concept",
                        "Concept",
                    )
                )


        strong_concepts = list(
            dict.fromkeys(
                [
                    str(x)
                    for x in strong_concepts
                ]
            )
        )


        if strong_concepts:

            for concept_name in strong_concepts:

                st.success(
                    f"✓ {concept_name}"
                )

        else:

            st.info(
                "No strongly mastered concepts recorded yet."
            )


    with col2:

        st.subheader(
            "⚠️ Areas to Improve"
        )


        weak_concepts = []


        if isinstance(
            weak_from_memory,
            list,
        ):

            weak_concepts.extend(
                weak_from_memory
            )


        for result in (
            st.session_state.assessment_results
        ):

            if not result.get(
                "correct"
            ):

                weak_concepts.append(

                    result.get(
                        "concept",
                        "Concept",
                    )
                )


        weak_concepts = list(
            dict.fromkeys(
                [
                    str(x)
                    for x in weak_concepts
                ]
            )
        )


        if weak_concepts:

            for concept_name in weak_concepts:

                st.warning(
                    f"• {concept_name}"
                )

        else:

            st.success(
                "No major weak areas detected."
            )


    st.markdown("---")


    # ============================================================
    # MISCONCEPTIONS
    # ============================================================
    st.subheader(
        "🧠 Misconceptions Detected & Addressed"
    )


    session_misconceptions = (
        st.session_state.misconceptions
    )


    if session_misconceptions:

        for index, item in enumerate(

            session_misconceptions,

            start=1,
        ):

            with st.expander(

                f"{index}. "
                f"{item.get('concept', item.get('module', 'Concept'))}"
            ):

                st.write(

                    f"**Question:** "
                    f"{item.get('question', '')}"
                )

                st.write(

                    f"**Your answer:** "
                    f"{item.get('wrong_answer', '')}"
                )

                st.write(

                    f"**Explanation:** "
                    f"{item.get('explanation', '')}"
                )

                if item.get(
                    "analogy"
                ):

                    st.write(

                        f"**New analogy:** "
                        f"{item.get('analogy')}"
                    )

    elif isinstance(
        active_misconceptions,
        list,
    ) and active_misconceptions:

        st.warning(
            "The learner has active misconceptions "
            "stored from previous learning sessions."
        )

        for item in active_misconceptions:

            st.write(
                f"• {item}"
            )

    else:

        st.success(
            "🎉 No active misconceptions detected."
        )


    st.markdown("---")


    # ============================================================
    # ASSESSMENT HISTORY
    # ============================================================
    st.subheader(
        "📝 Assessment History"
    )


    if st.session_state.assessment_results:

        for result in (
            st.session_state.assessment_results
        ):

            if result.get(
                "correct"
            ):

                st.success(

                    f"✓ "
                    f"{result.get('concept', 'Concept')} "
                    f"— Correct"
                )

            else:

                st.error(

                    f"✗ "
                    f"{result.get('concept', 'Concept')} "
                    f"— Needs Review"
                )

    else:

        st.info(
            "No assessment history available."
        )


    st.markdown("---")


    # ============================================================
    # NEXT LEARNING RECOMMENDATION
    # ============================================================
    st.subheader(
        "🚀 Recommended Next Learning"
    )


    recommendation = safe_method_call(

        profile_manager,

        "recommend_next_learning",
    )


    recommendation_text = ""


    if isinstance(
        recommendation,
        dict,
    ):

        recommendation_text = (

            recommendation.get(

                "recommendation",

                recommendation.get(
                    "next_topic",
                    "",
                ),
            )
        )


    if not recommendation_text:

        if mastery >= 0.85:

            recommendation_text = (

                f"Continue to a more advanced "
                f"concept related to {topic}."
            )

        elif mastery >= 0.70:

            recommendation_text = (

                f"Practice {topic} with "
                f"application-based questions."
            )

        else:

            recommendation_text = (

                f"Review the weak concepts in "
                f"{topic} before moving ahead."
            )


    st.info(
        f"🎯 {recommendation_text}"
    )


    # ============================================================
    # VIDEO PLAN
    # ============================================================
    if st.session_state.video_plan:

        with st.expander(
            "🎬 AI Teaching Video Plan"
        ):

            video_plan = (
                st.session_state.video_plan
            )


            if isinstance(
                video_plan,
                dict,
            ):

                scenes = video_plan.get(
                    "scenes",
                    [],
                )


                st.write(
                    f"Planned scenes: "
                    f"{len(scenes)}"
                )


                for scene in scenes:

                    if isinstance(
                        scene,
                        dict,
                    ):

                        scene_type = (
                            scene.get(
                                "scene_type",
                                "Scene",
                            )
                        )

                        st.write(
                            f"• {scene_type}"
                        )

                    else:

                        st.write(
                            f"• {scene}"
                        )


            st.info(

                "The VideoEngine has prepared the "
                "teaching sequence. Actual MP4 rendering "
                "requires FFmpeg."
            )


    # ============================================================
    # SESSION MEMORY
    # ============================================================
    memory = get_learning_memory()

    recent_memory = safe_method_call(

        memory,

        "get_recent_memory",
    )


    if recent_memory:

        with st.expander(
            "🧠 Learning Memory"
        ):

            st.write(
                recent_memory
            )


    # ============================================================
    # NEW SESSION
    # ============================================================
    st.markdown("---")


    if st.button(

        "🔄 Start New Learning Session",

        type="primary",

        use_container_width=True,
    ):

        reset_session_for_new_lesson()

        st.rerun()


# ================================================================
# RESET SESSION
# ================================================================
def reset_session_for_new_lesson():

    st.session_state.page = (
        "home"
    )

    st.session_state.lesson_plan = (
        None
    )

    st.session_state.current_module_idx = (
        0
    )

    st.session_state.user_score = (
        0
    )

    st.session_state.total_questions = (
        0
    )

    st.session_state.misconceptions = (
        []
    )

    st.session_state.assessment_results = (
        []
    )

    st.session_state.grounded_context = (
        ""
    )

    st.session_state.source_name = (
        ""
    )

    st.session_state.source_type = (
        "topic"
    )

    st.session_state.audio_file = (
        None
    )
    st.session_state.audio_requested = False
    st.session_state.selected_voice = None
    st.session_state.voice_rate = "normal"
    st.session_state.avatar_expression = "explaining"
    st.session_state.current_video_file = None

    st.session_state.current_question_answered = (
        False
    )

    st.session_state.current_question_correct = (
        False
    )

    st.session_state.remediation = (
        None
    )

    st.session_state.retest_mode = (
        False
    )

    st.session_state.retest_answered = (
        False
    )

    st.session_state.lesson_started = (
        False
    )

    st.session_state.lesson_completed = (
        False
    )

    st.session_state.video_plan = (
        None
    )

    st.session_state.video_manifest = (
        None
    )

    st.session_state.topic_mastery = (
        0.0
    )

    st.session_state.chat_messages = (
        []
    )

    st.session_state.generation_timings = {}
    st.session_state.audio_requested = False
    st.session_state.video_plan_requested = False


# ================================================================
# SIDEBAR
# ================================================================
def render_sidebar():

    with st.sidebar:

        st.markdown(
            "## 🎓 PedagogyEngine AI"
        )

        st.caption(
            "Human-like adaptive AI Teacher"
        )

        st.markdown("---")


        if st.session_state.lesson_plan:

            st.markdown(
                "**Current Lesson**"
            )

            st.write(

                st.session_state.lesson_plan.get(

                    "topic_title",

                    "Unknown",
                )
            )


            st.markdown(
                "**Learner**"
            )

            st.write(
                st.session_state.learner_name
            )


            st.markdown(
                "**Level**"
            )

            st.write(
                st.session_state.learner_level
            )


            st.markdown(
                "**Language**"
            )

            st.write(
                st.session_state.learner_language
            )

            st.markdown("---")

            # --------------------------------------------------
            # MODULE PROGRESS DASHBOARD
            # --------------------------------------------------
            st.markdown("**📋 Lesson Progress**")

            modules = st.session_state.lesson_plan.get("modules", [])
            completed = set(st.session_state.get("completed_modules", []))
            current_idx = st.session_state.get("current_module_idx", 0)

            rows_html = []
            for i, module in enumerate(modules):
                title = module.get(
                    "module_title", f"Module {i + 1}"
                )
                title = (title[:26] + "…") if len(title) > 27 else title

                if i in completed:
                    icon, color = "✅", "#10B981"
                elif i == current_idx:
                    icon, color = "🔵", "#6366F1"
                else:
                    icon, color = "⚪", "#9CA3AF"

                rows_html.append(
                    f"<div style='display:flex;align-items:center;"
                    f"gap:.5rem;padding:.25rem 0;color:{color};'>"
                    f"<span>{icon}</span>"
                    f"<span style='color:#E5E7EB;font-size:.85rem;'>"
                    f"{title}</span></div>"
                )

            st.markdown(
                "<div style='max-height:260px;overflow-y:auto;'>"
                + "".join(rows_html)
                + "</div>",
                unsafe_allow_html=True,
            )

            done_count = len(completed)
            st.progress(
                done_count / max(len(modules), 1)
            )
            st.caption(
                f"{done_count}/{len(modules)} modules completed"
            )

            st.markdown("---")


            if st.button(

                "🏠 Back to Setup",

                use_container_width=True,
            ):

                reset_session_for_new_lesson()

                st.rerun()


        st.markdown("---")


        st.caption(

            "Understand → Plan → Explain → "
            "Demonstrate → Question → Evaluate → "
            "Adapt → Continue"
        )


# ================================================================
st.markdown("<div style='text-align:center;opacity:.55;padding:1.2rem 0 .4rem;font-size:.78rem'>Built for students • RAG-grounded • Adaptive assessment • Voice-enabled • Learning memory</div>",unsafe_allow_html=True)


# MAIN ROUTER
# ================================================================
render_sidebar()


if st.session_state.page == "home":

    render_home()


elif st.session_state.page == "teaching":

    render_teaching()


elif st.session_state.page == "report":

    render_report()


else:

    st.session_state.page = (
        "home"
    )

    st.rerun()