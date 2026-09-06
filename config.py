"""
PedagogyEngine AI
Central configuration for the AI Teacher system.

This file contains:
- Application settings
- Gemini configuration
- RAG configuration
- Lesson generation settings
- Learner personalization defaults
- Supported languages
- Teaching actions
- Assessment settings
- Database configuration
- Video/audio configuration

IMPORTANT:
Do NOT hard-code your Gemini API key in this file.
The API key should be supplied through Streamlit secrets or
an environment variable.
"""

import os
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"

DATA_DIR.mkdir(exist_ok=True)
PROMPTS_DIR.mkdir(exist_ok=True)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "PedagogyEngine AI"
APP_TITLE = "PedagogyEngine AI — Interactive AI Teacher"

APP_VERSION = "1.0.0"

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


# ============================================================
# GEMINI / LLM CONFIGURATION
# ============================================================

# Do NOT put your real API key here.
#
# The application will first look for:
# 1. Streamlit secrets
# 2. GEMINI_API_KEY environment variable
#
# app.py can also pass the key supplied by the user.

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# Model can be overridden using an environment variable.
#
# Keep this configurable because Gemini model availability can
# change over time.
DEFAULT_GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# Temperature controls creativity.
#
# Lower value:
#   More deterministic / factual
#
# Higher value:
#   More creative / varied
#
# Educational explanations should generally remain fairly
# deterministic.

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.3")
)

LLM_MAX_OUTPUT_TOKENS = int(
    os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu"
)

VECTOR_DATABASE_DIR = DATA_DIR / "chroma_db"

VECTOR_COLLECTION_NAME = "educational_material"

# Chunking configuration

RAG_CHUNK_SIZE = int(
    os.getenv("RAG_CHUNK_SIZE", "700")
)

RAG_CHUNK_OVERLAP = int(
    os.getenv("RAG_CHUNK_OVERLAP", "120")
)

# Number of relevant chunks retrieved for an explanation.

RAG_TOP_K = int(
    os.getenv("RAG_TOP_K", "5")
)


# ============================================================
# SUPPORTED INPUT MATERIALS
# ============================================================

SUPPORTED_FILE_TYPES = [
    "pdf",
    "txt",
    "docx",
    "pptx",
]

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".pptx",
}


# ============================================================
# LEARNER LEVELS
# ============================================================

LEARNER_LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced",
]


# ============================================================
# LEARNING GOALS
# ============================================================

LEARNING_GOALS = [
    "Concept Understanding",
    "Exam Preparation",
    "Interview Preparation",
    "Practical Application",
    "Project Learning",
    "Revision",
]


# ============================================================
# TEACHING STYLES
# ============================================================

TEACHING_STYLES = [
    "Visual",
    "Step-by-Step",
    "Example-Based",
    "Socratic",
    "Story-Based",
    "Exam-Focused",
]


# ============================================================
# DEPTH LEVELS
# ============================================================

DEPTH_LEVELS = [
    "Concise",
    "Standard",
    "Detailed",
    "Deep Dive",
]


# ============================================================
# SUPPORTED TEACHING LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = [
    "English",
    "Hinglish",
    "Hindi",
    "Tamil",
    "Spanish",
]


# ============================================================
# SESSION DURATIONS
# ============================================================

SESSION_DURATIONS = [
    5,
    10,
    20,
    30,
    45,
    60,
]


# ============================================================
# TIME-BASED TEACHING MODES
# ============================================================

TIME_MODES = {
    5: "Quick Concept",
    10: "Focused Lesson",
    20: "Structured Lesson",
    30: "Detailed Lesson",
    45: "Deep Learning",
    60: "Deep Dive",
}


# ============================================================
# HUMAN-LIKE TEACHER ACTIONS
# ============================================================

TEACHER_ACTIONS = [
    "INTRODUCE",
    "EXPLAIN",
    "DEMONSTRATE",
    "ASK",
    "EVALUATE",
    "SIMPLIFY",
    "REEXPLAIN",
    "REMEDIATE",
    "INCREASE_DIFFICULTY",
    "RECAP",
    "CONTINUE",
    "ASSESS",
]


# ============================================================
# DIFFICULTY LEVELS
# ============================================================

DIFFICULTY_LEVELS = [
    "EASY",
    "MEDIUM",
    "HARD",
]


# ============================================================
# QUESTION TYPES
# ============================================================

QUESTION_TYPES = [
    "MCQ",
    "SHORT_ANSWER",
    "CONCEPTUAL",
    "PROBLEM_SOLVING",
    "APPLICATION",
    "OWN_WORDS",
]


# ============================================================
# ASSESSMENT SETTINGS
# ============================================================

# Initial mastery threshold.

INITIAL_MASTERY_THRESHOLD = 0.70

# Strong mastery.

STRONG_MASTERY_THRESHOLD = 0.85

# Below this level the teacher should normally consider
# remediation.

WEAK_MASTERY_THRESHOLD = 0.50

# Maximum number of remediation attempts before moving to
# another strategy.

MAX_REMEDIATION_ATTEMPTS = 2

# Number of questions used in final assessment.

FINAL_ASSESSMENT_QUESTIONS = 5


# ============================================================
# MISCONCEPTION TYPES
# ============================================================

MISCONCEPTION_TYPES = [
    "CONCEPTUAL_GAP",
    "CALCULATION_ERROR",
    "TERMINOLOGY_CONFUSION",
    "PARTIAL_UNDERSTANDING",
    "GUESS",
    "MISAPPLIED_CONCEPT",
    "UNKNOWN",
]


# ============================================================
# VISUAL TYPES
# ============================================================

VISUAL_TYPES = [
    "EQUATION",
    "GRAPH",
    "DIAGRAM",
    "PROCESS",
    "TIMELINE",
    "MAP",
    "CODE",
    "FLOWCHART",
    "SIMULATION",
    "TABLE",
    "IMAGE",
    "TEXT",
]


# ============================================================
# SUBJECT → VISUAL PREFERENCES
# ============================================================

SUBJECT_VISUAL_MAPPING = {
    "mathematics": [
        "EQUATION",
        "GRAPH",
        "DIAGRAM",
        "TABLE",
    ],

    "physics": [
        "EQUATION",
        "DIAGRAM",
        "GRAPH",
        "SIMULATION",
    ],

    "chemistry": [
        "DIAGRAM",
        "PROCESS",
        "EQUATION",
        "TABLE",
    ],

    "biology": [
        "DIAGRAM",
        "PROCESS",
        "FLOWCHART",
        "IMAGE",
    ],

    "history": [
        "TIMELINE",
        "MAP",
        "IMAGE",
        "TEXT",
    ],

    "geography": [
        "MAP",
        "DIAGRAM",
        "GRAPH",
        "IMAGE",
    ],

    "computer science": [
        "CODE",
        "FLOWCHART",
        "DIAGRAM",
        "TABLE",
    ],

    "programming": [
        "CODE",
        "FLOWCHART",
        "DIAGRAM",
        "TEXT",
    ],

    "machine learning": [
        "DIAGRAM",
        "GRAPH",
        "FLOWCHART",
        "EQUATION",
    ],

    "artificial intelligence": [
        "DIAGRAM",
        "FLOWCHART",
        "GRAPH",
        "CODE",
    ],
}


# ============================================================
# AVATAR CONFIGURATION
# ============================================================

DEFAULT_TEACHER_NAME = "Prof. AI"

AVATAR_EMOTIONS = [
    "friendly",
    "happy",
    "curious",
    "encouraging",
    "thinking",
    "confident",
]


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

DEFAULT_AUDIO_FORMAT = "mp3"

AUDIO_OUTPUT_DIR = DATA_DIR / "audio"
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# VIDEO CONFIGURATION
# ============================================================

DEFAULT_VIDEO_FORMAT = "mp4"

VIDEO_OUTPUT_DIR = DATA_DIR / "videos"
VIDEO_OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_FPS = 24

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


# ============================================================
# LEARNING MEMORY / DATABASE
# ============================================================

DATABASE_PATH = DATA_DIR / "pedagogy_engine.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


# ============================================================
# LEARNING PATH CONFIGURATION
# ============================================================

MAX_NEXT_TOPIC_RECOMMENDATIONS = 3

MIN_MASTERY_FOR_NEXT_TOPIC = 0.75


# ============================================================
# PROGRESS CONFIGURATION
# ============================================================

PROGRESS_STATES = [
    "NOT_STARTED",
    "IN_PROGRESS",
    "STRUGGLING",
    "IMPROVING",
    "MASTERED",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_gemini_api_key():
    """
    Retrieve the Gemini API key from the environment.

    Streamlit secrets are intentionally handled separately
    inside app.py because config.py should remain usable
    outside Streamlit as well.

    Returns:
        str | None
    """

    return os.getenv(GEMINI_API_KEY_ENV)


def get_time_mode(minutes: int) -> str:
    """
    Return the appropriate teaching mode for the given
    session duration.
    """

    if minutes <= 5:
        return TIME_MODES[5]

    if minutes <= 10:
        return TIME_MODES[10]

    if minutes <= 20:
        return TIME_MODES[20]

    if minutes <= 30:
        return TIME_MODES[30]

    if minutes <= 45:
        return TIME_MODES[45]

    return TIME_MODES[60]


def validate_learner_profile(
    level: str,
    goal: str,
    language: str,
    teaching_style: str,
    depth: str,
    available_time: int,
) -> bool:
    """
    Validate the learner personalization settings.
    """

    if level not in LEARNER_LEVELS:
        return False

    if goal not in LEARNING_GOALS:
        return False

    if language not in SUPPORTED_LANGUAGES:
        return False

    if teaching_style not in TEACHING_STYLES:
        return False

    if depth not in DEPTH_LEVELS:
        return False

    if available_time <= 0:
        return False

    return True


def get_subject_visuals(subject: str):
    """
    Return recommended visual types for a subject.

    Uses a flexible keyword match rather than requiring an
    exact subject name.
    """

    subject_lower = subject.lower().strip()

    for key, visuals in SUBJECT_VISUAL_MAPPING.items():
        if key in subject_lower:
            return visuals

    # Generic fallback
    return [
        "DIAGRAM",
        "GRAPH",
        "TABLE",
        "TEXT",
    ]


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def get_config_summary():
    """
    Return a safe configuration summary.

    The API key is intentionally NOT included.
    """

    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "rag_top_k": RAG_TOP_K,
        "supported_languages": SUPPORTED_LANGUAGES,
        "learner_levels": LEARNER_LEVELS,
        "learning_goals": LEARNING_GOALS,
        "teaching_styles": TEACHING_STYLES,
        "depth_levels": DEPTH_LEVELS,
        "question_types": QUESTION_TYPES,
        "visual_types": VISUAL_TYPES,
        "database": str(DATABASE_PATH),
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PedagogyEngine AI - Configuration Test")
    print("=" * 60)

    print("\nApplication:")
    print(f"  Name: {APP_NAME}")
    print(f"  Version: {APP_VERSION}")

    print("\nLLM:")
    print(f"  Gemini Model: {DEFAULT_GEMINI_MODEL}")
    print(f"  Temperature: {LLM_TEMPERATURE}")

    print("\nRAG:")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Chunk Size: {RAG_CHUNK_SIZE}")
    print(f"  Chunk Overlap: {RAG_CHUNK_OVERLAP}")
    print(f"  Top K: {RAG_TOP_K}")

    print("\nLanguages:")
    print(f"  {', '.join(SUPPORTED_LANGUAGES)}")

    print("\nTeacher Actions:")
    print(f"  {', '.join(TEACHER_ACTIONS)}")

    print("\nQuestion Types:")
    print(f"  {', '.join(QUESTION_TYPES)}")

    print("\nVisual Types:")
    print(f"  {', '.join(VISUAL_TYPES)}")

    print("\nTime Modes:")

    for minutes, mode in TIME_MODES.items():
        print(f"  {minutes} min → {mode}")

    print("\nSubject Visual Test:")

    for subject in [
        "Physics",
        "Mathematics",
        "Machine Learning",
        "Programming",
        "History",
    ]:
        print(
            f"  {subject}: "
            f"{get_subject_visuals(subject)}"
        )

    print("\nGemini API key:")
    print(
        "  Available"
        if get_gemini_api_key()
        else "  Not configured "
        "(this is okay for config.py testing)"
    )

    print("\nConfiguration loaded successfully.")
    print("=" * 60)