import os
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI

from llm_provider import get_chat_llm

from config import (
    DEFAULT_GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)


# ============================================================================
# STRUCTURED LESSON SCHEMAS
# ============================================================================

class InteractiveQuestion(BaseModel):
    question: str = Field(
        description="The question to ask the student during the lesson."
    )

    options: List[str] = Field(
        description="List of 3-4 multiple-choice options."
    )

    correct_answer: str = Field(
        description="The exact text of the correct option."
    )

    explanation_on_misconception: str = Field(
        description=(
            "Simplified explanation if the student selects "
            "the wrong answer."
        )
    )


class VisualData(BaseModel):
    """
    Structured, topic-specific data behind a module's visual.

    Only the fields relevant to ``visual_type`` need to be populated;
    the rendering layer falls back gracefully when a field is empty.
    This exists so charts/flowcharts/timelines/tables are built from the
    actual lesson content instead of generic placeholder shapes.
    """

    x_label: str = Field(
        default="",
        description="X-axis label, for GRAPH visuals only.",
    )

    y_label: str = Field(
        default="",
        description="Y-axis label, for GRAPH visuals only.",
    )

    chart_points: List[List[float]] = Field(
        default_factory=list,
        description=(
            "For GRAPH visuals: 6-12 real [x, y] data points that "
            "illustrate the actual relationship being taught "
            "(e.g. current vs resistance for Ohm's Law), not a "
            "generic curve."
        ),
    )

    flow_steps: List[str] = Field(
        default_factory=list,
        description=(
            "For FLOWCHART/PROCESS visuals: 3-6 short ordered step "
            "labels that are specific to this concept, e.g. "
            "['Water absorbed by roots', 'Transported to leaves', "
            "'Light reaction', 'Glucose produced']."
        ),
    )

    timeline_events: List[str] = Field(
        default_factory=list,
        description=(
            "For TIMELINE visuals: 3-6 short 'label: detail' entries "
            "specific to this concept in chronological order."
        ),
    )

    table_rows: List[List[str]] = Field(
        default_factory=list,
        description=(
            "For TABLE visuals: rows of [aspect, explanation] specific "
            "to this concept."
        ),
    )


class LessonModule(BaseModel):
    module_title: str = Field(
        description="Title of this specific teaching module."
    )

    spoken_script: str = Field(
        description=(
            "Natural, human-like explanation script for "
            "the AI teacher/avatar to speak."
        )
    )

    visual_type: str = Field(
        description=(
            "Type of visual: latex, code, markdown, diagram, "
            "graph, process, timeline, map, flowchart, table, "
            "image, simulation, or text."
        )
    )

    visual_content: str = Field(
        description=(
            "Exact content to display in the visual."
        )
    )

    visual_data: VisualData = Field(
        default_factory=VisualData,
        description=(
            "Structured, topic-specific data for the visual "
            "(see VisualData). Populate only the fields relevant "
            "to visual_type."
        ),
    )

    checkpoint_question: InteractiveQuestion


class CompleteLessonPlan(BaseModel):
    topic_title: str = Field(
        description="Title of the lesson."
    )

    target_level: str = Field(
        description=(
            "Target learner level: Beginner, Intermediate, "
            "or Advanced."
        )
    )

    language: str = Field(
        description="Teaching language requested by the student."
    )

    time_allocation_mins: int = Field(
        description="Total lesson time in minutes."
    )

    modules: List[LessonModule] = Field(
        description=(
            "Sequential list of teaching modules."
        )
    )


class DynamicRemediation(BaseModel):
    misconception_reason: str = Field(
        description=(
            "Direct and constructive explanation of why "
            "the student's answer was incorrect."
        )
    )

    new_analogy: str = Field(
        description=(
            "A fresh real-world analogy that explains "
            "the concept from another perspective."
        )
    )

    spoken_remediation_script: str = Field(
        description=(
            "Empathetic spoken script for the AI teacher "
            "to address the misconception."
        )
    )


# ============================================================================
# LESSON GENERATOR ENGINE
# ============================================================================

class LessonGeneratorEngine:
    """
    AI Teacher lesson-generation engine.

    Converts:

        Topic
        +
        Learner profile
        +
        RAG grounded context
        +
        Time mode

    into:

        Structured interactive lesson.

    Features:
        - Gemini structured output
        - RAG/source grounding
        - Time-adaptive content
        - Multilingual teaching
        - Interactive checkpoints
        - Dynamic remediation
        - Safe operation without an API key

    The original generate_lesson_plan() interface remains compatible.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
    ):
        # --------------------------------------------------------------------
        # API KEY
        # --------------------------------------------------------------------

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )

        # --------------------------------------------------------------------
        # MODEL
        # --------------------------------------------------------------------

        self.model_name = (
            model_name
            or DEFAULT_GEMINI_MODEL
        )

        # --------------------------------------------------------------------
        # Initialize LLM objects safely
        #
        # IMPORTANT:
        # ChatGoogleGenerativeAI throws a validation error when initialized
        # without an API key. Therefore we only create the LLM when a key
        # actually exists.
        # --------------------------------------------------------------------

        self.llm_base = None
        self.lesson_llm = None
        self.remediation_llm = None

        groq_available = bool(os.getenv("GROQ_API_KEY", ""))

        if not self.api_key and not groq_available:
            print(
                "Warning: No LLM API key is set (GROQ_API_KEY or "
                "GEMINI_API_KEY). Lesson generation will be unavailable."
            )
            return

        try:
            self.llm_base = get_chat_llm(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                gemini_api_key=self.api_key,
                gemini_model=self.model_name,
            )

            # Structured lesson output
            self.lesson_llm = (
                self.llm_base.with_structured_output(
                    CompleteLessonPlan
                )
            )

            # Structured remediation output
            self.remediation_llm = (
                self.llm_base.with_structured_output(
                    DynamicRemediation
                )
            )

        except Exception as exc:
            print(
                "Warning: Gemini initialization failed."
            )
            print(
                f"Reason: {exc}"
            )

            self.llm_base = None
            self.lesson_llm = None
            self.remediation_llm = None

    # =========================================================================
    # STATUS
    # =========================================================================

    def is_available(self) -> bool:
        """
        Return True when Gemini is ready for generation.
        """

        return (
            self.lesson_llm is not None
            and self.remediation_llm is not None
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Return engine status without exposing the API key.
        """

        return {
            "available": self.is_available(),
            "model": self.model_name,
            "api_key_configured": bool(
                self.api_key
            ),
        }

    # =========================================================================
    # RAG SOURCE FORMATTING
    # =========================================================================

    @staticmethod
    def _format_source_references(
        source_references: Optional[
            List[Dict[str, Any]]
        ],
    ) -> str:
        """
        Convert RAG source metadata into readable prompt text.
        """

        if not source_references:
            return (
                "No explicit source-reference list was provided. "
                "Use only the grounded material supplied above."
            )

        lines = []

        for index, source in enumerate(
            source_references,
            start=1,
        ):
            if not isinstance(
                source,
                dict,
            ):
                continue

            filename = source.get(
                "filename",
                "Unknown source",
            )

            location = source.get(
                "location",
                "",
            )

            file_type = source.get(
                "file_type",
                "",
            )

            reference = str(
                filename
            )

            if location:
                reference += (
                    f" — {location}"
                )

            if file_type:
                reference += (
                    f" [{file_type}]"
                )

            lines.append(
                f"{index}. {reference}"
            )

        if not lines:
            return (
                "No usable source-reference metadata was provided."
            )

        return "\n".join(
            lines
        )

    # =========================================================================
    # RAG CONTEXT NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_grounded_context(
        grounded_context: str,
    ) -> str:
        """
        Normalize the RAG context before giving it to Gemini.
        """

        if not grounded_context:
            return (
                "NO_RELEVANT_SOURCE_MATERIAL_FOUND."
            )

        return str(
            grounded_context
        ).strip()

    # =========================================================================
    # TIME MODE INSTRUCTIONS
    # =========================================================================

    @staticmethod
    def _get_time_mode_instruction(
        available_time: int,
    ) -> str:
        """
        Create genuinely different content requirements
        for each time mode.

        The model is instructed to change the actual teaching
        depth rather than simply shortening the same script.
        """

        try:
            minutes = int(
                available_time
            )
        except Exception:
            minutes = 20

        # --------------------------------------------------------------------
        # 5 MINUTES
        # --------------------------------------------------------------------

        if minutes <= 5:
            return """
TIME MODE: QUICK 5-MINUTE CONCEPT

Create a genuinely concise teaching session.

Focus ONLY on the highest-value concept(s) needed for
the learner to understand the topic at a basic level.

Include:
- Core idea
- Essential intuition
- Key terminology
- One compact example
- One important visual
- One common misconception
- One comprehension checkpoint

Do NOT attempt to teach the entire topic.

The learner should finish with a clear mental model,
not a long list of facts.

Keep the explanation efficient and focused.
"""

        # --------------------------------------------------------------------
        # 20 MINUTES
        # --------------------------------------------------------------------

        if minutes <= 20:
            return """
TIME MODE: STRUCTURED 20-MINUTE LEARNING

Create a substantially deeper lesson than the 5-minute mode.

Progress through the concept logically.

Include:
- Foundation and terminology
- Core concepts
- Intuition and why the concept works
- Important relationships
- At least one worked example
- Appropriate subject-specific visuals
- Common misconceptions
- Guided understanding
- Multiple meaningful checkpoints
- A short recap

Do NOT simply expand the 5-minute explanation with filler.

Introduce additional conceptual depth and examples that
would not be present in the 5-minute version.
"""

        # --------------------------------------------------------------------
        # 60 MINUTES
        # --------------------------------------------------------------------

        if minutes <= 60:
            return """
TIME MODE: DEEP 60-MINUTE LEARNING

Create a comprehensive deep-learning lesson.

The lesson must contain multiple coherent modules and
substantially more conceptual content than the 5-minute
and 20-minute versions.

Progress through:

1. Foundation
2. Terminology
3. Core concepts
4. Detailed intuition
5. Important relationships
6. Worked examples
7. Subject-specific demonstrations
8. Visual explanations
9. Common misconceptions
10. Guided practice
11. Application
12. Higher-order reasoning
13. Recap
14. Mastery checkpoint

Do NOT produce a 20-minute lesson and merely label it
as a 60-minute lesson.

The actual concepts, explanations, examples, and depth
must increase substantially.

Use multiple modules that build on each other.
"""

        # --------------------------------------------------------------------
        # EXTENDED
        # --------------------------------------------------------------------

        return """
TIME MODE: EXTENDED LEARNING

Create a comprehensive learning sequence appropriate
for the available time.

Prioritize:
- Progressive conceptual development
- Multiple examples
- Practice
- Application
- Misconception handling
- Revision
- Mastery

The content itself must expand with the available time.
Do not pad the lesson with repetitive explanations.
"""

    # =========================================================================
    # LESSON PROMPT
    # =========================================================================

    def _build_lesson_prompt(
        self,
        topic: str,
        grounded_context: str,
        level: str,
        language: str,
        available_time: int,
        source_references: Optional[
            List[Dict[str, Any]]
        ],
    ) -> str:
        """
        Build the complete RAG-grounded lesson-generation prompt.
        """

        grounded_context = (
            self._normalize_grounded_context(
                grounded_context
            )
        )

        source_text = (
            self._format_source_references(
                source_references
            )
        )

        time_instruction = (
            self._get_time_mode_instruction(
                available_time
            )
        )

        return f"""
You are an expert AI Teacher preparing a personalized,
interactive teaching session.

Your goal is not merely to summarize a textbook.

Your goal is to TEACH the learner.

============================================================
STUDENT PROFILE
============================================================

Topic:
{topic}

Learner Level:
{level}

Teaching Language:
{language}

Available Time:
{available_time} minutes

============================================================
RETRIEVED EDUCATIONAL MATERIAL
============================================================

The following material was retrieved from the learner's
uploaded educational resources.

Treat this retrieved material as the PRIMARY SOURCE OF TRUTH
for factual claims about the requested topic.

Use it to ground:
- Definitions
- Concepts
- Formulas
- Examples
- Terminology
- Explanations
- Relationships

Do NOT invent information and claim that it came from
the uploaded material.

If the source material does not contain enough information,
do not fabricate a source-backed claim.

You may use basic pedagogical reasoning to explain the
retrieved material more clearly, but preserve the factual
meaning of the source.

---------------- SOURCE CONTEXT ----------------

{grounded_context}

---------------- END SOURCE CONTEXT ----------------

============================================================
SOURCE REFERENCES
============================================================

{source_text}

============================================================
TIME-ADAPTIVE CONTENT
============================================================

{time_instruction}

============================================================
PEDAGOGICAL REQUIREMENTS
============================================================

1. Adapt the explanation to the learner level.

2. Use the requested teaching language.

3. If Hindi or Hinglish is requested, use natural,
   conversational Hindi/Hinglish suitable for speaking.

4. Build the lesson progressively.

5. Every module must contain:
   - Module title
   - Natural spoken explanation
   - Meaningful visual
   - Interactive MCQ checkpoint
   - Misconception explanation

6. The spoken script should sound like a human teacher:
   - conversational
   - warm
   - encouraging
   - clear
   - progressive
   - engaging

7. Do NOT simply read textbook sentences.

8. Transform source material into a teaching experience.

9. Avoid unnecessary repetition.

10. Each checkpoint should test understanding,
    not merely word-for-word recall.

11. The correct answer must be unambiguous.

12. For mathematics:
    preserve equations accurately.

13. For programming:
    provide technically meaningful code and explain it.

14. For science:
    prefer diagrams, processes, formulas, graphs,
    or simulations where appropriate.

15. For history/social sciences:
    prefer timelines, maps, relationships,
    and structured explanations where appropriate.

16. Select visuals that match the actual subject.

16a. ALWAYS populate `visual_data` with real, concept-specific data
     (not placeholders). For a GRAPH give real [x, y] points for the
     actual relationship being taught. For a FLOWCHART/PROCESS give
     the real ordered steps of the actual process. For a TIMELINE give
     the real chronological events. For a TABLE give real comparison
     rows. Never reuse a generic "Input/Process/Output" or
     "Beginning/Development/Outcome" style placeholder — it must be
     obviously specific to this exact concept.

17. The lesson must adapt to the learner's available time.

18. The actual CONTENT must change with the time mode.

19. Do not create one generic lesson and simply
    pretend it fits every duration.

20. Do not fabricate references or page numbers.

21. Keep the lesson coherent from module to module.

22. Prepare the learner for active participation.

23. Return only information compatible with the
    requested structured lesson schema.
"""

    # =========================================================================
    # GENERATE LESSON
    # =========================================================================

    def generate_lesson_plan(
        self,
        topic: str,
        grounded_context: str,
        level: str = "Beginner",
        language: str = "English",
        available_time: int = 20,
        source_references: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured, personalized lesson.

        Backward compatible with the previous interface.

        Existing call:

            generate_lesson_plan(
                topic,
                grounded_context,
                level,
                language,
                available_time
            )

        New optional parameter:

            source_references
        """

        # --------------------------------------------------------------------
        # API availability check
        # --------------------------------------------------------------------

        if self.lesson_llm is None:
            return {
                "error": "Gemini API key is not configured.",
                "message": (
                    "Set GEMINI_API_KEY or GOOGLE_API_KEY "
                    "before generating a lesson."
                ),
            }

        # --------------------------------------------------------------------
        # Build prompt
        # --------------------------------------------------------------------

        prompt = self._build_lesson_prompt(
            topic=topic,
            grounded_context=grounded_context,
            level=level,
            language=language,
            available_time=available_time,
            source_references=source_references,
        )

        # --------------------------------------------------------------------
        # Gemini generation
        # --------------------------------------------------------------------

        try:
            structured_plan = (
                self.lesson_llm.invoke(
                    prompt
                )
            )

            # Pydantic structured response
            if hasattr(
                structured_plan,
                "model_dump",
            ):
                return (
                    structured_plan.model_dump()
                )

            # Dictionary fallback
            if isinstance(
                structured_plan,
                dict,
            ):
                return structured_plan

            # Preserve unexpected result for app-level handling
            return structured_plan

        except Exception as exc:
            print(
                "Error generating lesson plan:"
            )
            print(
                repr(exc)
            )

            # IMPORTANT:
            # Return the exception rather than crashing the application.
            # app.py already contains defensive handling for non-dict
            # lesson-generation responses.
            return exc

    # =========================================================================
    # REMEDIATION PROMPT
    # =========================================================================

    def _build_remediation_prompt(
        self,
        topic: str,
        question: str,
        wrong_answer: str,
        language: str,
        grounded_context: str = "",
    ) -> str:
        """
        Build a misconception-remediation prompt.
        """

        source_section = ""

        if grounded_context:
            source_section = f"""
============================================================
RELEVANT SOURCE MATERIAL
============================================================

{grounded_context}

Use this source material to keep the remediation
factually consistent with the learner's material.
"""

        return f"""
You are an empathetic AI Teacher helping a learner
recover from a misconception.

============================================================
LEARNING CONTEXT
============================================================

Topic:
{topic}

Checkpoint Question:
{question}

Student's Incorrect Answer:
{wrong_answer}

Preferred Language:
{language}

{source_section}

============================================================
YOUR TASK
============================================================

1. Identify the likely misconception.

2. Explain WHY the student's answer is incorrect.

3. Be constructive and non-judgmental.

4. Use a completely NEW analogy or explanation.

5. Do NOT simply repeat the previous explanation.

6. Create a natural spoken script suitable for
   an AI teacher/avatar.

7. Keep the explanation appropriate for the learner level.

8. If source material is supplied, remain consistent
   with that material.

9. Prepare the student for another attempt.

The student should feel:

"I understand what I misunderstood,
and I can try the question again."

Return only information compatible with
the DynamicRemediation schema.
"""

    # =========================================================================
    # GENERATE REMEDIATION
    # =========================================================================

    def generate_remediation(
        self,
        topic: str,
        question: str,
        wrong_answer: str,
        language: str = "English",
        grounded_context: str = "",
    ) -> Dict[str, Any]:
        """
        Dynamically generate misconception remediation.

        grounded_context is optional for backward compatibility.
        """

        # --------------------------------------------------------------------
        # API availability
        # --------------------------------------------------------------------

        if self.remediation_llm is None:
            return {
                "misconception_reason": (
                    "Gemini is not currently configured. "
                    "Let's revisit the concept step by step."
                ),
                "new_analogy": (
                    "Think about the concept from a different "
                    "real-world perspective before trying again."
                ),
                "spoken_remediation_script": (
                    "That's okay. Mistakes are part of learning. "
                    "Let's pause, look at the idea from another "
                    "angle, and then try a similar question again."
                ),
            }

        # --------------------------------------------------------------------
        # Build prompt
        # --------------------------------------------------------------------

        prompt = self._build_remediation_prompt(
            topic=topic,
            question=question,
            wrong_answer=wrong_answer,
            language=language,
            grounded_context=grounded_context,
        )

        # --------------------------------------------------------------------
        # Generate
        # --------------------------------------------------------------------

        try:
            remediation_plan = (
                self.remediation_llm.invoke(
                    prompt
                )
            )

            if hasattr(
                remediation_plan,
                "model_dump",
            ):
                return (
                    remediation_plan.model_dump()
                )

            if isinstance(
                remediation_plan,
                dict,
            ):
                return remediation_plan

            return remediation_plan

        except Exception as exc:
            print(
                "Error generating remediation:"
            )
            print(
                repr(exc)
            )

            return {
                "misconception_reason": (
                    "Let's look at the concept from "
                    "a different perspective."
                ),
                "new_analogy": (
                    "Imagine the concept using a simple "
                    "real-world situation and compare each "
                    "part step by step."
                ),
                "spoken_remediation_script": (
                    "That's completely okay. Let's pause "
                    "and look at this idea from another angle. "
                    "I'll explain it step by step, and then "
                    "you can try a similar question again."
                ),
            }


# ============================================================================
# LOCAL VERIFICATION
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LESSON GENERATOR — RAG AWARE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Sample RAG context
    # ------------------------------------------------------------------------

    sample_context = """
[SOURCE 1: Physics_Chapter_3.pdf — Page 4]

Ohm's law describes the relationship between voltage,
current, and resistance.

The formula is:

V = I × R

Voltage is measured in volts.
Current is measured in amperes.
Resistance is measured in ohms.

Resistance opposes current flow.
When voltage remains constant, increasing resistance
reduces current.

[END SOURCE 1]

[SOURCE 2: Physics_Chapter_3.pdf — Page 5]

Example:

If voltage is 10 volts and resistance is 5 ohms,
current is:

I = V / R
I = 10 / 5
I = 2 amperes.

[END SOURCE 2]
"""

    source_references = [
        {
            "filename": "Physics_Chapter_3.pdf",
            "location": "Page 4",
            "file_type": "pdf",
        },
        {
            "filename": "Physics_Chapter_3.pdf",
            "location": "Page 5",
            "file_type": "pdf",
        },
    ]

    # ------------------------------------------------------------------------
    # API key
    # ------------------------------------------------------------------------

    gemini_key = (
        os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    # ------------------------------------------------------------------------
    # Initialize engine
    # ------------------------------------------------------------------------

    generator = LessonGeneratorEngine(
        api_key=gemini_key
    )

    print(
        "\nEngine status:"
    )

    print(
        json.dumps(
            generator.get_status(),
            indent=2,
        )
    )

    # ------------------------------------------------------------------------
    # TEST 1 — Time modes
    # ------------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TEST 1: Time-mode instructions"
    )

    for minutes in [5, 20, 60]:

        instruction = (
            generator._get_time_mode_instruction(
                minutes
            )
        )

        assert (
            "TIME MODE"
            in instruction
        )

        print(
            f"PASS — {minutes}-minute mode"
        )

    # ------------------------------------------------------------------------
    # TEST 2 — Source formatting
    # ------------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TEST 2: RAG source formatting"
    )

    formatted_sources = (
        generator._format_source_references(
            source_references
        )
    )

    assert (
        "Physics_Chapter_3.pdf"
        in formatted_sources
    )

    assert (
        "Page 4"
        in formatted_sources
    )

    assert (
        "Page 5"
        in formatted_sources
    )

    print(
        "PASS — source references formatted"
    )

    # ------------------------------------------------------------------------
    # TEST 3 — Context normalization
    # ------------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TEST 3: Grounded context normalization"
    )

    normalized_context = (
        generator._normalize_grounded_context(
            sample_context
        )
    )

    assert (
        "Ohm's law"
        in normalized_context
    )

    assert (
        "V = I"
        in normalized_context
    )

    print(
        "PASS — grounded context preserved"
    )

    # ------------------------------------------------------------------------
    # TEST 4 — Prompt generation
    # ------------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TEST 4: RAG lesson prompt generation"
    )

    prompt = generator._build_lesson_prompt(
        topic="Ohm's Law",
        grounded_context=sample_context,
        level="Beginner",
        language="English",
        available_time=5,
        source_references=source_references,
    )

    assert (
        "Ohm's Law"
        in prompt
    )

    assert (
        "Physics_Chapter_3.pdf"
        in prompt
    )

    assert (
        "Page 4"
        in prompt
    )

    assert (
        "PRIMARY SOURCE OF TRUTH"
        in prompt
    )

    assert (
        "TIME MODE: QUICK 5-MINUTE CONCEPT"
        in prompt
    )

    print(
        "PASS — RAG-aware prompt created"
    )

    # ------------------------------------------------------------------------
    # TEST 5 — Gemini generation
    # ------------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TEST 5: Full Gemini lesson generation"
    )

    if not gemini_key:

        print(
            "SKIPPED — GEMINI_API_KEY / GOOGLE_API_KEY "
            "is not configured."
        )

    else:

        plan = generator.generate_lesson_plan(
            topic="Ohm's Law",
            grounded_context=sample_context,
            level="Beginner",
            language="English",
            available_time=5,
            source_references=source_references,
        )

        if isinstance(
            plan,
            dict,
        ):

            print(
                "\nGenerated lesson:"
            )

            print(
                json.dumps(
                    plan,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            assert (
                plan.get("modules")
            )

            print(
                "\nPASS — Gemini returned structured lesson"
            )

        else:

            print(
                "\nGemini generation returned:"
            )

            print(
                repr(plan)
            )

            print(
                "\nGemini generation could not be verified."
            )

    # ------------------------------------------------------------------------
    # TEST 6 — Remediation prompt
    # ------------------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TEST 6: Remediation prompt generation"
    )

    remediation_prompt = (
        generator._build_remediation_prompt(
            topic="Ohm's Law",
            question=(
                "If resistance doubles while voltage "
                "remains constant, what happens to current?"
            ),
            wrong_answer="Current doubles",
            language="English",
            grounded_context=sample_context,
        )
    )

    assert (
        "Current doubles"
        in remediation_prompt
    )

    assert (
        "Ohm's Law"
        in remediation_prompt
    )

    print(
        "PASS — remediation prompt created"
    )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL LOCAL LESSON GENERATOR TESTS PASSED"
    )

    print(
        "=" * 70
    )

    print(
        "\nGemini availability:",
        generator.is_available(),
    )

    if not generator.is_available():
        print(
            "\nGemini generation was skipped because "
            "no API key is configured."
        )

    print(
        "\nRAG-aware lesson generation is ready "
        "for app.py integration."
    )