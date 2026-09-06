"""
PedagogyEngine AI
Teacher Agent

The Teacher Agent is the decision-making brain of the AI Teacher.

It controls the human-like teaching loop:

Understand
    ↓
Plan
    ↓
Explain
    ↓
Demonstrate
    ↓
Question
    ↓
Evaluate
    ↓
Adapt
    ↓
Continue

The agent does NOT directly handle:
- PDF processing
- Vector database operations
- Audio generation
- Video rendering
- Database persistence

Those responsibilities belong to other modules.

The Teacher Agent decides WHAT should happen next.
"""

from typing import Any, Dict, Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI

from llm_provider import get_chat_llm

from config import (
    DEFAULT_GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    TEACHER_ACTIONS,
    DIFFICULTY_LEVELS,
    QUESTION_TYPES,
    MISCONCEPTION_TYPES,
    INITIAL_MASTERY_THRESHOLD,
    STRONG_MASTERY_THRESHOLD,
    WEAK_MASTERY_THRESHOLD,
)


# ============================================================
# TEACHER AGENT
# ============================================================

class TeacherAgent:
    """
    AI Teacher decision engine.

    The agent observes:

        - learner profile
        - current concept
        - previous teaching
        - student response
        - assessment result
        - misconception information
        - current mastery
        - remaining time

    and decides the next pedagogical action.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        """
        Initialize the Teacher Agent.

        Args:
            api_key:
                Gemini API key.

            model_name:
                Gemini model to use.

            temperature:
                LLM creativity level.
        """

        import os as _os
        if not api_key and not _os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "An API key is required to initialize TeacherAgent — "
                "set GROQ_API_KEY (recommended, free) or pass a "
                "Gemini API key."
            )

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

        self.llm = get_chat_llm(
            temperature=self.temperature,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
            gemini_api_key=self.api_key,
            gemini_model=self.model_name,
        )

    # ========================================================
    # BASIC HELPERS
    # ========================================================

    @staticmethod
    def _clean_json_response(response_text: str) -> str:
        """
        Remove markdown code fences from an LLM JSON response.
        """

        text = response_text.strip()

        if text.startswith("```json"):
            text = text[7:]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    # ========================================================
    # LLM INVOCATION
    # ========================================================

    def _invoke_llm(self, prompt: str) -> str:
        """
        Send a prompt to Gemini and return plain text.
        """

        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Some LangChain versions return a list of
        # content blocks instead of a simple string.
        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        parts.append(
                            str(item.get("text", ""))
                        )

                else:
                    parts.append(str(item))

            content = "".join(parts)

        return str(content).strip()

    # ========================================================
    # LEARNER CONTEXT
    # ========================================================

    @staticmethod
    def build_learner_context(
        learner_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Convert learner profile information into a compact
        textual representation for the Teacher Agent.
        """

        if not learner_profile:
            return """
Learner information is not available.

Use a beginner-friendly teaching strategy:
- clear explanation
- simple examples
- minimal jargon
- frequent checks for understanding
"""

        level = learner_profile.get(
            "level",
            "Beginner"
        )

        prior_knowledge = learner_profile.get(
            "prior_knowledge",
            "Basic"
        )

        goal = learner_profile.get(
            "goal",
            "Concept Understanding"
        )

        language = learner_profile.get(
            "language",
            "English"
        )

        teaching_style = learner_profile.get(
            "teaching_style",
            "Example-Based"
        )

        available_time = learner_profile.get(
            "time",
            20
        )

        depth = learner_profile.get(
            "depth",
            "Standard"
        )

        return f"""
Learner Level: {level}
Prior Knowledge: {prior_knowledge}
Learning Goal: {goal}
Preferred Language: {language}
Teaching Style: {teaching_style}
Available Time: {available_time} minutes
Desired Depth: {depth}
"""

    # ========================================================
    # INITIAL TEACHING DECISION
    # ========================================================

    def plan_teaching_action(
        self,
        topic: str,
        concept: str,
        learner_profile: Optional[Dict[str, Any]] = None,
        previous_context: str = "",
        mastery: float = 0.0,
        remaining_time: int = 20,
    ) -> Dict[str, Any]:
        """
        Decide what the teacher should do next.

        This is primarily used before teaching a concept.
        """

        learner_context = self.build_learner_context(
            learner_profile
        )

        prompt = f"""
You are the pedagogical decision engine of an AI Teacher.

Your job is NOT simply to generate a lesson.

Your job is to decide the best NEXT TEACHING ACTION
for the learner.

You must behave like an intelligent human teacher.

Possible actions:

{TEACHER_ACTIONS}

Possible difficulties:

{DIFFICULTY_LEVELS}

Possible question types:

{QUESTION_TYPES}

--------------------------------------------------
TOPIC
--------------------------------------------------

{topic}

--------------------------------------------------
CURRENT CONCEPT
--------------------------------------------------

{concept}

--------------------------------------------------
LEARNER
--------------------------------------------------

{learner_context}

--------------------------------------------------
CURRENT MASTERY
--------------------------------------------------

{mastery}

--------------------------------------------------
REMAINING TIME
--------------------------------------------------

{remaining_time} minutes

--------------------------------------------------
PREVIOUS CONTEXT
--------------------------------------------------

{previous_context}

--------------------------------------------------
PEDAGOGICAL RULES
--------------------------------------------------

1. If the concept has not been introduced:
   use INTRODUCE or EXPLAIN.

2. If explanation alone is insufficient:
   use DEMONSTRATE.

3. After meaningful teaching:
   use ASK.

4. If mastery is high:
   use INCREASE_DIFFICULTY or CONTINUE.

5. If mastery is moderate:
   use another example, DEMONSTRATE, or ASK.

6. If mastery is weak:
   use SIMPLIFY or REEXPLAIN.

7. If a misconception is detected:
   use REMEDIATE.

8. Do not repeatedly explain the same thing in exactly
   the same way.

9. Adapt the strategy to the learner's teaching style.

10. Respect the learner's remaining time.

11. The goal is genuine understanding, not merely finishing
    the lesson.

Return ONLY valid JSON.

Required format:

{{
    "action": "EXPLAIN",
    "concept": "{concept}",
    "difficulty": "EASY",
    "strategy": "STEP_BY_STEP",
    "question_type": "MCQ",
    "reason": "Brief pedagogical reason",
    "next_step": "What the teacher should do after this action"
}}
"""

        try:

            response = self._invoke_llm(prompt)

            cleaned = self._clean_json_response(
                response
            )

            result = json.loads(cleaned)

            return self._validate_decision(
                result,
                concept
            )

        except Exception as error:

            print(
                f"Teacher Agent planning error: {error}"
            )

            return self._fallback_initial_decision(
                concept=concept,
                mastery=mastery,
            )

    # ========================================================
    # RESPONSE ADAPTATION
    # ========================================================

    def adapt_after_response(
        self,
        topic: str,
        concept: str,
        question: str,
        student_answer: str,
        assessment_result: Dict[str, Any],
        learner_profile: Optional[Dict[str, Any]] = None,
        previous_explanation: str = "",
        remaining_time: int = 15,
    ) -> Dict[str, Any]:
        """
        Decide how the teacher should react after a student
        answers a question.

        This is the core adaptive teaching loop.
        """

        learner_context = self.build_learner_context(
            learner_profile
        )

        correctness = assessment_result.get(
            "correctness",
            "unknown"
        )

        mastery = float(
            assessment_result.get(
                "mastery",
                0.0
            )
        )

        misconception = assessment_result.get(
            "misconception",
            "UNKNOWN"
        )

        confidence = assessment_result.get(
            "confidence",
            0.0
        )

        prompt = f"""
You are an adaptive AI Teacher.

A student has just answered a question.

You must decide what the teacher should do NEXT.

--------------------------------------------------
TOPIC
--------------------------------------------------

{topic}

--------------------------------------------------
CONCEPT
--------------------------------------------------

{concept}

--------------------------------------------------
QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
STUDENT ANSWER
--------------------------------------------------

{student_answer}

--------------------------------------------------
ASSESSMENT
--------------------------------------------------

Correctness:
{correctness}

Mastery:
{mastery}

Misconception:
{misconception}

Assessment Confidence:
{confidence}

--------------------------------------------------
LEARNER PROFILE
--------------------------------------------------

{learner_context}

--------------------------------------------------
PREVIOUS EXPLANATION
--------------------------------------------------

{previous_explanation}

--------------------------------------------------
REMAINING TIME
--------------------------------------------------

{remaining_time} minutes

--------------------------------------------------
ADAPTIVE TEACHING RULES
--------------------------------------------------

If the student is clearly correct and mastery is strong:
    prefer CONTINUE or INCREASE_DIFFICULTY.

If the student is correct but mastery is uncertain:
    prefer another ASK or DEMONSTRATE.

If the student is partially correct:
    prefer EXPLAIN or DEMONSTRATE.

If the student has a conceptual misunderstanding:
    prefer REMEDIATE or REEXPLAIN.

If the student makes a calculation mistake but understands
the concept:
    explain the calculation mistake instead of restarting
    the entire concept.

If terminology is confused:
    clarify the terminology with a simple comparison.

If the student appears to have guessed:
    ask a conceptual follow-up question.

If the student is repeatedly struggling:
    SIMPLIFY the explanation and use a new analogy.

Do NOT use the same explanation strategy repeatedly.

The goal is to diagnose WHY the student struggled and
choose the most effective next teaching action.

Return ONLY valid JSON.

Required format:

{{
    "action": "REMEDIATE",
    "concept": "{concept}",
    "difficulty": "EASY",
    "strategy": "ANALOGY",
    "question_type": "CONCEPTUAL",
    "reason": "Why this action is appropriate",
    "misconception": "{misconception}",
    "next_step": "What the teacher should do next"
}}
"""

        try:

            response = self._invoke_llm(prompt)

            cleaned = self._clean_json_response(
                response
            )

            result = json.loads(cleaned)

            return self._validate_decision(
                result,
                concept
            )

        except Exception as error:

            print(
                f"Teacher Agent adaptation error: {error}"
            )

            return self._fallback_adaptive_decision(
                correctness=correctness,
                mastery=mastery,
                misconception=misconception,
                concept=concept,
            )

    # ========================================================
    # MASTERY-BASED DECISION
    # ========================================================

    def decide_from_mastery(
        self,
        mastery: float,
        misconception: Optional[str] = None,
        attempts: int = 0,
    ) -> str:
        """
        Deterministic safety layer for mastery decisions.

        The LLM should not be the only component determining
        whether a learner has mastered a concept.
        """

        if misconception and misconception != "UNKNOWN":

            if attempts < 2:
                return "REMEDIATE"

            return "REEXPLAIN"

        if mastery >= STRONG_MASTERY_THRESHOLD:
            return "INCREASE_DIFFICULTY"

        if mastery >= INITIAL_MASTERY_THRESHOLD:
            return "CONTINUE"

        if mastery >= WEAK_MASTERY_THRESHOLD:
            return "DEMONSTRATE"

        return "SIMPLIFY"

    # ========================================================
    # DECISION VALIDATION
    # ========================================================

    @staticmethod
    def _validate_decision(
        decision: Dict[str, Any],
        concept: str,
    ) -> Dict[str, Any]:
        """
        Validate and normalize the decision returned by Gemini.
        """

        if not isinstance(decision, dict):
            raise ValueError(
                "Teacher Agent response is not a JSON object."
            )

        action = str(
            decision.get(
                "action",
                "EXPLAIN"
            )
        ).upper()

        if action not in TEACHER_ACTIONS:
            action = "EXPLAIN"

        difficulty = str(
            decision.get(
                "difficulty",
                "EASY"
            )
        ).upper()

        if difficulty not in DIFFICULTY_LEVELS:
            difficulty = "EASY"

        question_type = str(
            decision.get(
                "question_type",
                "MCQ"
            )
        ).upper()

        if question_type not in QUESTION_TYPES:
            question_type = "MCQ"

        strategy = str(
            decision.get(
                "strategy",
                "STEP_BY_STEP"
            )
        )

        misconception = decision.get(
            "misconception",
            None
        )

        if misconception:

            misconception = str(
                misconception
            ).upper()

            if misconception not in MISCONCEPTION_TYPES:
                misconception = "UNKNOWN"

        return {
            "action": action,
            "concept": decision.get(
                "concept",
                concept
            ),
            "difficulty": difficulty,
            "strategy": strategy,
            "question_type": question_type,
            "reason": str(
                decision.get(
                    "reason",
                    ""
                )
            ),
            "misconception": misconception,
            "next_step": str(
                decision.get(
                    "next_step",
                    ""
                )
            ),
        }

    # ========================================================
    # FALLBACK INITIAL DECISION
    # ========================================================

    @staticmethod
    def _fallback_initial_decision(
        concept: str,
        mastery: float,
    ) -> Dict[str, Any]:
        """
        Safe fallback if Gemini fails.
        """

        if mastery >= STRONG_MASTERY_THRESHOLD:

            action = "INCREASE_DIFFICULTY"
            difficulty = "HARD"

        elif mastery >= INITIAL_MASTERY_THRESHOLD:

            action = "CONTINUE"
            difficulty = "MEDIUM"

        elif mastery >= WEAK_MASTERY_THRESHOLD:

            action = "DEMONSTRATE"
            difficulty = "EASY"

        else:

            action = "EXPLAIN"
            difficulty = "EASY"

        return {
            "action": action,
            "concept": concept,
            "difficulty": difficulty,
            "strategy": "STEP_BY_STEP",
            "question_type": "MCQ",
            "reason": (
                "Fallback pedagogical decision based "
                "on current mastery."
            ),
            "misconception": None,
            "next_step": (
                "Check learner understanding "
                "before continuing."
            ),
        }

    # ========================================================
    # FALLBACK ADAPTIVE DECISION
    # ========================================================

    @staticmethod
    def _fallback_adaptive_decision(
        correctness: str,
        mastery: float,
        misconception: str,
        concept: str,
    ) -> Dict[str, Any]:
        """
        Safe fallback after an assessment.
        """

        if (
            misconception
            and misconception != "UNKNOWN"
        ):

            action = "REMEDIATE"
            strategy = "ANALOGY"
            difficulty = "EASY"

        elif correctness == "correct":

            if mastery >= STRONG_MASTERY_THRESHOLD:

                action = "INCREASE_DIFFICULTY"
                strategy = "CHALLENGE"
                difficulty = "HARD"

            else:

                action = "CONTINUE"
                strategy = "EXAMPLE"
                difficulty = "MEDIUM"

        elif correctness in (
            "partial",
            "partially_correct",
        ):

            action = "DEMONSTRATE"
            strategy = "STEP_BY_STEP"
            difficulty = "EASY"

        else:

            action = "REEXPLAIN"
            strategy = "ANALOGY"
            difficulty = "EASY"

        return {
            "action": action,
            "concept": concept,
            "difficulty": difficulty,
            "strategy": strategy,
            "question_type": "CONCEPTUAL",
            "reason": (
                "Fallback adaptive decision based "
                "on the assessment result."
            ),
            "misconception": (
                misconception
                if misconception != "UNKNOWN"
                else None
            ),
            "next_step": (
                "Reassess the learner after "
                "the selected intervention."
            ),
        }

    # ========================================================
    # TEACHING LOOP STATE
    # ========================================================

    @staticmethod
    def create_teaching_state(
        topic: str,
        concept: str,
        learner_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a state object for an individual concept.

        This state can later be stored in learning_memory.py.
        """

        return {
            "topic": topic,
            "concept": concept,
            "current_action": "INTRODUCE",
            "mastery": 0.0,
            "attempts": 0,
            "remediation_attempts": 0,
            "questions_asked": 0,
            "correct_answers": 0,
            "misconceptions": [],
            "history": [],
            "learner_profile": (
                learner_profile
                if learner_profile
                else {}
            ),
        }

    # ========================================================
    # RECORD ACTION
    # ========================================================

    @staticmethod
    def record_action(
        state: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Add the Teacher Agent's decision to the teaching
        history.
        """

        action = decision.get(
            "action",
            "EXPLAIN"
        )

        state["current_action"] = action

        state["history"].append({
            "action": action,
            "concept": decision.get(
                "concept"
            ),
            "difficulty": decision.get(
                "difficulty"
            ),
            "strategy": decision.get(
                "strategy"
            ),
            "question_type": decision.get(
                "question_type"
            ),
            "reason": decision.get(
                "reason"
            ),
        })

        return state

    # ========================================================
    # UPDATE STATE AFTER ASSESSMENT
    # ========================================================

    @staticmethod
    def update_state_after_assessment(
        state: Dict[str, Any],
        assessment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update the current teaching state using assessment
        information.
        """

        mastery = float(
            assessment_result.get(
                "mastery",
                state.get("mastery", 0.0)
            )
        )

        state["mastery"] = max(
            0.0,
            min(1.0, mastery)
        )

        state["attempts"] += 1

        correctness = assessment_result.get(
            "correctness",
            "unknown"
        )

        if correctness == "correct":
            state["correct_answers"] += 1

        misconception = assessment_result.get(
            "misconception"
        )

        if misconception:

            if misconception not in state["misconceptions"]:
                state["misconceptions"].append(
                    misconception
                )

        if (
            misconception
            and misconception != "UNKNOWN"
        ):
            state["remediation_attempts"] += 1

        return state


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PedagogyEngine AI - Teacher Agent Test")
    print("=" * 70)

    print("\nTeacher Agent module loaded successfully.")

    print("\nAvailable actions:")

    for action in TEACHER_ACTIONS:
        print(f"  - {action}")

    print("\nAvailable difficulty levels:")

    for difficulty in DIFFICULTY_LEVELS:
        print(f"  - {difficulty}")

    print("\nAvailable question types:")

    for question_type in QUESTION_TYPES:
        print(f"  - {question_type}")

    # --------------------------------------------------------
    # Create a test object WITHOUT initializing Gemini.
    #
    # The tests below only test local logic and do not make
    # an API call.
    # --------------------------------------------------------

    test_agent = TeacherAgent.__new__(TeacherAgent)

    # --------------------------------------------------------
    # MASTERY DECISION TEST
    # --------------------------------------------------------

    print("\nMastery decision test:")

    test_cases = [
        {
            "mastery": 0.25,
            "misconception": None,
            "attempts": 0,
        },
        {
            "mastery": 0.45,
            "misconception": None,
            "attempts": 1,
        },
        {
            "mastery": 0.72,
            "misconception": None,
            "attempts": 1,
        },
        {
            "mastery": 0.91,
            "misconception": None,
            "attempts": 1,
        },
        {
            "mastery": 0.30,
            "misconception": "CONCEPTUAL_GAP",
            "attempts": 0,
        },
    ]

    for case in test_cases:

        decision = test_agent.decide_from_mastery(
            mastery=case["mastery"],
            misconception=case["misconception"],
            attempts=case["attempts"],
        )

        print(
            f"  Mastery={case['mastery']:.2f}, "
            f"Misconception={case['misconception']} "
            f"→ {decision}"
        )

    # --------------------------------------------------------
    # TEACHING STATE TEST
    # --------------------------------------------------------

    print("\nState test:")

    state = test_agent.create_teaching_state(
        topic="Ohm's Law",
        concept="Resistance",
        learner_profile={
            "level": "Beginner",
            "prior_knowledge": "Basic",
            "goal": "Concept Understanding",
            "language": "Hinglish",
            "teaching_style": "Visual",
            "time": 20,
            "depth": "Standard",
        },
    )

    print(
        json.dumps(
            state,
            indent=2
        )
    )

    # --------------------------------------------------------
    # STATE UPDATE TEST
    # --------------------------------------------------------

    print("\nState update test:")

    assessment_result = {
        "correctness": "incorrect",
        "mastery": 0.35,
        "misconception": "CONCEPTUAL_GAP",
        "confidence": 0.85,
    }

    updated_state = test_agent.update_state_after_assessment(
        state,
        assessment_result
    )

    print(
        json.dumps(
            updated_state,
            indent=2
        )
    )

    # --------------------------------------------------------
    # ACTION RECORD TEST
    # --------------------------------------------------------

    print("\nAction recording test:")

    sample_decision = {
        "action": "REMEDIATE",
        "concept": "Resistance",
        "difficulty": "EASY",
        "strategy": "ANALOGY",
        "question_type": "CONCEPTUAL",
        "reason": (
            "The learner appears to have "
            "a conceptual gap."
        ),
        "misconception": "CONCEPTUAL_GAP",
        "next_step": "Ask a re-test question.",
    }

    updated_state = test_agent.record_action(
        updated_state,
        sample_decision
    )

    print(
        json.dumps(
            updated_state,
            indent=2
        )
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\nTeacher Agent basic tests passed.")
    print("=" * 70)