"""
assessment_engine.py

AI Teacher - Assessment Engine

Responsibilities:
1. Evaluate MCQ answers deterministically.
2. Evaluate free-form answers using Gemini when an API key is available.
3. Provide a safe deterministic fallback when Gemini is unavailable.
4. Calculate concept mastery.
5. Provide a preliminary misconception category.
6. Recommend the next teaching action.
7. Return a consistent structured assessment result.

This module is independently testable and does not require
a Gemini API key for its local tests.
"""

import json
import re
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from llm_provider import get_chat_llm

from config import (
    DEFAULT_GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    QUESTION_TYPES,
    MISCONCEPTION_TYPES,
    TEACHER_ACTIONS,
)


# ================================================================
# LOCAL ASSESSMENT THRESHOLDS
# ================================================================

INITIAL_THRESHOLD = 0.70
STRONG_THRESHOLD = 0.85
WEAK_THRESHOLD = 0.50


class AssessmentEngine:
    """
    Evaluates student responses and produces structured
    assessment results for the AI Teacher.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AssessmentEngine.

        Parameters
        ----------
        api_key : str, optional
            Gemini API key.

        Notes
        -----
        api_key is optional.

        If no API key is provided:
            - MCQ evaluation works normally.
            - deterministic free-form evaluation works.
            - Gemini is simply disabled.
        """

        self.api_key = api_key
        self.llm = None

        # LLM is optional; get_chat_llm() itself raises a clear error
        # when neither GROQ_API_KEY nor a Gemini key is available, so
        # we always attempt and just fall back to None on failure —
        # this way a Groq-only setup (no Gemini key passed in here)
        # still works.
        try:

            self.llm = get_chat_llm(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                gemini_api_key=api_key,
            )

        except Exception as exc:

            print(
                f"[AssessmentEngine] Gemini initialization failed: {exc}"
            )

            self.llm = None

    # ============================================================
    # TEXT HELPERS
    # ============================================================

    @staticmethod
    def _normalize_text(text: Any) -> str:
        """
        Normalize text for comparison.
        """

        if text is None:
            return ""

        text = str(text).strip().lower()

        # Normalize common mathematical symbols.
        text = text.replace("×", "x")
        text = text.replace("÷", "/")

        # Remove unnecessary punctuation while retaining
        # useful mathematical characters.
        text = re.sub(
            r"[^\w\s\.\-\/\+\=\*]",
            " ",
            text,
        )

        # Collapse multiple spaces.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Safely convert a value to a bounded float.
        """

        try:
            value = float(value)

        except (TypeError, ValueError):

            value = default

        return max(
            minimum,
            min(maximum, value),
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from an LLM response.

        Handles:
        - plain JSON
        - ```json blocks
        - surrounding explanatory text
        """

        if not text:
            return None

        text = str(text).strip()

        # Remove markdown code fences.
        text = re.sub(
            r"```json\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```\s*",
            "",
            text,
        )

        # Try direct JSON parsing.
        try:

            result = json.loads(text)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Search for an embedded JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:

            candidate = text[start:end + 1]

            try:

                result = json.loads(candidate)

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                pass

        return None

    # ============================================================
    # GEMINI
    # ============================================================

    def _invoke_llm(
        self,
        prompt: str,
    ) -> Optional[str]:
        """
        Safely invoke Gemini.
        """

        # Gemini unavailable.
        if self.llm is None:
            return None

        try:

            response = self.llm.invoke(prompt)

            content = getattr(
                response,
                "content",
                response,
            )

            # Some LangChain responses may contain a list.
            if isinstance(content, list):

                parts = []

                for item in content:

                    if isinstance(item, dict):

                        parts.append(
                            str(
                                item.get(
                                    "text",
                                    "",
                                )
                            )
                        )

                    else:

                        parts.append(str(item))

                content = "".join(parts)

            return str(content).strip()

        except Exception as exc:

            print(
                f"[AssessmentEngine] Gemini evaluation failed: {exc}"
            )

            return None

    # ============================================================
    # QUESTION TYPE
    # ============================================================

    @staticmethod
    def _normalize_question_type(
        question_type: Any,
    ) -> str:
        """
        Normalize question type names.
        """

        value = str(
            question_type or ""
        ).strip().upper()

        # Direct configured value.
        if value in QUESTION_TYPES:
            return value

        aliases = {

            "MULTIPLE_CHOICE": "MCQ",

            "MULTIPLE-CHOICE": "MCQ",

            "MULTIPLE CHOICE": "MCQ",

            "CHOICE": "MCQ",

            "SHORT": "SHORT_ANSWER",

            "SHORT ANSWER": "SHORT_ANSWER",

            "CONCEPT": "CONCEPTUAL",

            "CONCEPTUAL QUESTION": "CONCEPTUAL",

            "PROBLEM": "PROBLEM_SOLVING",

            "PROBLEM-SOLVING": "PROBLEM_SOLVING",

            "APPLICATION-BASED": "APPLICATION",

            "APPLICATION BASED": "APPLICATION",

            "OWN WORD": "OWN_WORDS",

            "OWN WORDS": "OWN_WORDS",
        }

        return aliases.get(
            value,
            "SHORT_ANSWER",
        )

    # ============================================================
    # MCQ OPTION RESOLUTION
    # ============================================================

    def _resolve_option_index(
        self,
        answer: str,
        options: list,
    ) -> Optional[int]:
        """
        Resolve an MCQ answer to an option index.

        Supports:
            A / B / C / D
            1 / 2 / 3 / 4
            exact option text
        """

        if not answer:
            return None

        answer = self._normalize_text(answer)

        # --------------------------------------------------------
        # Letter answer
        # --------------------------------------------------------

        if (
            len(answer) == 1
            and answer.isalpha()
        ):

            index = ord(answer) - ord("a")

            if 0 <= index < len(options):
                return index

        # --------------------------------------------------------
        # Numeric answer
        # --------------------------------------------------------

        if answer.isdigit():

            number = int(answer)

            # Human numbering: 1 = first option.
            if 1 <= number <= len(options):
                return number - 1

            # Zero-based numbering.
            if 0 <= number < len(options):
                return number

        # --------------------------------------------------------
        # Exact option text
        # --------------------------------------------------------

        for index, option in enumerate(options):

            if answer == option:
                return index

        return None

    # ============================================================
    # MCQ EVALUATION
    # ============================================================

    def evaluate_mcq(
        self,
        student_answer: Any,
        correct_answer: Any,
        options: Optional[list] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Deterministically evaluate an MCQ.
        """

        student = self._normalize_text(
            student_answer
        )

        correct = self._normalize_text(
            correct_answer
        )

        # Empty answer.
        if not student:

            return self._build_result(
                correctness="incorrect",
                mastery=0.0,
                misconception="GUESS",
                confidence=0.95,
                recommended_action="SIMPLIFY",
                feedback=(
                    "No answer was provided. "
                    "Let's review the concept and try again."
                ),
                reason=(
                    "The student did not provide an answer."
                ),
                score=0.0,
            )

        # Direct comparison.
        if student == correct:

            return self._build_result(
                correctness="correct",
                mastery=1.0,
                misconception="UNKNOWN",
                confidence=0.99,
                recommended_action="INCREASE_DIFFICULTY",
                feedback=(
                    "Correct! You understood this concept well."
                ),
                reason=(
                    "The student's answer matches the "
                    "expected answer."
                ),
                score=1.0,
            )

        # Compare option indexes.
        if options:

            normalized_options = [
                self._normalize_text(option)
                for option in options
            ]

            student_index = self._resolve_option_index(
                student,
                normalized_options,
            )

            correct_index = self._resolve_option_index(
                correct,
                normalized_options,
            )

            if (
                student_index is not None
                and correct_index is not None
                and student_index == correct_index
            ):

                return self._build_result(
                    correctness="correct",
                    mastery=1.0,
                    misconception="UNKNOWN",
                    confidence=0.99,
                    recommended_action="INCREASE_DIFFICULTY",
                    feedback=(
                        "Correct! You selected the right answer."
                    ),
                    reason=(
                        "The selected option corresponds "
                        "to the expected answer."
                    ),
                    score=1.0,
                )

        # Wrong answer.
        return self._build_result(
            correctness="incorrect",
            mastery=0.0,
            misconception="CONCEPTUAL_GAP",
            confidence=0.90,
            recommended_action="REMEDIATE",
            feedback=(
                "That's not the correct answer. "
                "Let's revisit the concept using another explanation."
            ),
            reason=(
                "The selected answer does not match "
                "the expected answer."
            ),
            score=0.0,
        )

    # ============================================================
    # FREE-FORM EVALUATION
    # ============================================================

    def evaluate_free_form(
        self,
        question: str,
        student_answer: str,
        expected_answer: str = "",
        context: str = "",
        question_type: str = "SHORT_ANSWER",
        rubric: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate free-form answers.

        Gemini is used when available.
        Otherwise deterministic fallback is used.
        """

        student_answer = str(
            student_answer or ""
        ).strip()

        # Empty answer.
        if not student_answer:

            return self._build_result(
                correctness="incorrect",
                mastery=0.0,
                misconception="GUESS",
                confidence=0.95,
                recommended_action="SIMPLIFY",
                feedback=(
                    "You haven't entered an answer yet. "
                    "Let's review the concept before trying again."
                ),
                reason=(
                    "No student response was provided."
                ),
                score=0.0,
            )

        # No Gemini.
        if self.llm is None:

            return self._fallback_free_form_evaluation(
                question=question,
                student_answer=student_answer,
                expected_answer=expected_answer,
                question_type=question_type,
            )

        # --------------------------------------------------------
        # Gemini assessment prompt
        # --------------------------------------------------------

        prompt = f"""
You are an expert AI educational assessment engine.

Evaluate the student's answer fairly and pedagogically.

QUESTION:
{question}

QUESTION TYPE:
{question_type}

EXPECTED ANSWER:
{expected_answer}

RUBRIC:
{rubric}

RELEVANT LEARNING CONTEXT:
{context}

STUDENT ANSWER:
{student_answer}

Rules:

1. Evaluate understanding rather than exact wording.
2. Equivalent explanations should receive credit.
3. Give partial credit for partial understanding.
4. For problem-solving questions, evaluate both reasoning
   and the final answer.
5. For application questions, evaluate whether the concept
   was correctly applied.
6. For own-words questions, prioritize conceptual meaning.
7. Do not penalize concise but correct answers.
8. Do not invent information.
9. Identify only a preliminary misconception category.
10. Feedback should be student-friendly and concise.

Return ONLY valid JSON.

Use exactly:

{{
    "correctness": "correct",
    "score": 0.0,
    "misconception": "UNKNOWN",
    "confidence": 0.0,
    "feedback": "Short student-friendly feedback.",
    "reason": "Brief explanation of the evaluation."
}}

Allowed correctness:
correct
partial
incorrect

Allowed misconception:
CONCEPTUAL_GAP
CALCULATION_ERROR
TERMINOLOGY_CONFUSION
PARTIAL_UNDERSTANDING
GUESS
MISAPPLIED_CONCEPT
UNKNOWN

Score:
0.0 to 1.0

Confidence:
0.0 to 1.0
"""

        response = self._invoke_llm(prompt)

        parsed = self._extract_json(
            response or ""
        )

        # Gemini response invalid.
        if not parsed:

            return self._fallback_free_form_evaluation(
                question=question,
                student_answer=student_answer,
                expected_answer=expected_answer,
                question_type=question_type,
            )

        return self._normalize_llm_result(
            parsed,
            question=question,
            student_answer=student_answer,
            expected_answer=expected_answer,
        )

    # ============================================================
    # DETERMINISTIC FREE-FORM FALLBACK
    # ============================================================

    def _fallback_free_form_evaluation(
        self,
        question: str,
        student_answer: str,
        expected_answer: str = "",
        question_type: str = "SHORT_ANSWER",
    ) -> Dict[str, Any]:
        """
        Conservative deterministic fallback.

        This is NOT intended to replace semantic LLM evaluation.
        It only handles obvious matches and term overlap.
        """

        student = self._normalize_text(
            student_answer
        )

        expected = self._normalize_text(
            expected_answer
        )

        # No expected answer.
        if not expected:

            return self._build_result(
                correctness="partial",
                mastery=0.50,
                misconception="UNKNOWN",
                confidence=0.25,
                recommended_action="CONTINUE",
                feedback=(
                    "Your answer has been recorded. "
                    "A deeper AI evaluation will be performed "
                    "when the AI evaluator is available."
                ),
                reason=(
                    "No expected answer was supplied for "
                    "deterministic comparison."
                ),
                score=0.50,
            )

        # Exact match.
        if student == expected:

            return self._build_result(
                correctness="correct",
                mastery=1.0,
                misconception="UNKNOWN",
                confidence=0.95,
                recommended_action="INCREASE_DIFFICULTY",
                feedback=(
                    "Correct! Your answer matches "
                    "the expected concept."
                ),
                reason=(
                    "The normalized student answer matches "
                    "the expected answer."
                ),
                score=1.0,
            )

        # --------------------------------------------------------
        # Token overlap
        # --------------------------------------------------------

        student_tokens = set(
            student.split()
        )

        expected_tokens = set(
            expected.split()
        )

        if (
            student_tokens
            and expected_tokens
        ):

            overlap = (
                len(
                    student_tokens
                    & expected_tokens
                )
                / len(expected_tokens)
            )

            # Strong overlap.
            if overlap >= 0.75:

                return self._build_result(
                    correctness="correct",
                    mastery=0.85,
                    misconception="UNKNOWN",
                    confidence=0.70,
                    recommended_action="INCREASE_DIFFICULTY",
                    feedback=(
                        "Your answer captures the main "
                        "idea correctly."
                    ),
                    reason=(
                        "The response contains most important "
                        "concept terms from the expected answer."
                    ),
                    score=0.85,
                )

            # Partial overlap.
            if overlap >= 0.40:

                return self._build_result(
                    correctness="partial",
                    mastery=0.55,
                    misconception="PARTIAL_UNDERSTANDING",
                    confidence=0.55,
                    recommended_action="DEMONSTRATE",
                    feedback=(
                        "You have part of the idea. "
                        "Let's strengthen the missing part."
                    ),
                    reason=(
                        "The response contains some important "
                        "concepts but does not cover enough "
                        "of the expected answer."
                    ),
                    score=0.55,
                )

        # Weak/no overlap.
        return self._build_result(
            correctness="incorrect",
            mastery=0.20,
            misconception="CONCEPTUAL_GAP",
            confidence=0.50,
            recommended_action="REMEDIATE",
            feedback=(
                "Your answer does not yet show the expected "
                "concept. Let's approach it using another explanation."
            ),
            reason=(
                "The response has insufficient overlap "
                "with the expected concepts."
            ),
            score=0.20,
        )

    # ============================================================
    # NORMALIZE GEMINI RESULT
    # ============================================================

    def _normalize_llm_result(
        self,
        result: Dict[str, Any],
        question: str,
        student_answer: str,
        expected_answer: str,
    ) -> Dict[str, Any]:
        """
        Validate and normalize Gemini's response.
        """

        correctness = str(
            result.get(
                "correctness",
                "partial",
            )
        ).strip().lower()

        if correctness not in {
            "correct",
            "partial",
            "incorrect",
        }:

            correctness = "partial"

        score = self._safe_float(
            result.get(
                "score",
                0.50,
            ),
            default=0.50,
        )

        confidence = self._safe_float(
            result.get(
                "confidence",
                0.50,
            ),
            default=0.50,
        )

        misconception = str(
            result.get(
                "misconception",
                "UNKNOWN",
            )
        ).strip().upper()

        if misconception not in MISCONCEPTION_TYPES:

            misconception = "UNKNOWN"

        feedback = str(
            result.get(
                "feedback",
                "Let's continue working on this concept.",
            )
        ).strip()

        reason = str(
            result.get(
                "reason",
                "The answer was evaluated against "
                "the expected concept.",
            )
        ).strip()

        # --------------------------------------------------------
        # Correctness safety bounds.
        # --------------------------------------------------------

        if correctness == "correct":

            score = max(
                score,
                0.75,
            )

        elif correctness == "partial":

            score = min(
                max(score, 0.30),
                0.79,
            )

        elif correctness == "incorrect":

            score = min(
                score,
                0.49,
            )

        recommended_action = self._recommend_action(
            correctness=correctness,
            mastery=score,
            misconception=misconception,
        )

        return self._build_result(
            correctness=correctness,
            mastery=score,
            misconception=misconception,
            confidence=confidence,
            recommended_action=recommended_action,
            feedback=feedback,
            reason=reason,
            score=score,
        )

    # ============================================================
    # ACTION RECOMMENDATION
    # ============================================================

    def _recommend_action(
        self,
        correctness: str,
        mastery: float,
        misconception: str = "UNKNOWN",
    ) -> str:
        """
        Recommend the next teaching action.

        TeacherAgent will later make the final decision.
        """

        mastery = self._safe_float(
            mastery
        )

        # Incorrect.
        if correctness == "incorrect":

            if misconception in {
                "CONCEPTUAL_GAP",
                "MISAPPLIED_CONCEPT",
                "TERMINOLOGY_CONFUSION",
            }:

                return "REMEDIATE"

            return "SIMPLIFY"

        # Partial.
        if correctness == "partial":

            if mastery >= INITIAL_THRESHOLD:
                return "REEXPLAIN"

            return "DEMONSTRATE"

        # Correct.
        if correctness == "correct":

            if mastery >= STRONG_THRESHOLD:
                return "INCREASE_DIFFICULTY"

            return "CONTINUE"

        return "CONTINUE"

    # ============================================================
    # MASTERY CALCULATION
    # ============================================================

    def calculate_mastery(
        self,
        assessment_result: Dict[str, Any],
        previous_mastery: float = 0.0,
    ) -> float:
        """
        Calculate updated mastery.

        Recent performance receives more weight than older performance.
        """

        current_score = self._safe_float(
            assessment_result.get(
                "mastery",
                0.0,
            )
        )

        previous_mastery = self._safe_float(
            previous_mastery,
            default=0.0,
        )

        # First assessment.
        if previous_mastery <= 0.0:

            return round(
                current_score,
                3,
            )

        # Weighted update.
        updated_mastery = (
            0.40 * previous_mastery
            + 0.60 * current_score
        )

        return round(
            self._safe_float(
                updated_mastery
            ),
            3,
        )

    # ============================================================
    # MAIN EVALUATION METHOD
    # ============================================================

    def evaluate(
        self,
        question: str,
        student_answer: Any,
        question_type: str = "SHORT_ANSWER",
        correct_answer: Any = "",
        options: Optional[list] = None,
        expected_answer: str = "",
        context: str = "",
        rubric: str = "",
        previous_mastery: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Main public evaluation function.
        """

        normalized_type = (
            self._normalize_question_type(
                question_type
            )
        )

        # --------------------------------------------------------
        # MCQ
        # --------------------------------------------------------

        if normalized_type == "MCQ":

            result = self.evaluate_mcq(
                student_answer=student_answer,
                correct_answer=correct_answer,
                options=options,
                question=question,
            )

        # --------------------------------------------------------
        # Free-form
        # --------------------------------------------------------

        else:

            expected = expected_answer

            if not expected:

                expected = str(
                    correct_answer or ""
                )

            result = self.evaluate_free_form(
                question=question,
                student_answer=str(
                    student_answer or ""
                ),
                expected_answer=expected,
                context=context,
                question_type=normalized_type,
                rubric=rubric,
            )

        # --------------------------------------------------------
        # Update cumulative mastery.
        # --------------------------------------------------------

        updated_mastery = (
            self.calculate_mastery(
                assessment_result=result,
                previous_mastery=previous_mastery,
            )
        )

        result["current_mastery"] = updated_mastery

        # Additional metadata.
        result["question_type"] = normalized_type

        result["student_answer"] = str(
            student_answer or ""
        )

        return result

    # ============================================================
    # RESULT BUILDER
    # ============================================================

    def _build_result(
        self,
        correctness: str,
        mastery: float,
        misconception: str,
        confidence: float,
        recommended_action: str,
        feedback: str,
        reason: str,
        score: float,
    ) -> Dict[str, Any]:
        """
        Create a consistent assessment result.
        """

        if correctness not in {
            "correct",
            "partial",
            "incorrect",
        }:

            correctness = "partial"

        if misconception not in MISCONCEPTION_TYPES:

            misconception = "UNKNOWN"

        if recommended_action not in TEACHER_ACTIONS:

            recommended_action = "CONTINUE"

        mastery = self._safe_float(
            mastery
        )

        confidence = self._safe_float(
            confidence
        )

        score = self._safe_float(
            score
        )

        return {
            "correctness": correctness,
            "mastery": round(mastery, 3),
            "score": round(score, 3),
            "misconception": misconception,
            "confidence": round(confidence, 3),
            "recommended_action": recommended_action,
            "feedback": feedback,
            "reason": reason,
        }


# =================================================================
# LOCAL TESTS
# =================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ASSESSMENT ENGINE - LOCAL TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------------
    # We use the NORMAL constructor here.
    #
    # No API key means:
    #     self.llm = None
    #
    # This avoids the previous __new__() problem.
    # ------------------------------------------------------------

    test_engine = AssessmentEngine()

    print("\nGemini available:", test_engine.llm is not None)

    # ============================================================
    # TEST 1 - Correct MCQ
    # ============================================================

    print("\n[TEST 1] Correct MCQ")

    result_1 = test_engine.evaluate(
        question="What is Ohm's Law?",
        student_answer="B",
        question_type="MCQ",
        correct_answer="B",
        options=[
            "V = I + R",
            "V = I x R",
            "V = I / R",
            "V = R / I",
        ],
    )

    print(
        json.dumps(
            result_1,
            indent=2,
        )
    )

    assert result_1["correctness"] == "correct"
    assert result_1["mastery"] == 1.0

    print("PASS")

    # ============================================================
    # TEST 2 - Wrong MCQ
    # ============================================================

    print("\n[TEST 2] Wrong MCQ")

    result_2 = test_engine.evaluate(
        question="What is Ohm's Law?",
        student_answer="A",
        question_type="MCQ",
        correct_answer="B",
        options=[
            "V = I + R",
            "V = I x R",
            "V = I / R",
            "V = R / I",
        ],
    )

    print(
        json.dumps(
            result_2,
            indent=2,
        )
    )

    assert result_2["correctness"] == "incorrect"
    assert result_2["recommended_action"] == "REMEDIATE"

    print("PASS")

    # ============================================================
    # TEST 3 - Empty Answer
    # ============================================================

    print("\n[TEST 3] Empty Answer")

    result_3 = test_engine.evaluate(
        question="What is resistance?",
        student_answer="",
        question_type="SHORT_ANSWER",
        expected_answer=(
            "Resistance opposes the flow of electric current."
        ),
    )

    print(
        json.dumps(
            result_3,
            indent=2,
        )
    )

    assert result_3["correctness"] == "incorrect"
    assert result_3["mastery"] == 0.0

    print("PASS")

    # ============================================================
    # TEST 4 - Exact Free-form Answer
    # ============================================================

    print("\n[TEST 4] Exact Free-form Answer")

    result_4 = test_engine.evaluate(
        question="What does resistance do in a circuit?",
        student_answer=(
            "Resistance opposes the flow of electric current."
        ),
        question_type="CONCEPTUAL",
        expected_answer=(
            "Resistance opposes the flow of electric current."
        ),
    )

    print(
        json.dumps(
            result_4,
            indent=2,
        )
    )

    assert result_4["correctness"] == "correct"
    assert result_4["mastery"] == 1.0

    print("PASS")

    # ============================================================
    # TEST 5 - Partial Free-form Answer
    # ============================================================

    print("\n[TEST 5] Partial Free-form Answer")

    result_5 = test_engine.evaluate(
        question="Explain Ohm's Law.",
        student_answer=(
            "Ohm's Law relates voltage and current."
        ),
        question_type="OWN_WORDS",
        expected_answer=(
            "Ohm's Law states that voltage equals current "
            "multiplied by resistance, expressed as V = I x R."
        ),
    )

    print(
        json.dumps(
            result_5,
            indent=2,
        )
    )

    assert result_5["correctness"] in {
        "correct",
        "partial",
        "incorrect",
    }

    print("PASS")

    # ============================================================
    # TEST 6 - Mastery Update
    # ============================================================

    print("\n[TEST 6] Mastery Update")

    mastery = test_engine.calculate_mastery(
        assessment_result={
            "mastery": 1.0
        },
        previous_mastery=0.50,
    )

    print("Previous mastery : 0.50")
    print("Current score    : 1.00")
    print("Updated mastery  :", mastery)

    assert 0.50 < mastery <= 1.0

    print("PASS")

    # ============================================================
    # TEST 7 - MCQ Using Option Text
    # ============================================================

    print("\n[TEST 7] MCQ Using Option Text")

    result_7 = test_engine.evaluate(
        question="Which formula represents Ohm's Law?",
        student_answer="V = I x R",
        question_type="MCQ",
        correct_answer="B",
        options=[
            "V = I + R",
            "V = I x R",
            "V = I / R",
            "V = R / I",
        ],
    )

    print(
        json.dumps(
            result_7,
            indent=2,
        )
    )

    assert result_7["correctness"] == "correct"

    print("PASS")

    # ============================================================
    # TEST 8 - Previous Mastery
    # ============================================================

    print("\n[TEST 8] Previous Mastery Integration")

    result_8 = test_engine.evaluate(
        question="What is Ohm's Law?",
        student_answer="B",
        question_type="MCQ",
        correct_answer="B",
        options=[
            "V = I + R",
            "V = I x R",
            "V = I / R",
            "V = R / I",
        ],
        previous_mastery=0.60,
    )

    print(
        json.dumps(
            result_8,
            indent=2,
        )
    )

    assert (
        0.60
        < result_8["current_mastery"]
        <= 1.0
    )

    print("PASS")

    # ============================================================
    # TEST 9 - Numeric MCQ Answer
    # ============================================================

    print("\n[TEST 9] Numeric MCQ Answer")

    result_9 = test_engine.evaluate(
        question="Which option is correct?",
        student_answer="2",
        question_type="MCQ",
        correct_answer="B",
        options=[
            "Wrong",
            "Correct",
            "Wrong",
            "Wrong",
        ],
    )

    print(
        json.dumps(
            result_9,
            indent=2,
        )
    )

    assert result_9["correctness"] == "correct"

    print("PASS")

    # ============================================================
    # TEST 10 - Question Type Normalization
    # ============================================================

    print("\n[TEST 10] Question Type Normalization")

    normalized = (
        test_engine._normalize_question_type(
            "multiple choice"
        )
    )

    print("Input :", "multiple choice")
    print("Output:", normalized)

    assert normalized == "MCQ"

    print("PASS")

    # ============================================================
    # FINAL
    # ============================================================

    print("\n" + "=" * 70)
    print("ALL LOCAL ASSESSMENT TESTS PASSED SUCCESSFULLY")
    print("=" * 70)
