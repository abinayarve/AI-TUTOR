"""
misconception_engine.py

Step 4 of the AI Teacher system.

Responsibilities:
1. Diagnose the likely reason behind an incorrect/partial answer.
2. Classify the misconception.
3. Generate targeted remediation.
4. Provide a new analogy and example.
5. Generate a re-test question.
6. Recommend difficulty adjustment.
7. Work without Gemini for local testing.
8. Use Gemini when an API key is available.

This file is intentionally independent from app.py.
"""

import json
import re
from typing import Any, Dict, Optional

from config import (
    DEFAULT_GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    MISCONCEPTION_TYPES,
    TEACHER_ACTIONS,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from llm_provider import get_chat_llm


class MisconceptionEngine:
    """
    Detects and remediates learner misconceptions.

    An LLM (Groq preferred, Gemini as fallback) is optional:
    - Without any API key -> deterministic fallback logic is used.
    - With a key -> the LLM provides deeper semantic diagnosis.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.llm = None

        try:
            self.llm = get_chat_llm(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                gemini_api_key=api_key,
            )
        except Exception:
            self.llm = None

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize text for safe comparison."""
        if value is None:
            return ""

        text = str(value).strip().lower()

        text = re.sub(r"\s+", " ", text)

        return text

    @staticmethod
    def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
        """Safely clamp a numeric value."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = minimum

        return max(minimum, min(maximum, number))

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Clean Gemini output so JSON can be parsed even when
        Gemini surrounds it with markdown fences.
        """
        if not text:
            return ""

        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        return text.strip()

    def _extract_json(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from an LLM response."""
        if response is None:
            return None

        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                else:
                    parts.append(str(item))

            content = "".join(parts)

        content = self._clean_json_response(str(content))

        try:
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to locate the first JSON object.
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)

        if match:
            try:
                parsed = json.loads(match.group(0))

                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None

        return None

    def _invoke_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Invoke Gemini safely."""
        if self.llm is None:
            return None

        try:
            response = self.llm.invoke(prompt)
            return self._extract_json(response)
        except Exception:
            return None

    # ============================================================
    # MISCONCEPTION NORMALIZATION
    # ============================================================

    def _normalize_misconception(
        self,
        misconception: Any,
    ) -> str:
        """
        Normalize a misconception into one of the configured values.
        """
        if not misconception:
            return "UNKNOWN"

        value = str(misconception).strip().upper()

        aliases = {
            "CONCEPTUAL": "CONCEPTUAL_GAP",
            "CONCEPTUAL_ERROR": "CONCEPTUAL_GAP",
            "CONCEPT_GAP": "CONCEPTUAL_GAP",
            "CALCULATION": "CALCULATION_ERROR",
            "CALCULATION_MISTAKE": "CALCULATION_ERROR",
            "TERM_CONFUSION": "TERMINOLOGY_CONFUSION",
            "TERMINOLOGY": "TERMINOLOGY_CONFUSION",
            "PARTIAL": "PARTIAL_UNDERSTANDING",
            "GUESSING": "GUESS",
            "MISAPPLICATION": "MISAPPLIED_CONCEPT",
            "MISAPPLIED": "MISAPPLIED_CONCEPT",
        }

        value = aliases.get(value, value)

        if value in MISCONCEPTION_TYPES:
            return value

        return "UNKNOWN"

    # ============================================================
    # DIFFICULTY NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_difficulty(value: Any) -> str:
        """Normalize difficulty."""
        if not value:
            return "MEDIUM"

        value = str(value).strip().upper()

        aliases = {
            "EASY": "EASY",
            "BEGINNER": "EASY",
            "SIMPLE": "EASY",
            "MEDIUM": "MEDIUM",
            "MODERATE": "MEDIUM",
            "HARD": "HARD",
            "DIFFICULT": "HARD",
            "ADVANCED": "HARD",
        }

        return aliases.get(value, "MEDIUM")

    # ============================================================
    # DETERMINISTIC MISCONCEPTION DETECTION
    # ============================================================

    def detect_misconception(
        self,
        assessment_result: Dict[str, Any],
        question: Optional[Dict[str, Any]] = None,
        topic: str = "",
        concept: str = "",
        student_answer: str = "",
    ) -> Dict[str, Any]:
        """
        Determine the likely misconception.

        Gemini is preferred when available.

        If Gemini is unavailable, deterministic rules are used.
        """

        assessment_result = assessment_result or {}
        question = question or {}

        correctness = str(
            assessment_result.get("correctness", "incorrect")
        ).strip().lower()

        existing = self._normalize_misconception(
            assessment_result.get("misconception")
        )

        # Correct answers normally do not require misconception remediation.
        if correctness == "correct":
            return {
                "misconception": "UNKNOWN",
                "confidence": 0.95,
                "reason": "The learner answered correctly, so no misconception was detected.",
            }

        # If assessment already identified a specific misconception,
        # preserve it unless it is UNKNOWN.
        if existing != "UNKNOWN":
            return self._fallback_diagnosis(
                assessment_result=assessment_result,
                question=question,
                topic=topic,
                concept=concept,
                student_answer=student_answer,
                preferred_misconception=existing,
            )

        # Try Gemini for semantic diagnosis.
        llm_result = self._llm_diagnosis(
            assessment_result=assessment_result,
            question=question,
            topic=topic,
            concept=concept,
            student_answer=student_answer,
        )

        if llm_result:
            return llm_result

        # Deterministic fallback.
        return self._fallback_diagnosis(
            assessment_result=assessment_result,
            question=question,
            topic=topic,
            concept=concept,
            student_answer=student_answer,
        )

    # ============================================================
    # GEMINI DIAGNOSIS
    # ============================================================

    def _llm_diagnosis(
        self,
        assessment_result: Dict[str, Any],
        question: Dict[str, Any],
        topic: str,
        concept: str,
        student_answer: str,
    ) -> Optional[Dict[str, Any]]:
        """Use Gemini to semantically diagnose the misconception."""

        prompt = f"""
You are an expert AI tutor diagnosing a student's misunderstanding.

TOPIC:
{topic}

CONCEPT:
{concept}

QUESTION:
{json.dumps(question, ensure_ascii=False)}

STUDENT ANSWER:
{student_answer}

ASSESSMENT RESULT:
{json.dumps(assessment_result, ensure_ascii=False)}

Classify the MOST LIKELY misconception.

Allowed misconception values:
{json.dumps(MISCONCEPTION_TYPES)}

Possible categories:
- CONCEPTUAL_GAP: student does not understand the underlying idea.
- CALCULATION_ERROR: concept seems understood but arithmetic/calculation is wrong.
- TERMINOLOGY_CONFUSION: student confuses terms, definitions, symbols, or vocabulary.
- PARTIAL_UNDERSTANDING: student understands part of the idea but misses an important component.
- GUESS: answer appears random, unsupported, or shows no meaningful reasoning.
- MISAPPLIED_CONCEPT: student knows a concept but applies it in the wrong situation.
- UNKNOWN: insufficient evidence.

Return ONLY valid JSON:

{{
    "misconception": "CONCEPTUAL_GAP",
    "confidence": 0.0,
    "reason": "Brief explanation of why this misconception is likely."
}}

Do not include markdown.
"""

        result = self._invoke_llm(prompt)

        if not result:
            return None

        misconception = self._normalize_misconception(
            result.get("misconception")
        )

        confidence = self._clamp(result.get("confidence", 0.70))

        reason = str(
            result.get(
                "reason",
                "The student's response suggests a misunderstanding of the concept.",
            )
        ).strip()

        return {
            "misconception": misconception,
            "confidence": confidence,
            "reason": reason,
        }

    # ============================================================
    # FALLBACK DIAGNOSIS
    # ============================================================

    def _fallback_diagnosis(
        self,
        assessment_result: Dict[str, Any],
        question: Dict[str, Any],
        topic: str,
        concept: str,
        student_answer: str,
        preferred_misconception: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """Deterministic misconception diagnosis."""

        answer = self._normalize_text(student_answer)

        if not answer:
            return {
                "misconception": "GUESS",
                "confidence": 0.95,
                "reason": "No meaningful answer was provided, so the response cannot demonstrate conceptual understanding.",
            }

        if preferred_misconception != "UNKNOWN":
            return {
                "misconception": preferred_misconception,
                "confidence": 0.85,
                "reason": self._default_reason(preferred_misconception),
            }

        score = self._clamp(
            assessment_result.get(
                "score",
                assessment_result.get("mastery", 0.0),
            )
        )

        question_type = str(
            assessment_result.get(
                "question_type",
                question.get("type", ""),
            )
        ).upper()

        # Very low score + no useful response -> likely guess.
        if score <= 0.10 and len(answer.split()) <= 2:
            return {
                "misconception": "GUESS",
                "confidence": 0.80,
                "reason": "The response is very short and does not provide enough evidence of conceptual reasoning.",
            }

        # Numerical-looking questions.
        if self._looks_like_calculation_question(question, concept):
            if self._contains_number(answer):
                return {
                    "misconception": "CALCULATION_ERROR",
                    "confidence": 0.68,
                    "reason": "The response contains a numerical result but does not match the expected result, suggesting a possible calculation or substitution error.",
                }

        # Terminology questions.
        if question_type in {
            "CONCEPTUAL",
            "SHORT_ANSWER",
            "OWN_WORDS",
        }:
            if self._looks_like_term_confusion(
                answer,
                question,
                concept,
            ):
                return {
                    "misconception": "TERMINOLOGY_CONFUSION",
                    "confidence": 0.62,
                    "reason": "The response appears to use related terminology incorrectly or interchangeably.",
                }

        # Partial score.
        if 0.30 <= score < 0.70:
            return {
                "misconception": "PARTIAL_UNDERSTANDING",
                "confidence": 0.72,
                "reason": "The response shows some understanding but misses an important part of the concept.",
            }

        # Default.
        return {
            "misconception": "CONCEPTUAL_GAP",
            "confidence": 0.70,
            "reason": "The response does not demonstrate the expected understanding of the underlying concept.",
        }

    @staticmethod
    def _default_reason(misconception: str) -> str:
        reasons = {
            "CONCEPTUAL_GAP": (
                "The learner appears to be missing the underlying concept."
            ),
            "CALCULATION_ERROR": (
                "The learner appears to understand the idea but may have made a calculation error."
            ),
            "TERMINOLOGY_CONFUSION": (
                "The learner appears to be confusing related terms or definitions."
            ),
            "PARTIAL_UNDERSTANDING": (
                "The learner understands part of the concept but is missing an important component."
            ),
            "GUESS": (
                "The answer does not provide reliable evidence of understanding."
            ),
            "MISAPPLIED_CONCEPT": (
                "The learner appears to know the concept but applied it to the wrong situation."
            ),
            "UNKNOWN": (
                "There is not enough evidence to determine the exact misconception."
            ),
        }

        return reasons.get(misconception, reasons["UNKNOWN"])

    # ============================================================
    # FALLBACK HELPER DETECTION
    # ============================================================

    @staticmethod
    def _contains_number(text: str) -> bool:
        return bool(re.search(r"\d", text))

    @staticmethod
    def _looks_like_calculation_question(
        question: Dict[str, Any],
        concept: str,
    ) -> bool:
        combined = " ".join(
            [
                str(question.get("question", "")),
                str(question.get("prompt", "")),
                str(question.get("text", "")),
                str(concept),
            ]
        ).lower()

        calculation_words = [
            "calculate",
            "calculation",
            "solve",
            "find the value",
            "compute",
            "how much",
            "what is the value",
            "equation",
            "formula",
            "numerical",
        ]

        return any(word in combined for word in calculation_words)

    @staticmethod
    def _looks_like_term_confusion(
        answer: str,
        question: Dict[str, Any],
        concept: str,
    ) -> bool:
        combined = " ".join(
            [
                str(question.get("question", "")),
                str(question.get("prompt", "")),
                str(concept),
            ]
        ).lower()

        terminology_words = [
            "define",
            "definition",
            "meaning",
            "term",
            "difference between",
            "distinguish",
            "what is",
        ]

        return any(word in combined for word in terminology_words) and len(
            answer.split()
        ) <= 8

    # ============================================================
    # REMEDIATION
    # ============================================================

    def generate_remediation(
        self,
        topic: str,
        concept: str,
        question: Dict[str, Any],
        student_answer: str,
        assessment_result: Dict[str, Any],
        misconception_result: Dict[str, Any],
        learner_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate targeted remediation.

        Gemini is used when available.
        Otherwise a safe deterministic remediation is returned.
        """

        learner_profile = learner_profile or {}

        misconception = self._normalize_misconception(
            misconception_result.get("misconception")
        )

        # Gemini remediation.
        llm_result = self._llm_remediation(
            topic=topic,
            concept=concept,
            question=question,
            student_answer=student_answer,
            assessment_result=assessment_result,
            misconception_result=misconception_result,
            learner_profile=learner_profile,
        )

        if llm_result:
            return self._normalize_remediation(
                llm_result,
                misconception,
            )

        # Deterministic fallback.
        return self._fallback_remediation(
            topic=topic,
            concept=concept,
            question=question,
            student_answer=student_answer,
            misconception=misconception,
        )

    # ============================================================
    # GEMINI REMEDIATION
    # ============================================================

    def _llm_remediation(
        self,
        topic: str,
        concept: str,
        question: Dict[str, Any],
        student_answer: str,
        assessment_result: Dict[str, Any],
        misconception_result: Dict[str, Any],
        learner_profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate targeted remediation using Gemini."""

        prompt = f"""
You are a patient expert AI teacher.

The learner has misunderstood a concept.

TOPIC:
{topic}

CONCEPT:
{concept}

QUESTION:
{json.dumps(question, ensure_ascii=False)}

STUDENT ANSWER:
{student_answer}

ASSESSMENT:
{json.dumps(assessment_result, ensure_ascii=False)}

MISCONCEPTION:
{json.dumps(misconception_result, ensure_ascii=False)}

LEARNER PROFILE:
{json.dumps(learner_profile, ensure_ascii=False)}

Create a remediation lesson that does NOT simply repeat the previous explanation.

The remediation must:
1. Explain why the student's thinking may have gone wrong.
2. Explain the concept again in simpler/different words.
3. Give a fresh analogy.
4. Give a fresh concrete example.
5. Ask exactly ONE new re-test question.
6. Recommend whether the next question should be EASY, MEDIUM, or HARD.
7. Suggest a teacher action.

Allowed teacher actions:
{json.dumps(TEACHER_ACTIONS)}

Return ONLY valid JSON:

{{
    "misconception": "{misconception_result.get('misconception', 'UNKNOWN')}",
    "why_wrong": "...",
    "re_explanation": "...",
    "analogy": "...",
    "example": "...",
    "retest_question": "...",
    "retest_type": "CONCEPTUAL",
    "retest_expected_answer": "...",
    "difficulty": "EASY",
    "teacher_action": "REMEDIATE",
    "encouragement": "..."
}}

Do not include markdown.
"""

        result = self._invoke_llm(prompt)

        return result

    # ============================================================
    # REMEDIATION NORMALIZATION
    # ============================================================

    def _normalize_remediation(
        self,
        result: Dict[str, Any],
        misconception: str,
    ) -> Dict[str, Any]:
        """Normalize Gemini remediation output."""

        teacher_action = str(
            result.get("teacher_action", "REMEDIATE")
        ).upper()

        if teacher_action not in TEACHER_ACTIONS:
            teacher_action = "REMEDIATE"

        retest_type = str(
            result.get("retest_type", "CONCEPTUAL")
        ).upper()

        normalized = {
            "misconception": misconception,
            "why_wrong": str(
                result.get(
                    "why_wrong",
                    "Your answer suggests that this concept needs another explanation.",
                )
            ).strip(),
            "re_explanation": str(
                result.get(
                    "re_explanation",
                    "Let's explain the concept using a different approach.",
                )
            ).strip(),
            "analogy": str(
                result.get(
                    "analogy",
                    "Think of the concept using a familiar real-world situation.",
                )
            ).strip(),
            "example": str(
                result.get(
                    "example",
                    "Let's work through a simple example together.",
                )
            ).strip(),
            "retest_question": str(
                result.get(
                    "retest_question",
                    "Can you explain the concept in your own words?",
                )
            ).strip(),
            "retest_type": retest_type,
            "retest_expected_answer": str(
                result.get(
                    "retest_expected_answer",
                    "",
                )
            ).strip(),
            "difficulty": self._normalize_difficulty(
                result.get("difficulty", "EASY")
            ),
            "teacher_action": teacher_action,
            "encouragement": str(
                result.get(
                    "encouragement",
                    "It's okay to make mistakes. Let's try it again.",
                )
            ).strip(),
        }

        return normalized

    # ============================================================
    # DETERMINISTIC REMEDIATION
    # ============================================================

    def _fallback_remediation(
        self,
        topic: str,
        concept: str,
        question: Dict[str, Any],
        student_answer: str,
        misconception: str,
    ) -> Dict[str, Any]:
        """Create useful remediation without Gemini."""

        topic_text = topic or "this topic"
        concept_text = concept or topic_text

        templates = {
            "CONCEPTUAL_GAP": {
                "why_wrong": (
                    f"Your answer suggests that the basic idea behind "
                    f"{concept_text} is not fully clear yet."
                ),
                "re_explanation": (
                    f"Let's start from the core idea. {concept_text} "
                    f"means understanding what happens, why it happens, "
                    f"and how the important parts are connected."
                ),
                "analogy": (
                    "Think of learning this concept like learning a route. "
                    "Before taking shortcuts, you first need to know where "
                    "the starting point, destination, and important turns are."
                ),
                "example": (
                    f"Let's use a simple example from {topic_text}. "
                    "First identify the known information, then identify "
                    "what the concept is asking you to explain."
                ),
            },
            "CALCULATION_ERROR": {
                "why_wrong": (
                    "Your response may contain the right idea, but the "
                    "calculation or substitution appears to need checking."
                ),
                "re_explanation": (
                    "When solving a numerical problem, first write the "
                    "formula, substitute the known values carefully, and "
                    "then calculate one step at a time."
                ),
                "analogy": (
                    "A calculation is like following a recipe: even if you "
                    "know the recipe, adding an ingredient incorrectly can "
                    "change the final result."
                ),
                "example": (
                    "Take a simple numerical example and solve it in three "
                    "steps: formula, substitution, and final calculation."
                ),
            },
            "TERMINOLOGY_CONFUSION": {
                "why_wrong": (
                    "Your answer suggests that two related terms or "
                    "definitions may be getting mixed up."
                ),
                "re_explanation": (
                    f"Let's separate the important terms in {concept_text}. "
                    "For each term, remember its definition, role, and one "
                    "simple example."
                ),
                "analogy": (
                    "Think of similar technical terms like two people with "
                    "similar names: they may look related, but they have "
                    "different roles."
                ),
                "example": (
                    "Write the two related terms side by side and describe "
                    "one key difference between them."
                ),
            },
            "PARTIAL_UNDERSTANDING": {
                "why_wrong": (
                    "You have part of the idea, but one or more important "
                    "pieces are still missing."
                ),
                "re_explanation": (
                    f"You're on the right track. Let's connect the missing "
                    f"part of {concept_text} to what you already understand."
                ),
                "analogy": (
                    "Imagine assembling a puzzle. You already have several "
                    "pieces, and we only need to place the missing pieces "
                    "to see the complete picture."
                ),
                "example": (
                    f"Let's take a small example from {topic_text} and "
                    "identify both the part you already know and the missing part."
                ),
            },
            "GUESS": {
                "why_wrong": (
                    "The answer does not give enough reasoning to show "
                    "whether the concept is understood."
                ),
                "re_explanation": (
                    f"Let's slow down and rebuild {concept_text} from the "
                    "simplest idea before trying another question."
                ),
                "analogy": (
                    "Instead of guessing the destination, think of this like "
                    "using a map: identify the starting point and direction first."
                ),
                "example": (
                    "Let's work through one very simple example together "
                    "before you answer independently."
                ),
            },
            "MISAPPLIED_CONCEPT": {
                "why_wrong": (
                    "You may know the concept, but it appears to have been "
                    "used in a situation where it does not apply."
                ),
                "re_explanation": (
                    f"Let's identify when {concept_text} should be used and "
                    "when a different idea is more appropriate."
                ),
                "analogy": (
                    "A tool can be useful but still be the wrong tool for "
                    "a particular job. Concepts work in the same way."
                ),
                "example": (
                    "Compare one situation where the concept applies with "
                    "one similar situation where it does not."
                ),
            },
            "UNKNOWN": {
                "why_wrong": (
                    "The response is incorrect, but there is not enough "
                    "evidence to identify the exact misunderstanding yet."
                ),
                "re_explanation": (
                    f"Let's revisit the core idea of {concept_text} using "
                    "a simpler explanation."
                ),
                "analogy": (
                    "Think of the concept through a familiar everyday "
                    "situation before returning to the technical definition."
                ),
                "example": (
                    f"Let's work through a basic example from {topic_text}."
                ),
            },
        }

        template = templates.get(
            misconception,
            templates["UNKNOWN"],
        )

        return {
            "misconception": misconception,
            "why_wrong": template["why_wrong"],
            "re_explanation": template["re_explanation"],
            "analogy": template["analogy"],
            "example": template["example"],
            "retest_question": (
                f"After this explanation, can you explain "
                f"{concept_text} in your own words and give one simple example?"
            ),
            "retest_type": "OWN_WORDS",
            "retest_expected_answer": (
                f"A correct explanation of {concept_text} with a relevant example."
            ),
            "difficulty": "EASY",
            "teacher_action": "REMEDIATE",
            "encouragement": (
                "Don't worry about the mistake. "
                "Let's try the idea from a different angle."
            ),
        }

    # ============================================================
    # COMPLETE PIPELINE
    # ============================================================

    def analyze_and_remediate(
        self,
        topic: str,
        concept: str,
        question: Dict[str, Any],
        student_answer: str,
        assessment_result: Dict[str, Any],
        learner_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete misconception pipeline.

        Returns:
            diagnosis + remediation + re-test + adaptation information.
        """

        diagnosis = self.detect_misconception(
            assessment_result=assessment_result,
            question=question,
            topic=topic,
            concept=concept,
            student_answer=student_answer,
        )

        # Correct answer: no remediation required.
        correctness = str(
            assessment_result.get("correctness", "")
        ).lower()

        if correctness == "correct":
            return {
                "needed": False,
                "diagnosis": diagnosis,
                "remediation": None,
                "next_action": "CONTINUE",
                "retest_required": False,
            }

        remediation = self.generate_remediation(
            topic=topic,
            concept=concept,
            question=question,
            student_answer=student_answer,
            assessment_result=assessment_result,
            misconception_result=diagnosis,
            learner_profile=learner_profile,
        )

        return {
            "needed": True,
            "diagnosis": diagnosis,
            "remediation": remediation,
            "next_action": remediation.get(
                "teacher_action",
                "REMEDIATE",
            ),
            "retest_required": True,
        }

    # ============================================================
    # RETEST RESULT ANALYSIS
    # ============================================================

    def evaluate_retest_outcome(
        self,
        original_misconception: str,
        retest_assessment: Dict[str, Any],
        attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        Decide what should happen after a remediation re-test.

        This supports the human-like teaching loop.
        """

        original_misconception = self._normalize_misconception(
            original_misconception
        )

        correctness = str(
            retest_assessment.get("correctness", "incorrect")
        ).lower()

        mastery = self._clamp(
            retest_assessment.get(
                "current_mastery",
                retest_assessment.get(
                    "mastery",
                    retest_assessment.get("score", 0.0),
                ),
            )
        )

        if correctness == "correct" or mastery >= 0.70:
            return {
                "resolved": True,
                "action": "CONTINUE",
                "message": (
                    "Great! The re-test shows that the misconception "
                    "has been addressed."
                ),
                "difficulty": (
                    "MEDIUM" if mastery < 0.85 else "HARD"
                ),
            }

        if attempts < 2:
            return {
                "resolved": False,
                "action": "REEXPLAIN",
                "message": (
                    "The misconception may still be present. "
                    "Let's explain it using another approach."
                ),
                "difficulty": "EASY",
            }

        return {
            "resolved": False,
            "action": "SIMPLIFY",
            "message": (
                "The learner needs a simpler foundation before "
                "attempting the concept again."
            ),
            "difficulty": "EASY",
        }


# ================================================================
# LOCAL TESTS
# ================================================================

def run_local_tests():
    """
    Run independent tests without Gemini or an API key.
    """

    print("=" * 70)
    print("MISCONCEPTION ENGINE - LOCAL TEST")
    print("=" * 70)

    engine = MisconceptionEngine()

    print()
    print("Gemini available:", engine.llm is not None)

    # ------------------------------------------------------------
    # TEST 1
    # ------------------------------------------------------------
    print()
    print("[TEST 1] Correct Answer -> No Misconception")

    assessment = {
        "correctness": "correct",
        "score": 1.0,
        "mastery": 1.0,
        "misconception": "UNKNOWN",
    }

    result = engine.detect_misconception(
        assessment_result=assessment,
        question={"type": "MCQ", "question": "What is Ohm's Law?"},
        topic="Electricity",
        concept="Ohm's Law",
        student_answer="V = IR",
    )

    print(json.dumps(result, indent=2))

    assert result["misconception"] == "UNKNOWN"
    print("PASS")

    # ------------------------------------------------------------
    # TEST 2
    # ------------------------------------------------------------
    print()
    print("[TEST 2] Conceptual Gap")

    assessment = {
        "correctness": "incorrect",
        "score": 0.0,
        "mastery": 0.0,
        "misconception": "CONCEPTUAL_GAP",
    }

    result = engine.detect_misconception(
        assessment_result=assessment,
        question={
            "type": "CONCEPTUAL",
            "question": "What does resistance do in a circuit?",
        },
        topic="Electricity",
        concept="Resistance",
        student_answer="Resistance creates current.",
    )

    print(json.dumps(result, indent=2))

    assert result["misconception"] == "CONCEPTUAL_GAP"
    print("PASS")

    # ------------------------------------------------------------
    # TEST 3
    # ------------------------------------------------------------
    print()
    print("[TEST 3] Calculation Error")

    assessment = {
        "correctness": "incorrect",
        "score": 0.2,
        "mastery": 0.2,
        "misconception": "UNKNOWN",
    }

    result = engine.detect_misconception(
        assessment_result=assessment,
        question={
            "type": "PROBLEM_SOLVING",
            "question": "Calculate the current using V = IR.",
        },
        topic="Electricity",
        concept="Ohm's Law",
        student_answer="12 A",
    )

    print(json.dumps(result, indent=2))

    assert result["misconception"] == "CALCULATION_ERROR"
    print("PASS")

    # ------------------------------------------------------------
    # TEST 4
    # ------------------------------------------------------------
    print()
    print("[TEST 4] Terminology Confusion")

    assessment = {
        "correctness": "incorrect",
        "score": 0.2,
        "mastery": 0.2,
        "misconception": "UNKNOWN",
    }

    result = engine.detect_misconception(
        assessment_result=assessment,
        question={
            "type": "SHORT_ANSWER",
            "question": "What is the definition of voltage?",
        },
        topic="Electricity",
        concept="Voltage",
        student_answer="Voltage is current.",
    )

    print(json.dumps(result, indent=2))

    assert result["misconception"] == "TERMINOLOGY_CONFUSION"
    print("PASS")

    # ------------------------------------------------------------
    # TEST 5
    # ------------------------------------------------------------
    print()
    print("[TEST 5] Partial Understanding")

    assessment = {
        "correctness": "partial",
        "score": 0.55,
        "mastery": 0.55,
        "misconception": "UNKNOWN",
    }

    result = engine.detect_misconception(
        assessment_result=assessment,
        question={
            "type": "OWN_WORDS",
            "question": "Explain Ohm's Law.",
        },
        topic="Electricity",
        concept="Ohm's Law",
        student_answer="It relates voltage and current.",
    )

    print(json.dumps(result, indent=2))

    assert result["misconception"] == "PARTIAL_UNDERSTANDING"
    print("PASS")

    # ------------------------------------------------------------
    # TEST 6
    # ------------------------------------------------------------
    print()
    print("[TEST 6] Empty Answer -> Guess")

    assessment = {
        "correctness": "incorrect",
        "score": 0.0,
        "mastery": 0.0,
        "misconception": "GUESS",
    }

    result = engine.detect_misconception(
        assessment_result=assessment,
        question={
            "type": "SHORT_ANSWER",
            "question": "Explain resistance.",
        },
        topic="Electricity",
        concept="Resistance",
        student_answer="",
    )

    print(json.dumps(result, indent=2))

    assert result["misconception"] == "GUESS"
    print("PASS")

    # ------------------------------------------------------------
    # TEST 7
    # ------------------------------------------------------------
    print()
    print("[TEST 7] Remediation Generation")

    assessment = {
        "correctness": "incorrect",
        "score": 0.0,
        "mastery": 0.0,
        "misconception": "CONCEPTUAL_GAP",
    }

    diagnosis = engine.detect_misconception(
        assessment_result=assessment,
        question={
            "type": "CONCEPTUAL",
            "question": "What does resistance do?",
        },
        topic="Electricity",
        concept="Resistance",
        student_answer="Resistance increases current.",
    )

    remediation = engine.generate_remediation(
        topic="Electricity",
        concept="Resistance",
        question={
            "type": "CONCEPTUAL",
            "question": "What does resistance do?",
        },
        student_answer="Resistance increases current.",
        assessment_result=assessment,
        misconception_result=diagnosis,
    )

    print(json.dumps(remediation, indent=2))

    assert remediation["misconception"] == "CONCEPTUAL_GAP"
    assert remediation["retest_required"] if "retest_required" in remediation else True
    assert len(remediation["retest_question"]) > 0
    assert remediation["teacher_action"] == "REMEDIATE"

    print("PASS")

    # ------------------------------------------------------------
    # TEST 8
    # ------------------------------------------------------------
    print()
    print("[TEST 8] Complete Analyze + Remediate Pipeline")

    result = engine.analyze_and_remediate(
        topic="Electricity",
        concept="Ohm's Law",
        question={
            "type": "CONCEPTUAL",
            "question": "Explain Ohm's Law.",
        },
        student_answer="Ohm's Law only relates voltage and resistance.",
        assessment_result={
            "correctness": "incorrect",
            "score": 0.2,
            "mastery": 0.2,
            "misconception": "CONCEPTUAL_GAP",
        },
        learner_profile={
            "level": "BEGINNER",
            "goal": "UNDERSTAND",
        },
    )

    print(json.dumps(result, indent=2))

    assert result["needed"] is True
    assert result["retest_required"] is True
    assert result["diagnosis"]["misconception"] == "CONCEPTUAL_GAP"
    assert result["remediation"] is not None

    print("PASS")

    # ------------------------------------------------------------
    # TEST 9
    # ------------------------------------------------------------
    print()
    print("[TEST 9] Successful Re-test")

    result = engine.evaluate_retest_outcome(
        original_misconception="CONCEPTUAL_GAP",
        retest_assessment={
            "correctness": "correct",
            "score": 1.0,
            "mastery": 1.0,
            "current_mastery": 0.85,
        },
        attempts=1,
    )

    print(json.dumps(result, indent=2))

    assert result["resolved"] is True
    assert result["action"] == "CONTINUE"

    print("PASS")

    # ------------------------------------------------------------
    # TEST 10
    # ------------------------------------------------------------
    print()
    print("[TEST 10] Failed Re-test -> Re-explain")

    result = engine.evaluate_retest_outcome(
        original_misconception="CONCEPTUAL_GAP",
        retest_assessment={
            "correctness": "incorrect",
            "score": 0.2,
            "mastery": 0.2,
            "current_mastery": 0.2,
        },
        attempts=1,
    )

    print(json.dumps(result, indent=2))

    assert result["resolved"] is False
    assert result["action"] == "REEXPLAIN"

    print("PASS")

    # ------------------------------------------------------------
    # TEST 11
    # ------------------------------------------------------------
    print()
    print("[TEST 11] Repeated Failure -> Simplify")

    result = engine.evaluate_retest_outcome(
        original_misconception="CONCEPTUAL_GAP",
        retest_assessment={
            "correctness": "incorrect",
            "score": 0.1,
            "mastery": 0.1,
            "current_mastery": 0.1,
        },
        attempts=2,
    )

    print(json.dumps(result, indent=2))

    assert result["resolved"] is False
    assert result["action"] == "SIMPLIFY"
    assert result["difficulty"] == "EASY"

    print("PASS")

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL LOCAL MISCONCEPTION TESTS PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_local_tests()
