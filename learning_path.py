"""
Learning Path Engine
====================

Builds adaptive learning paths from learner progress, mastery,
misconceptions, assessment history, and learning goals.

This module is intentionally independent and can be tested without
a Gemini API key.
"""

import os
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)
from database import DatabaseManager
from learner_profile import LearnerProfileManager


class LearningPathEngine:
    """
    Adaptive learning-path engine.

    Responsibilities:
    - Analyze topic mastery
    - Detect weak/developing/strong concepts
    - Recommend the next concept
    - Recommend the next topic
    - Build a learning path
    - Build a 7-day learning plan
    - Build revision plans
    - Build teacher context
    - Reset learning progress
    """

    STRONG_MASTERY = 0.85
    GOOD_MASTERY = 0.70
    WEAK_MASTERY = 0.50

    def __init__(
        self,
        database: Optional[DatabaseManager] = None,
        profile_manager: Optional[LearnerProfileManager] = None,
        learner_id: str = "default_learner",
        api_key: Optional[str] = None,
    ):
        self.database = database or DatabaseManager()
        self.profile_manager = profile_manager or LearnerProfileManager(
            database=self.database
        )

        self.learner_id = learner_id
        self.api_key = api_key

        self.llm = None

        # LLM is optional.
        # The learning-path engine is fully functional without it.
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=DEFAULT_GEMINI_MODEL,
                    google_api_key=api_key,
                    temperature=LLM_TEMPERATURE,
                    max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                )
            except Exception:
                self.llm = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely convert a value to int."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        """Clamp a numeric value to a range."""
        return max(minimum, min(maximum, value))

    def _get_concept_name(self, concept: Dict[str, Any]) -> str:
        """
        Extract concept name regardless of which compatible key
        the learner-profile/database layer returns.
        """
        possible_keys = [
            "concept_name",
            "concept",
            "name",
            "concept_title",
        ]

        for key in possible_keys:
            value = concept.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    def _get_concept_mastery(self, concept: Dict[str, Any]) -> float:
        """
        Extract mastery regardless of the field name used by the
        learner-profile/database layer.
        """
        possible_keys = [
            "mastery_score",
            "mastery",
            "mastery_percentage",
            "score",
        ]

        for key in possible_keys:
            if key in concept and concept.get(key) is not None:
                value = self._safe_float(concept.get(key))

                # Convert percentage values such as 75 -> 0.75.
                if value > 1.0:
                    value = value / 100.0

                return self._clamp(value)

        return 0.0

    def _get_misconception(self, concept: Dict[str, Any]) -> Optional[str]:
        """Extract an active misconception from a concept record."""
        possible_keys = [
            "active_misconception",
            "misconception",
            "misconception_text",
        ]

        for key in possible_keys:
            value = concept.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return None

    def _get_topic_mastery(self, topic: Dict[str, Any]) -> float:
        """Extract topic mastery from a topic-progress record."""
        possible_keys = [
            "mastery_score",
            "mastery",
            "score",
        ]

        for key in possible_keys:
            if key in topic and topic.get(key) is not None:
                value = self._safe_float(topic.get(key))

                if value > 1.0:
                    value = value / 100.0

                return self._clamp(value)

        return 0.0

    # ------------------------------------------------------------------
    # Topic analysis
    # ------------------------------------------------------------------

    def analyze_topic(self, topic: str) -> Dict[str, Any]:
        """
        Analyze learner progress for a specific topic.
        """

        topic = (topic or "").strip()

        if not topic:
            return {
                "topic": "",
                "topic_mastery": 0.0,
                "weak_concepts": [],
                "developing_concepts": [],
                "strong_concepts": [],
                "active_misconceptions": [],
                "total_concepts": 0,
                "status": "unknown",
            }

        topic_progress = self.profile_manager.get_topic_progress(topic)

        if not topic_progress:
            topic_mastery = 0.0
        else:
            topic_mastery = self._get_topic_mastery(topic_progress)

        all_concepts = self.profile_manager.get_all_concepts(topic)

        weak_concepts: List[Dict[str, Any]] = []
        developing_concepts: List[Dict[str, Any]] = []
        strong_concepts: List[Dict[str, Any]] = []
        active_misconceptions: List[Dict[str, Any]] = []

        for concept in all_concepts or []:
            concept_name = self._get_concept_name(concept)
            mastery = self._get_concept_mastery(concept)
            misconception = self._get_misconception(concept)

            if not concept_name:
                continue

            normalized = {
                "concept": concept_name,
                "mastery": round(mastery, 4),
                "attempts": self._safe_int(
                    concept.get("attempts", concept.get("attempt_count", 0))
                ),
                "correct_count": self._safe_int(
                    concept.get("correct_count", concept.get("correct", 0))
                ),
                "incorrect_count": self._safe_int(
                    concept.get("incorrect_count", concept.get("incorrect", 0))
                ),
                "misconception": misconception,
            }

            if mastery < self.WEAK_MASTERY:
                weak_concepts.append(normalized)

            elif mastery < self.GOOD_MASTERY:
                developing_concepts.append(normalized)

            else:
                strong_concepts.append(normalized)

            if misconception:
                active_misconceptions.append(normalized)

        weak_concepts.sort(key=lambda item: (item["mastery"], item["concept"]))
        developing_concepts.sort(
            key=lambda item: (item["mastery"], item["concept"])
        )
        strong_concepts.sort(
            key=lambda item: (-item["mastery"], item["concept"])
        )

        if active_misconceptions:
            status = "needs_remediation"
        elif weak_concepts:
            status = "weak"
        elif developing_concepts:
            status = "developing"
        elif strong_concepts:
            status = "strong"
        else:
            status = "new"

        return {
            "topic": topic,
            "topic_mastery": round(topic_mastery, 4),
            "weak_concepts": weak_concepts,
            "developing_concepts": developing_concepts,
            "strong_concepts": strong_concepts,
            "active_misconceptions": active_misconceptions,
            "total_concepts": (
                len(weak_concepts)
                + len(developing_concepts)
                + len(strong_concepts)
            ),
            "status": status,
        }

    # ------------------------------------------------------------------
    # Next concept recommendation
    # ------------------------------------------------------------------

    def recommend_next_concept(self, topic: str) -> Dict[str, Any]:
        """
        Recommend the next concept to teach for a topic.

        Priority:
        1. Active misconception
        2. Weak concept
        3. Developing concept
        4. New concept/topic
        5. Mastered topic recap
        """

        topic = (topic or "").strip()

        if not topic:
            return {
                "topic": "",
                "concept": None,
                "reason": "no_topic",
                "priority": "low",
                "recommended_action": "select_topic",
                "difficulty": "easy",
                "mastery": 0.0,
            }

        analysis = self.analyze_topic(topic)

        # --------------------------------------------------------------
        # 1. Active misconception
        # --------------------------------------------------------------
        if analysis["active_misconceptions"]:
            item = min(
                analysis["active_misconceptions"],
                key=lambda x: (x["mastery"], x["concept"]),
            )

            return {
                "topic": topic,
                "concept": item["concept"],
                "reason": "active_misconception",
                "priority": "critical",
                "recommended_action": "remediate",
                "difficulty": "easy",
                "mastery": item["mastery"],
                "misconception": item.get("misconception"),
            }

        # --------------------------------------------------------------
        # 2. Weak concept
        # --------------------------------------------------------------
        if analysis["weak_concepts"]:
            item = min(
                analysis["weak_concepts"],
                key=lambda x: (x["mastery"], x["concept"]),
            )

            return {
                "topic": topic,
                "concept": item["concept"],
                "reason": "weak_mastery",
                "priority": "high",
                "recommended_action": "reteach",
                "difficulty": "easy",
                "mastery": item["mastery"],
            }

        # --------------------------------------------------------------
        # 3. Developing concept
        # --------------------------------------------------------------
        if analysis["developing_concepts"]:
            item = min(
                analysis["developing_concepts"],
                key=lambda x: (x["mastery"], x["concept"]),
            )

            return {
                "topic": topic,
                "concept": item["concept"],
                "reason": "developing_mastery",
                "priority": "medium",
                "recommended_action": "practice",
                "difficulty": "medium",
                "mastery": item["mastery"],
            }

        # --------------------------------------------------------------
        # 4. No concepts yet
        # --------------------------------------------------------------
        if analysis["total_concepts"] == 0:
            return {
                "topic": topic,
                "concept": None,
                "reason": "new_topic",
                "priority": "high",
                "recommended_action": "introduce",
                "difficulty": "easy",
                "mastery": analysis["topic_mastery"],
            }

        # --------------------------------------------------------------
        # 5. Everything is strong
        # --------------------------------------------------------------
        return {
            "topic": topic,
            "concept": None,
            "reason": "topic_mastered",
            "priority": "low",
            "recommended_action": "advance",
            "difficulty": "hard",
            "mastery": analysis["topic_mastery"],
        }

    # ------------------------------------------------------------------
    # Next topic recommendation
    # ------------------------------------------------------------------

    def recommend_next_topic(
        self,
        candidate_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend which topic should be studied next.
        """

        candidate_topics = candidate_topics or []

        cleaned_topics = []
        for topic in candidate_topics:
            topic = str(topic).strip()
            if topic and topic not in cleaned_topics:
                cleaned_topics.append(topic)

        if not cleaned_topics:
            return {
                "topic": None,
                "reason": "no_topics_available",
                "priority": "low",
                "recommended_action": "select_topic",
                "mastery": 0.0,
            }

        topic_results = []

        for topic in cleaned_topics:
            analysis = self.analyze_topic(topic)

            topic_results.append(
                {
                    "topic": topic,
                    "mastery": analysis["topic_mastery"],
                    "status": analysis["status"],
                    "weak_count": len(analysis["weak_concepts"]),
                    "misconception_count": len(
                        analysis["active_misconceptions"]
                    ),
                }
            )

        # Misconceptions have highest priority.
        with_misconceptions = [
            item
            for item in topic_results
            if item["misconception_count"] > 0
        ]

        if with_misconceptions:
            selected = min(
                with_misconceptions,
                key=lambda item: (
                    item["mastery"],
                    item["topic"],
                ),
            )

            return {
                "topic": selected["topic"],
                "reason": "active_misconception",
                "priority": "critical",
                "recommended_action": "remediate",
                "mastery": selected["mastery"],
            }

        # Then weak topics.
        weak_topics = [
            item
            for item in topic_results
            if item["mastery"] < self.WEAK_MASTERY
        ]

        if weak_topics:
            selected = min(
                weak_topics,
                key=lambda item: (
                    item["mastery"],
                    item["topic"],
                ),
            )

            return {
                "topic": selected["topic"],
                "reason": "weak_topic",
                "priority": "high",
                "recommended_action": "reteach",
                "mastery": selected["mastery"],
            }

        # Then developing topics.
        developing_topics = [
            item
            for item in topic_results
            if self.WEAK_MASTERY <= item["mastery"] < self.GOOD_MASTERY
        ]

        if developing_topics:
            selected = min(
                developing_topics,
                key=lambda item: (
                    item["mastery"],
                    item["topic"],
                ),
            )

            return {
                "topic": selected["topic"],
                "reason": "developing_topic",
                "priority": "medium",
                "recommended_action": "practice",
                "mastery": selected["mastery"],
            }

        # Everything is strong.
        selected = min(
            topic_results,
            key=lambda item: (
                item["mastery"],
                item["topic"],
            ),
        )

        return {
            "topic": selected["topic"],
            "reason": "all_topics_strong",
            "priority": "low",
            "recommended_action": "advance",
            "mastery": selected["mastery"],
        }

    # ------------------------------------------------------------------
    # Build learning path
    # ------------------------------------------------------------------

    def build_learning_path(
        self,
        topic: str,
        max_steps: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Build a concept-level learning path.

        The method creates an adaptive sequence based on the learner's
        current mastery.
        """

        topic = (topic or "").strip()

        if not topic:
            return []

        max_steps = max(1, int(max_steps))

        analysis = self.analyze_topic(topic)

        path: List[Dict[str, Any]] = []

        # Active misconceptions first.
        for item in analysis["active_misconceptions"]:
            if len(path) >= max_steps:
                break

            path.append(
                {
                    "step": len(path) + 1,
                    "topic": topic,
                    "concept": item["concept"],
                    "action": "remediate",
                    "difficulty": "easy",
                    "mastery": item["mastery"],
                    "reason": "active_misconception",
                }
            )

        # Weak concepts next.
        for item in analysis["weak_concepts"]:
            if len(path) >= max_steps:
                break

            if any(
                existing["concept"] == item["concept"]
                for existing in path
            ):
                continue

            path.append(
                {
                    "step": len(path) + 1,
                    "topic": topic,
                    "concept": item["concept"],
                    "action": "reteach",
                    "difficulty": "easy",
                    "mastery": item["mastery"],
                    "reason": "weak_mastery",
                }
            )

        # Developing concepts.
        for item in analysis["developing_concepts"]:
            if len(path) >= max_steps:
                break

            if any(
                existing["concept"] == item["concept"]
                for existing in path
            ):
                continue

            path.append(
                {
                    "step": len(path) + 1,
                    "topic": topic,
                    "concept": item["concept"],
                    "action": "practice",
                    "difficulty": "medium",
                    "mastery": item["mastery"],
                    "reason": "developing_mastery",
                }
            )

        # Strong concepts can be used for advanced practice.
        for item in analysis["strong_concepts"]:
            if len(path) >= max_steps:
                break

            if any(
                existing["concept"] == item["concept"]
                for existing in path
            ):
                continue

            path.append(
                {
                    "step": len(path) + 1,
                    "topic": topic,
                    "concept": item["concept"],
                    "action": "challenge",
                    "difficulty": "hard",
                    "mastery": item["mastery"],
                    "reason": "strong_mastery",
                }
            )

        # Completely new topic.
        if not path:
            path.append(
                {
                    "step": 1,
                    "topic": topic,
                    "concept": None,
                    "action": "introduce",
                    "difficulty": "easy",
                    "mastery": analysis["topic_mastery"],
                    "reason": "new_topic",
                }
            )

        return path[:max_steps]

    # ------------------------------------------------------------------
    # Seven-day learning plan
    # ------------------------------------------------------------------

    def build_7_day_plan(
        self,
        topic: str,
    ) -> List[Dict[str, Any]]:
        """
        Build a seven-day adaptive learning/revision plan.
        """

        topic = (topic or "").strip()

        if not topic:
            return []

        analysis = self.analyze_topic(topic)

        weak = analysis["weak_concepts"]
        developing = analysis["developing_concepts"]
        strong = analysis["strong_concepts"]

        plan: List[Dict[str, Any]] = []

        for day in range(1, 8):
            if day == 1:
                action = "introduce"
                focus = (
                    weak[0]["concept"]
                    if weak
                    else (
                        developing[0]["concept"]
                        if developing
                        else None
                    )
                )

            elif day == 2:
                action = "explain"
                focus = (
                    weak[min(1, len(weak) - 1)]["concept"]
                    if weak
                    else None
                )

            elif day == 3:
                action = "demonstrate"
                focus = (
                    developing[0]["concept"]
                    if developing
                    else (
                        weak[0]["concept"]
                        if weak
                        else None
                    )
                )

            elif day == 4:
                action = "practice"
                focus = (
                    developing[min(1, len(developing) - 1)]["concept"]
                    if developing
                    else None
                )

            elif day == 5:
                action = "assess"
                focus = topic

            elif day == 6:
                action = "remediate"
                focus = (
                    weak[0]["concept"]
                    if weak
                    else (
                        developing[0]["concept"]
                        if developing
                        else topic
                    )
                )

            else:
                action = "recap"
                focus = topic

            plan.append(
                {
                    "day": day,
                    "topic": topic,
                    "focus": focus,
                    "action": action,
                    "estimated_minutes": 20,
                    "objective": (
                        f"{action.capitalize()} {focus}"
                        if focus
                        else f"{action.capitalize()} {topic}"
                    ),
                }
            )

        return plan

    # ------------------------------------------------------------------
    # Revision plan
    # ------------------------------------------------------------------

    def build_revision_plan(
        self,
        topic: str,
        days: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Build a focused revision plan.
        """

        topic = (topic or "").strip()

        if not topic:
            return []

        days = max(1, int(days))

        analysis = self.analyze_topic(topic)

        priority_concepts = []

        for item in analysis["active_misconceptions"]:
            priority_concepts.append(
                (
                    item["concept"],
                    "misconception",
                    item["mastery"],
                )
            )

        for item in analysis["weak_concepts"]:
            priority_concepts.append(
                (
                    item["concept"],
                    "weak",
                    item["mastery"],
                )
            )

        for item in analysis["developing_concepts"]:
            priority_concepts.append(
                (
                    item["concept"],
                    "developing",
                    item["mastery"],
                )
            )

        plan = []

        for day in range(1, days + 1):
            if priority_concepts:
                concept, reason, mastery = priority_concepts[
                    min(day - 1, len(priority_concepts) - 1)
                ]

                action = (
                    "remediate"
                    if reason == "misconception"
                    else (
                        "reteach"
                        if reason == "weak"
                        else "practice"
                    )
                )

            else:
                concept = topic
                reason = "general_revision"
                mastery = analysis["topic_mastery"]
                action = "recap"

            plan.append(
                {
                    "day": day,
                    "topic": topic,
                    "concept": concept,
                    "action": action,
                    "reason": reason,
                    "mastery": mastery,
                    "estimated_minutes": 20,
                }
            )

        return plan

    # ------------------------------------------------------------------
    # Teacher learning context
    # ------------------------------------------------------------------

    def build_teacher_learning_context(
        self,
        topic: str,
    ) -> Dict[str, Any]:
        """
        Build a compact context object for the Teacher Agent.
        """

        topic = (topic or "").strip()

        analysis = self.analyze_topic(topic)

        recommendation = self.recommend_next_concept(topic)

        try:
            profile = self.profile_manager.get_profile()
        except Exception:
            profile = {}

        return {
            "learner_id": self.learner_id,
            "profile": profile or {},
            "topic": topic,
            "topic_mastery": analysis["topic_mastery"],
            "topic_status": analysis["status"],
            "weak_concepts": analysis["weak_concepts"],
            "developing_concepts": analysis["developing_concepts"],
            "strong_concepts": analysis["strong_concepts"],
            "active_misconceptions": analysis["active_misconceptions"],
            "next_recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Teacher-context recommendation
    # ------------------------------------------------------------------

    def recommend_from_teacher_context(
        self,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce a recommendation using an already-built teacher context.
        """

        context = context or {}

        topic = str(context.get("topic", "")).strip()

        if not topic:
            return {
                "topic": None,
                "concept": None,
                "reason": "no_topic",
                "priority": "low",
                "recommended_action": "select_topic",
                "difficulty": "easy",
                "mastery": 0.0,
            }

        recommendation = context.get("next_recommendation")

        if isinstance(recommendation, dict):
            return recommendation

        return self.recommend_next_concept(topic)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> bool:
        """
        Reset learner learning-path progress.

        The underlying LearnerProfileManager.reset_progress() may return
        a value that is not literally True. For the LearningPathEngine
        API, a successfully completed reset is represented by True.

        Returns:
            True  -> reset completed successfully
            False -> reset failed
        """

        try:
            self.profile_manager.reset_progress()
            return True

        except Exception as exc:
            print(f"Learning path reset failed: {exc}")
            return False


# ======================================================================
# TESTS
# ======================================================================

def run_tests():
    print("=" * 60)
    print("LEARNING PATH ENGINE TESTS")
    print("=" * 60)

    test_db = "learning_path_test.db"
    fresh_db = "learning_path_fresh_test.db"

    # Remove previous test databases so tests always start clean.
    for db_file in [test_db, fresh_db]:
        try:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"Removed old test database: {db_file}")
        except PermissionError:
            print(f"Could not remove locked database: {db_file}")

    database = DatabaseManager(db_path=test_db)

    profile_manager = LearnerProfileManager(
        database=database,
        learner_id="test_learner",
    )

    engine = LearningPathEngine(
        database=database,
        profile_manager=profile_manager,
        learner_id="test_learner",
        api_key=None,
    )

    # ------------------------------------------------------------------
    # TEST 1
    # ------------------------------------------------------------------
    print("\nTEST 1: Create learner profile")

    profile = profile_manager.create_or_update_profile(
        name="Test Learner",
        level="Intermediate",
        goal="Learn Machine Learning",
        language="English",
        learning_style="Visual",
        preferred_depth="Detailed",
    )

    assert profile is not None
    print("PASS")

    # ------------------------------------------------------------------
    # TEST 2
    # ------------------------------------------------------------------
    print("\nTEST 2: Empty topic analysis")

    result = engine.analyze_topic("Machine Learning")

    assert result["topic"] == "Machine Learning"
    assert result["weak_concepts"] == []
    assert result["developing_concepts"] == []
    assert result["strong_concepts"] == []

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 3
    # ------------------------------------------------------------------
    print("\nTEST 3: Weak concept detection")

    profile_manager.update_concept_progress(
        topic="Machine Learning",
        concept="Overfitting",
        mastery=0.30,
        attempts=3,
        correct_count=1,
        incorrect_count=2,
    )

    result = engine.analyze_topic("Machine Learning")

    assert any(
        item["concept"] == "Overfitting"
        for item in result["weak_concepts"]
    ), (
        "Weak concept was detected, but its name "
        "was not extracted correctly."
    )

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 4
    # ------------------------------------------------------------------
    print("\nTEST 4: Weak concept recommendation")

    result = engine.recommend_next_concept("Machine Learning")

    assert result["concept"] == "Overfitting", (
        f"Expected Overfitting, "
        f"got {result.get('concept')!r}. "
        f"Full result: {result}"
    )

    assert result["recommended_action"] == "reteach", (
        f"Expected reteach, "
        f"got {result.get('recommended_action')!r}. "
        f"Full result: {result}"
    )

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 5
    # ------------------------------------------------------------------
    print("\nTEST 5: Developing concept")

    profile_manager.update_concept_progress(
        topic="Machine Learning",
        concept="Regularization",
        mastery=0.60,
        attempts=3,
        correct_count=2,
        incorrect_count=1,
    )

    result = engine.analyze_topic("Machine Learning")

    assert any(
        item["concept"] == "Regularization"
        for item in result["developing_concepts"]
    ), (
        "Developing concept was not detected correctly."
    )

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 6
    # ------------------------------------------------------------------
    print("\nTEST 6: Strong concept")

    profile_manager.update_concept_progress(
        topic="Machine Learning",
        concept="Classification",
        mastery=0.90,
        attempts=5,
        correct_count=5,
        incorrect_count=0,
    )

    result = engine.analyze_topic("Machine Learning")

    assert any(
        item["concept"] == "Classification"
        for item in result["strong_concepts"]
    ), (
        "Strong concept was not detected correctly."
    )

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 7
    # ------------------------------------------------------------------
    print("\nTEST 7: New topic recommendation")

    result = engine.recommend_next_concept("Deep Learning")

    assert result["reason"] == "new_topic"
    assert result["recommended_action"] == "introduce"

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 8
    # ------------------------------------------------------------------
    print("\nTEST 8: Learning path")

    result = engine.build_learning_path(
        "Machine Learning",
        max_steps=5,
    )

    assert isinstance(result, list)
    assert len(result) > 0

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 9
    # ------------------------------------------------------------------
    print("\nTEST 9: Seven-day learning plan")

    result = engine.build_7_day_plan("Machine Learning")

    assert isinstance(result, list)
    assert len(result) == 7

    for index, day in enumerate(result, start=1):
        assert day["day"] == index

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 10
    # ------------------------------------------------------------------
    print("\nTEST 10: Revision plan")

    result = engine.build_revision_plan(
        "Machine Learning",
        days=3,
    )

    assert isinstance(result, list)
    assert len(result) == 3

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 11
    # ------------------------------------------------------------------
    print("\nTEST 11: Teacher learning context")

    result = engine.build_teacher_learning_context(
        "Machine Learning"
    )

    assert result["topic"] == "Machine Learning"
    assert "weak_concepts" in result
    assert "developing_concepts" in result
    assert "strong_concepts" in result
    assert "next_recommendation" in result

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 12
    # ------------------------------------------------------------------
    print("\nTEST 12: Topic recommendation")

    result = engine.recommend_next_topic(
        [
            "Machine Learning",
            "Deep Learning",
            "Natural Language Processing",
        ]
    )

    assert result["topic"] in [
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
    ]

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 13
    # ------------------------------------------------------------------
    print("\nTEST 13: Empty topic recommendation")

    result = engine.recommend_next_topic([])

    assert result["topic"] is None
    assert result["reason"] == "no_topics_available"

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 14
    # ------------------------------------------------------------------
    print("\nTEST 14: Teacher context recommendation")

    context = engine.build_teacher_learning_context(
        "Machine Learning"
    )

    result = engine.recommend_from_teacher_context(context)

    assert result["topic"] == "Machine Learning"

    if result.get("concept") is not None:
        assert result["concept"] in [
            "Overfitting",
            "Regularization",
            "Classification",
        ]

    print("PASS")

    # ------------------------------------------------------------------
    # TEST 15
    # ------------------------------------------------------------------
    print("\nTEST 15: Reset progress")

    reset_result = engine.reset()

    assert reset_result is True, (
        f"Expected reset() to return True, "
        f"got {reset_result!r}"
    )

    # Verify that reset actually cleared the concept progress.
    reset_analysis = engine.analyze_topic("Machine Learning")

    assert reset_analysis["weak_concepts"] == []
    assert reset_analysis["developing_concepts"] == []
    assert reset_analysis["strong_concepts"] == []

    print("PASS")

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()