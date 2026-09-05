"""
learning_memory.py
------------------
Persistent learning memory layer for the AI Teacher.

Responsibilities:
- Record teaching actions
- Record student assessments
- Record misconceptions
- Record remediation attempts
- Record re-test outcomes
- Update topic progress
- Update concept progress
- Retrieve learning history
- Build memory context for future teaching sessions

This module does NOT require a Gemini API key.
"""

from typing import Any, Dict, List, Optional

from database import DatabaseManager
from learner_profile import LearnerProfileManager


# ---------------------------------------------------------------------------
# MEMORY CONSTANTS
# ---------------------------------------------------------------------------

MASTERY_STRONG = 0.85
MASTERY_GOOD = 0.70
MASTERY_WEAK = 0.50


# ---------------------------------------------------------------------------
# LEARNING MEMORY
# ---------------------------------------------------------------------------

class LearningMemory:
    """
    Persistent memory manager for one learner.

    It connects:
        Teaching → Assessment → Misconception → Remediation
        → Re-test → Progress → Future Teaching
    """

    def __init__(
        self,
        learner_id: str,
        database: Optional[DatabaseManager] = None,
        learner_profile: Optional[LearnerProfileManager] = None,
    ):
        if not learner_id or not str(learner_id).strip():
            raise ValueError("learner_id must not be empty.")

        self.learner_id = str(learner_id).strip()

        self.db = database or DatabaseManager()

        self.learner_profile = learner_profile or LearnerProfileManager(
            learner_id=self.learner_id,
            database=self.db,
        )

    # -----------------------------------------------------------------------
    # TEACHING EVENTS
    # -----------------------------------------------------------------------

    def record_teaching_action(
        self,
        topic: str,
        concept: Optional[str],
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a teacher action.

        Examples:
        - INTRODUCE
        - EXPLAIN
        - DEMONSTRATE
        - ASK
        - EVALUATE
        - SIMPLIFY
        - REEXPLAIN
        - REMEDIATE
        - INCREASE_DIFFICULTY
        - RECAP
        - CONTINUE
        """

        event_details = details.copy() if details else {}

        event_details["action"] = action

        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="teaching_action",
            topic=topic,
            concept=concept,
            details=event_details,
        )

    def record_lesson_started(
        self,
        topic: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record lesson start."""
        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="lesson_started",
            topic=topic,
            details=details,
        )

    def record_lesson_completed(
        self,
        topic: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record lesson completion."""
        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="lesson_completed",
            topic=topic,
            details=details,
        )

    # -----------------------------------------------------------------------
    # ASSESSMENT MEMORY
    # -----------------------------------------------------------------------

    def record_assessment_result(
        self,
        topic: str,
        concept: str,
        question: str,
        student_answer: str,
        assessment_result: Dict[str, Any],
    ) -> int:
        """
        Persist an assessment result.

        The result dictionary is designed to work directly with
        AssessmentEngine.evaluate().
        """

        question_type = assessment_result.get(
            "question_type",
            assessment_result.get("type", ""),
        )

        correctness = assessment_result.get(
            "correctness",
            "unknown",
        )

        score = self._safe_float(
            assessment_result.get("score", 0.0)
        )

        mastery = self._safe_float(
            assessment_result.get(
                "current_mastery",
                assessment_result.get("mastery", 0.0),
            )
        )

        misconception = assessment_result.get(
            "misconception"
        )

        recommended_action = assessment_result.get(
            "recommended_action",
            assessment_result.get("action"),
        )

        feedback = assessment_result.get(
            "feedback",
            "",
        )

        assessment_id = self.db.record_assessment(
            learner_id=self.learner_id,
            topic=topic,
            concept=concept,
            question_type=question_type,
            question=question,
            student_answer=student_answer,
            correctness=correctness,
            score=score,
            mastery=mastery,
            misconception=misconception,
            recommended_action=recommended_action,
            feedback=feedback,
        )

        # Also create a learning event so the complete interaction
        # is visible in the learning timeline.
        self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="question_answered",
            topic=topic,
            concept=concept,
            details={
                "question_type": question_type,
                "correctness": correctness,
                "score": score,
                "mastery": mastery,
                "misconception": misconception,
                "recommended_action": recommended_action,
            },
        )

        return assessment_id

    # -----------------------------------------------------------------------
    # CONCEPT MEMORY
    # -----------------------------------------------------------------------

    def update_concept_from_assessment(
        self,
        topic: str,
        concept: str,
        assessment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update concept progress using one assessment result.

        The previous concept state is read first so that attempts,
        correct answers and incorrect answers accumulate correctly.
        """

        previous = self.learner_profile.get_concept_progress(
            topic,
            concept,
        )

        previous_attempts = (
            int(previous.get("attempts", 0))
            if previous
            else 0
        )

        previous_correct = (
            int(previous.get("correct_count", 0))
            if previous
            else 0
        )

        previous_incorrect = (
            int(previous.get("incorrect_count", 0))
            if previous
            else 0
        )

        attempts = previous_attempts + 1

        correctness = str(
            assessment_result.get(
                "correctness",
                "",
            )
        ).lower()

        score = self._safe_float(
            assessment_result.get(
                "score",
                0.0,
            )
        )

        # Treat both "correct" and high score as correct.
        is_correct = (
            correctness == "correct"
            or score >= 0.70
        )

        if is_correct:
            correct_count = previous_correct + 1
            incorrect_count = previous_incorrect
        else:
            correct_count = previous_correct
            incorrect_count = previous_incorrect + 1

        mastery = self._safe_float(
            assessment_result.get(
                "current_mastery",
                assessment_result.get(
                    "mastery",
                    score,
                ),
            )
        )

        misconception = assessment_result.get(
            "misconception"
        )

        if misconception == "UNKNOWN":
            misconception = None

        status = self._status_from_mastery(
            mastery
        )

        return self.learner_profile.update_concept_progress(
            topic=topic,
            concept=concept,
            mastery=mastery,
            attempts=attempts,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            misconception=misconception,
            status=status,
        )

    # -----------------------------------------------------------------------
    # TOPIC MEMORY
    # -----------------------------------------------------------------------

    def update_topic_from_assessment(
        self,
        topic: str,
        assessment_result: Dict[str, Any],
        difficulty: str = "MEDIUM",
    ) -> Dict[str, Any]:
        """
        Update topic-level progress from an assessment.
        """

        previous = self.learner_profile.get_topic_progress(
            topic
        )

        previous_attempts = (
            int(previous.get("attempts", 0))
            if previous
            else 0
        )

        attempts = previous_attempts + 1

        score = self._safe_float(
            assessment_result.get(
                "score",
                0.0,
            )
        )

        mastery = self._safe_float(
            assessment_result.get(
                "current_mastery",
                assessment_result.get(
                    "mastery",
                    score,
                ),
            )
        )

        correctness = str(
            assessment_result.get(
                "correctness",
                "",
            )
        ).lower()

        status = self._status_from_mastery(
            mastery
        )

        if correctness == "correct" and mastery >= MASTERY_STRONG:
            status = "mastered"

        return self.learner_profile.update_topic_progress(
            topic=topic,
            mastery=mastery,
            score=score * 100.0,
            attempts=attempts,
            status=status,
            difficulty=difficulty,
        )

    # -----------------------------------------------------------------------
    # MISCONCEPTION MEMORY
    # -----------------------------------------------------------------------

    def record_misconception(
        self,
        topic: str,
        concept: str,
        misconception_type: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a detected misconception.
        """

        event_details = details.copy() if details else {}

        event_details.update(
            {
                "misconception_type": misconception_type,
                "reason": reason,
            }
        )

        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="misconception_detected",
            topic=topic,
            concept=concept,
            details=event_details,
        )

    # -----------------------------------------------------------------------
    # REMEDIATION MEMORY
    # -----------------------------------------------------------------------

    def record_remediation_started(
        self,
        topic: str,
        concept: str,
        misconception_type: str,
        strategy: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record the beginning of remediation."""

        event_details = details.copy() if details else {}

        event_details.update(
            {
                "misconception_type": misconception_type,
                "strategy": strategy,
            }
        )

        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="remediation_started",
            topic=topic,
            concept=concept,
            details=event_details,
        )

    def record_remediation_completed(
        self,
        topic: str,
        concept: str,
        misconception_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record completed remediation."""

        event_details = details.copy() if details else {}

        event_details["misconception_type"] = (
            misconception_type
        )

        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type="remediation_completed",
            topic=topic,
            concept=concept,
            details=event_details,
        )

    # -----------------------------------------------------------------------
    # RETEST MEMORY
    # -----------------------------------------------------------------------

    def record_retest_result(
        self,
        topic: str,
        concept: str,
        retest_result: Dict[str, Any],
    ) -> int:
        """
        Record the outcome of a remediation re-test.
        """

        resolved = bool(
            retest_result.get(
                "resolved",
                False,
            )
        )

        action = retest_result.get(
            "action",
            "REEXPLAIN",
        )

        mastery = self._safe_float(
            retest_result.get(
                "mastery",
                0.0,
            )
        )

        details = {
            "resolved": resolved,
            "action": action,
            "mastery": mastery,
            "difficulty": retest_result.get(
                "difficulty"
            ),
            "feedback": retest_result.get(
                "feedback",
                "",
            ),
            "reason": retest_result.get(
                "reason",
                "",
            ),
        }

        event_type = (
            "retest_success"
            if resolved
            else "retest_failed"
        )

        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type=event_type,
            topic=topic,
            concept=concept,
            details=details,
        )

    # -----------------------------------------------------------------------
    # COMPLETE INTERACTION
    # -----------------------------------------------------------------------

    def process_assessment_cycle(
        self,
        topic: str,
        concept: str,
        question: str,
        student_answer: str,
        assessment_result: Dict[str, Any],
        difficulty: str = "MEDIUM",
    ) -> Dict[str, Any]:
        """
        Process one complete assessment cycle.

        Steps:
        1. Save assessment
        2. Update concept progress
        3. Update topic progress
        4. Record misconception if present
        5. Return updated learner state
        """

        assessment_id = self.record_assessment_result(
            topic=topic,
            concept=concept,
            question=question,
            student_answer=student_answer,
            assessment_result=assessment_result,
        )

        concept_progress = (
            self.update_concept_from_assessment(
                topic=topic,
                concept=concept,
                assessment_result=assessment_result,
            )
        )

        topic_progress = (
            self.update_topic_from_assessment(
                topic=topic,
                assessment_result=assessment_result,
                difficulty=difficulty,
            )
        )

        misconception = assessment_result.get(
            "misconception"
        )

        if (
            misconception
            and str(misconception).upper()
            != "UNKNOWN"
        ):
            self.record_misconception(
                topic=topic,
                concept=concept,
                misconception_type=str(
                    misconception
                ),
                reason=assessment_result.get(
                    "reason",
                    assessment_result.get(
                        "feedback",
                        "Misconception detected.",
                    ),
                ),
            )

        return {
            "assessment_id": assessment_id,
            "concept_progress": concept_progress,
            "topic_progress": topic_progress,
            "learner_state": self.learner_profile.get_current_learning_state(
                topic
            ),
        }

    # -----------------------------------------------------------------------
    # MEMORY RETRIEVAL
    # -----------------------------------------------------------------------

    def get_topic_memory(
        self,
        topic: str,
        assessment_limit: int = 20,
        event_limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Retrieve everything important about a topic.
        """

        return {
            "topic": topic,
            "topic_progress": (
                self.learner_profile.get_topic_progress(
                    topic
                )
            ),
            "concepts": (
                self.learner_profile.get_all_concepts(
                    topic
                )
            ),
            "strong_concepts": (
                self.learner_profile.get_strong_concepts(
                    topic
                )
            ),
            "weak_concepts": (
                self.learner_profile.get_weak_concepts(
                    topic
                )
            ),
            "misconceptions": (
                self.learner_profile.get_active_misconceptions(
                    topic
                )
            ),
            "assessments": (
                self.learner_profile.get_assessment_history(
                    topic,
                    assessment_limit,
                )
            ),
            "events": (
                self.learner_profile.get_learning_events(
                    topic,
                    event_limit,
                )
            ),
        }

    def get_recent_memory(
        self,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Retrieve recent learner activity across all topics.
        """

        return {
            "assessments": (
                self.learner_profile.get_assessment_history(
                    limit=limit
                )
            ),
            "events": (
                self.learner_profile.get_learning_events(
                    limit=limit
                )
            ),
            "summary": self.learner_profile.get_summary(),
        }

    # -----------------------------------------------------------------------
    # TEACHER CONTEXT
    # -----------------------------------------------------------------------

    def build_memory_context(
        self,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a compact context package for the Teacher Agent.

        This is the bridge between persistent learner memory and
        real-time adaptive teaching.
        """

        profile_context = (
            self.learner_profile.build_teacher_context(
                topic
            )
        )

        if topic:
            topic_memory = self.get_topic_memory(
                topic=topic,
                assessment_limit=10,
                event_limit=20,
            )
        else:
            topic_memory = {
                "topic": None,
                "topic_progress": None,
                "concepts": [],
                "strong_concepts": [],
                "weak_concepts": [],
                "misconceptions": [],
                "assessments": [],
                "events": [],
            }

        return {
            "learner": profile_context,
            "topic_memory": {
                "topic_progress": topic_memory[
                    "topic_progress"
                ],
                "concepts": topic_memory[
                    "concepts"
                ],
                "strong_concepts": [
                    item["concept"]
                    for item in topic_memory[
                        "strong_concepts"
                    ]
                ],
                "weak_concepts": [
                    item["concept"]
                    for item in topic_memory[
                        "weak_concepts"
                    ]
                ],
                "misconceptions": [
                    item["misconception"]
                    for item in topic_memory[
                        "misconceptions"
                    ]
                ],
                "recent_assessments": topic_memory[
                    "assessments"
                ],
                "recent_events": topic_memory[
                    "events"
                ],
            },
        }

    # -----------------------------------------------------------------------
    # NEXT LEARNING
    # -----------------------------------------------------------------------

    def get_next_learning_recommendation(
        self,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the next recommended learning action."""

        return self.learner_profile.recommend_next_learning(
            topic
        )

    # -----------------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Reset all persistent memory for this learner."""
        self.learner_profile.reset_progress()

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any
    ) -> float:
        """Safely convert a value to float and clamp it."""

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, value),
        )

    @staticmethod
    def _status_from_mastery(
        mastery: float
    ) -> str:
        """Convert mastery into a persistent status."""

        mastery = max(
            0.0,
            min(1.0, float(mastery)),
        )

        if mastery >= MASTERY_STRONG:
            return "mastered"

        if mastery >= MASTERY_GOOD:
            return "proficient"

        if mastery >= MASTERY_WEAK:
            return "developing"

        if mastery > 0.0:
            return "needs_remediation"

        return "not_started"


# ---------------------------------------------------------------------------
# LOCAL TESTS
# ---------------------------------------------------------------------------

def run_local_tests() -> None:
    """
    Independent local tests.

    No Gemini API key is required.
    """

    import os

    print("=" * 70)
    print("LEARNING MEMORY - LOCAL TEST")
    print("=" * 70)

    test_db_path = os.path.join(
        "data",
        "test_learning_memory.db",
    )

    # Remove old test database.
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except PermissionError:
            pass

    db = DatabaseManager(test_db_path)

    learner = LearnerProfileManager(
        learner_id="memory_test_001",
        database=db,
    )

    memory = LearningMemory(
        learner_id="memory_test_001",
        database=db,
        learner_profile=learner,
    )

    try:

        # -------------------------------------------------------------------
        # TEST 1 - Profile
        # -------------------------------------------------------------------

        print("\n[TEST 1] Learner profile")

        learner.create_or_update_profile(
            name="Memory Test Student",
            level="Intermediate",
            goal="Exam Preparation",
            language="English",
            learning_style="Interactive",
            preferred_depth="Detailed",
        )

        assert learner.get_profile() is not None

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 2 - Lesson started
        # -------------------------------------------------------------------

        print("[TEST 2] Lesson start memory")

        event_id = memory.record_lesson_started(
            topic="Ohm's Law",
            details={
                "duration": 20,
                "mode": "structured",
            },
        )

        assert event_id > 0

        events = learner.get_learning_events(
            "Ohm's Law"
        )

        assert len(events) == 1
        assert events[0]["event_type"] == "lesson_started"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 3 - Teaching action
        # -------------------------------------------------------------------

        print("[TEST 3] Teaching action memory")

        action_id = memory.record_teaching_action(
            topic="Ohm's Law",
            concept="Voltage",
            action="EXPLAIN",
            details={
                "strategy": "real_world_analogy",
                "difficulty": "EASY",
            },
        )

        assert action_id > 0

        events = learner.get_learning_events(
            "Ohm's Law"
        )

        teaching_events = [
            event
            for event in events
            if event["event_type"]
            == "teaching_action"
        ]

        assert len(teaching_events) == 1
        assert (
            teaching_events[0]["details"]["action"]
            == "EXPLAIN"
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 4 - Correct assessment
        # -------------------------------------------------------------------

        print("[TEST 4] Correct assessment memory")

        correct_result = {
            "question_type": "MCQ",
            "correctness": "correct",
            "score": 1.0,
            "current_mastery": 1.0,
            "misconception": "UNKNOWN",
            "recommended_action": "CONTINUE",
            "feedback": "Correct answer.",
        }

        assessment_id = memory.record_assessment_result(
            topic="Ohm's Law",
            concept="Voltage",
            question="What is voltage?",
            student_answer="Electrical potential difference",
            assessment_result=correct_result,
        )

        assert assessment_id > 0

        history = learner.get_assessment_history(
            "Ohm's Law"
        )

        assert len(history) == 1
        assert history[0]["correctness"] == "correct"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 5 - Concept update
        # -------------------------------------------------------------------

        print("[TEST 5] Concept progress update")

        concept_progress = (
            memory.update_concept_from_assessment(
                topic="Ohm's Law",
                concept="Voltage",
                assessment_result=correct_result,
            )
        )

        assert concept_progress["attempts"] == 1
        assert concept_progress["correct_count"] == 1
        assert concept_progress["incorrect_count"] == 0
        assert concept_progress["status"] == "mastered"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 6 - Topic update
        # -------------------------------------------------------------------

        print("[TEST 6] Topic progress update")

        topic_progress = (
            memory.update_topic_from_assessment(
                topic="Ohm's Law",
                assessment_result=correct_result,
                difficulty="MEDIUM",
            )
        )

        assert topic_progress["attempts"] == 1
        assert abs(topic_progress["mastery"] - 1.0) < 0.001
        assert topic_progress["status"] == "mastered"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 7 - Complete assessment cycle with misconception
        # -------------------------------------------------------------------

        print("[TEST 7] Misconception assessment cycle")

        wrong_result = {
            "question_type": "SHORT_ANSWER",
            "correctness": "incorrect",
            "score": 0.20,
            "current_mastery": 0.20,
            "misconception": "CONCEPTUAL_GAP",
            "recommended_action": "REMEDIATE",
            "feedback": (
                "The relationship between resistance "
                "and current is misunderstood."
            ),
            "reason": (
                "The learner thinks resistance increases "
                "current when voltage is constant."
            ),
        }

        cycle = memory.process_assessment_cycle(
            topic="Ohm's Law",
            concept="Resistance",
            question=(
                "What happens to current when resistance "
                "increases while voltage stays constant?"
            ),
            student_answer="Current increases.",
            assessment_result=wrong_result,
            difficulty="EASY",
        )

        assert cycle["assessment_id"] > 0

        resistance = learner.get_concept_progress(
            "Ohm's Law",
            "Resistance",
        )

        assert resistance is not None
        assert resistance["attempts"] == 1
        assert resistance["incorrect_count"] == 1
        assert resistance["misconception"] == "CONCEPTUAL_GAP"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 8 - Misconception event
        # -------------------------------------------------------------------

        print("[TEST 8] Misconception memory")

        events = learner.get_learning_events(
            "Ohm's Law"
        )

        misconception_events = [
            event
            for event in events
            if event["event_type"]
            == "misconception_detected"
        ]

        assert len(misconception_events) == 1
        assert (
            misconception_events[0]["details"][
                "misconception_type"
            ]
            == "CONCEPTUAL_GAP"
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 9 - Remediation
        # -------------------------------------------------------------------

        print("[TEST 9] Remediation memory")

        start_id = memory.record_remediation_started(
            topic="Ohm's Law",
            concept="Resistance",
            misconception_type="CONCEPTUAL_GAP",
            strategy="water_flow_analogy",
        )

        complete_id = memory.record_remediation_completed(
            topic="Ohm's Law",
            concept="Resistance",
            misconception_type="CONCEPTUAL_GAP",
            details={
                "new_explanation": True,
                "new_example": True,
            },
        )

        assert start_id > 0
        assert complete_id > 0

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 10 - Retest success
        # -------------------------------------------------------------------

        print("[TEST 10] Successful re-test memory")

        retest_id = memory.record_retest_result(
            topic="Ohm's Law",
            concept="Resistance",
            retest_result={
                "resolved": True,
                "action": "CONTINUE",
                "mastery": 0.75,
                "difficulty": "MEDIUM",
                "feedback": "Good recovery.",
            },
        )

        assert retest_id > 0

        events = learner.get_learning_events(
            "Ohm's Law"
        )

        success_events = [
            event
            for event in events
            if event["event_type"]
            == "retest_success"
        ]

        assert len(success_events) == 1
        assert (
            success_events[0]["details"]["resolved"]
            is True
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 11 - Topic memory
        # -------------------------------------------------------------------

        print("[TEST 11] Topic memory retrieval")

        topic_memory = memory.get_topic_memory(
            "Ohm's Law"
        )

        assert topic_memory["topic"] == "Ohm's Law"
        assert topic_memory["topic_progress"] is not None
        assert len(topic_memory["concepts"]) == 2
        assert len(topic_memory["assessments"]) == 2
        assert len(topic_memory["events"]) >= 6

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 12 - Teacher memory context
        # -------------------------------------------------------------------

        print("[TEST 12] Teacher memory context")

        context = memory.build_memory_context(
            "Ohm's Law"
        )

        assert "learner" in context
        assert "topic_memory" in context

        assert (
            "Voltage"
            in context["topic_memory"]["strong_concepts"]
        )

        assert (
            "Resistance"
            in context["topic_memory"]["weak_concepts"]
        )

        assert (
            "CONCEPTUAL_GAP"
            in context["topic_memory"]["misconceptions"]
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 13 - Next learning recommendation
        # -------------------------------------------------------------------

        print("[TEST 13] Next learning recommendation")

        recommendation = (
            memory.get_next_learning_recommendation(
                "Ohm's Law"
            )
        )

        assert recommendation["concept"] == "Resistance"

        # It should prioritize remediation because the
        # misconception is still present in concept memory.
        assert recommendation["action"] == "REMEDIATE"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 14 - Recent memory
        # -------------------------------------------------------------------

        print("[TEST 14] Recent memory")

        recent = memory.get_recent_memory(
            limit=20
        )

        assert "assessments" in recent
        assert "events" in recent
        assert "summary" in recent

        assert len(recent["assessments"]) == 2
        assert len(recent["events"]) >= 6

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 15 - Reset memory
        # -------------------------------------------------------------------

        print("[TEST 15] Reset learning memory")

        memory.reset_memory()

        assert learner.get_profile() is None
        assert learner.get_all_topic_progress() == []
        assert learner.get_all_concepts() == []
        assert learner.get_assessment_history() == []
        assert learner.get_learning_events() == []

        print("PASS")

        # -------------------------------------------------------------------
        # SUCCESS
        # -------------------------------------------------------------------

        print()
        print("=" * 70)
        print("ALL LEARNING MEMORY TESTS PASSED SUCCESSFULLY")
        print("=" * 70)

    finally:

        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
        except PermissionError:
            print(
                "\nWarning: Could not remove test database. "
                "Close any process using the SQLite file and "
                "delete it manually if necessary."
            )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_local_tests()