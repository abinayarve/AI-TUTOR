"""
learner_profile.py
------------------
Learner intelligence layer for the AI Teacher.

Responsibilities:
- Create and manage learner profiles
- Track topic mastery
- Track concept mastery
- Identify strong and weak concepts
- Track attempts and scores
- Build learner context for the Teacher Agent
- Recommend what the learner should study next
- Keep learner information persistent through DatabaseManager

This module does NOT require a Gemini API key.
"""

from typing import Any, Dict, List, Optional

from database import DatabaseManager


# ---------------------------------------------------------------------------
# MASTERY THRESHOLDS
# ---------------------------------------------------------------------------

STRONG_MASTERY = 0.85
GOOD_MASTERY = 0.70
WEAK_MASTERY = 0.50


# ---------------------------------------------------------------------------
# LEARNER PROFILE MANAGER
# ---------------------------------------------------------------------------

class LearnerProfileManager:
    """
    High-level learner profile manager.

    DatabaseManager handles storage.
    LearnerProfileManager handles learning intelligence and interpretation.
    """

    def __init__(
        self,
        learner_id: str,
        database: Optional[DatabaseManager] = None,
    ):
        if not learner_id or not str(learner_id).strip():
            raise ValueError("learner_id must not be empty.")

        self.learner_id = str(learner_id).strip()

        self.db = database or DatabaseManager()

    # -----------------------------------------------------------------------
    # PROFILE
    # -----------------------------------------------------------------------

    def create_or_update_profile(
        self,
        name: str = "",
        level: str = "Beginner",
        goal: str = "",
        language: str = "English",
        learning_style: str = "",
        preferred_depth: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new learner profile or update an existing one.
        """

        return self.db.upsert_learner_profile(
            learner_id=self.learner_id,
            name=name,
            level=level,
            goal=goal,
            language=language,
            learning_style=learning_style,
            preferred_depth=preferred_depth,
        )

    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Return the stored learner profile."""
        return self.db.get_learner_profile(self.learner_id)

    def ensure_profile(
        self,
        name: str = "Learner",
        level: str = "Beginner",
        goal: str = "General Learning",
        language: str = "English",
        learning_style: str = "Interactive",
        preferred_depth: str = "Standard",
    ) -> Dict[str, Any]:
        """
        Return an existing profile or create a sensible default profile.
        """

        profile = self.get_profile()

        if profile is not None:
            return profile

        return self.create_or_update_profile(
            name=name,
            level=level,
            goal=goal,
            language=language,
            learning_style=learning_style,
            preferred_depth=preferred_depth,
        )

    # -----------------------------------------------------------------------
    # TOPIC PROGRESS
    # -----------------------------------------------------------------------

    def update_topic_progress(
        self,
        topic: str,
        mastery: float,
        score: float,
        attempts: int,
        status: Optional[str] = None,
        difficulty: str = "MEDIUM",
    ) -> Dict[str, Any]:
        """
        Save topic-level learning progress.

        If status is not supplied, it is inferred from mastery.
        """

        if not topic or not str(topic).strip():
            raise ValueError("topic must not be empty.")

        mastery = self._clamp(mastery)
        score = self._clamp(score / 100.0) * 100.0
        attempts = max(0, int(attempts))

        if status is None:
            status = self._status_from_mastery(mastery)

        return self.db.upsert_topic_progress(
            learner_id=self.learner_id,
            topic=str(topic).strip(),
            mastery=mastery,
            score=score,
            attempts=attempts,
            status=status,
            difficulty=difficulty,
        )

    def get_topic_progress(
        self,
        topic: str,
    ) -> Optional[Dict[str, Any]]:
        """Return progress for one topic."""
        return self.db.get_topic_progress(
            self.learner_id,
            topic,
        )

    def get_all_topic_progress(self) -> List[Dict[str, Any]]:
        """Return all topics studied by this learner."""
        return self.db.get_all_topic_progress(
            self.learner_id
        )

    # -----------------------------------------------------------------------
    # CONCEPT PROGRESS
    # -----------------------------------------------------------------------

    def update_concept_progress(
        self,
        topic: str,
        concept: str,
        mastery: float,
        attempts: int,
        correct_count: int,
        incorrect_count: int,
        misconception: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save concept-level learning progress.
        """

        if not topic or not str(topic).strip():
            raise ValueError("topic must not be empty.")

        if not concept or not str(concept).strip():
            raise ValueError("concept must not be empty.")

        mastery = self._clamp(mastery)

        attempts = max(0, int(attempts))
        correct_count = max(0, int(correct_count))
        incorrect_count = max(0, int(incorrect_count))

        if status is None:
            status = self._status_from_mastery(mastery)

        return self.db.upsert_concept_progress(
            learner_id=self.learner_id,
            topic=str(topic).strip(),
            concept=str(concept).strip(),
            mastery=mastery,
            attempts=attempts,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            misconception=misconception,
            status=status,
        )

    def get_concept_progress(
        self,
        topic: str,
        concept: str,
    ) -> Optional[Dict[str, Any]]:
        """Return progress for one concept."""
        return self.db.get_concept_progress(
            self.learner_id,
            topic,
            concept,
        )

    def get_all_concepts(
        self,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all concept progress records."""
        return self.db.get_all_concept_progress(
            self.learner_id,
            topic,
        )

    # -----------------------------------------------------------------------
    # STRONG / WEAK CONCEPTS
    # -----------------------------------------------------------------------

    def get_strong_concepts(
        self,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return concepts with mastery >= 0.85.
        """

        concepts = self.get_all_concepts(topic)

        return [
            concept
            for concept in concepts
            if float(concept.get("mastery", 0.0)) >= STRONG_MASTERY
        ]

    def get_weak_concepts(
        self,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return concepts with mastery below 0.50.
        """

        concepts = self.get_all_concepts(topic)

        return [
            concept
            for concept in concepts
            if float(concept.get("mastery", 0.0)) < WEAK_MASTERY
        ]

    def get_developing_concepts(
        self,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return concepts that are developing but not yet strong.
        """

        concepts = self.get_all_concepts(topic)

        return [
            concept
            for concept in concepts
            if WEAK_MASTERY
            <= float(concept.get("mastery", 0.0))
            < STRONG_MASTERY
        ]

    # -----------------------------------------------------------------------
    # MISCONCEPTIONS
    # -----------------------------------------------------------------------

    def get_active_misconceptions(
        self,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return concepts where a misconception has been recorded.
        """

        concepts = self.get_all_concepts(topic)

        return [
            concept
            for concept in concepts
            if concept.get("misconception")
            and str(concept.get("misconception")).upper() != "UNKNOWN"
        ]

    # -----------------------------------------------------------------------
    # LEARNING HISTORY
    # -----------------------------------------------------------------------

    def get_assessment_history(
        self,
        topic: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent assessment history."""
        return self.db.get_assessment_history(
            self.learner_id,
            topic,
            limit,
        )

    def get_learning_events(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return recent learning events."""
        return self.db.get_learning_events(
            self.learner_id,
            topic,
            limit,
        )

    # -----------------------------------------------------------------------
    # OVERALL MASTERY
    # -----------------------------------------------------------------------

    def calculate_topic_mastery(
        self,
        topic: str,
    ) -> float:
        """
        Calculate topic mastery from concept-level mastery.

        If concept records exist, their average is used.
        Otherwise, stored topic mastery is returned.
        """

        concepts = self.get_all_concepts(topic)

        if concepts:
            values = [
                float(concept.get("mastery", 0.0))
                for concept in concepts
            ]

            return self._clamp(
                sum(values) / len(values)
            )

        topic_record = self.get_topic_progress(topic)

        if topic_record is None:
            return 0.0

        return self._clamp(
            float(topic_record.get("mastery", 0.0))
        )

    def calculate_overall_mastery(self) -> float:
        """
        Calculate overall learner mastery.

        Uses all tracked concepts when available.
        Otherwise uses topic-level mastery.
        """

        concepts = self.get_all_concepts()

        if concepts:
            values = [
                float(concept.get("mastery", 0.0))
                for concept in concepts
            ]

            return self._clamp(
                sum(values) / len(values)
            )

        topics = self.get_all_topic_progress()

        if not topics:
            return 0.0

        values = [
            float(topic.get("mastery", 0.0))
            for topic in topics
        ]

        return self._clamp(
            sum(values) / len(values)
        )

    # -----------------------------------------------------------------------
    # CURRENT LEARNING STATE
    # -----------------------------------------------------------------------

    def get_current_learning_state(
        self,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the learner's current learning state.

        This is designed to be consumed by the Teacher Agent.
        """

        profile = self.get_profile()

        if topic:
            topic_mastery = self.calculate_topic_mastery(topic)
            strong = self.get_strong_concepts(topic)
            weak = self.get_weak_concepts(topic)
            developing = self.get_developing_concepts(topic)
            misconceptions = self.get_active_misconceptions(topic)
        else:
            topic_mastery = None
            strong = self.get_strong_concepts()
            weak = self.get_weak_concepts()
            developing = self.get_developing_concepts()
            misconceptions = self.get_active_misconceptions()

        return {
            "learner_id": self.learner_id,
            "profile": profile,
            "topic": topic,
            "topic_mastery": topic_mastery,
            "overall_mastery": self.calculate_overall_mastery(),
            "strong_concepts": [
                item["concept"]
                for item in strong
            ],
            "weak_concepts": [
                item["concept"]
                for item in weak
            ],
            "developing_concepts": [
                item["concept"]
                for item in developing
            ],
            "active_misconceptions": [
                item["misconception"]
                for item in misconceptions
            ],
        }

    # -----------------------------------------------------------------------
    # TEACHER AGENT CONTEXT
    # -----------------------------------------------------------------------

    def build_teacher_context(
        self,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a clean learner context for the Teacher Agent.

        This allows the Teacher Agent to answer questions such as:

        - What does the learner already know?
        - What are they struggling with?
        - What misconceptions exist?
        - What difficulty should be used?
        - What should be taught next?
        """

        state = self.get_current_learning_state(topic)

        profile = state.get("profile") or {}

        topic_mastery = state.get("topic_mastery")

        if topic_mastery is None:
            recommended_difficulty = self._difficulty_from_mastery(
                state["overall_mastery"]
            )
        else:
            recommended_difficulty = self._difficulty_from_mastery(
                topic_mastery
            )

        return {
            "learner_id": self.learner_id,

            "learner_profile": {
                "name": profile.get("name", ""),
                "level": profile.get("level", "Beginner"),
                "goal": profile.get("goal", ""),
                "language": profile.get("language", "English"),
                "learning_style": profile.get(
                    "learning_style",
                    ""
                ),
                "preferred_depth": profile.get(
                    "preferred_depth",
                    ""
                ),
            },

            "current_topic": topic,

            "mastery": {
                "topic": topic_mastery,
                "overall": state["overall_mastery"],
            },

            "strong_concepts": state["strong_concepts"],
            "weak_concepts": state["weak_concepts"],
            "developing_concepts": state["developing_concepts"],
            "active_misconceptions": state[
                "active_misconceptions"
            ],

            "recommended_difficulty": recommended_difficulty,

            "teaching_guidance": self._build_teaching_guidance(
                state,
                recommended_difficulty,
            ),
        }

    # -----------------------------------------------------------------------
    # NEXT LEARNING RECOMMENDATION
    # -----------------------------------------------------------------------

    def recommend_next_learning(
        self,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recommend the next learning action.

        Priority:
        1. Active misconceptions
        2. Weak concepts
        3. Developing concepts
        4. New/unstarted topic
        5. Increase difficulty for mastered concepts
        """

        if topic:
            concepts = self.get_all_concepts(topic)

            misconceptions = [
                item
                for item in concepts
                if item.get("misconception")
                and str(
                    item.get("misconception")
                ).upper() != "UNKNOWN"
            ]

            if misconceptions:
                target = min(
                    misconceptions,
                    key=lambda x: float(
                        x.get("mastery", 0.0)
                    ),
                )

                return {
                    "topic": topic,
                    "concept": target["concept"],
                    "action": "REMEDIATE",
                    "reason": (
                        "An active misconception needs "
                        "to be corrected before moving on."
                    ),
                    "priority": "HIGH",
                }

            weak = [
                item
                for item in concepts
                if float(item.get("mastery", 0.0))
                < WEAK_MASTERY
            ]

            if weak:
                target = min(
                    weak,
                    key=lambda x: float(
                        x.get("mastery", 0.0)
                    ),
                )

                return {
                    "topic": topic,
                    "concept": target["concept"],
                    "action": "RETEACH",
                    "reason": (
                        "This concept has low mastery "
                        "and needs additional explanation."
                    ),
                    "priority": "HIGH",
                }

            developing = [
                item
                for item in concepts
                if WEAK_MASTERY
                <= float(item.get("mastery", 0.0))
                < STRONG_MASTERY
            ]

            if developing:
                target = min(
                    developing,
                    key=lambda x: float(
                        x.get("mastery", 0.0)
                    ),
                )

                return {
                    "topic": topic,
                    "concept": target["concept"],
                    "action": "PRACTICE",
                    "reason": (
                        "This concept is developing and "
                        "needs more practice."
                    ),
                    "priority": "MEDIUM",
                }

            if concepts:
                target = max(
                    concepts,
                    key=lambda x: float(
                        x.get("mastery", 0.0)
                    ),
                )

                return {
                    "topic": topic,
                    "concept": target["concept"],
                    "action": "INCREASE_DIFFICULTY",
                    "reason": (
                        "Tracked concepts are strong. "
                        "The learner can move to harder "
                        "application questions."
                    ),
                    "priority": "LOW",
                }

            return {
                "topic": topic,
                "concept": None,
                "action": "INTRODUCE",
                "reason": (
                    "No concept-level progress exists "
                    "for this topic yet."
                ),
                "priority": "MEDIUM",
            }

        # -------------------------------------------------------------------
        # No specific topic
        # -------------------------------------------------------------------

        weak = self.get_weak_concepts()

        if weak:
            target = min(
                weak,
                key=lambda x: float(
                    x.get("mastery", 0.0)
                ),
            )

            return {
                "topic": target["topic"],
                "concept": target["concept"],
                "action": "RETEACH",
                "reason": (
                    "The learner's weakest concept "
                    "should be reviewed next."
                ),
                "priority": "HIGH",
            }

        developing = self.get_developing_concepts()

        if developing:
            target = min(
                developing,
                key=lambda x: float(
                    x.get("mastery", 0.0)
                ),
            )

            return {
                "topic": target["topic"],
                "concept": target["concept"],
                "action": "PRACTICE",
                "reason": (
                    "A developing concept is the best "
                    "candidate for continued practice."
                ),
                "priority": "MEDIUM",
            }

        topics = self.get_all_topic_progress()

        if topics:
            target = min(
                topics,
                key=lambda x: float(
                    x.get("mastery", 0.0)
                ),
            )

            return {
                "topic": target["topic"],
                "concept": None,
                "action": "CONTINUE",
                "reason": (
                    "Continue with the topic having "
                    "the lowest current mastery."
                ),
                "priority": "LOW",
            }

        return {
            "topic": None,
            "concept": None,
            "action": "INTRODUCE",
            "reason": (
                "The learner has no recorded learning "
                "history yet."
            ),
            "priority": "MEDIUM",
        }

    # -----------------------------------------------------------------------
    # EVENT HELPERS
    # -----------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        topic: Optional[str] = None,
        concept: Optional[str] = None,
        details: Optional[Any] = None,
    ) -> int:
        """Record a learner activity event."""
        return self.db.record_learning_event(
            learner_id=self.learner_id,
            event_type=event_type,
            topic=topic,
            concept=concept,
            details=details,
        )

    # -----------------------------------------------------------------------
    # COMPLETE SUMMARY
    # -----------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a complete learner summary.
        """

        return self.db.get_learner_summary(
            self.learner_id
        )

    # -----------------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------------

    def reset_progress(self) -> None:
        """Delete all persistent data for this learner."""
        self.db.clear_learner_data(
            self.learner_id
        )

    # -----------------------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        """Keep mastery between 0 and 1."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, value))

    @staticmethod
    def _status_from_mastery(
        mastery: float
    ) -> str:
        """Convert mastery score into a learning status."""

        mastery = max(0.0, min(1.0, float(mastery)))

        if mastery >= STRONG_MASTERY:
            return "mastered"

        if mastery >= GOOD_MASTERY:
            return "proficient"

        if mastery >= WEAK_MASTERY:
            return "developing"

        if mastery > 0.0:
            return "needs_remediation"

        return "not_started"

    @staticmethod
    def _difficulty_from_mastery(
        mastery: float
    ) -> str:
        """Choose recommended difficulty from mastery."""

        mastery = max(
            0.0,
            min(1.0, float(mastery)),
        )

        if mastery < WEAK_MASTERY:
            return "EASY"

        if mastery < STRONG_MASTERY:
            return "MEDIUM"

        return "HARD"

    @staticmethod
    def _build_teaching_guidance(
        state: Dict[str, Any],
        difficulty: str,
    ) -> List[str]:
        """Create pedagogical guidance for the Teacher Agent."""

        guidance = []

        if state["weak_concepts"]:
            guidance.append(
                "Prioritize weak concepts before introducing "
                "new advanced material."
            )

        if state["active_misconceptions"]:
            guidance.append(
                "Address active misconceptions explicitly "
                "using a new explanation or analogy."
            )

        if state["developing_concepts"]:
            guidance.append(
                "Use guided practice and short checks "
                "for developing concepts."
            )

        if state["strong_concepts"]:
            guidance.append(
                "Use application or higher-order questions "
                "for strong concepts."
            )

        guidance.append(
            f"Recommended question difficulty: {difficulty}."
        )

        if not guidance:
            guidance.append(
                "Start with a simple conceptual explanation "
                "and establish baseline understanding."
            )

        return guidance


# ---------------------------------------------------------------------------
# LOCAL TESTS
# ---------------------------------------------------------------------------

def run_local_tests() -> None:
    """
    Independent local tests for LearnerProfileManager.

    No Gemini API key is required.
    """

    import os

    print("=" * 70)
    print("LEARNER PROFILE MANAGER - LOCAL TEST")
    print("=" * 70)

    test_db_path = os.path.join(
        "data",
        "test_learner_profile.db",
    )

    # Remove old test database.
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except PermissionError:
            pass

    db = DatabaseManager(test_db_path)

    learner = LearnerProfileManager(
        learner_id="profile_test_001",
        database=db,
    )

    try:

        # -------------------------------------------------------------------
        # TEST 1 - Profile creation
        # -------------------------------------------------------------------

        print("\n[TEST 1] Profile creation")

        profile = learner.create_or_update_profile(
            name="Abinaya Test",
            level="Intermediate",
            goal="Exam Preparation",
            language="English",
            learning_style="Visual",
            preferred_depth="Detailed",
        )

        assert profile["learner_id"] == "profile_test_001"
        assert profile["name"] == "Abinaya Test"
        assert profile["level"] == "Intermediate"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 2 - Profile retrieval
        # -------------------------------------------------------------------

        print("[TEST 2] Profile retrieval")

        retrieved = learner.get_profile()

        assert retrieved is not None
        assert retrieved["goal"] == "Exam Preparation"
        assert retrieved["language"] == "English"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 3 - Ensure profile
        # -------------------------------------------------------------------

        print("[TEST 3] Ensure existing profile")

        ensured = learner.ensure_profile(
            name="Should Not Replace",
            level="Beginner",
        )

        assert ensured["name"] == "Abinaya Test"
        assert ensured["level"] == "Intermediate"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 4 - Topic progress
        # -------------------------------------------------------------------

        print("[TEST 4] Topic progress")

        topic_record = learner.update_topic_progress(
            topic="Ohm's Law",
            mastery=0.60,
            score=60,
            attempts=2,
        )

        assert topic_record["topic"] == "Ohm's Law"
        assert abs(topic_record["mastery"] - 0.60) < 0.001
        assert topic_record["status"] == "developing"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 5 - Concept progress
        # -------------------------------------------------------------------

        print("[TEST 5] Concept progress")

        learner.update_concept_progress(
            topic="Ohm's Law",
            concept="Voltage",
            mastery=0.90,
            attempts=3,
            correct_count=3,
            incorrect_count=0,
            status="mastered",
        )

        learner.update_concept_progress(
            topic="Ohm's Law",
            concept="Resistance",
            mastery=0.35,
            attempts=3,
            correct_count=1,
            incorrect_count=2,
            misconception="CONCEPTUAL_GAP",
            status="needs_remediation",
        )

        learner.update_concept_progress(
            topic="Ohm's Law",
            concept="Current",
            mastery=0.65,
            attempts=2,
            correct_count=1,
            incorrect_count=1,
        )

        concepts = learner.get_all_concepts(
            "Ohm's Law"
        )

        assert len(concepts) == 3

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 6 - Strong concepts
        # -------------------------------------------------------------------

        print("[TEST 6] Strong concept detection")

        strong = learner.get_strong_concepts(
            "Ohm's Law"
        )

        assert len(strong) == 1
        assert strong[0]["concept"] == "Voltage"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 7 - Weak concepts
        # -------------------------------------------------------------------

        print("[TEST 7] Weak concept detection")

        weak = learner.get_weak_concepts(
            "Ohm's Law"
        )

        assert len(weak) == 1
        assert weak[0]["concept"] == "Resistance"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 8 - Developing concepts
        # -------------------------------------------------------------------

        print("[TEST 8] Developing concept detection")

        developing = learner.get_developing_concepts(
            "Ohm's Law"
        )

        assert len(developing) == 1
        assert developing[0]["concept"] == "Current"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 9 - Misconception detection
        # -------------------------------------------------------------------

        print("[TEST 9] Active misconception detection")

        misconceptions = learner.get_active_misconceptions(
            "Ohm's Law"
        )

        assert len(misconceptions) == 1
        assert (
            misconceptions[0]["concept"]
            == "Resistance"
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 10 - Topic mastery
        # -------------------------------------------------------------------

        print("[TEST 10] Topic mastery calculation")

        mastery = learner.calculate_topic_mastery(
            "Ohm's Law"
        )

        expected = (0.90 + 0.35 + 0.65) / 3

        assert abs(mastery - expected) < 0.001

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 11 - Overall mastery
        # -------------------------------------------------------------------

        print("[TEST 11] Overall mastery calculation")

        overall = learner.calculate_overall_mastery()

        assert abs(overall - expected) < 0.001

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 12 - Teacher context
        # -------------------------------------------------------------------

        print("[TEST 12] Teacher Agent context")

        context = learner.build_teacher_context(
            "Ohm's Law"
        )

        assert context["learner_id"] == "profile_test_001"
        assert (
            context["learner_profile"]["name"]
            == "Abinaya Test"
        )
        assert (
            "Voltage"
            in context["strong_concepts"]
        )
        assert (
            "Resistance"
            in context["weak_concepts"]
        )
        assert (
            "Current"
            in context["developing_concepts"]
        )
        assert (
            context["recommended_difficulty"]
            == "MEDIUM"
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 13 - Learning recommendation
        # -------------------------------------------------------------------

        print("[TEST 13] Next learning recommendation")

        recommendation = learner.recommend_next_learning(
            "Ohm's Law"
        )

        assert recommendation["action"] == "REMEDIATE"
        assert (
            recommendation["concept"]
            == "Resistance"
        )
        assert recommendation["priority"] == "HIGH"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 14 - Event recording
        # -------------------------------------------------------------------

        print("[TEST 14] Learning event recording")

        event_id = learner.record_event(
            event_type="remediation_started",
            topic="Ohm's Law",
            concept="Resistance",
            details={
                "attempt": 1,
                "strategy": "analogy",
            },
        )

        assert isinstance(event_id, int)
        assert event_id > 0

        events = learner.get_learning_events(
            "Ohm's Law"
        )

        assert len(events) == 1
        assert (
            events[0]["event_type"]
            == "remediation_started"
        )
        assert (
            events[0]["details"]["strategy"]
            == "analogy"
        )

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 15 - Complete summary
        # -------------------------------------------------------------------

        print("[TEST 15] Complete learner summary")

        summary = learner.get_summary()

        assert (
            summary["learner_id"]
            == "profile_test_001"
        )
        assert summary["profile"] is not None
        assert len(summary["topics"]) == 1
        assert len(summary["concepts"]) == 3

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 16 - Reset
        # -------------------------------------------------------------------

        print("[TEST 16] Reset learner progress")

        learner.reset_progress()

        assert learner.get_profile() is None
        assert learner.get_all_topic_progress() == []
        assert learner.get_all_concepts() == []

        print("PASS")

        # -------------------------------------------------------------------
        # SUCCESS
        # -------------------------------------------------------------------

        print()
        print("=" * 70)
        print("ALL LEARNER PROFILE TESTS PASSED SUCCESSFULLY")
        print("=" * 70)

    finally:

        # Make sure all connections are closed by the DatabaseManager
        # before deleting the test file.
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