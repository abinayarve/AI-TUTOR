"""
database.py
-----------
Persistent SQLite database layer for the AI Teacher project.

Stores:
- Learner profiles
- Topic progress
- Concept progress
- Assessment history
- Learning events

This module is intentionally independent from the rest of the application
so it can be tested before integration.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# DATABASE PATH
# ---------------------------------------------------------------------------

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_DATABASE_PATH = os.path.join(DATA_DIR, "ai_teacher.db")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _now() -> str:
    """Return current UTC timestamp in ISO format."""
    from datetime import timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    


def _json_dumps(value: Any) -> Optional[str]:
    """Safely convert a Python object to JSON."""
    if value is None:
        return None

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _json_loads(value: Optional[str]) -> Any:
    """Safely convert JSON string back to Python object."""
    if value is None:
        return None

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


# ---------------------------------------------------------------------------
# DATABASE MANAGER
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    SQLite database manager for persistent AI Teacher learning data.

    Parameters
    ----------
    db_path:
        Path to SQLite database file.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DATABASE_PATH

        # Make sure the parent directory exists.
        parent_dir = os.path.dirname(os.path.abspath(self.db_path))

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.initialize()

    # -----------------------------------------------------------------------
    # CONNECTION
    # -----------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a SQLite connection."""
        connection = sqlite3.connect(self.db_path)

        # Return rows that can behave like dictionaries.
        connection.row_factory = sqlite3.Row

        # Enable foreign-key constraints.
        connection.execute("PRAGMA foreign_keys = ON")

        return connection

    # -----------------------------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all required database tables."""
        with self.get_connection() as connection:

            # ---------------------------------------------------------------
            # Learner profiles
            # ---------------------------------------------------------------
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_profiles (
                    learner_id TEXT PRIMARY KEY,
                    name TEXT,
                    level TEXT,
                    goal TEXT,
                    language TEXT,
                    learning_style TEXT,
                    preferred_depth TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # ---------------------------------------------------------------
            # Topic-level progress
            # ---------------------------------------------------------------
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    mastery REAL DEFAULT 0.0,
                    score REAL DEFAULT 0.0,
                    attempts INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'not_started',
                    difficulty TEXT DEFAULT 'MEDIUM',
                    last_studied TEXT,
                    updated_at TEXT NOT NULL,

                    UNIQUE(learner_id, topic),

                    FOREIGN KEY(learner_id)
                        REFERENCES learner_profiles(learner_id)
                        ON DELETE CASCADE
                )
                """
            )

            # ---------------------------------------------------------------
            # Concept-level progress
            # ---------------------------------------------------------------
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    mastery REAL DEFAULT 0.0,
                    attempts INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    incorrect_count INTEGER DEFAULT 0,
                    misconception TEXT,
                    status TEXT DEFAULT 'not_started',
                    updated_at TEXT NOT NULL,

                    UNIQUE(learner_id, topic, concept),

                    FOREIGN KEY(learner_id)
                        REFERENCES learner_profiles(learner_id)
                        ON DELETE CASCADE
                )
                """
            )

            # ---------------------------------------------------------------
            # Assessment history
            # ---------------------------------------------------------------
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    concept TEXT,
                    question_type TEXT,
                    question TEXT,
                    student_answer TEXT,
                    correctness TEXT,
                    score REAL DEFAULT 0.0,
                    mastery REAL DEFAULT 0.0,
                    misconception TEXT,
                    recommended_action TEXT,
                    feedback TEXT,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY(learner_id)
                        REFERENCES learner_profiles(learner_id)
                        ON DELETE CASCADE
                )
                """
            )

            # ---------------------------------------------------------------
            # Learning events
            # ---------------------------------------------------------------
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    topic TEXT,
                    concept TEXT,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY(learner_id)
                        REFERENCES learner_profiles(learner_id)
                        ON DELETE CASCADE
                )
                """
            )

            # ---------------------------------------------------------------
            # Useful indexes
            # ---------------------------------------------------------------

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic_progress_learner
                ON topic_progress(learner_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_concept_progress_learner_topic
                ON concept_progress(learner_id, topic)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_assessment_learner_topic
                ON assessment_history(learner_id, topic)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_learner
                ON learning_events(learner_id)
                """
            )

    # -----------------------------------------------------------------------
    # GENERIC DATABASE METHODS
    # -----------------------------------------------------------------------

    def execute(
        self,
        query: str,
        parameters: tuple = ()
    ) -> int:
        """
        Execute a SQL statement.

        Returns
        -------
        int
            Last inserted row ID.
        """
        with self.get_connection() as connection:
            cursor = connection.execute(query, parameters)
            return cursor.lastrowid

    def fetch_one(
        self,
        query: str,
        parameters: tuple = ()
    ) -> Optional[Dict[str, Any]]:
        """Execute a query and return one row as a dictionary."""
        with self.get_connection() as connection:
            cursor = connection.execute(query, parameters)
            row = cursor.fetchone()

            if row is None:
                return None

            return dict(row)

    def fetch_all(
        self,
        query: str,
        parameters: tuple = ()
    ) -> List[Dict[str, Any]]:
        """Execute a query and return all rows as dictionaries."""
        with self.get_connection() as connection:
            cursor = connection.execute(query, parameters)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    # -----------------------------------------------------------------------
    # LEARNER PROFILE
    # -----------------------------------------------------------------------

    def upsert_learner_profile(
        self,
        learner_id: str,
        name: str = "",
        level: str = "",
        goal: str = "",
        language: str = "English",
        learning_style: str = "",
        preferred_depth: str = "",
    ) -> Dict[str, Any]:
        """
        Insert or update a learner profile.
        """

        now = _now()

        query = """
            INSERT INTO learner_profiles (
                learner_id,
                name,
                level,
                goal,
                language,
                learning_style,
                preferred_depth,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(learner_id)
            DO UPDATE SET
                name = excluded.name,
                level = excluded.level,
                goal = excluded.goal,
                language = excluded.language,
                learning_style = excluded.learning_style,
                preferred_depth = excluded.preferred_depth,
                updated_at = excluded.updated_at
        """

        with self.get_connection() as connection:
            connection.execute(
                query,
                (
                    learner_id,
                    name,
                    level,
                    goal,
                    language,
                    learning_style,
                    preferred_depth,
                    now,
                    now,
                ),
            )

        return self.get_learner_profile(learner_id)

    def get_learner_profile(
        self,
        learner_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a learner profile."""
        return self.fetch_one(
            """
            SELECT *
            FROM learner_profiles
            WHERE learner_id = ?
            """,
            (learner_id,),
        )

    # -----------------------------------------------------------------------
    # TOPIC PROGRESS
    # -----------------------------------------------------------------------

    def upsert_topic_progress(
        self,
        learner_id: str,
        topic: str,
        mastery: float = 0.0,
        score: float = 0.0,
        attempts: int = 0,
        status: str = "not_started",
        difficulty: str = "MEDIUM",
        last_studied: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Insert or update progress for a topic.
        """

        now = _now()
        last_studied = last_studied or now

        query = """
            INSERT INTO topic_progress (
                learner_id,
                topic,
                mastery,
                score,
                attempts,
                status,
                difficulty,
                last_studied,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(learner_id, topic)
            DO UPDATE SET
                mastery = excluded.mastery,
                score = excluded.score,
                attempts = excluded.attempts,
                status = excluded.status,
                difficulty = excluded.difficulty,
                last_studied = excluded.last_studied,
                updated_at = excluded.updated_at
        """

        with self.get_connection() as connection:
            connection.execute(
                query,
                (
                    learner_id,
                    topic,
                    float(mastery),
                    float(score),
                    int(attempts),
                    status,
                    difficulty,
                    last_studied,
                    now,
                ),
            )

        result = self.get_topic_progress(learner_id, topic)

        if result is None:
            raise RuntimeError("Failed to save topic progress.")

        return result

    def get_topic_progress(
        self,
        learner_id: str,
        topic: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve progress for one topic."""
        return self.fetch_one(
            """
            SELECT *
            FROM topic_progress
            WHERE learner_id = ?
              AND topic = ?
            """,
            (learner_id, topic),
        )

    def get_all_topic_progress(
        self,
        learner_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve all topic progress records for a learner."""
        return self.fetch_all(
            """
            SELECT *
            FROM topic_progress
            WHERE learner_id = ?
            ORDER BY updated_at DESC
            """,
            (learner_id,),
        )

    # -----------------------------------------------------------------------
    # CONCEPT PROGRESS
    # -----------------------------------------------------------------------

    def upsert_concept_progress(
        self,
        learner_id: str,
        topic: str,
        concept: str,
        mastery: float = 0.0,
        attempts: int = 0,
        correct_count: int = 0,
        incorrect_count: int = 0,
        misconception: Optional[str] = None,
        status: str = "not_started",
    ) -> Dict[str, Any]:
        """
        Insert or update progress for a concept.
        """

        now = _now()

        query = """
            INSERT INTO concept_progress (
                learner_id,
                topic,
                concept,
                mastery,
                attempts,
                correct_count,
                incorrect_count,
                misconception,
                status,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(learner_id, topic, concept)
            DO UPDATE SET
                mastery = excluded.mastery,
                attempts = excluded.attempts,
                correct_count = excluded.correct_count,
                incorrect_count = excluded.incorrect_count,
                misconception = excluded.misconception,
                status = excluded.status,
                updated_at = excluded.updated_at
        """

        with self.get_connection() as connection:
            connection.execute(
                query,
                (
                    learner_id,
                    topic,
                    concept,
                    float(mastery),
                    int(attempts),
                    int(correct_count),
                    int(incorrect_count),
                    misconception,
                    status,
                    now,
                ),
            )

        result = self.get_concept_progress(
            learner_id,
            topic,
            concept,
        )

        if result is None:
            raise RuntimeError("Failed to save concept progress.")

        return result

    def get_concept_progress(
        self,
        learner_id: str,
        topic: str,
        concept: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve one concept's progress."""
        return self.fetch_one(
            """
            SELECT *
            FROM concept_progress
            WHERE learner_id = ?
              AND topic = ?
              AND concept = ?
            """,
            (learner_id, topic, concept),
        )

    def get_all_concept_progress(
        self,
        learner_id: str,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve concept progress for a learner."""
        if topic is not None:
            return self.fetch_all(
                """
                SELECT *
                FROM concept_progress
                WHERE learner_id = ?
                  AND topic = ?
                ORDER BY mastery ASC, updated_at DESC
                """,
                (learner_id, topic),
            )

        return self.fetch_all(
            """
            SELECT *
            FROM concept_progress
            WHERE learner_id = ?
            ORDER BY mastery ASC, updated_at DESC
            """,
            (learner_id,),
        )

    # -----------------------------------------------------------------------
    # ASSESSMENT HISTORY
    # -----------------------------------------------------------------------

    def record_assessment(
        self,
        learner_id: str,
        topic: str,
        concept: Optional[str] = None,
        question_type: Optional[str] = None,
        question: Optional[str] = None,
        student_answer: Optional[str] = None,
        correctness: Optional[str] = None,
        score: float = 0.0,
        mastery: float = 0.0,
        misconception: Optional[str] = None,
        recommended_action: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> int:
        """Record one assessment attempt."""

        now = _now()

        return self.execute(
            """
            INSERT INTO assessment_history (
                learner_id,
                topic,
                concept,
                question_type,
                question,
                student_answer,
                correctness,
                score,
                mastery,
                misconception,
                recommended_action,
                feedback,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                topic,
                concept,
                question_type,
                question,
                student_answer,
                correctness,
                float(score),
                float(mastery),
                misconception,
                recommended_action,
                feedback,
                now,
            ),
        )

    def get_assessment_history(
        self,
        learner_id: str,
        topic: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent assessment history."""

        limit = max(1, int(limit))

        if topic is not None:
            return self.fetch_all(
                f"""
                SELECT *
                FROM assessment_history
                WHERE learner_id = ?
                  AND topic = ?
                ORDER BY created_at DESC
                LIMIT {limit}
                """,
                (learner_id, topic),
            )

        return self.fetch_all(
            f"""
            SELECT *
            FROM assessment_history
            WHERE learner_id = ?
            ORDER BY created_at DESC
            LIMIT {limit}
            """,
            (learner_id,),
        )

    # -----------------------------------------------------------------------
    # LEARNING EVENTS
    # -----------------------------------------------------------------------

    def record_learning_event(
        self,
        learner_id: str,
        event_type: str,
        topic: Optional[str] = None,
        concept: Optional[str] = None,
        details: Optional[Any] = None,
    ) -> int:
        """
        Record a learning event.

        Examples:
        - lesson_started
        - lesson_completed
        - question_answered
        - misconception_detected
        - remediation_started
        - remediation_completed
        - difficulty_increased
        - language_changed
        """

        now = _now()

        return self.execute(
            """
            INSERT INTO learning_events (
                learner_id,
                topic,
                concept,
                event_type,
                details,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                topic,
                concept,
                event_type,
                _json_dumps(details),
                now,
            ),
        )

    def get_learning_events(
        self,
        learner_id: str,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve learning events."""

        limit = max(1, int(limit))

        if topic is not None:
            rows = self.fetch_all(
                f"""
                SELECT *
                FROM learning_events
                WHERE learner_id = ?
                  AND topic = ?
                ORDER BY created_at DESC
                LIMIT {limit}
                """,
                (learner_id, topic),
            )
        else:
            rows = self.fetch_all(
                f"""
                SELECT *
                FROM learning_events
                WHERE learner_id = ?
                ORDER BY created_at DESC
                LIMIT {limit}
                """,
                (learner_id,),
            )

        # Convert JSON details back into Python objects.
        for row in rows:
            row["details"] = _json_loads(row.get("details"))

        return rows

    # -----------------------------------------------------------------------
    # LEARNER SUMMARY
    # -----------------------------------------------------------------------

    def get_learner_summary(
        self,
        learner_id: str
    ) -> Dict[str, Any]:
        """
        Return a compact summary of a learner's persistent progress.
        """

        profile = self.get_learner_profile(learner_id)
        topics = self.get_all_topic_progress(learner_id)
        concepts = self.get_all_concept_progress(learner_id)
        assessments = self.get_assessment_history(
            learner_id,
            limit=20,
        )

        strong_concepts = [
            item
            for item in concepts
            if float(item.get("mastery", 0.0)) >= 0.85
        ]

        weak_concepts = [
            item
            for item in concepts
            if float(item.get("mastery", 0.0)) < 0.50
        ]

        active_topics = [
            item
            for item in topics
            if item.get("status") != "completed"
        ]

        return {
            "learner_id": learner_id,
            "profile": profile,
            "topics": topics,
            "concepts": concepts,
            "recent_assessments": assessments,
            "strong_concepts": strong_concepts,
            "weak_concepts": weak_concepts,
            "active_topics": active_topics,
        }

    # -----------------------------------------------------------------------
    # DELETE / RESET
    # -----------------------------------------------------------------------

    def clear_learner_data(
        self,
        learner_id: str
    ) -> None:
        """
        Delete all persistent data belonging to one learner.

        Useful for testing and for future "Reset Learning Progress" UI.
        """

        with self.get_connection() as connection:

            # Explicit deletion keeps this safe even if foreign-key
            # behavior changes in the future.
            connection.execute(
                """
                DELETE FROM learning_events
                WHERE learner_id = ?
                """,
                (learner_id,),
            )

            connection.execute(
                """
                DELETE FROM assessment_history
                WHERE learner_id = ?
                """,
                (learner_id,),
            )

            connection.execute(
                """
                DELETE FROM concept_progress
                WHERE learner_id = ?
                """,
                (learner_id,),
            )

            connection.execute(
                """
                DELETE FROM topic_progress
                WHERE learner_id = ?
                """,
                (learner_id,),
            )

            connection.execute(
                """
                DELETE FROM learner_profiles
                WHERE learner_id = ?
                """,
                (learner_id,),
            )


# ---------------------------------------------------------------------------
# LOCAL TESTS
# ---------------------------------------------------------------------------

def run_local_tests() -> None:
    """
    Run database tests using a temporary SQLite database.

    No Gemini API key is required.
    """

    print("=" * 70)
    print("DATABASE MANAGER - LOCAL TEST")
    print("=" * 70)

    test_db_path = os.path.join(
        DATA_DIR,
        "test_ai_teacher.db",
    )

    # Remove previous test database.
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = DatabaseManager(test_db_path)

    learner_id = "test_learner_001"
    topic = "Ohm's Law"
    concept = "Voltage Current Resistance Relationship"

    try:

        # -------------------------------------------------------------------
        # TEST 1 - Database initialization
        # -------------------------------------------------------------------

        print("\n[TEST 1] Database initialization")

        assert os.path.exists(test_db_path)

        tables = db.fetch_all(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )

        table_names = {
            table["name"]
            for table in tables
        }

        required_tables = {
            "learner_profiles",
            "topic_progress",
            "concept_progress",
            "assessment_history",
            "learning_events",
        }

        assert required_tables.issubset(table_names)

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 2 - Learner profile insert
        # -------------------------------------------------------------------

        print("[TEST 2] Learner profile insert")

        profile = db.upsert_learner_profile(
            learner_id=learner_id,
            name="Test Student",
            level="Intermediate",
            goal="Exam Preparation",
            language="English",
            learning_style="Visual",
            preferred_depth="Detailed",
        )

        assert profile is not None
        assert profile["learner_id"] == learner_id
        assert profile["name"] == "Test Student"
        assert profile["level"] == "Intermediate"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 3 - Learner profile retrieval
        # -------------------------------------------------------------------

        print("[TEST 3] Learner profile retrieval")

        retrieved_profile = db.get_learner_profile(learner_id)

        assert retrieved_profile is not None
        assert retrieved_profile["goal"] == "Exam Preparation"
        assert retrieved_profile["language"] == "English"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 4 - Learner profile update
        # -------------------------------------------------------------------

        print("[TEST 4] Learner profile update")

        updated_profile = db.upsert_learner_profile(
            learner_id=learner_id,
            name="Test Student Updated",
            level="Advanced",
            goal="Competitive Exam",
            language="Tamil",
            learning_style="Interactive",
            preferred_depth="Deep",
        )

        assert updated_profile["name"] == "Test Student Updated"
        assert updated_profile["level"] == "Advanced"
        assert updated_profile["language"] == "Tamil"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 5 - Topic progress
        # -------------------------------------------------------------------

        print("[TEST 5] Topic progress insert")

        topic_record = db.upsert_topic_progress(
            learner_id=learner_id,
            topic=topic,
            mastery=0.60,
            score=60.0,
            attempts=2,
            status="in_progress",
            difficulty="MEDIUM",
        )

        assert topic_record["topic"] == topic
        assert abs(topic_record["mastery"] - 0.60) < 0.001
        assert topic_record["attempts"] == 2

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 6 - Topic progress update
        # -------------------------------------------------------------------

        print("[TEST 6] Topic progress update")

        topic_record = db.upsert_topic_progress(
            learner_id=learner_id,
            topic=topic,
            mastery=0.82,
            score=82.0,
            attempts=4,
            status="in_progress",
            difficulty="HARD",
        )

        assert abs(topic_record["mastery"] - 0.82) < 0.001
        assert topic_record["attempts"] == 4
        assert topic_record["difficulty"] == "HARD"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 7 - Concept progress
        # -------------------------------------------------------------------

        print("[TEST 7] Concept progress")

        concept_record = db.upsert_concept_progress(
            learner_id=learner_id,
            topic=topic,
            concept=concept,
            mastery=0.45,
            attempts=3,
            correct_count=1,
            incorrect_count=2,
            misconception="CONCEPTUAL_GAP",
            status="needs_remediation",
        )

        assert concept_record["concept"] == concept
        assert abs(concept_record["mastery"] - 0.45) < 0.001
        assert concept_record["incorrect_count"] == 2
        assert concept_record["misconception"] == "CONCEPTUAL_GAP"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 8 - Multiple concept retrieval
        # -------------------------------------------------------------------

        print("[TEST 8] Multiple concept retrieval")

        db.upsert_concept_progress(
            learner_id=learner_id,
            topic=topic,
            concept="Ohm's Law Formula",
            mastery=0.92,
            attempts=2,
            correct_count=2,
            incorrect_count=0,
            misconception=None,
            status="mastered",
        )

        concepts = db.get_all_concept_progress(
            learner_id,
            topic,
        )

        assert len(concepts) == 2

        # Lowest mastery should appear first.
        assert concepts[0]["mastery"] <= concepts[1]["mastery"]

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 9 - Assessment history
        # -------------------------------------------------------------------

        print("[TEST 9] Assessment history")

        assessment_id = db.record_assessment(
            learner_id=learner_id,
            topic=topic,
            concept=concept,
            question_type="MCQ",
            question="What happens to current when voltage increases?",
            student_answer="Current increases",
            correctness="correct",
            score=1.0,
            mastery=0.82,
            misconception="UNKNOWN",
            recommended_action="CONTINUE",
            feedback="Correct. Current increases when resistance is constant.",
        )

        assert isinstance(assessment_id, int)
        assert assessment_id > 0

        history = db.get_assessment_history(
            learner_id,
            topic,
        )

        assert len(history) == 1
        assert history[0]["correctness"] == "correct"

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 10 - Learning event with JSON details
        # -------------------------------------------------------------------

        print("[TEST 10] Learning event")

        event_id = db.record_learning_event(
            learner_id=learner_id,
            topic=topic,
            concept=concept,
            event_type="misconception_detected",
            details={
                "type": "CONCEPTUAL_GAP",
                "severity": "medium",
                "attempt": 1,
            },
        )

        assert isinstance(event_id, int)
        assert event_id > 0

        events = db.get_learning_events(
            learner_id,
            topic,
        )

        assert len(events) == 1
        assert events[0]["event_type"] == "misconception_detected"
        assert events[0]["details"]["type"] == "CONCEPTUAL_GAP"
        assert events[0]["details"]["attempt"] == 1

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 11 - Learner summary
        # -------------------------------------------------------------------

        print("[TEST 11] Learner summary")

        summary = db.get_learner_summary(learner_id)

        assert summary["learner_id"] == learner_id
        assert summary["profile"] is not None
        assert len(summary["topics"]) == 1
        assert len(summary["concepts"]) == 2
        assert len(summary["recent_assessments"]) == 1
        assert len(summary["strong_concepts"]) == 1
        assert len(summary["weak_concepts"]) == 1

        print("PASS")

        # -------------------------------------------------------------------
        # TEST 12 - Clear learner data
        # -------------------------------------------------------------------

        print("[TEST 12] Clear learner data")

        db.clear_learner_data(learner_id)

        assert db.get_learner_profile(learner_id) is None
        assert db.get_all_topic_progress(learner_id) == []
        assert db.get_all_concept_progress(learner_id) == []
        assert db.get_assessment_history(learner_id) == []
        assert db.get_learning_events(learner_id) == []

        print("PASS")

        # -------------------------------------------------------------------
        # SUCCESS
        # -------------------------------------------------------------------

        print()
        print("=" * 70)
        print("ALL DATABASE TESTS PASSED SUCCESSFULLY")
        print("=" * 70)

    finally:
        # Remove test database so the real application database remains clean.
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except PermissionError:
                print(
                    "\nWarning: Could not remove the test database because "
                    "another process is using it."
                )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_local_tests()