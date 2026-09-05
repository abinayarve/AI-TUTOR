"""
Visual Engine
=============

Subject-aware visual planning engine for the AI Teacher.

Responsibilities:
- Select appropriate visual types based on subject/concept.
- Create visual specifications for teaching.
- Generate visual prompts.
- Support equations, graphs, diagrams, processes, timelines,
  maps, code, flowcharts, simulations, tables, images and text.
- Work independently without requiring a Gemini API key.

The actual rendering of visuals can be handled later by the
video/Streamlit layer.
"""

from typing import Any, Dict, List, Optional


class VisualEngine:
    """
    Creates subject-aware visual plans for AI teaching.

    This module focuses on WHAT should be shown.
    Rendering is intentionally separated from planning.
    """

    # ---------------------------------------------------------------
    # Subject -> preferred visual types
    # ---------------------------------------------------------------

    SUBJECT_VISUALS = {
        "mathematics": [
            "equation",
            "graph",
            "diagram",
            "table",
            "text",
        ],
        "math": [
            "equation",
            "graph",
            "diagram",
            "table",
            "text",
        ],
        "physics": [
            "diagram",
            "equation",
            "graph",
            "process",
            "simulation",
        ],
        "chemistry": [
            "diagram",
            "process",
            "equation",
            "table",
            "image",
        ],
        "biology": [
            "diagram",
            "process",
            "image",
            "table",
            "text",
        ],
        "history": [
            "timeline",
            "map",
            "image",
            "table",
            "text",
        ],
        "geography": [
            "map",
            "diagram",
            "graph",
            "image",
            "table",
        ],
        "computer science": [
            "code",
            "flowchart",
            "diagram",
            "table",
            "text",
        ],
        "programming": [
            "code",
            "output",
            "flowchart",
            "diagram",
            "text",
        ],
        "machine learning": [
            "diagram",
            "graph",
            "flowchart",
            "code",
            "table",
        ],
        "artificial intelligence": [
            "diagram",
            "flowchart",
            "code",
            "graph",
            "image",
        ],
        "ai": [
            "diagram",
            "flowchart",
            "code",
            "graph",
            "image",
        ],
        "data science": [
            "graph",
            "table",
            "diagram",
            "flowchart",
            "code",
        ],
        "statistics": [
            "graph",
            "equation",
            "table",
            "diagram",
            "text",
        ],
        "english": [
            "text",
            "diagram",
            "table",
            "image",
        ],
        "language": [
            "text",
            "table",
            "diagram",
            "image",
        ],
        "general": [
            "diagram",
            "image",
            "table",
            "text",
        ],
    }

    # ---------------------------------------------------------------
    # Visual descriptions
    # ---------------------------------------------------------------

    VISUAL_DESCRIPTIONS = {
        "equation": "Mathematical or scientific equation with clearly labeled variables.",
        "graph": "Graph or chart showing relationships, trends, or numerical behavior.",
        "diagram": "Labeled conceptual diagram showing relationships between components.",
        "process": "Step-by-step process or mechanism with arrows and labels.",
        "timeline": "Chronological timeline showing important events and dates.",
        "map": "Map-based visual showing relevant geographic locations or movement.",
        "code": "Readable code example with syntax highlighting and explanatory labels.",
        "output": "Expected programming output or result displayed clearly.",
        "flowchart": "Flowchart showing algorithmic or logical decision flow.",
        "simulation": "Interactive or simulation-style representation of a physical or scientific process.",
        "table": "Structured comparison or data table.",
        "image": "Relevant illustrative image representing the concept.",
        "text": "Concise on-screen teaching text with key points.",
    }

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the visual engine.

        api_key is optional and reserved for future LLM-assisted
        visual planning.
        """

        self.api_key = api_key
        self.llm = None

        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.2,
                    max_output_tokens=2048,
                )
            except Exception:
                self.llm = None

    # ---------------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------------

    @staticmethod
    def _normalize(value: Any) -> str:
        """Normalize text for matching."""
        if value is None:
            return ""

        return str(value).strip().lower()

    # ---------------------------------------------------------------
    # Subject detection
    # ---------------------------------------------------------------

    def detect_subject(
        self,
        topic: str,
        subject: Optional[str] = None,
    ) -> str:
        """
        Detect the most likely subject.

        Explicit subject takes priority.
        """

        if subject and str(subject).strip():
            return str(subject).strip().lower()

        topic_text = self._normalize(topic)

        keyword_groups = {
            "mathematics": [
                "math",
                "mathematics",
                "algebra",
                "calculus",
                "geometry",
                "trigonometry",
                "probability",
                "permutation",
                "combination",
                "equation",
                "quadratic",
                "derivative",
                "integral",
            ],
            "physics": [
                "physics",
                "motion",
                "velocity",
                "acceleration",
                "force",
                "energy",
                "momentum",
                "electricity",
                "magnetism",
                "optics",
                "thermodynamics",
            ],
            "chemistry": [
                "chemistry",
                "atom",
                "molecule",
                "reaction",
                "periodic table",
                "bond",
                "acid",
                "base",
                "organic chemistry",
            ],
            "biology": [
                "biology",
                "cell",
                "dna",
                "rna",
                "gene",
                "genetics",
                "photosynthesis",
                "respiration",
                "ecosystem",
                "human body",
            ],
            "history": [
                "history",
                "war",
                "empire",
                "civilization",
                "revolution",
                "independence",
                "kingdom",
                "dynasty",
            ],
            "geography": [
                "geography",
                "continent",
                "climate",
                "river",
                "mountain",
                "population",
                "latitude",
                "longitude",
                "map",
            ],
            "programming": [
                "programming",
                "python",
                "java",
                "c++",
                "javascript",
                "algorithm",
                "function",
                "recursion",
                "array",
                "linked list",
                "database",
                "sql",
                "coding",
            ],
            "machine learning": [
                "machine learning",
                "supervised learning",
                "unsupervised learning",
                "classification",
                "regression",
                "clustering",
                "overfitting",
                "underfitting",
                "random forest",
                "decision tree",
                "xgboost",
            ],
            "artificial intelligence": [
                "artificial intelligence",
                "ai",
                "deep learning",
                "neural network",
                "llm",
                "large language model",
                "generative ai",
                "computer vision",
                "nlp",
                "natural language processing",
            ],
            "statistics": [
                "statistics",
                "mean",
                "median",
                "mode",
                "variance",
                "standard deviation",
                "hypothesis testing",
                "distribution",
                "correlation",
            ],
        }

        for detected_subject, keywords in keyword_groups.items():
            for keyword in keywords:
                if keyword in topic_text:
                    return detected_subject

        return "general"

    # ---------------------------------------------------------------
    # Visual type selection
    # ---------------------------------------------------------------

    def get_visual_types(
        self,
        subject: str,
    ) -> List[str]:
        """
        Return preferred visual types for a subject.
        """

        normalized_subject = self._normalize(subject)

        if normalized_subject in self.SUBJECT_VISUALS:
            return list(self.SUBJECT_VISUALS[normalized_subject])

        # Handle partial subject matches.
        for key, visuals in self.SUBJECT_VISUALS.items():
            if key in normalized_subject or normalized_subject in key:
                return list(visuals)

        return list(self.SUBJECT_VISUALS["general"])

    # ---------------------------------------------------------------
    # Best visual selection
    # ---------------------------------------------------------------

    def select_visual_type(
        self,
        topic: str,
        concept: str = "",
        subject: Optional[str] = None,
        requested_type: Optional[str] = None,
    ) -> str:
        """
        Select the most appropriate visual type.
        """

        detected_subject = self.detect_subject(
            topic=topic,
            subject=subject,
        )

        available = self.get_visual_types(detected_subject)

        if requested_type:
            requested = self._normalize(requested_type)

            aliases = {
                "formula": "equation",
                "chart": "graph",
                "algorithm": "flowchart",
                "program": "code",
                "coding": "code",
                "sequence": "process",
                "historical timeline": "timeline",
                "geographical map": "map",
            }

            requested = aliases.get(requested, requested)

            if requested in available:
                return requested

        topic_text = self._normalize(
            f"{topic} {concept}"
        )

        # Strong semantic keyword rules.
        if any(
            word in topic_text
            for word in [
                "equation",
                "formula",
                "calculate",
                "algebra",
                "derivative",
                "integral",
            ]
        ):
            if "equation" in available:
                return "equation"

        if any(
            word in topic_text
            for word in [
                "graph",
                "trend",
                "distribution",
                "relationship",
                "correlation",
            ]
        ):
            if "graph" in available:
                return "graph"

        if any(
            word in topic_text
            for word in [
                "steps",
                "process",
                "mechanism",
                "cycle",
                "workflow",
            ]
        ):
            if "process" in available:
                return "process"

        if any(
            word in topic_text
            for word in [
                "algorithm",
                "decision",
                "flow",
                "logic",
            ]
        ):
            if "flowchart" in available:
                return "flowchart"

        if any(
            word in topic_text
            for word in [
                "code",
                "python",
                "java",
                "javascript",
                "programming",
                "sql",
            ]
        ):
            if "code" in available:
                return "code"

        if any(
            word in topic_text
            for word in [
                "history",
                "event",
                "war",
                "revolution",
                "dynasty",
            ]
        ):
            if "timeline" in available:
                return "timeline"

        if any(
            word in topic_text
            for word in [
                "location",
                "country",
                "state",
                "region",
                "geography",
            ]
        ):
            if "map" in available:
                return "map"

        return available[0] if available else "text"

    # ---------------------------------------------------------------
    # Visual specification
    # ---------------------------------------------------------------

    def create_visual_spec(
        self,
        topic: str,
        concept: str = "",
        subject: Optional[str] = None,
        visual_type: Optional[str] = None,
        explanation: str = "",
    ) -> Dict[str, Any]:
        """
        Create a structured visual specification.

        This specification can later be consumed by a renderer,
        image-generation model, plotting library, or video engine.
        """

        topic = str(topic or "").strip()
        concept = str(concept or "").strip()
        explanation = str(explanation or "").strip()

        detected_subject = self.detect_subject(
            topic=topic,
            subject=subject,
        )

        selected_type = self.select_visual_type(
            topic=topic,
            concept=concept,
            subject=detected_subject,
            requested_type=visual_type,
        )

        description = self.VISUAL_DESCRIPTIONS.get(
            selected_type,
            "Educational visual supporting the explanation.",
        )

        title = concept if concept else topic

        prompt = self.generate_visual_prompt(
            topic=topic,
            concept=concept,
            subject=detected_subject,
            visual_type=selected_type,
            explanation=explanation,
        )

        return {
            "topic": topic,
            "concept": concept,
            "subject": detected_subject,
            "visual_type": selected_type,
            "title": title,
            "description": description,
            "prompt": prompt,
            "educational_purpose": (
                f"Help the learner understand {title} "
                f"through a {selected_type}."
            ),
            "show_during_explanation": True,
            "requires_rendering": True,
        }

    # ---------------------------------------------------------------
    # Prompt generation
    # ---------------------------------------------------------------

    def generate_visual_prompt(
        self,
        topic: str,
        concept: str = "",
        subject: Optional[str] = None,
        visual_type: Optional[str] = None,
        explanation: str = "",
    ) -> str:
        """
        Generate a clear prompt describing the educational visual.
        """

        topic = str(topic or "").strip()
        concept = str(concept or "").strip()

        detected_subject = self.detect_subject(
            topic=topic,
            subject=subject,
        )

        selected_type = visual_type or self.select_visual_type(
            topic=topic,
            concept=concept,
            subject=detected_subject,
        )

        focus = concept if concept else topic

        base = (
            f"Create a clean educational {selected_type} "
            f"for teaching {focus} in {detected_subject}. "
        )

        type_instructions = {
            "equation": (
                "Show the equation clearly, define every variable, "
                "and highlight the relationship being explained."
            ),
            "graph": (
                "Use clearly labeled axes, meaningful values, "
                "a readable title, and visually emphasize the key trend."
            ),
            "diagram": (
                "Use clearly labeled components, arrows where needed, "
                "and a logical top-to-bottom or left-to-right structure."
            ),
            "process": (
                "Show the process as numbered sequential stages "
                "with arrows and concise labels."
            ),
            "timeline": (
                "Arrange important events chronologically with "
                "clear dates and short descriptions."
            ),
            "map": (
                "Use clear geographic labels and highlight the "
                "locations relevant to the lesson."
            ),
            "code": (
                "Show a short readable code example with syntax "
                "highlighting and annotations explaining the key lines."
            ),
            "output": (
                "Show the expected program output clearly and "
                "connect it to the corresponding code behavior."
            ),
            "flowchart": (
                "Use standard flowchart logic with clear decisions, "
                "branches, arrows, and concise labels."
            ),
            "simulation": (
                "Represent the physical or scientific process dynamically "
                "with clearly labeled variables and stages."
            ),
            "table": (
                "Create a concise comparison or data table with "
                "clear column headings and only essential information."
            ),
            "image": (
                "Use a scientifically or historically appropriate "
                "illustration with clear labels where useful."
            ),
            "text": (
                "Display only concise key points suitable for "
                "on-screen teaching."
            ),
        }

        prompt = base + type_instructions.get(
            selected_type,
            type_instructions["text"],
        )

        if explanation:
            prompt += (
                f" The visual should support this explanation: "
                f"{explanation[:500]}"
            )

        prompt += (
            " Keep the visual uncluttered, accurate, educational, "
            "and suitable for a student learning for the first time."
        )

        return prompt

    # ---------------------------------------------------------------
    # Lesson visual planning
    # ---------------------------------------------------------------

    def plan_lesson_visuals(
        self,
        topic: str,
        concepts: List[Any],
        subject: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create visual specifications for multiple lesson concepts.
        """

        plans = []

        if not concepts:
            concepts = [topic]

        for item in concepts:
            if isinstance(item, dict):
                concept = (
                    item.get("concept")
                    or item.get("concept_name")
                    or item.get("title")
                    or topic
                )

                explanation = (
                    item.get("explanation")
                    or item.get("script")
                    or item.get("spoken_script")
                    or ""
                )

                requested_type = (
                    item.get("visual_type")
                    or item.get("visual")
                    or None
                )

            else:
                concept = str(item)
                explanation = ""
                requested_type = None

            spec = self.create_visual_spec(
                topic=topic,
                concept=str(concept),
                subject=subject,
                visual_type=requested_type,
                explanation=explanation,
            )

            spec["sequence"] = len(plans) + 1
            plans.append(spec)

        return plans

    # ---------------------------------------------------------------
    # Visual validation
    # ---------------------------------------------------------------

    def validate_visual_spec(
        self,
        spec: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Validate a visual specification.
        """

        if not isinstance(spec, dict):
            return False

        required_fields = [
            "topic",
            "visual_type",
            "title",
            "description",
            "prompt",
        ]

        for field in required_fields:
            value = spec.get(field)

            if value is None:
                return False

            if isinstance(value, str) and not value.strip():
                return False

        return True

    # ---------------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------------

    def summarize_visual_plan(
        self,
        specs: List[Dict[str, Any]],
    ) -> str:
        """
        Produce a concise summary of the planned visuals.
        """

        if not specs:
            return "No visuals planned."

        lines = []

        for index, spec in enumerate(specs, start=1):
            visual_type = spec.get("visual_type", "text")
            title = spec.get("title", "Concept")

            lines.append(
                f"{index}. {visual_type.title()}: {title}"
            )

        return "\n".join(lines)


# ======================================================================
# TESTS
# ======================================================================

def run_tests():
    print("=" * 60)
    print("VISUAL ENGINE TESTS")
    print("=" * 60)

    engine = VisualEngine(api_key=None)

    # ---------------------------------------------------------------
    # TEST 1
    # ---------------------------------------------------------------
    print("\nTEST 1: Engine initialization")

    assert engine is not None
    assert engine.api_key is None

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 2
    # ---------------------------------------------------------------
    print("\nTEST 2: Mathematics subject detection")

    result = engine.detect_subject(
        "Quadratic equations"
    )

    assert result == "mathematics"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 3
    # ---------------------------------------------------------------
    print("\nTEST 3: Physics subject detection")

    result = engine.detect_subject(
        "Newton's laws of motion"
    )

    assert result == "physics"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 4
    # ---------------------------------------------------------------
    print("\nTEST 4: Programming subject detection")

    result = engine.detect_subject(
        "Python functions and recursion"
    )

    assert result == "programming"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 5
    # ---------------------------------------------------------------
    print("\nTEST 5: Machine learning visual types")

    result = engine.get_visual_types(
        "Machine Learning"
    )

    assert "diagram" in result
    assert "graph" in result
    assert "flowchart" in result

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 6
    # ---------------------------------------------------------------
    print("\nTEST 6: Equation visual selection")

    result = engine.select_visual_type(
        topic="Mathematics",
        concept="Quadratic equation formula",
    )

    assert result == "equation"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 7
    # ---------------------------------------------------------------
    print("\nTEST 7: Code visual selection")

    result = engine.select_visual_type(
        topic="Programming",
        concept="Python function code",
    )

    assert result == "code"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 8
    # ---------------------------------------------------------------
    print("\nTEST 8: Timeline visual selection")

    result = engine.select_visual_type(
        topic="History",
        concept="Major events of Indian independence",
    )

    assert result == "timeline"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 9
    # ---------------------------------------------------------------
    print("\nTEST 9: Diagram specification")

    result = engine.create_visual_spec(
        topic="Machine Learning",
        concept="Supervised Learning",
    )

    assert result["subject"] == "machine learning"
    assert result["visual_type"] == "diagram"
    assert result["title"] == "Supervised Learning"
    assert result["prompt"]

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 10
    # ---------------------------------------------------------------
    print("\nTEST 10: Visual specification validation")

    result = engine.create_visual_spec(
        topic="Physics",
        concept="Newton's Second Law",
        visual_type="equation",
    )

    assert engine.validate_visual_spec(result) is True

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 11
    # ---------------------------------------------------------------
    print("\nTEST 11: Invalid visual specification")

    result = engine.validate_visual_spec(
        {
            "topic": "Physics",
        }
    )

    assert result is False

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 12
    # ---------------------------------------------------------------
    print("\nTEST 12: Multiple lesson visuals")

    result = engine.plan_lesson_visuals(
        topic="Machine Learning",
        concepts=[
            "Supervised Learning",
            "Classification",
            "Overfitting",
        ],
    )

    assert isinstance(result, list)
    assert len(result) == 3

    for index, spec in enumerate(result, start=1):
        assert spec["sequence"] == index
        assert engine.validate_visual_spec(spec)

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 13
    # ---------------------------------------------------------------
    print("\nTEST 13: Dictionary concept support")

    result = engine.plan_lesson_visuals(
        topic="Python Programming",
        concepts=[
            {
                "concept": "Functions",
                "explanation": "A function is a reusable block of code.",
                "visual_type": "flowchart",
            }
        ],
    )

    assert len(result) == 1
    assert result[0]["concept"] == "Functions"
    assert result[0]["visual_type"] == "flowchart"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 14
    # ---------------------------------------------------------------
    print("\nTEST 14: Visual prompt generation")

    result = engine.generate_visual_prompt(
        topic="Physics",
        concept="Newton's Second Law",
        visual_type="equation",
    )

    assert "Newton's Second Law" in result
    assert "equation" in result.lower()

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 15
    # ---------------------------------------------------------------
    print("\nTEST 15: Visual plan summary")

    specs = engine.plan_lesson_visuals(
        topic="Machine Learning",
        concepts=[
            "Classification",
            "Overfitting",
        ],
    )

    result = engine.summarize_visual_plan(specs)

    assert isinstance(result, str)
    assert "Classification" in result
    assert "Overfitting" in result

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 16
    # ---------------------------------------------------------------
    print("\nTEST 16: Explicit visual type override")

    result = engine.select_visual_type(
        topic="Machine Learning",
        concept="Classification",
        requested_type="graph",
    )

    assert result == "graph"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 17
    # ---------------------------------------------------------------
    print("\nTEST 17: Unknown subject fallback")

    result = engine.get_visual_types(
        "Astronomy"
    )

    assert isinstance(result, list)
    assert len(result) > 0

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 18
    # ---------------------------------------------------------------
    print("\nTEST 18: Empty concept handling")

    result = engine.create_visual_spec(
        topic="Photosynthesis",
        concept="",
    )

    assert result["title"] == "Photosynthesis"
    assert result["prompt"]

    print("PASS")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL VISUAL ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()