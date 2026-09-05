"""
Video Engine
============

Educational video planning and rendering engine for the AI Teacher.

Responsibilities:
- Convert lesson modules into educational video scenes.
- Combine narration, visuals, text, and avatar instructions.
- Create scene timelines.
- Validate video scene specifications.
- Detect rendering capabilities.
- Provide a rendering interface that can later be connected
  to FFmpeg, MoviePy, avatar generation, or external video APIs.

The independent test suite does NOT require:
- Gemini API
- API keys
- FFmpeg
- External video services
"""

import os
import shutil
from typing import Any, Dict, List, Optional


class VideoEngine:
    """
    Builds structured educational video plans.

    The engine separates:
        1. Teaching content
        2. Scene planning
        3. Rendering

    This makes the architecture easier to extend later.
    """

    DEFAULT_SCENE_DURATION = 8

    SCENE_TYPES = [
        "intro",
        "explanation",
        "demonstration",
        "visual",
        "question",
        "evaluation",
        "remediation",
        "recap",
        "outro",
    ]

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------

    def __init__(
        self,
        output_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data",
                "videos",
            )

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        self.ffmpeg_path = shutil.which("ffmpeg")

    # ---------------------------------------------------------------
    # Capability detection
    # ---------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Detect available rendering capabilities.
        """

        return {
            "ffmpeg": self.ffmpeg_path is not None,
            "ffmpeg_path": self.ffmpeg_path,
            "output_directory": self.output_dir,
            "can_render_basic_video": self.ffmpeg_path is not None,
            "api_key_available": bool(self.api_key),
        }

    # ---------------------------------------------------------------
    # Duration estimation
    # ---------------------------------------------------------------

    @staticmethod
    def estimate_duration(
        text: str,
        words_per_minute: int = 145,
    ) -> float:
        """
        Estimate narration duration in seconds.

        Uses an average educational speaking speed.
        """

        text = str(text or "").strip()

        if not text:
            return 0.0

        words = len(text.split())

        if words_per_minute <= 0:
            words_per_minute = 145

        minutes = words / words_per_minute

        return round(minutes * 60, 2)

    # ---------------------------------------------------------------
    # Scene duration
    # ---------------------------------------------------------------

    def calculate_scene_duration(
        self,
        narration: str = "",
        minimum_duration: float = 3.0,
        maximum_duration: float = 60.0,
    ) -> float:
        """
        Calculate a reasonable scene duration from narration.
        """

        estimated = self.estimate_duration(narration)

        if estimated <= 0:
            estimated = self.DEFAULT_SCENE_DURATION

        estimated = max(
            minimum_duration,
            estimated,
        )

        estimated = min(
            maximum_duration,
            estimated,
        )

        return round(estimated, 2)

    # ---------------------------------------------------------------
    # Scene creation
    # ---------------------------------------------------------------

    def create_scene(
        self,
        scene_type: str,
        title: str,
        narration: str = "",
        visual: Optional[Dict[str, Any]] = None,
        on_screen_text: Optional[List[str]] = None,
        avatar_action: str = "speaking",
        duration: Optional[float] = None,
        question: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a single educational video scene.
        """

        scene_type = str(scene_type or "explanation").strip().lower()
        title = str(title or "Lesson").strip()
        narration = str(narration or "").strip()

        if scene_type not in self.SCENE_TYPES:
            scene_type = "explanation"

        if duration is None:
            duration = self.calculate_scene_duration(
                narration=narration
            )

        if on_screen_text is None:
            on_screen_text = []

        if not isinstance(on_screen_text, list):
            on_screen_text = [str(on_screen_text)]

        scene = {
            "scene_type": scene_type,
            "title": title,
            "narration": narration,
            "visual": visual or {},
            "on_screen_text": on_screen_text,
            "avatar_action": avatar_action,
            "duration": float(duration),
            "question": question,
        }

        return scene

    # ---------------------------------------------------------------
    # Scene validation
    # ---------------------------------------------------------------

    def validate_scene(
        self,
        scene: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Validate an individual scene.
        """

        if not isinstance(scene, dict):
            return False

        required = [
            "scene_type",
            "title",
            "narration",
            "visual",
            "on_screen_text",
            "avatar_action",
            "duration",
        ]

        for field in required:
            if field not in scene:
                return False

        if scene["scene_type"] not in self.SCENE_TYPES:
            return False

        if not str(scene["title"]).strip():
            return False

        try:
            duration = float(scene["duration"])
        except (TypeError, ValueError):
            return False

        if duration <= 0:
            return False

        if not isinstance(scene["on_screen_text"], list):
            return False

        return True

    # ---------------------------------------------------------------
    # Lesson -> scenes
    # ---------------------------------------------------------------

    def create_lesson_scenes(
        self,
        topic: str,
        lesson: Optional[Dict[str, Any]] = None,
        visual_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert a lesson structure into educational video scenes.

        Supports common lesson keys:
        - title
        - introduction
        - explanation
        - spoken_script
        - script
        - visual
        - question
        - modules
        - concepts
        """

        topic = str(topic or "").strip()

        if not lesson:
            lesson = {}

        scenes: List[Dict[str, Any]] = []

        # -----------------------------------------------------------
        # Intro scene
        # -----------------------------------------------------------

        introduction = (
            lesson.get("introduction")
            or lesson.get("intro")
            or ""
        )

        if introduction:
            scenes.append(
                self.create_scene(
                    scene_type="intro",
                    title=topic or "Introduction",
                    narration=str(introduction),
                    on_screen_text=[topic] if topic else [],
                    avatar_action="welcoming",
                )
            )

        # -----------------------------------------------------------
        # Modules
        # -----------------------------------------------------------

        modules = lesson.get("modules")

        if not modules:
            modules = lesson.get("concepts")

        if not modules:
            modules = [lesson]

        if not isinstance(modules, list):
            modules = [modules]

        for index, module in enumerate(modules, start=1):

            if isinstance(module, dict):

                title = (
                    module.get("title")
                    or module.get("concept")
                    or module.get("concept_name")
                    or f"Concept {index}"
                )

                narration = (
                    module.get("spoken_script")
                    or module.get("script")
                    or module.get("explanation")
                    or module.get("narration")
                    or ""
                )

                visual = module.get("visual")

                if not visual and visual_specs:
                    visual_index = index - 1

                    if visual_index < len(visual_specs):
                        visual = visual_specs[visual_index]

                question = (
                    module.get("question")
                    or module.get("assessment")
                    or None
                )

                on_screen_text = module.get(
                    "on_screen_text",
                    [],
                )

                if isinstance(on_screen_text, str):
                    on_screen_text = [on_screen_text]

            else:
                title = f"Concept {index}"
                narration = str(module)
                visual = None
                question = None
                on_screen_text = []

            # Explanation scene.
            if narration or visual:

                scenes.append(
                    self.create_scene(
                        scene_type="explanation",
                        title=str(title),
                        narration=str(narration),
                        visual=visual,
                        on_screen_text=on_screen_text,
                        avatar_action="speaking",
                    )
                )

            # Question scene.
            if question:

                question_text = ""

                if isinstance(question, dict):
                    question_text = (
                        question.get("question")
                        or question.get("text")
                        or ""
                    )
                else:
                    question_text = str(question)

                scenes.append(
                    self.create_scene(
                        scene_type="question",
                        title="Check Your Understanding",
                        narration=question_text,
                        visual=None,
                        on_screen_text=[question_text]
                        if question_text
                        else [],
                        avatar_action="asking",
                        question=question
                        if isinstance(question, dict)
                        else {
                            "question": question_text
                        },
                    )
                )

        # -----------------------------------------------------------
        # Outro
        # -----------------------------------------------------------

        if scenes:
            scenes.append(
                self.create_scene(
                    scene_type="outro",
                    title="Lesson Recap",
                    narration=(
                        f"Let's quickly recap what we learned "
                        f"about {topic}."
                        if topic
                        else "Let's quickly recap what we learned."
                    ),
                    on_screen_text=["Recap"],
                    avatar_action="concluding",
                )
            )

        return scenes

    # ---------------------------------------------------------------
    # Timeline
    # ---------------------------------------------------------------

    def build_timeline(
        self,
        scenes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add start/end timestamps to scenes.
        """

        timeline = []
        current_time = 0.0

        for index, scene in enumerate(scenes, start=1):

            duration = float(scene.get("duration", 0))

            start = round(current_time, 2)
            end = round(current_time + duration, 2)

            timeline_scene = dict(scene)

            timeline_scene["scene_number"] = index
            timeline_scene["start_time"] = start
            timeline_scene["end_time"] = end

            timeline.append(timeline_scene)

            current_time = end

        return timeline

    # ---------------------------------------------------------------
    # Total duration
    # ---------------------------------------------------------------

    def get_total_duration(
        self,
        scenes: List[Dict[str, Any]],
    ) -> float:
        """
        Return total video duration.
        """

        total = 0.0

        for scene in scenes:
            try:
                total += float(scene.get("duration", 0))
            except (TypeError, ValueError):
                continue

        return round(total, 2)

    # ---------------------------------------------------------------
    # Complete video plan
    # ---------------------------------------------------------------

    def create_video_plan(
        self,
        topic: str,
        lesson: Optional[Dict[str, Any]] = None,
        visual_specs: Optional[List[Dict[str, Any]]] = None,
        target_duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete educational video plan.
        """

        scenes = self.create_lesson_scenes(
            topic=topic,
            lesson=lesson,
            visual_specs=visual_specs,
        )

        timeline = self.build_timeline(scenes)

        total_duration = self.get_total_duration(timeline)

        plan = {
            "topic": topic,
            "scenes": timeline,
            "scene_count": len(timeline),
            "duration_seconds": total_duration,
            "duration_minutes": round(
                total_duration / 60,
                2,
            ),
            "target_duration_minutes": target_duration,
            "has_visuals": any(
                bool(scene.get("visual"))
                for scene in timeline
            ),
            "has_questions": any(
                scene.get("scene_type") == "question"
                for scene in timeline
            ),
            "has_narration": any(
                bool(scene.get("narration"))
                for scene in timeline
            ),
            "render_ready": all(
                self.validate_scene(scene)
                for scene in timeline
            ),
        }

        return plan

    # ---------------------------------------------------------------
    # Adapt video plan to target duration
    # ---------------------------------------------------------------

    def adapt_to_duration(
        self,
        plan: Dict[str, Any],
        target_minutes: float,
    ) -> Dict[str, Any]:
        """
        Adapt a video plan to a requested duration.

        This does not delete educational content aggressively.
        Instead, it scales scene durations while keeping the
        teaching structure intact.
        """

        if not isinstance(plan, dict):
            return {}

        target_minutes = max(
            0.5,
            float(target_minutes),
        )

        target_seconds = target_minutes * 60

        scenes = plan.get("scenes", [])

        if not scenes:
            result = dict(plan)
            result["target_duration_minutes"] = target_minutes
            result["duration_seconds"] = 0.0
            result["duration_minutes"] = 0.0
            return result

        current_seconds = self.get_total_duration(scenes)

        if current_seconds <= 0:
            return dict(plan)

        scale = target_seconds / current_seconds

        # Keep scenes readable.
        adjusted_scenes = []

        for scene in scenes:
            new_scene = dict(scene)

            original_duration = float(
                scene.get("duration", 1)
            )

            adjusted_duration = original_duration * scale

            adjusted_duration = max(
                2.0,
                adjusted_duration,
            )

            new_scene["duration"] = round(
                adjusted_duration,
                2,
            )

            adjusted_scenes.append(new_scene)

        timeline = self.build_timeline(
            adjusted_scenes
        )

        result = dict(plan)

        result["scenes"] = timeline
        result["target_duration_minutes"] = target_minutes
        result["duration_seconds"] = self.get_total_duration(
            timeline
        )
        result["duration_minutes"] = round(
            result["duration_seconds"] / 60,
            2,
        )

        return result

    # ---------------------------------------------------------------
    # Rendering preparation
    # ---------------------------------------------------------------

    def prepare_render_manifest(
        self,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a renderer-friendly manifest.

        This separates educational planning from actual media
        rendering.
        """

        scenes = plan.get("scenes", [])

        manifest_scenes = []

        for scene in scenes:

            manifest_scenes.append(
                {
                    "scene_number": scene.get(
                        "scene_number"
                    ),
                    "start_time": scene.get(
                        "start_time",
                        0,
                    ),
                    "end_time": scene.get(
                        "end_time",
                        0,
                    ),
                    "duration": scene.get(
                        "duration",
                        0,
                    ),
                    "type": scene.get(
                        "scene_type",
                    ),
                    "narration": scene.get(
                        "narration",
                        "",
                    ),
                    "visual": scene.get(
                        "visual",
                        {},
                    ),
                    "text": scene.get(
                        "on_screen_text",
                        [],
                    ),
                    "avatar_action": scene.get(
                        "avatar_action",
                        "speaking",
                    ),
                }
            )

        return {
            "topic": plan.get("topic", ""),
            "output_dir": self.output_dir,
            "scenes": manifest_scenes,
            "duration_seconds": plan.get(
                "duration_seconds",
                0,
            ),
            "render_backend": (
                "ffmpeg"
                if self.ffmpeg_path
                else "unavailable"
            ),
        }

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    def render_video(
        self,
        plan: Dict[str, Any],
        output_filename: str = "lesson.mp4",
    ) -> Dict[str, Any]:
        """
        Rendering interface.

        Actual rendering is intentionally conservative:
        - If FFmpeg is unavailable, return a clear status.
        - If FFmpeg is available, return a renderer-ready status.

        The full media compositor can later be connected here
        without changing the lesson/video planning API.
        """

        if not isinstance(plan, dict):
            return {
                "success": False,
                "status": "invalid_plan",
                "output_path": None,
            }

        if not plan.get("render_ready", False):
            return {
                "success": False,
                "status": "invalid_plan",
                "output_path": None,
            }

        safe_filename = os.path.basename(
            output_filename
        )

        if not safe_filename.lower().endswith(".mp4"):
            safe_filename += ".mp4"

        output_path = os.path.join(
            self.output_dir,
            safe_filename,
        )

        if not self.ffmpeg_path:
            return {
                "success": False,
                "status": "ffmpeg_unavailable",
                "output_path": output_path,
                "message": (
                    "Video plan is valid, but FFmpeg is "
                    "not installed or not available on PATH."
                ),
            }

        return {
            "success": False,
            "status": "renderer_ready",
            "output_path": output_path,
            "message": (
                "Video plan is ready for FFmpeg rendering. "
                "Media compositor integration is required "
                "for final MP4 generation."
            ),
        }


# ======================================================================
# TESTS
# ======================================================================

def run_tests():
    print("=" * 60)
    print("VIDEO ENGINE TESTS")
    print("=" * 60)

    engine = VideoEngine(
        output_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "video_test_output",
        ),
        api_key=None,
    )

    # ---------------------------------------------------------------
    # TEST 1
    # ---------------------------------------------------------------

    print("\nTEST 1: Engine initialization")

    assert engine is not None
    assert engine.api_key is None
    assert os.path.isdir(engine.output_dir)

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 2
    # ---------------------------------------------------------------

    print("\nTEST 2: Capability detection")

    result = engine.get_capabilities()

    assert isinstance(result, dict)
    assert "ffmpeg" in result
    assert "output_directory" in result
    assert "can_render_basic_video" in result

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 3
    # ---------------------------------------------------------------

    print("\nTEST 3: Duration estimation")

    result = engine.estimate_duration(
        "Machine learning allows computers to learn patterns from data."
    )

    assert result > 0

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 4
    # ---------------------------------------------------------------

    print("\nTEST 4: Scene duration calculation")

    result = engine.calculate_scene_duration(
        narration="This is an educational explanation."
    )

    assert result >= 3
    assert result <= 60

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 5
    # ---------------------------------------------------------------

    print("\nTEST 5: Create explanation scene")

    scene = engine.create_scene(
        scene_type="explanation",
        title="Supervised Learning",
        narration=(
            "Supervised learning learns from labeled examples."
        ),
        visual={
            "visual_type": "diagram",
            "title": "Supervised Learning",
        },
        on_screen_text=[
            "Learns from labeled data"
        ],
        avatar_action="speaking",
    )

    assert scene["scene_type"] == "explanation"
    assert scene["title"] == "Supervised Learning"
    assert scene["duration"] > 0

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 6
    # ---------------------------------------------------------------

    print("\nTEST 6: Scene validation")

    assert engine.validate_scene(scene) is True

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 7
    # ---------------------------------------------------------------

    print("\nTEST 7: Invalid scene detection")

    invalid_scene = {
        "scene_type": "invalid_type",
        "title": "",
    }

    assert engine.validate_scene(invalid_scene) is False

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 8
    # ---------------------------------------------------------------

    print("\nTEST 8: Lesson scene generation")

    lesson = {
        "introduction": (
            "Today we will learn supervised learning."
        ),
        "modules": [
            {
                "concept": "Labeled Data",
                "spoken_script": (
                    "Labeled data contains examples with "
                    "known answers."
                ),
                "visual": {
                    "visual_type": "diagram",
                    "title": "Labeled Data",
                },
            },
            {
                "concept": "Classification",
                "spoken_script": (
                    "Classification predicts categories."
                ),
                "visual": {
                    "visual_type": "flowchart",
                    "title": "Classification",
                },
                "question": {
                    "question": (
                        "What does classification predict?"
                    ),
                    "options": [
                        "Categories",
                        "Only numbers",
                        "Nothing",
                    ],
                    "answer": "Categories",
                },
            },
        ],
    }

    scenes = engine.create_lesson_scenes(
        topic="Machine Learning",
        lesson=lesson,
    )

    assert isinstance(scenes, list)
    assert len(scenes) >= 3

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 9
    # ---------------------------------------------------------------

    print("\nTEST 9: Timeline generation")

    timeline = engine.build_timeline(
        scenes
    )

    assert len(timeline) == len(scenes)

    for index, item in enumerate(
        timeline,
        start=1,
    ):
        assert item["scene_number"] == index
        assert item["end_time"] >= item["start_time"]

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 10
    # ---------------------------------------------------------------

    print("\nTEST 10: Total duration")

    duration = engine.get_total_duration(
        timeline
    )

    assert duration > 0

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 11
    # ---------------------------------------------------------------

    print("\nTEST 11: Complete video plan")

    plan = engine.create_video_plan(
        topic="Machine Learning",
        lesson=lesson,
    )

    assert plan["topic"] == "Machine Learning"
    assert plan["scene_count"] > 0
    assert plan["duration_seconds"] > 0
    assert plan["has_narration"] is True
    assert plan["has_visuals"] is True
    assert plan["has_questions"] is True
    assert plan["render_ready"] is True

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 12
    # ---------------------------------------------------------------

    print("\nTEST 12: Duration adaptation")

    adapted = engine.adapt_to_duration(
        plan,
        target_minutes=5,
    )

    assert adapted["target_duration_minutes"] == 5
    assert adapted["duration_seconds"] > 0
    assert adapted["scenes"]

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 13
    # ---------------------------------------------------------------

    print("\nTEST 13: Render manifest")

    manifest = engine.prepare_render_manifest(
        plan
    )

    assert manifest["topic"] == "Machine Learning"
    assert isinstance(
        manifest["scenes"],
        list,
    )
    assert len(manifest["scenes"]) == plan["scene_count"]

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 14
    # ---------------------------------------------------------------

    print("\nTEST 14: Renderer interface")

    render_result = engine.render_video(
        plan,
        output_filename="test_lesson.mp4",
    )

    assert isinstance(
        render_result,
        dict,
    )

    assert "success" in render_result
    assert "status" in render_result
    assert "output_path" in render_result

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 15
    # ---------------------------------------------------------------

    print("\nTEST 15: Question scene")

    question_scene = engine.create_scene(
        scene_type="question",
        title="Quick Check",
        narration="What is supervised learning?",
        on_screen_text=[
            "What is supervised learning?"
        ],
        avatar_action="asking",
        question={
            "question": "What is supervised learning?"
        },
    )

    assert question_scene["scene_type"] == "question"
    assert question_scene["question"] is not None
    assert engine.validate_scene(
        question_scene
    )

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 16
    # ---------------------------------------------------------------

    print("\nTEST 16: Empty lesson handling")

    result = engine.create_video_plan(
        topic="Physics",
        lesson={},
    )

    assert isinstance(result, dict)
    assert result["topic"] == "Physics"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 17
    # ---------------------------------------------------------------

    print("\nTEST 17: Dictionary visual integration")

    visual_specs = [
        {
            "visual_type": "equation",
            "title": "F = ma",
            "description": "Newton's second law",
        }
    ]

    lesson_data = {
        "modules": [
            {
                "concept": "Newton's Second Law",
                "spoken_script": (
                    "Force equals mass multiplied by acceleration."
                ),
            }
        ]
    }

    scenes = engine.create_lesson_scenes(
        topic="Physics",
        lesson=lesson_data,
        visual_specs=visual_specs,
    )

    assert len(scenes) >= 1

    explanation_scene = next(
        (
            scene
            for scene in scenes
            if scene["scene_type"] == "explanation"
        ),
        None,
    )

    assert explanation_scene is not None
    assert explanation_scene["visual"]["visual_type"] == "equation"

    print("PASS")

    # ---------------------------------------------------------------
    # TEST 18
    # ---------------------------------------------------------------

    print("\nTEST 18: Render-ready validation")

    for scene in plan["scenes"]:
        assert engine.validate_scene(scene)

    assert plan["render_ready"] is True

    print("PASS")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL VIDEO ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()