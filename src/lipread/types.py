"""Shared data types for the LipRead pipeline.

Every module talks through these dataclasses. Keep them plain (no heavy imports)
so they can be used by the CLI, the web server and the tests alike.

Conventions
-----------
* Internal frame rate is always 25 fps (``FPS``). ``frame`` indices refer to the
  25 fps timeline that starts at ``Job.start`` seconds into the source video.
* Times in outputs are seconds on the ORIGINAL source timeline:
  ``t = job.start + frame / FPS``.
* Points are (x, y) pixel coordinates in the 25 fps frame, float.
* The 4 alignment keypoints are, in order: image-left eye, image-right eye,
  nose tip, mouth centre (this matches Auto-AVSR's MediaPipe preprocessing).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

FPS = 25                 # model frame rate
CROP = 88                # model input crop size (88x88 grayscale)
MAX_MODEL_FRAMES = 500   # hard cap of the ONNX position table
CHUNK_FRAMES = 400       # frames per inference window
CHUNK_OVERLAP = 50       # overlap between windows (frames)


class Readability(str, Enum):
    READABLE = "readable"
    MARGINAL = "marginal"
    UNREADABLE = "unreadable"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VideoInfo:
    path: str                 # local path of the (downloaded) source video
    width: int
    height: int
    src_fps: float            # source frame rate (may be 29.97, 50, 59.94 ...)
    duration: float           # seconds
    interlaced: bool = False
    has_audio: bool = False
    rotation: int = 0         # degrees from container metadata (0/90/180/270)


@dataclass
class Detection:
    """One face in one frame."""
    frame: int
    box: tuple[float, float, float, float]   # x, y, w, h in pixels
    score: float
    points5: np.ndarray                       # (5,2): l-eye, r-eye, nose, mouth-l, mouth-r (image left/right)
    keypoints4: Optional[np.ndarray] = None   # (4,2) alignment points (see module doc); None until refined


@dataclass
class FaceTrack:
    """A single person's face followed over consecutive frames inside one shot."""
    track_id: int
    shot_id: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame

    @property
    def end_frame(self) -> int:           # inclusive
        return self.detections[-1].frame

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def median_face_height(self) -> float:
        return float(np.median([d.box[3] for d in self.detections]))


@dataclass
class QualityReport:
    tier: Readability
    face_px: float            # median face box height (pixels)
    mouth_px: float           # median mouth width (pixels)
    yaw_deg: float            # median absolute yaw estimate
    sharpness: float          # median Laplacian variance of mouth region (higher = sharper)
    coverage: float           # fraction of frames in the track with a real (non-interpolated) detection
    motion: float             # mouth-motion energy (visual voice activity), 0..1
    reasons: list[str] = field(default_factory=list)   # human-readable reasons for the tier


@dataclass
class Word:
    text: str                 # UPPERCASE as emitted by the model (or the boosted roster name)
    start: float              # seconds, source timeline
    end: float
    confidence: float         # 0..1 (mean max-posterior over the word's frames)
    tier: Confidence
    boosted: bool = False     # True if the roster booster replaced the model's word
    original: Optional[str] = None   # the model's word before boosting


@dataclass
class Utterance:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def confidence(self) -> float:
        return float(np.mean([w.confidence for w in self.words])) if self.words else 0.0


@dataclass
class TrackResult:
    track_id: int
    shot_id: int
    start: float
    end: float
    quality: QualityReport
    utterances: list[Utterance] = field(default_factory=list)
    thumbnail: Optional[str] = None      # path to a face thumbnail (jpg) in the output dir
    crops_preview: Optional[str] = None  # path to a strip image of mouth crops (png)
    skipped: bool = False                # True if not decoded (unreadable / not selected)


@dataclass
class Job:
    source: str                          # file path or http(s) URL
    out_dir: str
    start: float = 0.0                   # trim start (s)
    end: Optional[float] = None          # trim end (s)
    faces: str = "readable"              # "all" | "readable" | "largest" | comma list of track ids
    names: list[str] = field(default_factory=list)   # roster / hotwords for the booster
    render_video: bool = True
    min_track_frames: int = 13           # ignore tracks shorter than ~0.5 s
    force_unreadable: bool = False       # decode even UNREADABLE tracks (labelled as such)


@dataclass
class Result:
    job: Job
    video: VideoInfo
    tracks: list[TrackResult]
    shots: list[tuple[int, int]]         # (start_frame, end_frame inclusive) per shot
    model: dict[str, Any]                # engine name, version, license note
    timings: dict[str, float]            # seconds per stage
    outputs: dict[str, str] = field(default_factory=dict)   # kind -> path (json, srt, vtt, txt, mp4)
    disclaimer: str = ("AI lip-reading estimate. Lip reading is uncertain; words can be wrong. "
                       "Do not treat as a quote.")

    def to_dict(self) -> dict[str, Any]:
        d = _jsonable(asdict(self))
        # asdict() skips properties: add each utterance's text and mean confidence for consumers
        for tr_obj, tr in zip(self.tracks, d["tracks"]):
            for u_obj, u in zip(tr_obj.utterances, tr["utterances"]):
                u["text"] = u_obj.text
                u["confidence"] = round(u_obj.confidence, 4)
        return d


# progress callback: (stage_name, fraction 0..1, message)
ProgressFn = Callable[[str, float, str], None]


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Enum):
        return o.value
    return o
