from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# VISION METADATA PRIMITIVES (Layer 2)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int]
    thickness: int = 2
    fill_alpha: float = 0.0
    label: str = ""
    label_color: Tuple[int, int, int] = (255, 255, 255)
    label_bg_color: Optional[Tuple[int, int, int]] = None

@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray  # Binary mask (HxW) uint8
    color: Tuple[int, int, int]
    alpha: float = 0.5

@dataclass(frozen=True)
class VisionMetadata:
    """
    Immutable, display-ready vision primitives required for rendering.
    Must never contain model instances, tensors, inference state, trackers, or business logic.
    """
    version: int = 1
    segmentations: List[SegmentationMask] = field(default_factory=list)
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    
    # Raw primitives needed for exact legacy CV algorithms
    pred_map: Optional[np.ndarray] = None
    mask_frozen_pred: Optional[np.ndarray] = None
    raw_pred: Optional[np.ndarray] = None
    socket_hit: Optional[Dict[str, Any]] = None
    hand_in_roi: bool = False
    has_tubes: bool = False
    sm_state: str = ""


# -----------------------------------------------------------------------------
# UI METADATA (Layer 3)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InspectionProgress:
    socket_current: int = 0
    socket_required: int = 0
    tube_current: int = 0
    tube_required: int = 0

@dataclass(frozen=True)
class FrameMetadata:
    """
    Immutable, rendering-only UI data contract.
    Contains only abstracted data required for HUD visualizations.
    Must NEVER contain ML states, model instances, or business logic.
    """
    fps: float
    frame_number: int
    elapsed_ms: float
    avg_elapsed_ms: float
    
    state: str
    cycle_number: int
    verdict: str
    order_status: str
    
    # Keeping these for legacy debug HUD components, though visual rendering moved to VisionOverlay
    socket_hit: Optional[Dict[str, Any]]
    detected_seq: List[int]
    status_dict: Dict[int, bool]
    
    hand_in_roi: bool
    mask_is_locked: bool
    
    version: int = 1
    ui_models: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
    
    counters: Dict[str, Any] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    inspection_progress: Optional[InspectionProgress] = None


# -----------------------------------------------------------------------------
# RENDERER INTERFACES
# -----------------------------------------------------------------------------

class BaseRenderer:
    """
    Stateless UI Renderer Interface.
    Must not cache frame history, counters, tracking information,
    cycle state, or inference results between frames.
    """
    def render(self, frame: np.ndarray, metadata: FrameMetadata) -> np.ndarray:
        raise NotImplementedError("Renderers must implement the render() method.")


class BaseVisionRenderer:
    """
    Stateless Vision Renderer Interface.
    """
    def render(self, frame: np.ndarray, vision_metadata: VisionMetadata) -> np.ndarray:
        raise NotImplementedError("Vision Renderers must implement the render() method.")


class NullRenderer(BaseRenderer):
    """
    Test renderer that returns the unmodified frame.
    Useful for verifying that rendering logic is completely decoupled.
    """
    def render(self, frame: np.ndarray, metadata: FrameMetadata) -> np.ndarray:
        return frame
