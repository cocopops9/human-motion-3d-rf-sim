"""Objects of interest in a sequence of frames: foreground objects followed
across frames (segmentation, tracking) and selected by tests (selectors:
near a point, the person cascade, the rotation test), all behind ObjectFinder."""

from bodyscan.detection.segmentation import (BackgroundSegmenter, Cluster, ObjectSegmenter, PointBackgroundSegmenter,
                                             SegmentationConfig, Segmenter, cluster_points)
from bodyscan.detection.tracking import Track, Tracker, TrackingConfig
from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig, RotationResult
from bodyscan.detection.selectors import (HumanSelector, NearSelector, RotationSelector, Selector, SelectorChain,
                                          Verdict)
from bodyscan.detection.finder import FoundObject, ObjectFinder, SelectConfig, object_table, selector_chain
from bodyscan.detection.human import (BodyView, Centered, ColumnStage, CurvedSurface, Feature, FrameResult,
                                      HaarFeature, HumanCascade, HumanConfig, HumanVerdict, OccupancyImage,
                                      RectangleContrast, ShapeStage, SilhouetteStage, SizeStage, Stage, StageResult,
                                      SurfaceStage, ViewSampling, WidthRatio, default_features, flat_fraction)

__all__ = ["BackgroundSegmenter", "Cluster", "ObjectSegmenter", "PointBackgroundSegmenter", "SegmentationConfig",
           "Segmenter", "cluster_points",
           "Track", "Tracker", "TrackingConfig", "RotationAnalyzer", "RotationConfig", "RotationResult",
           "HumanSelector", "NearSelector", "RotationSelector", "Selector", "SelectorChain", "Verdict",
           "FoundObject", "ObjectFinder", "SelectConfig", "object_table", "selector_chain",
           "BodyView", "Centered", "ColumnStage", "CurvedSurface", "Feature", "FrameResult", "HaarFeature",
           "HumanCascade", "HumanConfig", "HumanVerdict", "OccupancyImage", "RectangleContrast", "ShapeStage",
           "SilhouetteStage", "SizeStage", "Stage", "StageResult", "SurfaceStage", "ViewSampling", "WidthRatio",
           "default_features", "flat_fraction"]
