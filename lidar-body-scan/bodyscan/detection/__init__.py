"""Rotating objects and people in a sequence of frames."""

from bodyscan.detection.segmentation import (BackgroundSegmenter, Cluster, ObjectSegmenter, PointBackgroundSegmenter,
                                             SegmentationConfig, Segmenter, cluster_points)
from bodyscan.detection.tracking import Track, Tracker, TrackingConfig
from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig, RotationResult
from bodyscan.detection.human import (BodyView, Centered, ColumnStage, CurvedSurface, Feature, FrameResult,
                                      HaarFeature, HumanCascade, HumanConfig, HumanVerdict, OccupancyImage,
                                      RectangleContrast, ShapeStage, SilhouetteStage, SizeStage, Stage, StageResult,
                                      SurfaceStage, ViewSampling, WidthRatio, default_features, flat_fraction)

__all__ = ["BackgroundSegmenter", "Cluster", "ObjectSegmenter", "PointBackgroundSegmenter", "SegmentationConfig",
           "Segmenter", "cluster_points",
           "Track", "Tracker", "TrackingConfig", "RotationAnalyzer", "RotationConfig", "RotationResult",
           "BodyView", "Centered", "ColumnStage", "CurvedSurface", "Feature", "FrameResult", "HaarFeature",
           "HumanCascade", "HumanConfig", "HumanVerdict", "OccupancyImage", "RectangleContrast", "ShapeStage",
           "SilhouetteStage", "SizeStage", "Stage", "StageResult", "SurfaceStage", "ViewSampling", "WidthRatio",
           "default_features", "flat_fraction"]
