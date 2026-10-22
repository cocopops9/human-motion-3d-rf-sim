"""Evaluation of tracked motions without ground truth, and planning of the sensor layout.

    fit         pixel classes of the measurements against the posed body (fitted, violation,
                unexplained, silhouette, unobserved), the per-vertex error map, per-part tables
    take        a tracked single-LiDAR take evaluated from its segment-motion crops (evaluate-take)
    kinematics  speeds, jitter (high-frequency share), turn rates, hinges, joint speed limits
    heatmap     per-vertex values as colours on the body (coloured PLY)
    coverage    how much of the body each sensor of a rig sees, frame by frame (bench-rig)

Real-data errors here are consistencies with the measurements, not accuracies:
a region no sensor observed has no value.
"""

from bodyscan.evaluation.fit import (CLASS_NAMES, DARK, EMPTY, FITTED, NONE, OTHER, PERSON, SILHOUETTE, UNEXPLAINED,
                                     UNKNOWN_KIND, UNOBSERVED, VIOLATION, ClassifyConfig, RayBatch, VertexMap,
                                     class_counts, classify, part_table, point_rates, vertex_areas)

__all__ = ["CLASS_NAMES", "DARK", "EMPTY", "FITTED", "NONE", "OTHER", "PERSON", "SILHOUETTE", "UNEXPLAINED",
           "UNKNOWN_KIND", "UNOBSERVED", "VIOLATION", "ClassifyConfig", "RayBatch", "VertexMap", "class_counts",
           "classify", "part_table", "point_rates", "vertex_areas"]
