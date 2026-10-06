"""bodyscan: LiDAR scanning of people on a turntable, from capture to a mesh for ray tracing.

Layers (each one only uses the layers above it):

    geometry       transforms, fitting, nearest neighbours, clustering
    io             recordings on disk, sensor models (range image to points)
    scene          floor, static background, platform, regions of interest, foreground isolation
    registration   ICP variants and relative-turn solvers
    motion         platform angle versus time (stepper model, model-free solution)
    fusion         views, per-view corrections, robust surface fusion
    meshing        surface reconstruction, watertight remeshing, RF smoothing, quality metrics
    detection      rotating objects and people in a sequence of frames
    capture        recording with the Ouster sensor (optional dependency: ouster-sdk)
    pipelines      the steps above assembled for one setup (turntable, turning in place, meshing)
    commands       the command-line interface (python -m bodyscan ...)

Every tunable parameter is declared once in a configuration section (see
bodyscan.config) together with its help text and the effect of changing it.
"""

__version__ = "1.1.0"
