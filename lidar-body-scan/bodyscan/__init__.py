"""bodyscan: LiDAR scanning of people, from capture to meshes for ray tracing.

Standing people on a turntable (or turning in place) give one closed mesh;
people moving around the sensor give an animated mesh, from their body model
fitted to every frame.

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
    body           the body model (SMPL-X format): skeleton, rotations, skinning (needs PyTorch)
    dynamic        moving people: segmentation, avatar, tracking, export, simulation, evaluation
    commands       the command-line interface (python -m bodyscan ...)

Every tunable parameter is declared once in a configuration section (see
bodyscan.config) together with its help text and the effect of changing it.
"""

__version__ = "1.2.0"
