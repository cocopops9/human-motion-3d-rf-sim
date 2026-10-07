"""Moving people: from a recording of a person moving around the LiDAR to an animated body.

    image       range-image geometry: world point to pixel, pixel times (numpy)
    segment     the person in every frame, with a silhouette crop (numpy, Open3D)
    motions     pose sequences, smooth resampling, procedural walks and jumps, AMASS
    simulate    synthetic recordings of a moving body with truth (rolling shutter, footprint)
    fitting     losses, closest points, visibility, silhouette distance maps
    avatar      a person's body model fitted to their turntable scan
    track       the avatar fitted to every frame, then the whole sequence refined
    evaluate    accuracy against the truth of a synthetic recording
    export      animated meshes (PLY per step, velocities) and body tracks for the radio simulation
    review      pictures of the tracked body over the LiDAR points, for human review
    bench       synthetic test bench: takes over a grid, error tables

image and segment need numpy and Open3D only (they run on the capture PC);
the rest needs PyTorch (the processing PC).
"""
