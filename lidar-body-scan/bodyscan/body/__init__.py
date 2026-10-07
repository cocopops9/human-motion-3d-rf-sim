"""Parametric human bodies: the SMPL-X skeleton, the body model, a procedural test body.

    skeleton    joint names, parents, body parts, anatomical limits, axes (numpy only)
    rotations   axis-angle, matrices, quaternions, smooth interpolation (numpy and PyTorch)
    model       SMPL-X npz loader, shape, pose, linear blend skinning, subdivision (PyTorch)
    testbody    a procedural body in the SMPL-X file format (tests, demonstrations)

bodyscan.body.model needs PyTorch; the other modules do not, so that the
capture PC (numpy and Open3D only) can import the package.
"""
