"""Marching cubes with a case table built in code (numpy only), closed and edge-manifold."""

from __future__ import annotations

import numpy as np

# Cube corners, edges and faces (corners of a face in cyclic order).
CUBE_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
CUBE_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)])
CUBE_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
FACE_NORMALS = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]], dtype=np.float64)


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary dilation with the 6-neighbourhood."""
    mask = mask.copy()
    for _ in range(iterations):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown[:, :, 1:] |= mask[:, :, :-1]
        grown[:, :, :-1] |= mask[:, :, 1:]
        mask = grown
    return mask


def _midpoint(edge):
    return (CUBE_CORNERS[CUBE_EDGES[edge][0]] + CUBE_CORNERS[CUBE_EDGES[edge][1]]) / 2.0


def cube_cases():
    """Iso-surface triangles in a cube for each of the 256 inside/outside
    patterns of its corners. Vertex ids 0 to 11 are the crossing points on
    the cube edges, 12 and up the centres of the loops that need one.

    On every face the crossing points are joined into segments; a face with
    all four edges crossed (inside corners on a diagonal) is resolved by
    cutting off each inside corner, a rule that depends on the face alone, so
    the two cubes sharing a face agree and the surface is closed. The
    segments form closed loops. A loop is triangulated as a fan unless two of
    its non-consecutive points lie on one face (the neighbouring cube may then
    use the same chord, which would belong to four triangles); such a loop
    gets a centre point and a star of triangles instead.
    Returns, per pattern, (triangles, loops that have a centre)."""
    faces_of_edge = [{f for f, face in enumerate(CUBE_FACES)
                      if set(edge) <= set(face)} for edge in CUBE_EDGES.tolist()]
    edge_of = {frozenset(e): k for k, e in enumerate(CUBE_EDGES.tolist())}
    cases = []
    for pattern in range(256):
        inside = [bool(pattern >> c & 1) for c in range(8)]
        neighbours, outward_of = {}, {}
        for f, face in enumerate(CUBE_FACES):
            edges = [edge_of[frozenset((face[k], face[(k + 1) % 4]))] for k in range(4)]
            crossed = [k for k in range(4) if inside[face[k]] != inside[face[(k + 1) % 4]]]
            segments = []
            if len(crossed) == 2:
                inner = [CUBE_CORNERS[c] for c in face if inside[c]]
                outer = [CUBE_CORNERS[c] for c in face if not inside[c]]
                towards = np.mean(outer, axis=0) - np.mean(inner, axis=0)
                segments = [(edges[crossed[0]], edges[crossed[1]], towards)]
            elif len(crossed) == 4:
                for k in range(4):
                    if inside[face[k]]:
                        a, b = edges[(k - 1) % 4], edges[k]
                        towards = (_midpoint(a) + _midpoint(b)) / 2 - CUBE_CORNERS[face[k]]
                        segments.append((a, b, towards))
            for a, b, towards in segments:
                neighbours.setdefault(a, []).append(b)
                neighbours.setdefault(b, []).append(a)
                outward_of[frozenset((a, b))] = (f, towards)
        triangles, centred, seen = [], [], set()
        for start in sorted(neighbours):
            if start in seen:
                continue
            loop, previous, current = [start], None, start
            seen.add(start)
            while True:
                following = [n for n in neighbours[current] if n != previous][0]
                if following == start:
                    break
                loop.append(following)
                seen.add(following)
                previous, current = current, following
            # Seen from outside, the loop runs counterclockwise: on the face of
            # its first segment, the inside of the surface patch lies towards
            # the cube (-face normal) and the outside towards 'towards'; the
            # face's other cube makes the opposite choice, so the orientation
            # is consistent.
            f, towards = outward_of[frozenset((loop[0], loop[1]))]
            step = _midpoint(loop[1]) - _midpoint(loop[0])
            if np.cross(towards, step) @ (-FACE_NORMALS[f]) < 0:
                loop = [loop[0]] + loop[1:][::-1]
            count = len(loop)
            chord_on_face = any(faces_of_edge[loop[a]] & faces_of_edge[loop[b]]
                                for a in range(count) for b in range(a + 2, count)
                                if not (a == 0 and b == count - 1))
            if count == 3 or not chord_on_face:
                triangles += [(loop[0], loop[k], loop[k + 1]) for k in range(1, count - 1)]
            else:
                centre = 12 + len(centred)
                centred.append(loop)
                triangles += [(centre, loop[k], loop[(k + 1) % count]) for k in range(count)]
        cases.append((triangles, centred))
    return cases


CASES = cube_cases()
MAX_TRIANGLES = max(len(t) for t, _ in CASES)
MAX_CENTRES = max(len(c) for _, c in CASES)


def marching_cubes(sdf: np.ndarray, voxel: float):
    """Zero level set of a sampled field as a closed, edge-manifold triangle
    mesh, every triangle oriented from the negative (inside) to the positive
    side. Returns vertices (grid index times voxel) and triangles."""
    shape = np.array(sdf.shape)
    negative = sdf < 0
    pattern = np.zeros(shape - 1, dtype=np.uint8)            # 8 corners: 8 bits (a fine grid has 10^8 cubes)
    for c, corner in enumerate(CUBE_CORNERS):
        pattern |= negative[corner[0]:shape[0] - 1 + corner[0], corner[1]:shape[1] - 1 + corner[1],
                            corner[2]:shape[2] - 1 + corner[2]].astype(np.uint8) << np.uint8(c)
    cubes = np.argwhere((pattern > 0) & (pattern < 255))
    case = pattern[cubes[:, 0], cubes[:, 1], cubes[:, 2]].astype(np.int64)
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    corner_index = (cubes[:, None, :] + CUBE_CORNERS[None, :, :]) @ strides           # (cubes, 8)
    flat = sdf.ravel()

    # Crossing points on the 12 edges of every active cube, shared between
    # cubes through the key of the grid edge.
    ends_a, ends_b = corner_index[:, CUBE_EDGES[:, 0]], corner_index[:, CUBE_EDGES[:, 1]]
    crossed = negative.ravel()[ends_a] != negative.ravel()[ends_b]
    keys = np.where(crossed, np.minimum(ends_a, ends_b) * flat.size + np.maximum(ends_a, ends_b), -1)
    unique_keys, inverse = np.unique(keys[crossed], return_inverse=True)
    a, b = unique_keys // flat.size, unique_keys % flat.size
    t = flat[a] / (flat[a] - flat[b])
    pa = np.stack(np.unravel_index(a, sdf.shape), axis=-1).astype(np.float64)
    pb = np.stack(np.unravel_index(b, sdf.shape), axis=-1).astype(np.float64)
    vertices = [pa + t[:, None] * (pb - pa)]
    vertex_of = np.full((len(cubes), 12 + MAX_CENTRES), -1, dtype=np.int64)
    vertex_of[:, :12][crossed] = inverse.reshape(-1)

    # Loop centres, where a case needs them: the mean of the loop's points.
    centre_count = len(unique_keys)
    for k, (_, centred) in enumerate(CASES):
        if not centred:
            continue
        members = np.flatnonzero(case == k)
        for c, loop in enumerate(centred):
            centres = vertices[0][vertex_of[members][:, loop]].mean(axis=1)
            vertex_of[members, 12 + c] = centre_count + np.arange(len(members))
            centre_count += len(members)
            vertices.append(centres)
    vertices = np.concatenate(vertices)

    counts = np.array([len(t) for t, _ in CASES])
    table = np.zeros((256, MAX_TRIANGLES, 3), dtype=np.int64)
    for k, (triangles, _) in enumerate(CASES):
        if triangles:
            table[k, :len(triangles)] = triangles
    owners = np.repeat(np.arange(len(cubes)), counts[case])
    slot = np.arange(len(owners)) - np.repeat(np.cumsum(counts[case]) - counts[case], counts[case])
    local = table[case[owners], slot]
    faces = vertex_of[owners[:, None], local]
    return vertices * voxel, faces
