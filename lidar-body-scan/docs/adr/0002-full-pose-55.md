# ADR 0002: the jaw, the eyes and the hands in the tracker (pose model full55)

**Status:** Proposed (branch feature/full-pose-55); not validated on recorded takes
**Date:** 2026-10-26
**Deciders:** Giuseppe D'Auria

## Context

The tracker of 1.6.0 fits the root and the 21 body joints of SMPL-X. The jaw and the eyes stay at rest
and the hands at the mean hand of the model file; the SMPL-X files used for the s02 avatar
(SMPLX_MALE.npz, sha1 f53ed32) carry no hand mean, so in the 1.6.0 motions the hands are flat. For the
radio simulation the hands matter (they move fast and reflect), so the question is how much of the hand
pose a single LiDAR can recover, and whether fitting it changes the rest of the fit.

## Decision

1. A pose model option, `tracking.pose_model` (`--pose-model`): `body21` (default, the 1.6.0 behaviour
   unchanged) or `full55`. full55 adds a per-frame vector (bodyscan.dynamic.fullpose): jaw (3), eyes
   (2 x 3) and K MANO PCA coefficients per hand (default K = 6).
2. The hand components come from an SMPL-X file that has them (`--hand-pca`). The body stays that of the
   avatar's model file; the hand file is accepted only if its body arrays (v_template, f, shapedirs,
   posedirs, J_regressor, weights, kintree_table) are equal. motion.npz records both sha1.
3. Priors: the jaw and the eyes towards rest with a strong weight (face_weight 10 per rad^2: the LiDAR
   cannot see them); the hands towards the mean hand, on their joint rotations (hand_prior_weight); the
   stage-1 continuity and the stage-2 smoothness apply to them, with the hidden-part weighting of the
   wrist's part.
4. A warm start (`--init-motion`): every frame starts from its pose in an earlier motion (same
   frame_index), without the initialisation and the searches, so that body21 and full55 can be compared
   on the same starting motion.
5. A hold-out check (`--holdout-fraction`): a fraction of the points of every frame, chosen from a seed and the
   frame index, is kept out of the fit; their pixels count as unknown for the free-space terms.

## Options considered

### A. MANO PCA, K = 6 per hand (chosen)

A hand at 2 to 4 m from an OS0-128 at 1024 columns gets a few tens of points, at 1 to 2 cm spacing: enough
to tell an open from a closed hand, the spread and the thumb, not single fingers. Six components carry
those modes. Cons: the PCA cannot represent unusual hand poses (pointing with one finger).

### B. Full 45 axis-angle numbers per hand

Pros: any hand pose. Cons: about 40 directions per hand that no point constrains; the fit would be
decided by the prior alone and could drift with noise.

### C. Hands fixed at the mean (relaxed) pose

Pros: no extra unknowns; better than the flat hands of 1.6.0. Cons: no hand motion at all.

## Consequences

* motion.npz of full55 gains `left_hand`, `right_hand`, `jaw`, `eyes` (PoseSequence reads them, so
  export-motion and review-motion pose the hands), `hand_coefficients`, `jaw_deg`, `eye_deg`,
  `hand_pca_file`, `hand_pca_sha1`; every run gains `hand_points` and, with a hold-out, the held-out
  residuals. body21 motions keep their keys.
* Cost: 21 more unknowns per frame; the searches (correspondences, visibility) dominate the time and
  do not change.
* Not validated: the gain near the hands on recorded takes, and the prior weights, are to be measured
  (pilot on s02_natural01 and s02_walk01).
