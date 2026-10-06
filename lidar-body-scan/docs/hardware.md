# The sensor: what it can resolve, and where to put it

This note answers two questions for the Ouster OS0-128 (firmware 3.1.0, mode
2048 x 10 Hz) mounted upright, as now:

1. Can the scans separate the fingers of a hand?
2. Where must the sensor stand, and with which settings, so that a person up
   to 2.0 m tall is seen from the soles to the head?

Numbers marked *datasheet* come from the Ouster OS0 datasheet; *measured*
from the lab recordings; *computed* from the geometry below.

## 1. Specifications that matter

| Quantity | Value | Source |
|---|---|---|
| Vertical field of view | 90 deg (+45 to -45), 128 beams, evenly spaced: 0.71 deg between beams | datasheet; spacing measured from the sensor's lut.npz |
| Horizontal sampling | 2048 columns per turn: 0.176 deg | datasheet (mode 2048 x 10) |
| Beam diameter at the window | 5 mm | datasheet |
| Beam divergence | 0.35 deg, full width at half maximum | datasheet |
| Range precision (10 % reflectivity) | ±2 cm from 0.3 to 1 m, ±1 cm from 1 to 10 m | datasheet |
| Range noise on a person | 2.2 mm median at 1 to 1.5 m | measured (spread of single views around the fused surface) |
| Minimum range | 0.3 m | datasheet |
| Returns | strongest return; dual return (strongest and second strongest) in 1024 x 20 and 2048 x 10 with firmware 3.0 and later | Ouster firmware documentation |

## 2. Sampling on the body

At a distance *d* the samples on a surface facing the sensor are
*d* x 3.07 mm/m apart across (columns) and *d* x 12.4 mm/m apart along the
height (beams). The footprint of a beam is the 5 mm aperture widened by the
divergence: between √(5² + (6.1 d)²) mm (adding the two in quadrature) and
5 + 6.1 d mm (adding them linearly).

| Distance | Column spacing | Beam spacing | Footprint | Smallest gap across (vertical slit) | Smallest gap along the height (horizontal slit) |
|---|---|---|---|---|---|
| 0.6 m | 1.8 mm | 7.4 mm | 6 to 9 mm | 8 to 12 mm | 14 to 24 mm |
| 0.8 m | 2.5 mm | 9.9 mm | 7 to 10 mm | 9 to 15 mm | 17 to 30 mm |
| 0.9 m | 2.8 mm | 11.2 mm | 7 to 11 mm | 10 to 16 mm | 19 to 33 mm |
| 1.0 m | 3.1 mm | 12.4 mm | 8 to 11 mm | 11 to 17 mm | 20 to 36 mm |
| 1.2 m | 3.7 mm | 14.9 mm | 9 to 12 mm | 13 to 20 mm | 24 to 42 mm |
| 1.5 m | 4.6 mm | 18.6 mm | 10 to 14 mm | 15 to 23 mm | 29 to 51 mm |

**Criterion for a visible gap** (computed): a gap between two surfaces shows in
a scan only if at least one beam passes through it without touching either
side. In strongest-return mode a beam that touches a near finger reports the
finger even if most of its footprint passes by: the light returned by the far
background (the room, metres away) is weaker by the square of the distance
ratio. So the gap must be wider than the footprint, plus one to two sample
spacings so that a beam falls inside it whatever the phase of the sampling
grid. This gives the two right-hand columns. Fusing many views does not lower
this limit: every view blurs the gap by the same footprint, and the views only
average the range noise.

Confidence: moderate. The criterion is geometric; how the sensor's pulse
processing treats a beam split between a finger and a far wall is not
documented. The lab data agree with it (section 3).

## 3. Fingers: the answer

The fused clouds are not the limit. On the hands of tt14 the fused cloud has a
point every 3.9 mm, and every point has 65 views and more than 250 measured
points within 1 cm (measured, `person_tt14_confidence.ply`). The limit is the
beam footprint above, then the processing (section 5).

| Gap | Typical width (adult hand) | Resolved? |
|---|---|---|
| Thumb and index, A-pose, palm forward | 30 to 50 mm | **yes**, at any distance up to about 2 m (tt14 shows it) |
| Spread fingers, near the fingertips | 10 to 25 mm, closing to 0 at the knuckles | **marginal**: only the outer part of the gaps, only when the hand passes at 0.9 m or closer, and only if the processing keeps them (section 5) |
| Relaxed fingers (touching) | 0 to 3 mm | **no**, at any distance (the footprint is 6 mm or more even at the 0.3 m minimum range) |
| Fingers held horizontally (hands spread to the sides) | 10 to 25 mm, but the slit runs along the beams' spacing direction | **no**: along the height the beams are 11 mm apart at 0.9 m, so the smallest visible gap is 19 to 33 mm |

So, with the sensor upright: the thumb separates from the other fingers;
individual fingers do not separate reliably. Keep the hands hanging with the
fingers pointing down (A-pose, palms forward, fingers spread): the gaps are
then vertical slits, resolved by the 3.1 mm/m column spacing instead of the
12.4 mm/m beam spacing.

What would separate spread fingers: the hands at 0.6 m or closer (smallest
gap 8 to 12 mm). That is incompatible with seeing the whole body of a 2 m
person with an upright sensor (section 4: the platform axis must be 1.2 m
away). It would need a second, short capture of the hands close to the sensor,
registered onto the body: not implemented.

## 4. Where to put the sensor for a person up to 2.0 m

The vertical field of view is 90 deg. The beams must reach the top of the head
(the front of the head is about 0.10 m in front of the axis) and the front of
the feet (about 0.15 m in front of the axis, 0.03 m high on the platform),
with a margin of 3 deg at both ends. Computed:

| Stature | Lens height | Minimum distance sensor to platform axis | Downward tilt that fits |
|---|---|---|---|
| 2.0 m | 1.0 m | 1.22 m | 0 deg (none) |
| 2.0 m | 1.1 m | 1.22 m | 3 deg |
| 2.0 m | 1.2 m | 1.21 m | 6 deg |
| 1.8 m | 1.0 m | 1.11 m | 3.5 deg |
| 1.8 m | 1.1 m | 1.09 m | 6.7 deg |

Without any tilt: lens at 1.0 m, axis at 1.23 m or more (for 1.8 to 2.0 m).

**Recommended setup** (computed): lens (the centre of the sensor window) 1.0 m
above the floor, sensor upright and level, platform axis 1.25 m from the
sensor. The hands then pass the sensor at about 0.85 m when they face it
(A-pose hands about 0.4 m from the axis), the closest a whole-body setup
allows.

Your recent runs explain the crops you saw: with the lens at 1.14 to 1.20 m and
the axis at about 1.1 m (tt13 to tt15), a 1.8 m person needs 49 deg below the
horizon for the feet and 33 deg above it for the head. Level, the feet are cut
(tt13b); tilted down (tt14), the head is cut. `bodyscan check-view` makes this
check on a 20 s capture and prints the tilt that fits.

What no upright position shows: the top of the head (seen only at grazing
angles, so the mesh closes it by interpolation), the soles (on the platform)
and the underside of the chin and arms. A second sensor above the head would
be needed for the crown.

## 5. Settings

| Setting | Recommendation | Why |
|---|---|---|
| Lidar mode | 2048 x 10 (as now) | the finest column spacing (3.1 mm/m); 1024 x 20 halves it |
| Returns | strongest (as now) | dual return reports the two strongest returns of each beam; it could separate a finger from the wall behind it, but the separation needed between the two returns is not documented, and when the hand is in front of the body the two are a few cm apart. Low confidence that it helps: test it on a short capture before relying on it |
| Platform speed | 4 deg/s (one lap in 90 s), as now | the body turns 0.4 deg during one sweep (2 mm at 0.3 m from the axis); the fusion undoes it with the time of every column |
| Laps | 3 to 5 | every patch of skin is seen in more views; the fusion's surface fit becomes more precise (standard error about 1.25 x spread / √points) |
| Fusion for the hands | `configs/turntable_fingers.toml` (voxel 3 mm, support radius 12 mm, surface-fit radius 6 mm) | the defaults (5, 20, 10 mm) fill gaps narrower than about 1 cm |
| Mesh for the hands | `configs/mesh_detail_2mm.toml` (watertight grid 2 mm, Poisson depth 10) | the default 4 mm grid seals gaps narrower than about 6 mm; the 2 mm grid keeps gaps down to about 3 mm, as smooth as the defaults, with 4 times the triangles (about 1.6 million) |

The finer configurations keep narrower gaps only if the fused cloud has them:
with the default fusion, gaps narrower than about 1 cm are already filled.
`configs/turntable_fingers.toml` is experimental: it keeps narrower gaps and
also more noise, so check the mesh quality report (facet noise, bump height)
against the needs of the ray tracer. Smoothness for ray tracing comes first,
as discussed with Marco; the fingers are secondary.

## Sources

- [Ouster OS0 datasheet (via RobotShop)](https://cdn.robotshop.com/rbm/1452f63b-45fa-4c16-b9a5-44447596ee17/9/926a2a4b-0ddc-418d-846e-6e4dcc5a9173/85f35666_209129079.pdf): field of view, beam diameter, divergence, range precision
- [Ouster sensor documentation, lidar data](https://docs.ouster.com/sensor-docs/docs/sensor-data/lidar-data): dual return profile, valid for 1024 x 20 and 2048 x 10 from firmware 3.0
- [Clearpath technical data sheet of the Ouster OS series](https://docs.clearpathrobotics.com/assets/files/clearpath_robotics_028738-TDS1-436562d34fd910ad37b69ab8b8a8bb13.pdf): beam diameter and divergence (confirmation)
