# Legacy scripts

The single-file scripts used until 2026-10-04, replaced by the `bodyscan`
package. They are kept unchanged so that the earlier results can be
reproduced (each result's JSON report names the script version).

| Script (version) | Replaced by |
|---|---|
| `fuse_turntable.py` (2026-10-02d) | `python -m bodyscan fuse` |
| `fuse_person.py` (2026-10-01a) | `python -m bodyscan fuse-inplace` |
| `fuse_views.py` (2026-09-29b) | `python -m bodyscan fuse` (13 still views of a chair on the platform; superseded since the continuous turntable capture) |
| `pointcloud_to_mesh.py` (2026-10-02a) | `python -m bodyscan mesh` |
| `capture_turntable.py` (2026-10-01c) | `python -m bodyscan capture-turntable` |
| `capture_person.py` (2026-09-30a) | `python -m bodyscan capture-inplace` |
| `check_view.py` (2026-10-02a) | `python -m bodyscan check-view` |
| `check_sensor.py` | `python -m bodyscan check-sensor` |
| `ouster_extract.py` | `python -m bodyscan convert` (recordings), `capture-*` (live) |

The capture directories are the same for both: recordings made with the old
capture scripts are read by the new commands.

Differences in the results of the package, measured on the same data:

- `fuse` gives the same angles and cloud as `fuse_turntable.py` 2026-10-02d
  when started from the same platform centre (frame angles within 1e-12 deg on
  the synthetic run rsE); it now finds the centre around the person, which can
  change the start of the axis fit.
- `fuse-inplace` adds the surface fit and the confidence of the turntable
  fusion: on a synthetic in-place run the fused cloud is closer to the truth
  (median 1.6 mm instead of 2.2 mm).
- `mesh` converts without smoothing, with an orientation and inside test that
  no longer fail at random; `smooth` then smooths with a bilateral filter
  instead of Taubin, and no decimation by default (see CHANGELOG).
