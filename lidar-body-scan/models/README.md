# Body models (SMPL-X)

Put the SMPL-X model files here (`SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`,
`SMPLX_FEMALE.npz`; register at https://smpl-x.is.tue.mpg.de and download the
npz models).

The licence forbids redistribution: `.gitignore` excludes `SMPLX_*` and every
`.npz`, so the files stay on your PC and are never committed. Commands find a
model by its file name in this folder, so `--model SMPLX_MALE.npz` is enough.

The data (scans, takes, avatars, motions) stays in `C:\lidar`; the code, and
these model files, stay in the project folder.
