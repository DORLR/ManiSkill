# Navigation World Models (vendored, unchanged)

Source: https://github.com/facebookresearch/nwm, commit 3f6cd8e70d6f2d1e2b9684acff510710135f0f41
Files: `models.py` (CDiT), `diffusion/` (DiT/IDDPM Gaussian diffusion). License: CC BY-NC 4.0 (`LICENSE.md`).

Do not edit these files. Our adaptation lives in `wm/models/cdit.py`:
the action embedder is replaced (joint-target chunk instead of a 3-D navigation action);
latent tokens are laid out as an 8 x 8 grid with patch size 1, so the network itself is unchanged.
