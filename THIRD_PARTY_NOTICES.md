# Third-party materials

## Gameplay recordings

Both recordings are by **PantheraLeo1359531**. Their Wikimedia Commons description pages list **GPL-3.0-or-later**:

- [Super Tux — Welcome to Antarctica](https://commons.wikimedia.org/wiki/File:Super_Tux_-_Welcome_to_Antarctica_(Walkthrough_by_PantheraLeo1359531)_2023.01.29_-_12.27.00.08.webm), used for development.
- [Super Tux — The Journey Begins](https://commons.wikimedia.org/wiki/File:Super_Tux_-_The_Journey_Begins_(Walkthrough_by_PantheraLeo1359531)_2023.01.29_-_12.30.19.10.webm), used for held-out evaluation.

The downloaded derivatives are 480p VP9 recordings. The annotated demos are modified excerpts: 2–92 seconds, sampled to 15 fps, with tracking/event overlays and no audio. Source downloads and SHA-256 hashes are recorded in `data/sources.json`. Original recordings are not included in the submission ZIP. The upstream GNU GPL version 3 text is included at `assets/upstream/LICENSE.txt`; the recording pages govern their listed licensing and attribution.

## Sprite and font assets

Selected unmodified sprite/font files come from [SuperTux v0.6.3](https://github.com/SuperTux/supertux/tree/c1ddb4f28c54de77f07aa9965231a91c0ed06708), commit `c1ddb4f28c54de77f07aa9965231a91c0ed06708`. They are used as detector templates and font-geometry references. The included upstream `README.md` identifies the GNU GPL and notes that most data is also licensed under CC BY-SA. This submission preserves the upstream license and credits rather than assigning a new license to those materials. Exact file URLs/hashes are in `data/sources.json`.

The HUD classifier is built from development-video digit patches. Its training procedure and labels are in `scripts/build_hud_templates.py`.

## Software dependencies

This project uses Python, NumPy, OpenCV, SciPy, Pillow, imageio-ffmpeg/FFmpeg, and pytest. Dependencies are installed from their own distributions and retain their respective licenses. They are not vendored in the submission ZIP. Tested versions are in `requirements-lock.txt`.
