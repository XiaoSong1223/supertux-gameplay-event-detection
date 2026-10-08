# Delivery validation

Validation was performed on 9 October 2026 (Singapore time), on the tested macOS / Apple M4 / CPython 3.12 environment.

- Extracted the submission ZIP into a new temporary directory and verified every file against its SHA-256 manifest.
- Created a separate virtual environment, installed `requirements-lock.txt`, and installed the extracted project in editable mode.
- Ran all 52 tests successfully. Five expected warnings come from synthetic test frames whose height differs from the calibrated 480 pixels.
- Ran a fresh-video three-frame smoke test from the extracted checkout. It generated two enemy appearance events, two item appearance events, and a playable MP4.
- Ran `scripts/fetch_assets.py` from the extracted checkout. It downloaded both omitted source videos and verified all 81 pinned video/asset records.
- Verified the original full runs each contain 1,350 analysis/output frames at 15 fps; decoded dispersed MP4 frames and checked output dimensions.
- Checked JSON/CSV event-count agreement, event timestamps, confirmation ordering per event, confidence bounds, frozen core hashes, asset hashes, and overview-document links.

Final demos were re-rendered from saved tracks/events with compact ID labels and an explicit “last confirmed HUD” caption. This avoids interpreting a stale counter as a current read. Inference predictions and metric inputs were not changed. `scripts/render_demo.py` reproduces that presentation pass.

The ZIP excludes the original assignment PDF, source-video downloads, virtual environments, working annotation images, private Chinese interview notes, and the email draft. Those exclusions are also recorded in `SUBMISSION_MANIFEST.json`.

This validation covers the tested local environment. It does not claim testing on Windows, Linux, other Python versions, or independent human validation of visual labels.
