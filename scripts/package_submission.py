"""Build the review ZIP from an explicit allowlist, excluding private work files."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "Razer_CV_Gameplay_Events"


def main():
    files = set()

    def add(path):
        p = ROOT / path
        if not p.is_file():
            raise FileNotFoundError(f"Required submission artifact missing: {path}")
        files.add(p)

    for filename in ["README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements-lock.txt",
                     "data/sources.json", "data/freeze.json"]:
        add(filename)
    for dirname in ["src/gameplay_events", "tests", "scripts", "assets", "data/annotations", "outputs/metrics"]:
        for path in (ROOT / dirname).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.add(path)
    for name in ["EVALUATION.md", "VALIDATION.md", "ANNOTATION_PROTOCOL.md", "ANNOTATION_AUDIT.md", "HUD_NOTES.md", "HUD_EVALUATION.md"]:
        add(f"docs/{name}")
    for split in ["development", "evaluation"]:
        for filename in ["demo.mp4", "events.json", "events.csv", "run.json", "run_metadata_addendum.json", "raw_detections.jsonl", "tracks.jsonl"]:
            add(f"outputs/{split}/{filename}")
    for filename in ["events.json", "events.csv", "run.json"]:
        add(f"outputs/evaluation_no_smoothing/{filename}")
    for filename in ["demo_preview.jpg", "failure_examples.jpg", "detection_frame_metrics.json"]:
        add(f"outputs/review/{filename}")
    for path in (ROOT / "outputs/diagnostics").glob("*"):
        if path.is_file():
            files.add(path)

    manifest = {"format": 1, "root": NAME,
                "excluded": ["source videos (fetch using pinned manifest)", "virtual environment", "private candidate notes", "assignment PDF", "annotation working images"],
                "files": {str(p.relative_to(ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(files)}}
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    archive = destination / f"{NAME}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, f"{NAME}/{p.relative_to(ROOT)}")
        z.writestr(f"{NAME}/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    report = {"zip": str(archive), "bytes": archive.stat().st_size,
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "file_count": len(files) + 1}
    (destination / "package_info.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
