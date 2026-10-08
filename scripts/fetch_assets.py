"""Fetch public inputs via Wikimedia's API and version-pinned SuperTux assets."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c1ddb4f28c54de77f07aa9965231a91c0ed06708"
CLIPS = {
    "development": "Super Tux - Welcome to Antarctica (Walkthrough by PantheraLeo1359531) 2023.01.29 - 12.27.00.08.webm",
    "evaluation": "Super Tux - The Journey Begins (Walkthrough by PantheraLeo1359531) 2023.01.29 - 12.30.19.10.webm",
}


def read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "GameplayEventsAssignment/0.1 (research; source attribution preserved)"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        content = read_url(url)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(content)
        partial.replace(path)
    return {"path": str(path.relative_to(ROOT)), "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    pinned = ROOT / "data" / "sources.json"
    if pinned.exists():
        manifest = json.loads(pinned.read_text())
        entries = list(manifest["videos"].values()) + manifest["assets"]
        def pinned_fetch(entry):
            result = fetch(entry["url"], ROOT / entry["path"])
            if result["sha256"] != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {entry['path']}")
            return result["path"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for path in executor.map(pinned_fetch, entries):
                print("verified", path, flush=True)
        return
    manifest = {"supertux_commit": COMMIT, "videos": {}, "assets": []}
    for split, name in CLIPS.items():
        api = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "format": "json", "prop": "videoinfo", "viprop": "url|derivatives", "titles": "File:" + name})
        info = next(iter(json.loads(read_url(api))["query"]["pages"].values()))["videoinfo"][0]
        derivative = next(x for x in info["derivatives"] if x.get("transcodekey") == "480p.vp9.webm")
        result = fetch(derivative["src"], ROOT / "data" / "videos" / (split + ".webm"))
        result.update({"source_page": info["descriptionurl"], "title": name, "creator": "PantheraLeo1359531", "license_as_listed": "GPL-3.0-or-later", "transcode": "480p.vp9.webm", "role": split})
        manifest["videos"][split] = result
        print(split, result["bytes"], flush=True)
    paths = ([f"data/images/creatures/snowball/snowball-{i}.png" for i in range(8)]
             + [f"data/images/creatures/mr_iceblock/iceblock-{i}.png" for i in range(8)]
             + [f"data/images/objects/coin/coin-{i}.png" for i in range(16)]
             + ["data/images/creatures/snowball/snowball.sprite", "data/images/creatures/mr_iceblock/mr_iceblock.sprite", "data/images/objects/coin/coin.sprite", "data/fonts/white.stf", "data/fonts/README", "data/images/engine/fonts/white.png", "LICENSE.txt", "README.md"])
    def asset(path):
        return fetch(f"https://raw.githubusercontent.com/SuperTux/supertux/{COMMIT}/{path}", ROOT / "assets" / "upstream" / path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        manifest["assets"] = list(executor.map(asset, paths))
    (ROOT / "data" / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved data/sources.json", flush=True)


if __name__ == "__main__":
    main()
