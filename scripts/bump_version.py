#!/usr/bin/env python3
import argparse
import configparser
import json
from datetime import datetime, timezone
from pathlib import Path


def load_tally(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "current_version": "0.0.0",
        "major": 0,
        "minor": 0,
        "patch": 0,
        "rules": {
            "major": "Major structural changes (non-engine architecture shifts)",
            "minor": "Feature-level and other minor changes",
            "patch": "Bug fixes"
        },
        "history": []
    }


def parse_semver(value: str):
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semantic version: {value}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def update_setup_cfg(setup_cfg_path: Path, version: str):
    config = configparser.ConfigParser()
    config.read(setup_cfg_path, encoding="utf-8")
    if "metadata" not in config:
        config["metadata"] = {}
    config["metadata"]["version"] = version
    with setup_cfg_path.open("w", encoding="utf-8") as handle:
        config.write(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change", choices=["major", "minor", "patch"], required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    tally_path = repo_root / "version_tally.json"
    setup_cfg_path = repo_root / "setup.cfg"

    tally = load_tally(tally_path)
    major = int(tally.get("major", 0))
    minor = int(tally.get("minor", 0))
    patch = int(tally.get("patch", 0))
    previous = f"{major}.{minor}.{patch}"

    if args.change == "major":
        major += 1
        minor = 0
        patch = 0
    elif args.change == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    current = f"{major}.{minor}.{patch}"

    tally["major"] = major
    tally["minor"] = minor
    tally["patch"] = patch
    tally["current_version"] = current
    history = tally.setdefault("history", [])
    history.append(
        {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "change": args.change,
            "from": previous,
            "to": current,
            "reason": args.reason,
        }
    )

    tally_path.write_text(json.dumps(tally, indent=2) + "\n", encoding="utf-8")
    update_setup_cfg(setup_cfg_path, current)
    print(current)


if __name__ == "__main__":
    main()
