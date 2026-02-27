#!/usr/bin/env python3
import os
import sys


if __name__ == "__main__" and __package__ is None:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_this_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from core.engine_template import Engine


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("LIVE_TEST=SKIP (missing GITHUB_TOKEN/GH_TOKEN)")
        return 2

    os.environ.setdefault("EVOAI_BACKEND_PROVIDER", "github")
    os.environ.setdefault("EVOAI_COPILOT_CHAT", "1")
    os.environ.setdefault("EVOAI_USE_THESAURUS", "0")

    engine = Engine()
    status = engine.status()
    if status.get("backend_active") != "true":
        print("LIVE_TEST=FAIL (github backend not active)")
        return 1

    prompt = os.environ.get("EVOAI_LIVE_TEST_PROMPT", "Give a concise explanation of why unit tests matter.")
    reply = engine.respond(prompt)
    if not isinstance(reply, str) or not reply.strip():
        print("LIVE_TEST=FAIL (empty reply)")
        return 1

    print("LIVE_TEST=PASS")
    print(f"REPLY_PREVIEW={reply[:180].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
