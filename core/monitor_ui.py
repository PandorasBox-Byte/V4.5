#!/usr/bin/env python3
"""Simple ASCII dashboard for observing the engine.

This script can either start the engine itself (passing a progress callback) or
connect to an existing engine's REST API to poll status.  In both cases a grid
of status indicators is rendered, with green dots for healthy components and
red for missing/errored ones.

Usage:

    python core/monitor_ui.py [--api http://host:port] [--start-engine]

If ``--start-engine`` is supplied, the engine is created in-process; otherwise
``--api`` must point at a running server.  In the former mode you can still use
``--api`` to have the monitor poll a separately started engine for additional
reports.
"""
from __future__ import annotations

import argparse
import curses
import json
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

from core.engine_template import Engine


class Dashboard:
    def __init__(self, engine: Engine | None = None, api_url: str | None = None):
        self.engine = engine
        self.api_url = api_url
        self.progress = {"frac": 0.0, "msg": ""}
        self.status = {}

    def start_engine(self):
        def cb(frac, msg=""):
            self.progress["frac"] = frac
            self.progress["msg"] = msg

        # instantiate engine in a background thread so curses main loop can run
        def _target():
            self.engine = Engine(progress_cb=cb)

        t = threading.Thread(target=_target, name="EngineStarter", daemon=True)
        t.start()
        return t

    def _fetch_status(self) -> dict[str, Any]:
        if self.api_url and not self.engine:
            try:
                req = Request(self.api_url.rstrip("/") + "/status", headers={"Accept": "application/json"})
                with urlopen(req, timeout=1) as resp:
                    payload = resp.read().decode("utf-8")
                return json.loads(payload)
            except Exception:
                return {"error": "unreachable"}
        elif self.engine:
            try:
                return self.engine.status()
            except Exception:
                return {"error": "exception"}
        else:
            return {}

    def run(self):
        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, 0)
        curses.init_pair(2, curses.COLOR_GREEN, 0)
        maxy, maxx = stdscr.getmaxyx()
        while True:
            stdscr.erase()
            # progress bar area
            if self.progress["frac"] < 1.0:
                bar_width = maxx - 20
                filled = int(bar_width * self.progress["frac"])
                stdscr.addstr(0, 0, "Loading: [" + "#" * filled + " " * (bar_width - filled) + "]")
                stdscr.addstr(1, 0, f"{self.progress['msg']}")
            # status grid below
            self.status = self._fetch_status()
            row = 3
            for key, val in sorted(self.status.items()):
                color = (
                    curses.color_pair(2)
                    if str(val).lower() in ("ok", "loaded", "running", "yes", "true", "enabled")
                    else curses.color_pair(1)
                )
                stdscr.addstr(row, 0, f"{key:15}")
                stdscr.addstr(row, 16, "●", color)
                stdscr.addstr(row, 18, f"{val}")
                row += 1
                if row >= maxy - 1:
                    break
            stdscr.refresh()
            time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="URL of engine REST API (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--start-engine", action="store_true", help="Instantiate engine in this process")
    args = parser.parse_args()
    dash = Dashboard(api_url=args.api, engine=None)
    if args.start_engine:
        dash.start_engine()
    dash.run()
