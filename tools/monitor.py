#!/usr/bin/env python3
"""
Lightweight ASCII monitor for EvoAI V4.5.
Displays CPU load, simple memory info for the engine PID, active modules, uptime, and heartbeat.

Designed to be dependency-free (standard library only).
"""
import argparse
import datetime
import os
import subprocess
import sys
import time


def clear():
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def read_pid(pidfile):
    try:
        with open(pidfile, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def pid_stats(pid):
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "%cpu=,rss=,etime="]).decode().strip()
        if not out:
            return None
        cpu, rss, etime = [p.strip() for p in out.split(None, 2)]
        return {"cpu": cpu, "rss_kb": rss, "etime": etime}
    except Exception:
        return None


def active_modules(core_dir="core"):
    try:
        files = [f for f in os.listdir(core_dir) if f.endswith(".py")]
        return files
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pidfile", default="data/engine.pid")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    start = time.time()

    try:
        while True:
            clear()
            now = datetime.datetime.now()
            uptime = int(time.time() - start)
            loadavg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)

            pid = read_pid(args.pidfile)
            if pid:
                stats = pid_stats(pid)
                pid_status = f"PID {pid} running" if stats else f"PID {pid} (not responding)"
            else:
                stats = None
                pid_status = "No engine PID file"

            modules = active_modules()

            print("EvoAI V4.5 Monitor")
            print("------------------")
            print(f"Time: {now.isoformat()}  |  Uptime: {uptime}s")
            print(f"Load Avg: {loadavg[0]:.2f} {loadavg[1]:.2f} {loadavg[2]:.2f}")
            print("")
            print("Engine:")
            print(f"  Status: {pid_status}")
            if stats:
                print(f"  CPU% : {stats['cpu']}")
                print(f"  RSS  : {stats['rss_kb']} KB")
                print(f"  Elap : {stats['etime']}")
            print("")
            print("Modules (core/*.py):")
            for m in modules:
                print(f"  - {m}")
            print("")
            print("Heartbeat: ." + "." * ((int(time.time()) % 3)))
            print("")
            print("Press Ctrl+C to exit monitor.")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitor exiting...")


if __name__ == '__main__':
    main()
