#!/usr/bin/env python3
"""Brain Monitor - Real-time Neural Activity Visualization.

Displays an ASCII tree of core/ files and lights them up (green flash) when
code from those files executes - like watching neurons fire in a brain.
"""

import os
import sys
import time
import curses
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Set, Optional


class BrainMonitor:
    """Monitor and visualize Python file execution as neural activity."""
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.activity_log = deque(maxlen=1000)
        self.active_files: Dict[str, float] = {}
        self.fire_counts: Dict[str, int] = defaultdict(int)
        self.flash_duration = 0.3  # seconds
        self.monitoring = False
        self.lock = threading.Lock()
        self.files = self._scan_core_files()
        self.activity_file = os.path.join(self.workspace_root, "data", "brain_activity.log")
        self.last_read_pos = 0
        
    def _scan_core_files(self) -> list:
        """Scan core/ directory for Python files."""
        core_path = Path(self.workspace_root) / "core"
        files = []
        
        if core_path.exists():
            for py_file in sorted(core_path.glob("*.py")):
                if py_file.name not in ("__init__.py", "__pycache__"):
                    files.append(py_file.name)
        
        return files
        
    def record_activity(self, filename: str):
        """Record that code from filename was just executed."""
        if not filename or not filename.endswith(".py"):
            return
            
        base_name = os.path.basename(filename)
        
        # Only track core/ files - check if basename is in our files list
        if base_name not in self.files:
            return
            
        with self.lock:
            timestamp = time.time()
            self.active_files[base_name] = timestamp
            self.fire_counts[base_name] += 1
            self.activity_log.append((timestamp, base_name))
    
    def get_active_files(self) -> Set[str]:
        """Return set of files currently 'firing' (recently active)."""
        current_time = time.time()
        active = set()
        
        with self.lock:
            for filename, last_access in list(self.active_files.items()):
                if current_time - last_access < self.flash_duration:
                    active.add(filename)
                else:
                    del self.active_files[filename]
                    
        return active
    
    def _read_activity_from_file(self):
        """Read new activity entries from log file (IPC mechanism)."""
        try:
            if not os.path.exists(self.activity_file):
                return
            
            # Check if file was recreated (smaller than our last read position)
            file_size = os.path.getsize(self.activity_file)
            if file_size < self.last_read_pos:
                self.last_read_pos = 0  # File was recreated, reset position
            
            with open(self.activity_file, 'r') as f:
                f.seek(self.last_read_pos)
                new_lines = f.readlines()
                self.last_read_pos = f.tell()
                
                for line in new_lines:
                    line = line.strip()
                    if line:
                        # Format: timestamp|filename
                        parts = line.split('|')
                        if len(parts) == 2:
                            filename = parts[1]
                            self.record_activity(filename)
        except Exception:
            pass
    
    def run_text_visualization(self):
        """Simple text-based visualization for Terminal windows."""
        self.monitoring = True
        print("\n" + "=" * 60)
        print("  BRAIN MONITOR - Neural Activity Visualization")
        print("=" * 60)
        print(f"\nMonitoring {len(self.files)} core modules")
        print("(Reading activity from engine process)")
        print("\nPress Ctrl+C to stop\n")
        
        import signal
        
        def signal_handler(sig, frame):
            self.monitoring = False
            print("\n\nBrain monitor stopped.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Text-based update loop
        last_update = {}
        while self.monitoring:
            try:
                # Read activity from file (IPC)
                self._read_activity_from_file()
                
                active = self.get_active_files()
                
                # Print updates for active files
                for filename in active:
                    if filename not in last_update or time.time() - last_update[filename] > 1.0:
                        fire_count = self.fire_counts.get(filename, 0)
                        print(f"✓ {filename} ({fire_count} fires)")
                        last_update[filename] = time.time()
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                self.monitoring = False
                break
            except Exception:
                pass

    def run_visualization(self, stdscr):
        """Main visualization loop with curses."""
        curses.curs_set(0)
        stdscr.nodelay(True)
        
        # Colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Active/firing
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Headers
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Stats
        curses.init_pair(4, curses.COLOR_WHITE, -1)   # Inactive
        
        self.monitoring = True
        
        while self.monitoring:
            try:
                # Read activity from file (IPC)
                self._read_activity_from_file()
                
                stdscr.clear()
                maxy, maxx = stdscr.getmaxyx()
                
                # Header
                title = "BRAIN MONITOR - Neural Activity Visualization"
                stdscr.addstr(0, 2, title[:maxx-4], curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(1, 2, "=" * min(len(title), maxx-4), curses.color_pair(2))
                
                active = self.get_active_files()
                
                # Draw tree
                row = 3
                stdscr.addstr(row, 2, "core/", curses.color_pair(2) | curses.A_BOLD)
                row += 1
                
                for i, filename in enumerate(self.files):
                    if row >= maxy - 7:
                        break
                        
                    is_last = (i == len(self.files) - 1)
                    connector = "└── " if is_last else "├── "
                    
                    is_active = filename in active
                    fire_count = self.fire_counts.get(filename, 0)
                    
                    if is_active:
                        color = curses.color_pair(1) | curses.A_BOLD
                        indicator = "⚡"
                    else:
                        color = curses.color_pair(4)
                        indicator = " "
                    
                    display = f"  {connector}{indicator} {filename}"
                    if fire_count > 0:
                        display += f" ({fire_count})"
                    
                    stdscr.addstr(row, 2, display[:maxx-4], color)
                    row += 1
                
                # Stats
                if row < maxy - 5:
                    row += 1
                    stdscr.addstr(row, 2, "-" * min(40, maxx-4), curses.color_pair(3))
                    row += 1
                    
                    total_fires = sum(self.fire_counts.values())
                    active_count = len(active)
                    
                    stdscr.addstr(row, 2, f"Active modules: {active_count}", curses.color_pair(3))
                    row += 1
                    stdscr.addstr(row, 2, f"Total neural fires: {total_fires}", curses.color_pair(3))
                    row += 1
                    
                    if self.fire_counts:
                        top = sorted(self.fire_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        stdscr.addstr(row, 2, "Top firing neurons:", curses.color_pair(3))
                        row += 1
                        for fname, count in top:
                            if row < maxy - 1:
                                stdscr.addstr(row, 4, f"  {fname}: {count} fires"[:maxx-6], curses.color_pair(3))
                                row += 1
                
                # Footer
                if maxy > 2:
                    footer = "Press q to quit | Green = Active neurons firing"
                    stdscr.addstr(maxy-1, 2, footer[:maxx-4], curses.color_pair(4))
                
                stdscr.refresh()
                
                # Check for quit
                try:
                    key = stdscr.getch()
                    if key in (ord('q'), ord('Q')):
                        self.monitoring = False
                except:
                    pass
                    
                time.sleep(0.05)  # 20 FPS
                
            except KeyboardInterrupt:
                self.monitoring = False
                break
            except Exception:
                pass


def start_monitor_window(workspace_root: Optional[str] = None, use_curses: bool = True):
    """Start brain monitor visualization.
    
    Args:
        workspace_root: Path to workspace directory
        use_curses: If True, use curses UI. If False, use text-based output.
    """
    monitor = BrainMonitor(workspace_root)
    
    try:
        if use_curses:
            curses.wrapper(monitor.run_visualization)
        else:
            monitor.run_text_visualization()
    except Exception as e:
        # If curses fails, try to write error to file for debugging
        try:
            with open("/tmp/brain_monitor_error.log", "a") as f:
                f.write(f"Brain monitor error: {e}\n")
                import traceback
                f.write(traceback.format_exc())
        except:
            pass
        print(f"Brain monitor error: {e}", file=sys.stderr)
    
    return monitor


def launch_brain_monitor_async(workspace_root: Optional[str] = None):
    """Launch brain monitor in separate Terminal window (macOS).
    
    Returns:
        subprocess.Popen object for the Terminal.app process (for cleanup), or None if launch failed
    """
    import subprocess
    import tempfile
    
    workspace = workspace_root or os.getcwd()
    
    # Find the correct Python interpreter in the venv
    # Try .venv311 first, fall back to system python3
    python_paths = [
        os.path.join(workspace, ".venv311", "bin", "python3"),
        os.path.join(workspace, ".venv", "bin", "python3"),
        "/usr/local/bin/python3",
        "/usr/bin/python3"
    ]
    
    python_exe = None
    for path in python_paths:
        if os.path.exists(path):
            python_exe = path
            break
    
    if not python_exe:
        print("Error: Could not find Python executable", file=sys.stderr)
        return None
    
    # Create a temporary shell script to run the monitor
    # When the monitor is killed (e.g., by Ctrl+C in main window),
    # the bash script exits and closes the Terminal window without prompting
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(f'''#!/bin/bash
cd "{workspace}"
"{python_exe}" -m core.brain_monitor
# Close the Terminal window when the monitor exits (without confirmation)
osascript -e 'tell application "Terminal" to close front window without saving' 2>/dev/null || true
exit
''')
        script_path = f.name
    
    try:
        os.chmod(script_path, 0o755)
        
        # Use AppleScript to open Terminal and run the script
        applescript = f'''
tell application "Terminal"
    -- Open a new window and run the brain monitor script
    do script "bash '{script_path}'"
    -- Activate Terminal and bring to front
    activate
end tell
'''
        
        # Use subprocess.Popen to capture the process for later cleanup
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            timeout=5
        )
        
        # Return a dummy process-like object since we can't directly track the Terminal window
        # Instead, we'll store metadata about the monitor for cleanup
        class MonitorProcess:
            def __init__(self, script_path):
                self.script_path = script_path
                self.pid = None
                
            def terminate(self):
                """Kill any remaining brain_monitor processes"""
                try:
                    subprocess.run(
                        ["pkill", "-f", "core.brain_monitor"],
                        timeout=2
                    )
                except Exception:
                    pass
                    
            def kill(self):
                """Force kill brain_monitor processes"""
                try:
                    subprocess.run(
                        ["pkill", "-9", "-f", "core.brain_monitor"],
                        timeout=2
                    )
                except Exception:
                    pass
                    
            def wait(self, timeout=None):
                """Wait for monitor to exit"""
                pass
        
        if result.returncode != 0 and result.stderr:
            stderr = result.stderr.decode() if result.stderr else ""
            print(f"Warning: Brain monitor launch had issues: {stderr}", file=sys.stderr)
            return None
        
        # Clean up script file after a delay and return monitor process handle
        def cleanup():
            import time
            time.sleep(10)
            try:
                os.unlink(script_path)
            except:
                pass
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
        
        return MonitorProcess(script_path)
        
    except subprocess.TimeoutExpired:
        pass  # AppleScript timed out, but window should be open
        return None
    except Exception as e:
        print(f"Could not launch brain monitor: {e}", file=sys.stderr)
        return None


# Global monitor instance for trace hook
_monitor_instance: Optional[BrainMonitor] = None


def install_trace_hook(monitor: BrainMonitor):
    """Install sys.settrace hook to monitor code execution."""
    global _monitor_instance
    _monitor_instance = monitor
    
    def trace_calls(frame, event, arg):
        if event == 'call' and _monitor_instance:
            filename = frame.f_code.co_filename
            if filename and 'core/' in filename:
                _monitor_instance.record_activity(filename)
        return trace_calls
    
    sys.settrace(trace_calls)


if __name__ == "__main__":
    # When run as main, use curses UI if available, fall back to text mode
    try:
        start_monitor_window(use_curses=True)
    except Exception:
        # Fall back to text-based visualization
        start_monitor_window(use_curses=False)
