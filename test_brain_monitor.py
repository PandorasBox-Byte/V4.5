#!/usr/bin/env python3
"""Simple test for brain monitor - simulates some file execution."""
import sys
import time
from core.brain_monitor import BrainMonitor, install_trace_hook

def test_brain_monitor():
    """Test the brain monitor by simulating code execution."""
    print("Initializing brain monitor...")
    monitor = BrainMonitor()
    
    print(f"Found {len(monitor.files)} core files to monitor")
    print(f"Files: {', '.join(monitor.files)}")
    
    # Install trace hook
    install_trace_hook(monitor)
    print("Trace hook installed")
    
    # Simulate some activity by importing and using core modules
    print("\nSimulating neural activity...")
    time.sleep(1)
    
    # Import some core modules (this will trigger the trace hook)
    from core import decision_policy
    from core import memory
    from core import embeddings_cache
    
    time.sleep(0.5)
    
    # Show activity log
    print(f"\nRecorded {len(monitor.activity_log)} neural fires")
    if monitor.fire_counts:
        print("Fire counts:")
        for fname, count in sorted(monitor.fire_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fname}: {count} fires")
    
    print("\n✓ Brain monitor test complete!")
    print("To see live visualization, run: python3 core/brain_monitor.py")

if __name__ == "__main__":
    test_brain_monitor()
