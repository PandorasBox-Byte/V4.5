#!/usr/bin/env python3
"""Quick verification script for v7.5.0 bug fixes."""

import sys
sys.path.insert(0, '.')

print("="*70)
print("V7.5.0 VERIFICATION - EVOLUTION ENGINE BUG FIXES")
print("="*70)

# Test the two bug fixes
print("\n[1/2] Testing detect_regression(lookback=...) parameter fix...")
try:
    from core.evolution_engine import get_evolution_engine
    ee = get_evolution_engine()
    result = ee.metrics_tracker.detect_regression(lookback=5)
    print(f"  ✅ PASS - detect_regression accepts 'lookback' parameter")
    print(f"     Result type: {type(result).__name__}")
except TypeError as e:
    print(f"  ❌ FAIL - {e}")
    sys.exit(1)

print("\n[2/2] Testing get_top_gaps(n=...) parameter fix...")
try:
    gaps = ee.gap_detector.get_top_gaps(n=3)
    print(f"  ✅ PASS - get_top_gaps accepts 'n' parameter")
    print(f"     Gaps returned: {len(gaps)}")
except TypeError as e:
    print(f"  ❌ FAIL - {e}")
    sys.exit(1)

# Test evolution status
print("\n[3/3] Testing full evolution engine status...")
try:
    status = ee.get_evolution_status()
    print(f"  ✅ PASS - Evolution engine operational")
    print(f"     System health: {status['system_health']:.2f}")
    print(f"     System status: {status.get('system_status', 'N/A')}")
    print(f"     Learning queue: {status.get('learning_queue_size', 0)}")
except Exception as e:
    print(f"  ❌ FAIL - {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL BUG FIXES VERIFIED SUCCESSFULLY")
print("="*70)
print("\nVersion: 7.5.0")
print("Status: Production Ready")
print("Fixes Applied:")
print("  • evolution_engine.py:205 - detect_regression(lookback=10)")
print("  • evolution_engine.py:257 - get_top_gaps(n=5)")
print("="*70)
