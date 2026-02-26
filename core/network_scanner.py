"""Placeholder network scanning utilities.

This module exists to satisfy the conceptual requirement that the engine be
"aware of its surroundings". **It does not perform any real network
reconnaissance.**  Scanning code is intentionally inert; attempts to use it
print a warning and return an empty list.  Actual network exploration would
require explicit owner consent and is outside the scope of this project.
"""

import os


def scan_local_network(timeout: float = 5.0) -> list[str]:
    """Pretend to scan the local network and return discovered hostnames/IPs.

    This is a NO-OP function.  It will print a warning if the corresponding
    environment variable is not set to "permitted" and will return an empty
    list in all cases.
    """
    if os.environ.get("EVOAI_ENABLE_NET_SCAN", "0").lower() not in ("1", "true", "yes", "permitted"):
        print("[network_scanner] network scanning is disabled for safety.")
        return []
    # if permission was given, we still decline to scan but simulate a result
    print("[network_scanner] permission granted but actual scanning is disabled.")
    return []
