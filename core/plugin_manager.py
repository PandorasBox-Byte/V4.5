"""Lightweight research plugin framework.

Plugins allow arbitrary code to inspect queries and, if appropriate, return a
text response.  They are discovered at startup by scanning a directory of
Python modules and instantiating subclasses of :class:`ResearchPlugin`.

This module is intentionally small and avoids pulling in any heavy
dependencies to keep the core engine lightweight.
"""

import importlib
import os
import sys
from typing import List, Optional, Type


class ResearchPlugin:
    """Base class for research-oriented plugins.

    Subclasses should override ``can_handle`` and ``handle``.  The engine
    instance is provided to give access to memory, embeddings, etc.  Returning
    ``None`` from ``handle`` means the plugin chose not to answer.
    """

    name: str = "base"

    def can_handle(self, query: str) -> bool:
        """Return ``True`` if this plugin should be given a chance to answer.

        The default implementation always returns ``False``; subclasses must
        implement their own detection logic.
        """
        return False

    def handle(self, query: str, engine) -> Optional[str]:
        """Produce a response for ``query`` or ``None`` to pass.

        ``engine`` is the :class:`core.engine_template.Engine` instance.
        """
        return None


def load_plugins(directory: str = "plugins") -> List[ResearchPlugin]:
    """Import and instantiate all plugins found under ``directory``.

    The directory is added to ``sys.path`` temporarily to allow simple
    ``import`` statements.  Files beginning with an underscore are ignored.
    """

    plugins: List[ResearchPlugin] = []
    if not os.path.isdir(directory):
        return plugins

    # temporarily add plugin directory for imports
    abs_dir = os.path.abspath(directory)
    sys.path.insert(0, abs_dir)

    for fname in os.listdir(abs_dir):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        mod_name = os.path.splitext(fname)[0]
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            # ignore modules that fail to import
            continue
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, ResearchPlugin)
                and obj is not ResearchPlugin
            ):
                try:
                    instance = obj()
                except Exception:
                    continue
                plugins.append(instance)
    sys.path.pop(0)
    return plugins
