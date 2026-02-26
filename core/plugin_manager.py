"""Lightweight research plugin framework.

Plugins allow arbitrary code to inspect queries and, if appropriate, return a
text response.  They are discovered at startup by scanning a directory of
Python modules and instantiating subclasses of :class:`ResearchPlugin`.

This module is intentionally small and avoids pulling in any heavy
dependencies to keep the core engine lightweight.
"""

import importlib
import importlib.util
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

    abs_dir = os.path.abspath(directory)
    pkg_name = os.path.basename(abs_dir)

    # If the directory is a proper package (has __init__), import modules
    # as `plugins.<module>`; otherwise fall back to loading by file.
    files = [f for f in os.listdir(abs_dir) if f.endswith(".py") and not f.startswith("_")]
    for fname in files:
        mod_name = os.path.splitext(fname)[0]
        module = None
        try:
            if os.path.exists(os.path.join(abs_dir, "__init__.py")):
                module = importlib.import_module(f"{pkg_name}.{mod_name}")
            else:
                # load module directly from file without modifying sys.path
                path = os.path.join(abs_dir, fname)
                spec = importlib.util.spec_from_file_location(mod_name, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
        except Exception:
            # ignore modules that fail to import
            module = None

        if not module:
            continue

        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, ResearchPlugin) and obj is not ResearchPlugin:
                try:
                    instance = obj()
                except Exception:
                    continue
                plugins.append(instance)

    return plugins
