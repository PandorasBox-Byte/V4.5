import os
import unittest

from core.plugin_manager import ResearchPlugin, load_plugins


def create_dummy_plugin(tmp_dir, name="dummy"):
    path = os.path.join(tmp_dir, f"{name}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            """from core.plugin_manager import ResearchPlugin


class Dummy(ResearchPlugin):
    def can_handle(self, query):
        return query == 'hello'

    def handle(self, query, engine):
        return 'hi'
"""
        )
    return path


class PluginTests(unittest.TestCase):
    def test_load_no_directory(self):
        # should return empty list when directory missing
        plugins = load_plugins("nonexistent_dir")
        self.assertEqual(plugins, [])

    def test_load_dummy(self, tmp_path):
        tmpdir = str(tmp_path)
        create_dummy_plugin(tmpdir)
        plugins = load_plugins(tmpdir)
        self.assertEqual(len(plugins), 1)
        p = plugins[0]
        self.assertTrue(isinstance(p, ResearchPlugin))
        self.assertEqual(p.handle('hello', None), 'hi')


if __name__ == "__main__":
    unittest.main(verbosity=2)
