"""Exercise search aggregation with lightweight stubs, without importing torch."""
import ast
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SearchTests(unittest.TestCase):
    def test_image_weighted_average_and_empty_input(self):
        source = ast.parse((ROOT / "runtime/scripts/binary_rate_search.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run_one_search")
        function.decorator_list = []
        namespace = {"tqdm": lambda data, **kwargs: data}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "search_stub", "exec"), namespace)
        model = types.SimpleNamespace(run_model=lambda **kw: {"bpp": types.SimpleNamespace(item=lambda: kw["real_images"].value)})
        data = [{"real_images": types.SimpleNamespace(shape=(size,), value=value)} for size, value in [(8, 0.1), (2, 0.2)]]
        self.assertAlmostEqual(namespace["run_one_search"](model, data, 1, 1), 0.12)
        with self.assertRaises(ValueError):
            namespace["run_one_search"](model, [], 1, 1)

    def test_candidate_reconstruction_disables_gradients(self):
        tree = ast.parse((ROOT / "runtime/scripts/beta_selection.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_reconstructions")
        self.assertTrue(any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "no_grad" for d in function.decorator_list))


if __name__ == "__main__":
    unittest.main()
