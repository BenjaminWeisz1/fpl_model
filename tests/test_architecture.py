"""Structural rules for the repository, checked on every test run."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "fpl_model"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


class PackageBoundaries(unittest.TestCase):
    def package_files(self):
        return [p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts and "open_fpl_solver" not in p.parts]

    def test_production_never_imports_research_or_tools(self):
        offenders = [
            f"{path.relative_to(ROOT)} imports {module}"
            for path in self.package_files()
            for module in imported_modules(path)
            if module.split(".")[0] in ("research", "tools")
        ]
        self.assertEqual(offenders, [])

    def test_no_sys_path_manipulation_in_the_package(self):
        # The vendored solver is the one exception: it is imported as `dev.solver`, outside the package.
        offenders = [
            str(path.relative_to(ROOT))
            for path in self.package_files()
            if "sys.path.insert" in path.read_text(encoding="utf-8-sig")
            and "open_fpl_solver" not in path.read_text(encoding="utf-8-sig")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
