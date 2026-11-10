"""Manual requirements installs must include the DeltaNet runtime used by Qwen3.8."""

import ast
from pathlib import Path
import unittest

from packaging.requirements import InvalidRequirement, Requirement


ROOT = Path(__file__).resolve().parents[1]


class Qwen38RequirementsTests(unittest.TestCase):
    def test_base_requirements_include_manager_fla_pin_without_backend_extras(self):
        requirements = {}
        for line in (ROOT / "requirements_base.txt").read_text(encoding="utf-8").splitlines():
            try:
                requirement = Requirement(line)
            except InvalidRequirement:
                continue
            requirements[requirement.name] = requirement
        self.assertIn("flash-linear-attention", requirements)
        requirement = requirements["flash-linear-attention"]
        self.assertFalse(requirement.extras, "FLA must not replace the platform Torch/Triton stack")
        tree = ast.parse((ROOT / "manager/spec.py").read_text(encoding="utf-8"))
        pin = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "FLA" for target in node.targets))
        self.assertEqual(str(requirement), pin)
        self.assertIn(pin, (ROOT / "docker/Dockerfile").read_text(encoding="utf-8"))

    def test_standard_requirements_include_base_dependencies(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("-r requirements_base.txt", lines)


if __name__ == "__main__":
    unittest.main()
