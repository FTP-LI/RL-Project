import unittest
from pathlib import Path

from pendulum_lab.config import load_config
from pendulum_lab.utils.paths import resolve_project_path


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config(Path("configs/double_rotary_pendulum.json"))
        self.assertEqual(config["model_path"], "assets/double_rotary_pendulum.xml")
        self.assertGreater(config["env"]["torque_limit"], 0)

    def test_model_file_exists(self) -> None:
        config = load_config(Path("configs/double_rotary_pendulum.json"))
        self.assertTrue(resolve_project_path(config["model_path"]).exists())

    def test_motor_profile_exists(self) -> None:
        config = load_config(Path("configs/double_rotary_pendulum.json"))
        self.assertTrue(resolve_project_path(config["motor_profile"]).exists())


if __name__ == "__main__":
    unittest.main()
