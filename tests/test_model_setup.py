import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "src" / "train.py"


class TestTrainingConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRAIN_FILE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_training_script_exists(self):
        self.assertTrue(TRAIN_FILE.exists())

    def test_random_seed_is_42(self):
        self.assertIn("set_random_seed(42)", self.source)

    def test_image_size_is_160(self):
        self.assertIn("image_size=(160, 160)", self.source)

    def test_batch_size_is_32(self):
        self.assertIn("batch_size=32", self.source)

    def test_training_runs_for_5_epochs(self):
        self.assertIn("epochs=5", self.source)

    def test_mobile_net_v2_is_used(self):
        self.assertIn("MobileNetV2", self.source)

    def test_base_model_is_frozen(self):
        self.assertIn("base.trainable = False", self.source)

    def test_binary_crossentropy_is_used(self):
        self.assertIn('loss="binary_crossentropy"', self.source)

    def test_best_checkpoint_uses_validation_loss(self):
        self.assertIn('monitor="val_loss"', self.source)
        self.assertIn("save_best_only=True", self.source)


if __name__ == "__main__":
    unittest.main()
