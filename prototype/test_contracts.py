"""Check inference equivalence against the actual study source, without importing training code."""
import bootstrap
import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from bootstrap import ROOT
from classification import build_transforms, decision
from prepare_models import download


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT.parent/'tyrelib'/'tyrelib.py').read_text(encoding='utf-8'))
        selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and n.name in ('CoralHead', 'build_transforms')]
        cls.reference = {'np': np}
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<study reference>', 'exec'), cls.reference)

    def test_coral_threshold_decision_matches_study(self):
        for values in ([.8, -.2], [-2, -3], [3, 2], [-1, 1], [0, 0]):
            logits = torch.tensor([values], dtype=torch.float32)
            probs, pred = decision(logits, 'coral')
            reference = self.reference['CoralHead']
            torch.testing.assert_close(probs, reference.probs(logits)[0], rtol=0, atol=0)
            self.assertEqual(pred, reference.predict(logits).item())
        probs, pred = decision(torch.tensor([[.8, -.2]]), 'coral')
        self.assertNotEqual(pred, probs.argmax().item())

    def test_preprocessing_matches_study(self):
        im = Image.fromarray(np.random.default_rng(4).integers(0, 256, (79, 103, 3), dtype=np.uint8))
        for mode in ('raw', 'grayscale', 'clahe'):
            expected = self.reference['build_transforms'](64, False, mode)(im)
            torch.testing.assert_close(build_transforms(64, mode)(im), expected, rtol=0, atol=0)

    def test_rotated_masks_return_to_source_coordinates(self):
        from engine import Engine
        image = np.zeros((19, 31, 3), dtype=np.uint8)
        image[2:9, 7:15, 0] = 255
        image[12:17, 1:5, 1] = 255
        engine = Engine('cpu')
        def predict(rgb, name, threshold=.25):
            if name == 'mobilenetv4':
                return {'task': 'classifier', 'prediction': 0}, None
            return {'task': 'regions'}, np.stack([rgb[:, :, 0]>0, rgb[:, :, 1]>0])
        engine.predict = predict
        expected = np.stack([image[:, :, 0]>0, image[:, :, 1]>0])
        for rotation in (0, 90, 180, 270):
            result, masks = engine.inspect(image, rotation=rotation)
            np.testing.assert_array_equal(masks['segformer_b0'], expected)
            self.assertEqual(result['analysis_rotation_degrees'], rotation)

    def test_completed_partial_recovers_without_get(self):
        content = b'checkpoint fixture'
        metadata = SimpleNamespace(commit_hash='a'*40, etag=hashlib.sha256(content).hexdigest(), size=len(content))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            dest = Path(temp)/'model.pt'
            dest.with_suffix('.pt.part').write_bytes(content)
            with patch('prepare_models.get_hf_file_metadata', return_value=metadata), patch('prepare_models.requests.get') as get:
                download('example/repo', 'model.pt', 'a'*40, dest)
                self.assertEqual(dest.read_bytes(), content)
                get.assert_not_called()

    def test_cached_checkpoint_is_verified_without_get(self):
        content = b'verified local file'
        metadata = SimpleNamespace(commit_hash='a'*40, etag=hashlib.sha256(content).hexdigest(), size=len(content))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            dest = Path(temp)/'model.pt'
            dest.write_bytes(content)
            with patch('prepare_models.get_hf_file_metadata', return_value=metadata), patch('prepare_models.requests.get') as get:
                self.assertEqual(download('example/repo', 'model.pt', 'a'*40, dest), dest)
                get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
