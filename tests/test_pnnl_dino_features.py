import unittest

import torch
from PIL import Image

from pnnl_pilot.dino_features import (
    build_official_dinov3_transform,
    pool_official_dinov3_outputs,
    resolve_official_dinov3_entrypoint,
)


class OfficialDinoV3FeatureTests(unittest.TestCase):
    def test_transform_produces_frozen_224_tensor(self):
        transform = build_official_dinov3_transform()
        tensor = transform(Image.new("RGB", (320, 160), "white"))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(tensor).all())

    def test_pooling_keeps_cls_and_averages_patch_tokens(self):
        outputs = {
            "x_norm_clstoken": torch.tensor([[1.0, 2.0]]),
            "x_norm_patchtokens": torch.tensor(
                [[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]]
            ),
        }
        cls, mean_patch = pool_official_dinov3_outputs(outputs)
        torch.testing.assert_close(cls, torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(mean_patch, torch.tensor([[3.0, 5.0]]))

    def test_entrypoint_is_restricted_to_frozen_matrix(self):
        self.assertEqual(
            resolve_official_dinov3_entrypoint("dinov3_vitb16"),
            "dinov3_vitb16",
        )
        with self.assertRaisesRegex(ValueError, "unsupported official DINOv3"):
            resolve_official_dinov3_entrypoint("arbitrary_python")


if __name__ == "__main__":
    unittest.main()
