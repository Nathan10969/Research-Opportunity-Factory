import csv
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pnnl_pilot.contracts import (
    ContractViolation,
    load_assets,
    select_optical_sources,
    validate_member_path,
)
from pnnl_pilot.extract import extract_selected_members
from pnnl_pilot.cleaning import (
    ContaminatedInput,
    apply_fixed_metadata_crop,
    assert_training_path,
)
from pnnl_pilot.metrics import sample_balanced_mae
from pnnl_pilot.spatial import CropGeometryError, crop_box_for_mm, project_mm_to_pixel
from pnnl_pilot.splits import build_loso_folds


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_CSV = (
    REPO_ROOT
    / "data_contracts"
    / "pnnl_binding_contract"
    / "eligible_optical_assets.csv"
)


class ContractTests(unittest.TestCase):
    def test_selects_exactly_nine_eligible_10x_data_members(self):
        rows = load_assets(ASSETS_CSV)
        selected = select_optical_sources(rows)
        self.assertEqual(
            {row.sample_id for row in selected},
            {"SS01", "SS02", "SS03", "SS04", "SS05", "SS06", "SS07", "SS08", "SS31"},
        )
        self.assertTrue(all("/DATA/" in row.member_path for row in selected))
        self.assertTrue(
            all(
                row.member_path.endswith("_10X_Nugget-Region-Etched.jpg")
                for row in selected
            )
        )

    def test_rejects_overlay_and_quarantined_members(self):
        for forbidden in ("/VISUALIZATION/MARKER/", "/VISUALIZATION/crop-7mm/"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ContractViolation):
                    validate_member_path(forbidden + "bad.jpg", "SS01")
        for sample_id in ("SS28", "SS32", "SS36"):
            with self.subTest(sample_id=sample_id):
                with self.assertRaises(ContractViolation):
                    validate_member_path(
                        "/DATA/x_10X_Nugget-Region-Etched.jpg", sample_id
                    )

    def test_duplicate_sample_is_rejected(self):
        rows = load_assets(ASSETS_CSV)
        with self.assertRaises(ContractViolation):
            select_optical_sources(rows + [rows[0]])


class ExtractionTests(unittest.TestCase):
    def test_extracts_streams_and_hashes_only_selected_members(self):
        rows = load_assets(ASSETS_CSV)[:2]
        payloads = {
            rows[0].member_path: b"first-image",
            rows[1].member_path: b"second-image",
            "root/VISUALIZATION/MARKER/forbidden.jpg": b"overlay",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            tar_path = temp / "fixture.tar"
            with tarfile.open(tar_path, "w") as archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

            output_dir = temp / "extracted"
            records = extract_selected_members(tar_path, rows, output_dir)

            self.assertEqual(len(records), 2)
            self.assertEqual({record.sample_id for record in records}, {"SS01", "SS02"})
            for row in rows:
                output_path = output_dir / row.sample_id / row.file_name
                self.assertTrue(output_path.is_file())
                expected = hashlib.sha256(payloads[row.member_path]).hexdigest()
                actual = next(
                    record.sha256 for record in records if record.sample_id == row.sample_id
                )
                self.assertEqual(actual, expected)
            self.assertFalse((output_dir / "forbidden.jpg").exists())

    def test_missing_selected_member_fails_closed(self):
        row = load_assets(ASSETS_CSV)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            tar_path = temp / "empty.tar"
            with tarfile.open(tar_path, "w"):
                pass
            with self.assertRaises(ContractViolation):
                extract_selected_members(tar_path, [row], temp / "out")


class CleaningTests(unittest.TestCase):
    def test_fixed_metadata_band_is_removed_without_labels(self):
        image = Image.new("RGB", (100, 100), "gray")
        for y in range(90, 100):
            for x in range(100):
                image.putpixel((x, y), (255, 0, 0))
        cleaned, policy = apply_fixed_metadata_crop(image, bottom_fraction=0.10)
        self.assertEqual(cleaned.size, (100, 90))
        self.assertEqual(policy["crop_box"], [0, 0, 100, 90])
        self.assertNotIn((255, 0, 0), cleaned.get_flattened_data())

    def test_training_loader_rejects_qa_overlay_and_quarantined_paths(self):
        for path in (
            "qa_overlays/SS01.jpg",
            "VISUALIZATION/MARKER/SS01.jpg",
            "VISUALIZATION/crop-7mm/SS01.jpg",
            "clean_images/SS28.jpg",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ContaminatedInput):
                    assert_training_path(path)


class SpatialTests(unittest.TestCase):
    def test_projection_uses_image_y_axis_direction(self):
        self.assertEqual(project_mm_to_pixel(0, 0, 100, 200, 0, 10), (100, 200))
        self.assertEqual(project_mm_to_pixel(1, 0, 100, 200, 0, 10), (110, 200))
        self.assertEqual(project_mm_to_pixel(0, 1, 100, 200, 0, 10), (100, 190))

    def test_crop_box_matches_physical_size_and_rejects_metadata_band(self):
        self.assertEqual(crop_box_for_mm(500, 400, 1.0, 100, 1000, 800, 760), (450, 350, 550, 450))
        with self.assertRaises(CropGeometryError):
            crop_box_for_mm(500, 740, 1.0, 100, 1000, 800, 760)


class SplitAndMetricTests(unittest.TestCase):
    def test_loso_has_zero_sample_overlap(self):
        sample_ids = ["SS01", "SS01", "SS02", "SS03"]
        folds = build_loso_folds(sample_ids)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertTrue(set(fold.train_samples).isdisjoint(fold.test_samples))
            self.assertEqual(len(fold.test_samples), 1)

    def test_sample_balanced_mae_weights_samples_equally(self):
        y_true = [0.0, 0.0, 0.0, 0.0]
        y_pred = [1.0, 1.0, 1.0, 9.0]
        sample_ids = ["dense", "dense", "dense", "sparse"]
        self.assertEqual(sample_balanced_mae(y_true, y_pred, sample_ids), 5.0)


if __name__ == "__main__":
    unittest.main()
