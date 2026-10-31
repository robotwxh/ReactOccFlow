import hashlib
import tempfile
import unittest
from pathlib import Path


class CopyResourceTests(unittest.TestCase):
    # Broken copies, aliasing, and corrupted existing destinations must be caught.
    def test_copy_has_independent_storage_and_verified_content(self):
        from reactoccflow.provenance import copy_resource
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(b"pretrained weights")
            target = root / "assets" / "weights.bin"
            record = copy_resource(source, target)
            self.assertEqual(target.read_bytes(), b"pretrained weights")
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(record["sha256"], hashlib.sha256(b"pretrained weights").hexdigest())
            source.unlink()
            self.assertEqual(target.read_bytes(), b"pretrained weights")

    def test_symlink_source_and_destination_parent_are_rejected(self):
        from reactoccflow.provenance import copy_resource
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(b"weights")
            link = root / "source-link.bin"
            link.symlink_to(source)
            with self.assertRaises(ValueError):
                copy_resource(link, root / "copy.bin")
            (root / "real").mkdir()
            (root / "alias").symlink_to(root / "real", target_is_directory=True)
            with self.assertRaises(ValueError):
                copy_resource(source, root / "alias" / "copy.bin")

    def test_corrupt_existing_target_is_not_silently_replaced(self):
        from reactoccflow.provenance import copy_resource
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "source", root / "target"
            source.write_bytes(b"original")
            target.write_bytes(b"different")
            with self.assertRaises(FileExistsError):
                copy_resource(source, target)
            self.assertEqual(target.read_bytes(), b"different")

    def test_same_path_and_hardlinked_target_are_rejected(self):
        from reactoccflow.provenance import copy_resource
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_bytes(b"data")
            with self.assertRaises(ValueError):
                copy_resource(source, source)
            os.link(source, root / "alias")
            with self.assertRaises(ValueError):
                copy_resource(source, root / "alias")

    def test_repeated_valid_copy_reuses_ordinary_independent_file(self):
        from reactoccflow.provenance import copy_resource
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "source", root / "target"
            source.write_bytes(b"data")
            copy_resource(source, target)
            result = copy_resource(source, target)
            self.assertTrue(result["verified"])
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)

    def test_snapshot_detects_modified_and_added_reference_files(self):
        from reactoccflow.provenance import snapshot_tree, compare_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.py").write_text("original")
            old = snapshot_tree(root)
            self.assertEqual(compare_snapshot(old, snapshot_tree(root)), [])
            (root / "model.py").write_text("modified")
            (root / "extra.py").write_text("generated")
            problems = compare_snapshot(old, snapshot_tree(root))
            self.assertTrue(any("model.py" in p for p in problems))
            self.assertTrue(any("extra.py" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
