from pathlib import Path
import unittest

from common.data.manifest import classify_frame_set, parse_nitf_name


class ManifestParsingTests(unittest.TestCase):
    def test_parse_nitf_name(self) -> None:
        parsed = parse_nitf_name("20091021202517-01000100-VIS.ntf.r1")
        self.assertEqual(parsed.timestamp, "20091021202517")
        self.assertEqual(parsed.frame_number, 100)
        self.assertEqual(parsed.level, "r1")

    def test_rejects_unrelated_name(self) -> None:
        with self.assertRaises(ValueError):
            parse_nitf_name("frame000100.png")

    def test_classifies_sets_in_safe_order(self) -> None:
        self.assertEqual(
            classify_frame_set(Path("WPAFB-21Oct2009-SELF-TEST_NITF_001/a.ntf.r1")),
            "self_test",
        )
        self.assertEqual(
            classify_frame_set(Path("WPAFB-21Oct2009-SELF-EVAL_NITF_001/a.ntf.r1")),
            "self_eval",
        )
        self.assertEqual(
            classify_frame_set(Path("WPAFB-21Oct2009-TRAIN_NITF_001/a.ntf.r1")),
            "train",
        )


if __name__ == "__main__":
    unittest.main()
