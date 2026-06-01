"""
Tests for src/imageprocessor.py - ImageProcessor class
"""
import math
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np


# ---------------------------------------------------------------------------
# Stub out cv2 before importing the module under test so that OpenCV is not
# required as a real dependency during unit tests.
# ---------------------------------------------------------------------------
_cv2_stub = types.ModuleType("cv2")
_cv2_stub.resize = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
sys.modules.setdefault("cv2", _cv2_stub)

# Now we can safely import the module.
sys.path.insert(0, "/home/jailuser/git")
from src.imageprocessor import ImageProcessor  # noqa: E402


def _make_image(width=100, height=80, channels=3):
    """Return a deterministic NumPy array shaped (height, width, channels)."""
    return np.zeros((height, width, channels), dtype=np.uint8)


class TestLuminance(unittest.TestCase):
    """Tests for ImageProcessor.luminance()"""

    def setUp(self):
        self.proc = ImageProcessor()

    def test_pure_red(self):
        result = self.proc.luminance(255, 0, 0)
        self.assertAlmostEqual(result, 0.2126 * 255, places=5)

    def test_pure_green(self):
        result = self.proc.luminance(0, 255, 0)
        self.assertAlmostEqual(result, 0.7152 * 255, places=5)

    def test_pure_blue(self):
        result = self.proc.luminance(0, 0, 255)
        self.assertAlmostEqual(result, 0.0722 * 255, places=5)

    def test_black(self):
        self.assertAlmostEqual(self.proc.luminance(0, 0, 0), 0.0, places=10)

    def test_white(self):
        result = self.proc.luminance(255, 255, 255)
        # Coefficients sum to 1.0, so white should be exactly 255.
        self.assertAlmostEqual(result, 255.0, places=4)

    def test_mixed_values(self):
        r, g, b = 100, 150, 200
        expected = 0.2126 * 100 + 0.7152 * 150 + 0.0722 * 200
        self.assertAlmostEqual(self.proc.luminance(r, g, b), expected, places=5)

    def test_returns_float(self):
        self.assertIsInstance(self.proc.luminance(128, 128, 128), float)


class TestGetChar(unittest.TestCase):
    """Tests for ImageProcessor.getchar()"""

    def setUp(self):
        self.proc = ImageProcessor()
        self.scale = "@#$%&"  # length 5

    def test_black_pixel_returns_first_char(self):
        # greyvalue=0 → index = floor(0 / 255 * 5) = 0
        self.assertEqual(self.proc.getchar(self.scale, 0), "@")

    def test_low_greyvalue(self):
        # greyvalue=50 → index = floor(50/255 * 5) = floor(0.9804) = 0
        self.assertEqual(self.proc.getchar(self.scale, 50), "@")

    def test_mid_greyvalue(self):
        # greyvalue=128 → index = floor(128/255 * 5) = floor(2.5098) = 2
        self.assertEqual(self.proc.getchar(self.scale, 128), "$")

    def test_near_max_greyvalue(self):
        # greyvalue=254 → index = floor(254/255 * 5) = floor(4.9804) = 4
        self.assertEqual(self.proc.getchar(self.scale, 254), "&")

    def test_single_char_scale(self):
        # With a one-char scale any value in [0, 254] should return that char.
        for greyvalue in (0, 127, 200, 254):
            self.assertEqual(self.proc.getchar("X", greyvalue), "X")

    def test_index_uses_floor(self):
        # Verify that floor (not round) is used for index calculation.
        # greyvalue=102 → 102/255 * 5 = 2.0, floor = 2
        self.assertEqual(self.proc.getchar(self.scale, 102), "$")

    def test_boundary_254(self):
        # 254/255 * 5 = 4.980…, floor = 4 → last valid index for len-5 scale
        char = self.proc.getchar(self.scale, 254)
        self.assertEqual(char, self.scale[4])

    def test_grey_value_255_out_of_bounds(self):
        # greyvalue=255 produces index == len(scale), which is out of range.
        # This documents the known boundary behaviour of the implementation.
        with self.assertRaises(IndexError):
            self.proc.getchar(self.scale, 255)

    def test_longer_scale(self):
        long_scale = "ABCDEFGHIJ"  # length 10
        # greyvalue=25 → floor(25/255 * 10) = floor(0.98) = 0 → 'A'
        self.assertEqual(self.proc.getchar(long_scale, 25), "A")
        # greyvalue=200 → floor(200/255 * 10) = floor(7.84) = 7 → 'H'
        self.assertEqual(self.proc.getchar(long_scale, 200), "H")


class TestSetParams(unittest.TestCase):
    """Tests for ImageProcessor.setparams()"""

    CHAR_WIDTH = 10
    CHAR_HEIGHT = 18

    def setUp(self):
        self.proc = ImageProcessor()

    def _call(self, width=200, height=100, quality=1.0):
        image = _make_image(width=width, height=height)
        self.proc.setparams(image, quality)

    def test_stores_quality(self):
        self._call(quality=0.5)
        self.assertEqual(self.proc.quality, 0.5)

    def test_stores_old_dimensions(self):
        self._call(width=320, height=240)
        self.assertEqual(self.proc.oldwidth, 320)
        self.assertEqual(self.proc.oldheight, 240)

    def test_stores_old_size_tuple(self):
        self._call(width=320, height=240)
        self.assertEqual(self.proc.oldsize, (320, 240))

    def test_newsize_full_quality(self):
        # quality=1.0, width=200, height=100
        # adjustedwidth  = int(200 * 1.0)       = 200
        # adjustedheight = int(100 * 1.0 * 10/18) = int(55.55) = 55
        # newsize = (200 * 10, 55 * 18) = (2000, 990)
        self._call(width=200, height=100, quality=1.0)
        aspect = self.CHAR_WIDTH / self.CHAR_HEIGHT
        aw = int(200 * 1.0)
        ah = int(100 * 1.0 * aspect)
        expected = (aw * self.CHAR_WIDTH, ah * self.CHAR_HEIGHT)
        self.assertEqual(self.proc.newsize, expected)

    def test_newsize_half_quality(self):
        self._call(width=200, height=100, quality=0.5)
        aspect = self.CHAR_WIDTH / self.CHAR_HEIGHT
        aw = int(200 * 0.5)
        ah = int(100 * 0.5 * aspect)
        expected = (aw * self.CHAR_WIDTH, ah * self.CHAR_HEIGHT)
        self.assertEqual(self.proc.newsize, expected)

    def test_newsize_low_quality(self):
        self._call(width=640, height=480, quality=0.1)
        aspect = self.CHAR_WIDTH / self.CHAR_HEIGHT
        aw = int(640 * 0.1)
        ah = int(480 * 0.1 * aspect)
        expected = (aw * self.CHAR_WIDTH, ah * self.CHAR_HEIGHT)
        self.assertEqual(self.proc.newsize, expected)

    def test_quality_zero_gives_zero_newsize(self):
        # Edge case: quality=0 → adjusted dimensions are 0
        self._call(width=100, height=100, quality=0.0)
        self.assertEqual(self.proc.newsize, (0, 0))

    def test_setparams_overwrites_previous(self):
        self._call(width=100, height=100, quality=1.0)
        self._call(width=50, height=50, quality=0.5)
        self.assertEqual(self.proc.oldwidth, 50)
        self.assertEqual(self.proc.oldheight, 50)
        self.assertEqual(self.proc.quality, 0.5)


class TestFontResize(unittest.TestCase):
    """Tests for ImageProcessor.fontresize()"""

    def setUp(self):
        self.proc = ImageProcessor()

    def test_calls_cv2_resize_with_newsize(self):
        image = _make_image(200, 100)
        self.proc.setparams(image, 1.0)

        fake_result = np.zeros((990, 2000, 3), dtype=np.uint8)
        with patch("src.imageprocessor.cv2.resize", return_value=fake_result) as mock_resize:
            result = self.proc.fontresize(image)
            mock_resize.assert_called_once_with(image, self.proc.newsize)
            self.assertIs(result, fake_result)

    def test_returns_resized_image(self):
        image = _make_image(100, 80)
        self.proc.setparams(image, 0.5)

        returned_image = np.zeros((10, 20, 3), dtype=np.uint8)
        with patch("src.imageprocessor.cv2.resize", return_value=returned_image):
            result = self.proc.fontresize(image)
        self.assertIs(result, returned_image)

    def test_fontresize_passes_original_image(self):
        """fontresize must pass the unmodified image object to cv2.resize."""
        image = _make_image(100, 80)
        self.proc.setparams(image, 0.5)

        with patch("src.imageprocessor.cv2.resize", return_value=image) as mock_resize:
            self.proc.fontresize(image)
        args, _ = mock_resize.call_args
        self.assertIs(args[0], image)


class TestImageToAscii(unittest.TestCase):
    """Tests for ImageProcessor.imageToAscii()"""

    def setUp(self):
        self.proc = ImageProcessor()

    def test_calls_setparams(self):
        image = _make_image(100, 80)
        with patch.object(self.proc, "setparams") as mock_sp, \
             patch.object(self.proc, "fontresize", return_value=image):
            self.proc.imageToAscii(image, quality=0.5, scale="@#")
            mock_sp.assert_called_once_with(image, 0.5)

    def test_calls_fontresize(self):
        image = _make_image(100, 80)
        with patch.object(self.proc, "setparams"), \
             patch.object(self.proc, "fontresize", return_value=image) as mock_fr:
            self.proc.imageToAscii(image, quality=0.5, scale="@#")
            mock_fr.assert_called_once()

    def test_returns_none_incomplete_implementation(self):
        # The current implementation is incomplete (TODOs present) and
        # returns None.  This test documents that behaviour so that it is
        # detected if the implementation changes.
        image = _make_image(100, 80)
        with patch.object(self.proc, "setparams"), \
             patch.object(self.proc, "fontresize", return_value=image):
            result = self.proc.imageToAscii(image, quality=0.5, scale="@#")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()