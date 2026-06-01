"""
Tests for src/videoprocessor.py - VideoProcessor class

Because VideoProcessor depends on cv2 (OpenCV) and the legacy
AsciiArt.imageprocessor package, both are stubbed out with lightweight mocks
so that no real video files or OpenCV installation are required.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call, PropertyMock
import numpy as np


# ---------------------------------------------------------------------------
# Build minimal stubs for cv2 and for the old AsciiArt package that
# videoprocessor.py still imports from.
# ---------------------------------------------------------------------------

# ---- cv2 stub ----
_cv2_stub = types.ModuleType("cv2")
_cv2_stub.CAP_PROP_FPS = 5
_cv2_stub.CAP_PROP_FRAME_COUNT = 7
_cv2_stub.COLOR_BGR2RGB = 4

_cv2_stub.VideoCapture = MagicMock()
_cv2_stub.VideoWriter = MagicMock()
_cv2_stub.VideoWriter_fourcc = MagicMock(return_value=0x7634706D)
_cv2_stub.cvtColor = MagicMock(side_effect=lambda frame, code: frame)
_cv2_stub.imshow = MagicMock()
_cv2_stub.waitKey = MagicMock(return_value=0)
_cv2_stub.destroyAllWindows = MagicMock()

sys.modules["cv2"] = _cv2_stub

# ---- AsciiArt package stub (old import path used in videoprocessor.py) ----
_ascii_pkg = types.ModuleType("AsciiArt")
_ascii_ip = types.ModuleType("AsciiArt.imageprocessor")


class _FakeImageProcessor:
    """Minimal stand-in for ImageProcessor."""

    def imageToAscii(self, image, quality, scale):
        return image


_ascii_ip.ImageProcessor = _FakeImageProcessor
sys.modules["AsciiArt"] = _ascii_pkg
sys.modules["AsciiArt.imageprocessor"] = _ascii_ip

# ---- tqdm stub ----
_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda iterable, **kw: iterable
sys.modules.setdefault("tqdm", _tqdm_stub)

# Now import the module under test.
sys.path.insert(0, "/home/jailuser/git")
from src.videoprocessor import VideoProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video_capture(width=640, height=480, fps=30.0, frame_count=100):
    """Return a MagicMock that behaves like cv2.VideoCapture."""
    cap = MagicMock()

    def _get(prop_id):
        return {
            3: float(width),
            4: float(height),
            _cv2_stub.CAP_PROP_FPS: fps,
            _cv2_stub.CAP_PROP_FRAME_COUNT: float(frame_count),
        }.get(prop_id, 0.0)

    cap.get.side_effect = _get
    cap.read.return_value = (True, np.zeros((height, width, 3), dtype=np.uint8))
    cap.set.return_value = True
    cap.release.return_value = None
    return cap


def _make_processor(width=640, height=480, fps=30.0, frame_count=100):
    """Instantiate a VideoProcessor with a mocked video capture."""
    cap = _make_video_capture(width, height, fps, frame_count)
    with patch("src.videoprocessor.cv2.VideoCapture", return_value=cap):
        vp = VideoProcessor("fake_video.mp4")
    vp.video = cap  # keep reference for assertions
    return vp


# ---------------------------------------------------------------------------
# Test __init__
# ---------------------------------------------------------------------------

class TestVideoProcessorInit(unittest.TestCase):

    def test_width_extracted(self):
        vp = _make_processor(width=1280)
        self.assertEqual(vp.width, 1280)

    def test_height_extracted(self):
        vp = _make_processor(height=720)
        self.assertEqual(vp.height, 720)

    def test_size_tuple(self):
        vp = _make_processor(width=1920, height=1080)
        self.assertEqual(vp.size, (1920, 1080))

    def test_fps_extracted(self):
        vp = _make_processor(fps=24.0)
        self.assertAlmostEqual(vp.fps, 24.0)

    def test_length_extracted(self):
        vp = _make_processor(frame_count=250)
        self.assertEqual(vp.length, 250)

    def test_video_object_stored(self):
        cap = _make_video_capture()
        with patch("src.videoprocessor.cv2.VideoCapture", return_value=cap):
            vp = VideoProcessor("fake.mp4")
        self.assertIs(vp.video, cap)

    def test_width_height_are_ints(self):
        vp = _make_processor(width=640, height=480)
        self.assertIsInstance(vp.width, int)
        self.assertIsInstance(vp.height, int)

    def test_length_is_int(self):
        vp = _make_processor(frame_count=99)
        self.assertIsInstance(vp.length, int)


# ---------------------------------------------------------------------------
# Test format()
# ---------------------------------------------------------------------------

class TestFormat(unittest.TestCase):

    def setUp(self):
        self.vp = _make_processor()

    def test_calls_cvtColor_with_bgr2rgb(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.videoprocessor.cv2.cvtColor", return_value=frame) as mock_cvt:
            self.vp.format(frame)
            mock_cvt.assert_called_once_with(frame, _cv2_stub.COLOR_BGR2RGB)

    def test_returns_converted_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        converted = np.ones((480, 640, 3), dtype=np.uint8) * 128
        with patch("src.videoprocessor.cv2.cvtColor", return_value=converted):
            result = self.vp.format(frame)
        self.assertIs(result, converted)

    def test_does_not_mutate_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        with patch("src.videoprocessor.cv2.cvtColor", return_value=frame):
            self.vp.format(frame)
        np.testing.assert_array_equal(frame, original)

    def test_format_passes_frame_to_cvtColor(self):
        frame = np.full((10, 10, 3), 42, dtype=np.uint8)
        with patch("src.videoprocessor.cv2.cvtColor", return_value=frame) as mock_cvt:
            self.vp.format(frame)
        args, _ = mock_cvt.call_args
        self.assertIs(args[0], frame)


# ---------------------------------------------------------------------------
# Test setparams()
# ---------------------------------------------------------------------------

class TestSetParams(unittest.TestCase):

    def setUp(self):
        self.vp = _make_processor(fps=25.0)
        self.mock_writer = MagicMock()

    def test_default_scale_is_used_when_empty_string(self):
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=self.mock_writer), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0):
            self.vp.setparams("out.mp4", "", 0)
        expected_default = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"'. "
        self.assertEqual(self.vp.scale, expected_default)

    def test_custom_scale_is_used(self):
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=self.mock_writer), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0):
            self.vp.setparams("out.mp4", "@#$", 0)
        self.assertEqual(self.vp.scale, "@#$")

    def test_result_writer_created(self):
        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer) as mock_vw, \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0):
            self.vp.setparams("output.mp4", "", 0)
        self.assertIs(self.vp.result, mock_writer)

    def test_videowriter_receives_output_filename(self):
        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer) as mock_vw, \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=99):
            self.vp.setparams("my_output.mp4", "", 0)
        args, _ = mock_vw.call_args
        self.assertEqual(args[0], "my_output.mp4")

    def test_videowriter_receives_fps(self):
        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer) as mock_vw, \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0):
            self.vp.setparams("out.mp4", "", 0)
        args, _ = mock_vw.call_args
        self.assertAlmostEqual(args[2], 25.0)

    def test_videowriter_receives_size(self):
        vp = _make_processor(width=320, height=240)
        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer) as mock_vw, \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0):
            vp.setparams("out.mp4", "", 0)
        args, _ = mock_vw.call_args
        self.assertEqual(args[3], (320, 240))

    def test_video_set_called_with_start(self):
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=self.mock_writer), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0):
            self.vp.setparams("out.mp4", "", 10)
        # CAP_PROP_FRAME_COUNT used as seek property; value is start - 1
        self.vp.video.set.assert_called_with(_cv2_stub.CAP_PROP_FRAME_COUNT, 9)


# ---------------------------------------------------------------------------
# Test videoToAscii()
# ---------------------------------------------------------------------------

class TestVideoToAscii(unittest.TestCase):
    """Tests for the high-level videoToAscii method."""

    def _make_vp_with_mocks(self, frame_count=3, width=64, height=48, fps=10.0):
        vp = _make_processor(width=width, height=height, fps=fps, frame_count=frame_count)
        return vp

    def _run_videoToAscii(self, vp, start=0, end=2, output="out.mp4",
                          quality=0.1, asciiscale="@#$", show=False):
        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows"), \
             patch("builtins.print"):
            vp.videoToAscii(output=output, quality=quality, start=start,
                            end=end, show=show, asciiscale=asciiscale)
        return mock_writer

    def test_video_released_after_processing(self):
        vp = self._make_vp_with_mocks()
        self._run_videoToAscii(vp, start=0, end=2)
        vp.video.release.assert_called_once()

    def test_result_released_after_processing(self):
        vp = self._make_vp_with_mocks()
        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows"), \
             patch("builtins.print"):
            vp.videoToAscii(output="out.mp4", quality=0.1, start=0, end=2,
                            show=False, asciiscale="@#$")
        mock_writer.release.assert_called_once()

    def test_destroyAllWindows_called(self):
        vp = self._make_vp_with_mocks()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=MagicMock()), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows") as mock_daw, \
             patch("builtins.print"):
            vp.videoToAscii(output="out.mp4", quality=0.1, start=0, end=2,
                            show=False, asciiscale="@#$")
        mock_daw.assert_called_once()

    def test_frame_read_called_for_each_frame(self):
        vp = self._make_vp_with_mocks(frame_count=5)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        vp.video.read.return_value = (True, frame)

        with patch("src.videoprocessor.cv2.VideoWriter", return_value=MagicMock()), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows"), \
             patch("builtins.print"):
            vp.videoToAscii(output="out.mp4", quality=0.1, start=0, end=4,
                            show=False, asciiscale="@#$")

        self.assertEqual(vp.video.read.call_count, 4)

    def test_stops_processing_when_no_frame(self):
        vp = self._make_vp_with_mocks(frame_count=10)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        # Fail after 2 successful reads
        vp.video.read.side_effect = [(True, frame), (True, frame), (False, None)]

        mock_writer = MagicMock()
        with patch("src.videoprocessor.cv2.VideoWriter", return_value=mock_writer), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows"), \
             patch("builtins.print"):
            vp.videoToAscii(output="out.mp4", quality=0.1, start=0, end=10,
                            show=False, asciiscale="@#$")

        # Only 2 frames should be written (the ones where exists=True)
        self.assertEqual(mock_writer.write.call_count, 2)

    def test_custom_scale_propagated_to_imageprocessor(self):
        """The asciiscale argument should end up in self.scale used by imageprocessor."""
        vp = self._make_vp_with_mocks()

        with patch("src.videoprocessor.cv2.VideoWriter", return_value=MagicMock()), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows"), \
             patch("builtins.print"):
            vp.videoToAscii(output="out.mp4", quality=0.1, start=0, end=0,
                            show=False, asciiscale="CUSTOM")

        self.assertEqual(vp.scale, "CUSTOM")

    def test_zero_frame_range_does_not_read_frames(self):
        """start == end means no frames to process."""
        vp = self._make_vp_with_mocks()

        with patch("src.videoprocessor.cv2.VideoWriter", return_value=MagicMock()), \
             patch("src.videoprocessor.cv2.VideoWriter_fourcc", return_value=0), \
             patch("src.videoprocessor.cv2.destroyAllWindows"), \
             patch("builtins.print"):
            vp.videoToAscii(output="out.mp4", quality=0.1, start=5, end=5,
                            show=False, asciiscale="@#$")

        vp.video.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
