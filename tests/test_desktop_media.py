"""Photo and simulated camera tests; no real camera or model is contacted."""
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop_media import CameraSession, MAX_IMAGE_BYTES, analyze_photo, prepare_photo
from deployment import DEFAULTS

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Install requirements-desktop.txt for photo tests")
class PhotoTests(unittest.TestCase):
    def image(self, mode="RGB", size=(80, 60), exif=None):
        data = io.BytesIO()
        Image.new(mode, size, "red").save(data, "PNG", exif=exif or b"")
        return data.getvalue()

    def test_large_photo_is_resized_and_metadata_is_removed(self):
        exif = Image.Exif()
        exif[270] = "private photo metadata"
        prepared = prepare_photo(self.image(size=(3000, 1500), exif=exif))
        with Image.open(io.BytesIO(prepared)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (1280, 640))
            self.assertFalse(image.getexif())
        self.assertLessEqual(len(prepared), MAX_IMAGE_BYTES)

    def test_phone_photo_orientation_is_applied(self):
        exif = Image.Exif()
        exif[274] = 6
        prepared = prepare_photo(self.image(size=(80, 40), exif=exif))
        with Image.open(io.BytesIO(prepared)) as image:
            self.assertEqual(image.size, (40, 80))

    def test_transparent_png_and_invalid_file(self):
        with Image.open(io.BytesIO(prepare_photo(self.image(mode="RGBA")))) as image:
            self.assertEqual(image.mode, "RGB")
        with self.assertRaisesRegex(ValueError, "valid JPEG"):
            prepare_photo(b"not a photo")


class CameraTests(unittest.TestCase):
    def fake_cv(self, device):
        return SimpleNamespace(VideoCapture=Mock(return_value=device), CAP_DSHOW=1, CAP_ANY=0,
                               CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                               IMWRITE_JPEG_QUALITY=1, imencode=Mock(return_value=(True, SimpleNamespace(tobytes=lambda: b"jpeg"))))

    def test_frame_can_be_captured_and_device_is_released_on_close(self):
        session = CameraSession()
        device = Mock()
        device.isOpened.return_value = True

        def read():
            session.stop()
            return True, SimpleNamespace(shape=(540, 960, 3))
        device.read.side_effect = read
        cv = self.fake_cv(device)
        with patch.dict(sys.modules, {"cv2": cv}):
            session._run()
        self.assertEqual(session.snapshot(), (1, b"jpeg"))
        device.release.assert_called_once()
        self.assertIsNone(session.error)

    def test_unavailable_camera_reports_error_and_releases_device(self):
        session = CameraSession()
        device = Mock()
        device.isOpened.return_value = False
        with patch.dict(sys.modules, {"cv2": self.fake_cv(device)}):
            session._run()
        self.assertIn("Cannot open", session.error)
        device.release.assert_called_once()

    def test_canceled_open_does_not_touch_camera(self):
        session = CameraSession()
        session.stop()
        cv = self.fake_cv(Mock())
        with patch.dict(sys.modules, {"cv2": cv}):
            session._run()
        cv.VideoCapture.assert_not_called()


class VisionTransferTests(unittest.TestCase):
    def test_selected_bytes_go_through_stdin_without_disk_files(self):
        result = SimpleNamespace(returncode=0, stdout=json.dumps({"observation": {"summary": "A tomato."}}).encode())
        with patch("desktop_media.subprocess.run", return_value=result) as run:
            observation = analyze_photo({**DEFAULTS, "OPENAI_API_KEY": "test-key"}, b"selected-jpeg", "What is this?")
        self.assertEqual(observation["summary"], "A tomato.")
        self.assertEqual(run.call_args.kwargs["input"], b"selected-jpeg")
        self.assertNotIn("test-key", " ".join(run.call_args.args[0]))
        self.assertEqual(run.call_args.kwargs["env"]["OPENAI_API_KEY"], "test-key")

    def test_provider_failure_is_shown_and_oversize_photo_is_not_sent(self):
        result = SimpleNamespace(returncode=1, stdout=b'{"error":"Vision unavailable"}')
        with patch("desktop_media.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "Vision unavailable"):
                analyze_photo(DEFAULTS, b"jpeg", "Describe")
            run.reset_mock()
            with self.assertRaises(ValueError):
                analyze_photo(DEFAULTS, b"x" * (MAX_IMAGE_BYTES + 1), "Describe")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
