"""Desktop photo preparation, camera lifecycle, and isolated vision requests."""
import argparse
import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

from deployment import ROOT, MODEL_KEYS, HIDDEN, project_python

MAX_IMAGE_BYTES = 1_500_000


def prepare_photo(source):
    """Read a photo, orient and shrink it, then return a metadata-free JPEG."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Install / repair dependencies, then reopen the launcher for photo support.") from exc
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > 30_000_000:
            raise ValueError("Choose a photo smaller than 30 MB.")
        source = path.read_bytes()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source)) as original:
                if original.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError("Choose a JPEG, PNG or WebP photo.")
                image = ImageOps.exif_transpose(original)
                image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                rgba = image.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                # Fresh pixels exclude EXIF, location tags, and other metadata.
                for quality in (85, 70, 50):
                    output = io.BytesIO()
                    flattened.save(output, "JPEG", quality=quality)
                    if output.tell() <= MAX_IMAGE_BYTES:
                        return output.getvalue()
    except (Image.UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("This photo cannot be opened. Choose a valid JPEG, PNG or WebP image.") from exc
    raise ValueError("This photo could not be reduced enough for the vision model.")


class CameraSession:
    """The worker owns the camera; Tk only reads immutable JPEG snapshots."""
    def __init__(self, index=0):
        self.index = int(index)
        self.stop_requested = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def running(self):
        return self.thread.is_alive()

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.sequence, self.frame

    def stop(self):
        self.stop_requested.set()

    def _run(self):
        camera = None
        try:
            if self.stop_requested.is_set():
                return
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Install / repair dependencies and reopen the launcher for camera support.") from exc
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.index, backend)
            if self.stop_requested.is_set():
                return
            if not camera.isOpened():
                raise RuntimeError("Cannot open this camera. Check camera access, close other camera apps, or try another camera number.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            while not self.stop_requested.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames. Reconnect it and try again.")
                height, width = frame.shape[:2]
                if max(height, width) > 1280:
                    scale = 1280 / max(height, width)
                    frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("Cannot read a photo from this camera.")
                with self.lock:
                    self.frame = jpeg.tobytes()
                    self.sequence += 1
                self.stop_requested.wait(0.10)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if camera is not None:
                camera.release()


def analyze_photo(settings, photo, question):
    """Use the project interpreter; no photo is written to disk or command args."""
    if not photo or len(photo) > MAX_IMAGE_BYTES:
        raise ValueError("Capture or choose a photo first.")
    env = dict(os.environ)
    for key in MODEL_KEYS:
        env.pop(key, None)
    env.update({key: str(value) for key, value in settings.items() if key in MODEL_KEYS})
    result = subprocess.run(
        [project_python(), str(Path(__file__).resolve()), "--analyze", "--question", question],
        input=photo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        cwd=ROOT, timeout=90, **HIDDEN)
    try:
        payload = json.loads(result.stdout)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Vision could not start. Install / repair dependencies and check your model settings.") from exc
    if result.returncode or "error" in payload:
        raise RuntimeError(payload.get("error", "Vision request failed. Check your Azure settings."))
    return payload["observation"]


async def _observe(photo, question):
    # Load the existing provider only in this isolated process, after the
    # caller has supplied its current settings in the process environment.
    sys.path.insert(0, str(ROOT / "chatbot"))
    import config  # noqa: F401 - loads chatbot/.env
    from vision import VisionRequest, load_vision_provider
    provider = load_vision_provider()
    try:
        if not provider.available:
            raise RuntimeError("Enable vision and enter your HKUST API key in model settings first.")
        observation = await provider.observe(photo, "image/jpeg", VisionRequest(question=question))
        if observation is None:
            raise RuntimeError(provider.last_error or "Vision returned no result. Check deployment access, image support and network connectivity.")
        return observation.to_dict()
    finally:
        if getattr(provider, "client", None):
            await provider.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--question", default="What food can you see?")
    args = parser.parse_args()
    if not args.analyze:
        parser.error("Use this module through the desktop launcher's Camera / photo tab.")
    logging.disable(logging.CRITICAL)
    try:
        photo = sys.stdin.buffer.read(MAX_IMAGE_BYTES + 1)
        if not photo or len(photo) > MAX_IMAGE_BYTES:
            raise RuntimeError("The selected photo is empty or too large.")
        observation = asyncio.run(_observe(photo, args.question))
        print(json.dumps({"observation": observation}, ensure_ascii=True))
        return 0
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"error": "Vision request failed. Check dependencies, network access and Azure settings."}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
