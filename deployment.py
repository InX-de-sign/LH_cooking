"""Local process management and SSH deployment for the desktop launcher.

Only the Python standard library is needed to open the launcher. The cooking
service uses its existing requirements.txt in the project's .venv.
"""
import io
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "OPENAI_API_KEY": "",
    "AZURE_OPENAI_ENDPOINT": "https://hkust.azure-api.net",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_DEPLOYMENT_NAME": "gpt-5-mini",
    "AZURE_OPENAI_TIMEOUT_SECONDS": "30",
    "VLM_PROVIDER": "azure",
    "VLM_MODEL": "",
    "VLM_BASE_URL": "",
    "VLM_API_KEY": "",
    "VLM_MODE": "auto",
    "VLM_TIMEOUT_SECONDS": "10",
    "WEB_SEARCH_ENABLED": "true",
    "AZURE_RESPONSES_BASE_URL": "",
    "WEB_SEARCH_TIMEOUT_SECONDS": "60",
}
MODEL_KEYS = set(DEFAULTS) | {
    "AZURE_MAX_COMPLETION_TOKENS", "AZURE_REASONING_EFFORT", "AZURE_VERBOSITY",
    "VLM_MAX_TOKENS", "VLM_IMAGE_MAX_BYTES",
}
SECRET_KEYS = ("OPENAI_API_KEY", "VLM_API_KEY")
HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def read_env(path):
    path = Path(path)
    if not path.exists():
        return {}
    # Prefer the project's full parser when installed. The fallback also reads
    # the templates and the single-line format written by this launcher.
    try:
        from dotenv import dotenv_values
    except ImportError:
        values = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", line)
            if not match:
                continue
            name, value = match.groups()
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
            elif value.startswith('"'):
                value, _ = json.JSONDecoder().raw_decode(value)
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            values[name] = value
        return values
    return {key: value or "" for key, value in dotenv_values(path, interpolate=False).items()}


def env_line(name, value):
    value = str(value)
    if not re.fullmatch(r"[A-Z][A-Z_0-9]*", name) or any(c in value for c in "\r\n\0"):
        raise ValueError("Settings must use single-line values.")
    # Literal quotes also prevent Docker Compose from expanding $ in API keys.
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"{name}='{escaped}'\n"


def save_env(path, updates):
    path = Path(path)
    remaining = dict(updates)
    lines = path.read_text(encoding="utf-8-sig").splitlines(True) if path.exists() else []
    result = []
    for line in lines:
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=", line)
        if match and match[1] in updates:
            if match[1] in remaining:
                result.append(env_line(match[1], remaining.pop(match[1])))
        else:
            result.append(line if line.endswith("\n") else line + "\n")
    result.extend(env_line(key, value) for key, value in remaining.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".env-save-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(result))
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_settings(root=ROOT):
    return {**DEFAULTS, **read_env(Path(root) / "chatbot" / ".env")}


def validate_models(settings, require_key=False):
    for name, value in settings.items():
        if name in MODEL_KEYS:
            env_line(name, value)
    endpoint = urlparse(settings["AZURE_OPENAI_ENDPOINT"])
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.query or endpoint.fragment:
        raise ValueError("Use an HTTPS Azure endpoint, such as https://hkust.azure-api.net.")
    if not settings["AZURE_DEPLOYMENT_NAME"].strip():
        raise ValueError("Enter the chat deployment name.")
    responses_base = settings.get("AZURE_RESPONSES_BASE_URL", "").strip()
    if responses_base:
        parsed = urlparse(responses_base.rstrip("/"))
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or not parsed.path.endswith("/v1")):
            raise ValueError("Use an HTTPS Responses base URL ending in /v1, without credentials or query parameters.")
    try:
        search_timeout = float(settings.get("WEB_SEARCH_TIMEOUT_SECONDS", "60"))
        if not 5 <= search_timeout <= 180:
            raise ValueError
    except ValueError:
        raise ValueError("Web search timeout must be between 5 and 180 seconds.") from None
    key = settings.get("OPENAI_API_KEY", "").strip()
    if require_key and (not key or key.startswith(("<", "your_", "replace_with_"))):
        raise ValueError("Enter your HKUST API key in Azure settings before deploying.")


def redact(text, settings):
    for name in SECRET_KEYS:
        secret = settings.get(name, "")
        if secret:
            text = text.replace(secret, "[hidden]")
    return text


def project_python(root=ROOT):
    candidate = Path(root) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def search_online(settings, query):
    """Run a Responses web search in the service environment, without exposing keys."""
    validate_models(settings)
    env = {key: value for key, value in os.environ.items() if key not in MODEL_KEYS}
    env.update({key: value for key, value in settings.items() if key in MODEL_KEYS})
    try:
        result = subprocess.run(
            [project_python(), str(ROOT / "scripts" / "check_search.py"), "--json", "--query", query],
            env=env, cwd=ROOT, capture_output=True, timeout=float(settings.get("WEB_SEARCH_TIMEOUT_SECONDS", "60")) + 15,
            **HIDDEN,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Online search timed out. Please try again.") from None
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError
    except ValueError:
        raise RuntimeError("Cannot start online search. Install / repair dependencies in Local app and retry.") from None
    if result.returncode or payload.get("error"):
        raise RuntimeError(redact(payload.get("error") or "Online search failed.", settings))
    return payload


def run_command(args, log, *, data=None, env=None, cwd=ROOT, timeout=900):
    process = subprocess.Popen(args, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, cwd=cwd, **HIDDEN)
    timed_out = threading.Event()

    def expire():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()

    def write_input():
        try:
            process.stdin.write(data)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    if data is not None:
        threading.Thread(target=write_input, daemon=True).start()
    try:
        for line in process.stdout:
            log(line.decode("utf-8", "replace").rstrip())
        code = process.wait()
        if timed_out.is_set():
            raise RuntimeError("Operation timed out. For SSH tasks, check the server before retrying.")
        if code:
            raise RuntimeError(f"Command failed (exit {code}). See the activity log.")
    finally:
        timer.cancel()
        process.stdout.close()


def install_dependencies(log, root=ROOT):
    root = Path(root)
    if not (root / ".venv" / "pyvenv.cfg").exists():
        log("Creating the project Python environment...")
        run_command([sys.executable, "-m", "venv", str(root / ".venv")], log, cwd=root)
    log("Installing the cooking service dependencies...")
    requirements = root / "requirements-desktop.txt"
    run_command([project_python(root), "-m", "pip", "install", "-r", str(requirements)], log, cwd=root)


def health(url, timeout=2):
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
        result = json.load(response)
    if result.get("service") != "LH Cooking Coach" or result.get("status") != "healthy":
        raise ValueError("This address did not return the cooking coach health response.")
    return result


class LocalServer:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.process = None
        self.url = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, settings, port, log):
        if self.running:
            raise ValueError("The local app is already running. Stop it before restarting.")
        port = int(port)
        if not 1024 <= port <= 65535:
            raise ValueError("Choose a local port between 1024 and 65535.")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"Port {port} is already in use. Choose another port.")
        env = dict(os.environ)
        # Explicit app settings take precedence over stale exported model keys.
        for key in MODEL_KEYS:
            env.pop(key, None)
        env.update({key: str(settings[key]) for key in MODEL_KEYS if key in settings})
        env.update(HOST="127.0.0.1", PORT=str(port), PYTHONUNBUFFERED="1")
        env.setdefault("MEMORY_DB_PATH", str(self.root / "chatbot" / "memory.db"))
        self.url = f"http://localhost:{port}"
        self.process = subprocess.Popen([project_python(self.root), "-u", str(self.root / "chatbot" / "cook_api.py")],
                                        cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **HIDDEN)
        process = self.process

        def read_log():
            try:
                for line in process.stdout:
                    log(redact(line.decode("utf-8", "replace").rstrip(), settings))
            finally:
                process.stdout.close()
        threading.Thread(target=read_log, daemon=True).start()
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("The service did not start. Try Install dependencies and check the log.")
                try:
                    health(self.url, timeout=0.5)
                    return self.url
                except (OSError, ValueError):
                    time.sleep(0.25)
            raise RuntimeError("Local startup timed out. See the activity log.")
        except Exception:
            self.stop()
            raise

    def stop(self):
        if self.running:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)


@dataclass
class RemoteSettings:
    host: str = ""
    user: str = "ubuntu"
    port: int = 22
    identity: str = ""
    folder: str = "lh_cooking"
    domain: str = ""

    def validate(self, require_ssh=True):
        self.domain = self.domain.strip().lower().rstrip(".")
        labels = self.domain.split(".")
        if len(labels) < 2 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels):
            raise ValueError("Enter a domain such as cook.example.com, without https:// or a path.")
        try:
            ipaddress.ip_address(self.domain)
        except ValueError:
            pass
        else:
            raise ValueError("HTTPS deployment needs a DNS domain, not an IP address.")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.folder):
            raise ValueError("Server folder must be a single name, such as lh_cooking.")
        if not require_ssh:
            return
        self.host = self.host.strip()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}", self.host):
            raise ValueError("Enter the SSH server's IPv4 address or hostname.")
        if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", self.user):
            raise ValueError("Enter a Linux SSH username, such as ubuntu.")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")
        if self.identity and not Path(self.identity).expanduser().is_file():
            raise ValueError("The selected SSH private-key file was not found.")

    def ssh_args(self):
        self.validate()
        ssh = shutil.which("ssh")
        if not ssh:
            raise ValueError("OpenSSH client is missing. Install it before using Lighthouse deployment.")
        args = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=12",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-p", str(self.port)]
        if self.identity:
            args += ["-i", str(Path(self.identity).expanduser()), "-o", "IdentitiesOnly=yes"]
        return args + [f"{self.user}@{self.host}"]


PREFLIGHT = r'''set -eu
command -v tar >/dev/null
command -v curl >/dev/null
docker info >/dev/null
engine=$(docker version --format '{{.Server.Version}}')
test "${engine%%.*}" -ge 28 || { echo 'Docker Engine 28+ is required.'; exit 1; }
compose=$(docker compose version --short | sed 's/^v//')
test "$(printf '%s\n' 2.33.1 "$compose" | sort -V | head -n1)" = 2.33.1 || { echo 'Docker Compose 2.33.1+ is required.'; exit 1; }
echo 'Server ready: SSH, Docker Engine and Compose are available.'
'''

DEPLOY_SCRIPT = '''#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
chmod 600 .env.lighthouse
unset ''' + " ".join(sorted(MODEL_KEYS | {"DOMAIN"})) + '''
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml config --quiet
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml up -d --build --force-recreate --wait --wait-timeout 240
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml ps
'''


def build_bundle(settings, remote, root=ROOT):
    remote.validate(require_ssh=False)
    validate_models(settings, require_key=True)
    root = Path(root).resolve()
    names = ["docker-compose.lighthouse.yml", "Caddyfile", "chatbot/Dockerfile",
             "chatbot/.dockerignore", "chatbot/requirements.txt", "chatbot/recipes.yaml", "chatbot/glossary.yaml"]
    names += [str(path.relative_to(root)).replace("\\", "/") for path in sorted((root / "chatbot").glob("*.py"))]
    names += [str(path.relative_to(root)).replace("\\", "/") for path in sorted((root / "chatbot" / "static").rglob("*"))
              if path.is_file() and path.suffix.lower() in {".html", ".js", ".css", ".svg", ".png", ".jpg", ".ico", ".woff2"}]
    env = {key: settings[key] for key in settings if key in MODEL_KEYS}
    env["DOMAIN"] = remote.domain
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in names:
            source = root / name
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError(f"Deployment source must be a regular project file: {name}")
            archive.add(source, arcname=name, recursive=False)
        for name, content, mode in [(".env.lighthouse", "".join(env_line(k, v) for k, v in env.items()), 0o600),
                                    ("deploy.sh", DEPLOY_SCRIPT, 0o700)]:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def check_remote(remote, log):
    run_command(remote.ssh_args() + ["sh -c " + shlex.quote(PREFLIGHT)], log, timeout=45)


def deploy_remote(settings, remote, log, root=ROOT):
    remote.validate()
    bundle = build_bundle(settings, remote, root)
    safe_log = lambda text: log(redact(text, settings))
    safe_log("Checking SSH and Docker before uploading...")
    check_remote(remote, safe_log)
    folder = '"$HOME"/' + shlex.quote(remote.folder)
    command = (f"set -eu; umask 077; mkdir -p {folder}; "
               f"tar -xzf - -C {folder}; cd {folder}; sh ./deploy.sh")
    safe_log("Uploading application and settings over SSH; building containers...")
    run_command(remote.ssh_args() + ["sh -c " + shlex.quote(command)], safe_log, data=bundle, timeout=1200)
    url = "https://" + remote.domain
    safe_log("Containers started. Checking public HTTPS readiness...")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            health(url, timeout=3)
            safe_log("Deployment healthy at " + url)
            return url
        except (OSError, ValueError):
            time.sleep(2)
    raise RuntimeError("Containers started, but public HTTPS is not ready. Check DNS, ports 80/443 and Caddy logs before retrying.")
