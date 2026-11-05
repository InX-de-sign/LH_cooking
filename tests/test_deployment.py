"""Offline launcher/deployment checks; no API calls or SSH connections."""
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import deployment
from deployment import (
    DEFAULTS, LocalServer, RemoteSettings, build_bundle, check_remote,
    deploy_remote, env_line, read_env, redact, save_env,
)


class SettingsTests(unittest.TestCase):
    def test_search_worker_uses_saved_environment_and_reports_route_failure(self):
        settings = {**DEFAULTS, "OPENAI_API_KEY": "test-only", "AZURE_RESPONSES_BASE_URL": "https://example.com/openai/v1"}
        result = Mock(returncode=1, stdout=json.dumps({"error": "The gateway returned HTTP 404"}).encode())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "stale-key"}), \
                patch("deployment.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                deployment.search_online(settings, "Find a tofu recipe")
        self.assertEqual(run.call_args.kwargs["env"]["OPENAI_API_KEY"], "test-only")
        self.assertNotIn("test-only", str(run.call_args.args))

    def test_invalid_search_settings_are_rejected(self):
        for updates in ({"AZURE_RESPONSES_BASE_URL": "http://example.com/v1"},
                        {"AZURE_RESPONSES_BASE_URL": "https://user:pass@example.com/v1"},
                        {"WEB_SEARCH_TIMEOUT_SECONDS": "nan"}):
            with self.assertRaises(ValueError):
                deployment.validate_models({**DEFAULTS, **updates})

    def test_save_preserves_comments_and_unknown_settings_and_roundtrips_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("# Keep this comment\nCUSTOM_SETTING=keep\nOPENAI_API_KEY=old\n", encoding="utf-8")
            secret = "fake-key$literal#token'quoted"
            save_env(path, {"OPENAI_API_KEY": secret, "VLM_PROVIDER": "azure"})
            self.assertIn("# Keep this comment", path.read_text())
            self.assertEqual(read_env(path)["OPENAI_API_KEY"], secret)
            self.assertEqual(read_env(path)["CUSTOM_SETTING"], "keep")
            with patch.dict(sys.modules, {"dotenv": None}):
                self.assertEqual(read_env(path)["OPENAI_API_KEY"], secret)

    def test_multiline_values_are_rejected_without_corrupting_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("OPENAI_API_KEY=original\n")
            with self.assertRaises(ValueError):
                save_env(path, {"OPENAI_API_KEY": "bad\nINJECTED=value"})
            self.assertEqual(path.read_text(), "OPENAI_API_KEY=original\n")

    def test_redaction_masks_both_model_secrets(self):
        self.assertEqual(redact("key secret1 and secret2", {
            "OPENAI_API_KEY": "secret1", "VLM_API_KEY": "secret2"}), "key [hidden] and [hidden]")


class BundleTests(unittest.TestCase):
    def settings(self):
        return {**DEFAULTS, "OPENAI_API_KEY": "test-only-not-a-real-key"}

    def test_bundle_has_only_deployment_files_and_private_environment(self):
        remote = RemoteSettings(domain="cook.example.com")
        data = build_bundle(self.settings(), remote)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            names = archive.getnames()
            self.assertIn("chatbot/static/index.html", names)
            self.assertIn("chatbot/cook_api.py", names)
            self.assertIn("deploy.sh", names)
            self.assertIn(".env.lighthouse", names)
            for name in names:
                self.assertFalse(any(part in name for part in (".venv", "memory.db", "knowledge.db", ".git/", "server.json")))
                self.assertNotEqual(name, "chatbot/.env")
                self.assertFalse(name.startswith("/"))
                self.assertNotIn("..", Path(name).parts)
            env = archive.extractfile(".env.lighthouse").read().decode()
            self.assertIn("DOMAIN='cook.example.com'", env)
            self.assertEqual(archive.getmember(".env.lighthouse").mode, 0o600)
            script = archive.extractfile("deploy.sh").read().decode()
            self.assertIn("--wait", script)
            self.assertNotIn("down -v", script)

    def test_empty_key_cannot_produce_deployable_bundle(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            build_bundle(DEFAULTS, RemoteSettings(domain="cook.example.com"))

    def test_shell_injection_and_invalid_domain_are_rejected(self):
        for remote in (
            RemoteSettings(domain="https://cook.example.com"),
            RemoteSettings(domain="127.0.0.1"),
            RemoteSettings(domain="cook.example.com", folder="../../outside"),
            RemoteSettings(domain="cook.example.com", host="server;touch bad"),
            RemoteSettings(domain="cook.example.com", host="-oProxyCommand=evil"),
            RemoteSettings(domain="cook.example.com", host="server", user="ubuntu;bad"),
        ):
            with self.assertRaises(ValueError):
                remote.validate()

    def test_ssh_uses_existing_trust_and_no_password_prompts(self):
        remote = RemoteSettings(domain="cook.example.com", host="192.0.2.10")
        with patch("deployment.shutil.which", return_value="ssh"):
            args = remote.ssh_args()
        self.assertIn("StrictHostKeyChecking=yes", args)
        self.assertIn("BatchMode=yes", args)
        self.assertEqual(args[-1], "ubuntu@192.0.2.10")

    def test_deploy_checks_before_upload_and_verifies_https(self):
        remote = RemoteSettings(domain="cook.example.com", host="192.0.2.10")
        calls = []

        def run(args, log, **kwargs):
            calls.append(kwargs)
            log("Do not show test-only-not-a-real-key")

        logs = []
        with patch("deployment.shutil.which", return_value="ssh"), \
                patch("deployment.run_command", side_effect=run), \
                patch("deployment.health", return_value={"status": "healthy"}) as health:
            url = deploy_remote(self.settings(), remote, logs.append)
        self.assertEqual(url, "https://cook.example.com")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("data", calls[0])
        self.assertIsInstance(calls[1]["data"], bytes)
        self.assertTrue(all("test-only-not-a-real-key" not in line for line in logs))
        health.assert_called_once_with(url, timeout=3)

    def test_failed_preflight_does_not_upload(self):
        remote = RemoteSettings(domain="cook.example.com", host="192.0.2.10")
        with patch("deployment.shutil.which", return_value="ssh"), \
                patch("deployment.run_command", side_effect=RuntimeError("SSH failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "SSH failed"):
                deploy_remote(self.settings(), remote, lambda line: None)
        self.assertEqual(run.call_count, 1)


class LocalTests(unittest.TestCase):
    def test_launcher_settings_override_stale_os_key_without_empty_numeric_fields(self):
        process = Mock()
        process.poll.return_value = None
        process.stdout = io.BytesIO()
        server = LocalServer()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "stale", "AZURE_MAX_COMPLETION_TOKENS": "999"}), \
                patch("deployment.socket.socket") as socket, \
                patch("deployment.subprocess.Popen", return_value=process) as popen, \
                patch("deployment.health", return_value={"status": "healthy"}):
            os.environ.pop("MEMORY_DB_PATH", None)
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            url = server.start(DEFAULTS, 8123, lambda line: None)
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["OPENAI_API_KEY"], "")
        self.assertNotIn("AZURE_MAX_COMPLETION_TOKENS", env)
        self.assertEqual(env["HOST"], "127.0.0.1")
        self.assertEqual(env["MEMORY_DB_PATH"], str(ROOT / "chatbot" / "memory.db"))
        self.assertEqual(url, "http://localhost:8123")
        server.stop()
        process.terminate.assert_called_once()

    def test_occupied_port_does_not_start_or_stop_another_service(self):
        server = LocalServer()
        with patch("deployment.socket.socket") as socket, patch("deployment.subprocess.Popen") as popen:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 0
            with self.assertRaisesRegex(ValueError, "already in use"):
                server.start(DEFAULTS, 8000, lambda line: None)
            server.stop()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
