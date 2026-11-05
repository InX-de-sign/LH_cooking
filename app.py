"""Cooking Coach desktop launcher: python app.py (or open Start Cooking Coach.pyw)."""
import json
import io
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import urlparse

from deployment import (
    ROOT, DEFAULTS, MODEL_KEYS, HIDDEN, LocalServer, RemoteSettings, build_bundle,
    check_remote, deploy_remote, install_dependencies, model_settings,
    project_python, redact, run_command, save_env, search_online, validate_models,
)
from desktop_media import CameraSession, prepare_photo, analyze_photo


class CookingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cooking Coach — Run & Deploy")
        self.geometry("900x820")
        self.minsize(800, 760)
        self.configure(background="#f3f6f5")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.local = LocalServer()
        self.busy = False
        self.buttons = []
        self.settings = model_settings()
        self.fields = {}
        self.remote_fields = {}
        self.status = tk.StringVar(value="Ready to run locally")
        self.local_status = tk.StringVar(value="Stopped · Runs on this computer")
        self.remote_status = tk.StringVar(value="Ready when your server and domain are available")
        self.port = tk.StringVar(value="8000")
        self.show_key = tk.BooleanVar(value=False)
        self.camera = None
        self.camera_sequence = 0
        self.selected_photo = None
        self.preview_image = None
        self.camera_index = tk.StringVar(value="0")
        self.photo_question = tk.StringVar(value="What food can you see?")
        self.search_question = tk.StringVar(value="Find a simple tomato and egg stir-fry recipe online")
        self.search_pending = False
        self.media_status = tk.StringVar(value="Open your camera or choose a photo")
        self._theme()
        self._layout()
        self.after(100, self._poll)

    def _theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6f5")
        style.configure("TLabel", background="#f3f6f5", foreground="#203c38", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.TLabel", foreground="#5a706a")
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Primary.TButton", foreground="white", background="#176859")
        style.map("Primary.TButton", background=[("active", "#125346"), ("disabled", "#9aaba5")])
        style.configure("TEntry", padding=6)
        style.configure("TNotebook", background="#f3f6f5", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(18, 10))
        style.configure("TCheckbutton", background="#f3f6f5", font=("Segoe UI", 10))

    def _button(self, parent, text, command, primary=False):
        button = ttk.Button(parent, text=text, command=command,
                            style="Primary.TButton" if primary else "TButton")
        self.buttons.append(button)
        return button

    def _entry(self, parent, row, label, name, value, fields, secret=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 18), pady=6)
        variable = tk.StringVar(value=value)
        fields[name] = variable
        entry = ttk.Entry(parent, textvariable=variable, show="\u2022" if secret else "")
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        return entry

    def _layout(self):
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        ttk.Label(outer, text="Cooking Coach", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="Run locally today. Deploy to Lighthouse when you're ready.",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 18))
        self.tabs = ttk.Notebook(outer)
        self.tabs.grid(row=2, column=0, sticky="nsew")
        self.canvases = {}
        local_tab, _ = self._tab("Local app")
        media_tab, self.media_tab = self._tab("Camera / photo")
        search_tab, _ = self._tab("Recipe search")
        settings_tab, self.settings_tab = self._tab("Azure settings")
        remote_tab, _ = self._tab("Lighthouse")
        self._local_tab(local_tab)
        self._media_tab(media_tab)
        self._search_tab(search_tab)
        self._settings_tab(settings_tab)
        self._remote_tab(remote_tab)
        progress_row = ttk.Frame(outer)
        progress_row.grid(row=3, column=0, sticky="ew", pady=(14, 8))
        ttk.Label(progress_row, textvariable=self.status).pack(side="left")
        self.progress = ttk.Progressbar(progress_row, mode="indeterminate", length=120)
        self.progress.pack(side="right")
        ttk.Label(outer, text="Activity", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(4, 6))
        self.log_view = scrolledtext.ScrolledText(outer, height=6, wrap="word", state="disabled",
                                                font=("Consolas", 9), background="#182c28", foreground="#e0ece7",
                                                relief="flat", padx=10, pady=10)
        self.log_view.grid(row=5, column=0, sticky="ew")
        self.bind("<MouseWheel>", self._scroll_tab)
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        self._log("Choose Azure settings to enter your HKUST API key, or start locally in recipe-notes mode.")

    def _tab(self, title):
        page = ttk.Frame(self.tabs)
        page.rowconfigure(0, weight=1)
        page.columnconfigure(0, weight=1)
        canvas = tk.Canvas(page, background="#f3f6f5", borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(canvas, padding=20)
        item = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        self.tabs.add(page, text=title)
        self.canvases[str(page)] = canvas
        return content, page

    def _scroll_tab(self, event):
        if event.widget not in (self.log_view, self.vision_result, self.search_result):
            self.canvases[self.tabs.select()].yview_scroll(-int(event.delta / 120), "units")

    def _tab_changed(self, event):
        if self.camera and self.tabs.select() != str(self.media_tab):
            self.close_camera()

    def _local_tab(self, tab):
        ttk.Label(tab, text="Your kitchen, on this computer", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, textvariable=self.local_status, style="Subtitle.TLabel").pack(anchor="w", pady=(8, 20))
        ttk.Label(tab, text="Start the service, then use the coach in your browser.\n"
                  "Camera and microphone work on localhost; no cloud server is needed.",
                  wraplength=700).pack(anchor="w", pady=(0, 16))
        row = ttk.Frame(tab)
        row.pack(anchor="w", pady=(0, 18))
        ttk.Label(row, text="Local port").pack(side="left", padx=(0, 12))
        ttk.Spinbox(row, from_=1024, to=65535, textvariable=self.port, width=9).pack(side="left")
        actions = ttk.Frame(tab)
        actions.pack(anchor="w")
        self._button(actions, "Start & open coach", self.start_local, True).pack(side="left", padx=(0, 8))
        self._button(actions, "Stop", self.stop_local).pack(side="left", padx=(0, 8))
        self._button(actions, "Open browser", self.open_local).pack(side="left")
        ttk.Separator(tab).pack(fill="x", pady=20)
        ttk.Label(tab, text="First-time setup", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, text="The launcher creates a project environment and installs the service dependencies.\n"
                  "Use this if the service reports missing packages.", wraplength=700,
                  style="Subtitle.TLabel").pack(anchor="w", pady=(6, 12))
        self._button(tab, "Install / repair dependencies", self.install).pack(anchor="w")
        ttk.Label(tab, text="Closing this window stops the local service it started.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(16, 0))

    def _settings_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="HKUST Azure connection", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.key_entry = self._entry(tab, 1, "API key", "OPENAI_API_KEY", self.settings.get("OPENAI_API_KEY", ""), self.fields, True)
        ttk.Checkbutton(tab, text="Show key", variable=self.show_key,
                        command=lambda: self.key_entry.configure(show="" if self.show_key.get() else "\u2022")).grid(row=2, column=1, sticky="w")
        for row, (name, label) in enumerate([
            ("AZURE_OPENAI_ENDPOINT", "Endpoint"), ("AZURE_OPENAI_API_VERSION", "API version"),
            ("AZURE_DEPLOYMENT_NAME", "Chat deployment"), ("VLM_MODEL", "Vision deployment"),
        ], start=3):
            self._entry(tab, row, label, name, self.settings.get(name, DEFAULTS[name]), self.fields)
        ttk.Label(tab, text="Vision provider").grid(row=7, column=0, sticky="w", pady=6)
        self.fields["VLM_PROVIDER"] = tk.StringVar(value=self.settings.get("VLM_PROVIDER", "azure"))
        ttk.Combobox(tab, textvariable=self.fields["VLM_PROVIDER"], values=("azure", "none"),
                     state="readonly").grid(row=7, column=1, sticky="ew", pady=6)
        ttk.Label(tab, text="Leave vision deployment blank to reuse chat. It must accept images.\n"
                  "The key is saved in chatbot/.env. Restart the local app after changing settings.",
                  style="Subtitle.TLabel", wraplength=690).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 12))
        row = ttk.Frame(tab)
        row.grid(row=9, column=0, columnspan=2, sticky="w")
        self._button(row, "Save settings", self.save, True).pack(side="left", padx=(0, 8))
        self._button(row, "Test Azure chat", self.test_chat).pack(side="left", padx=(0, 8))
        self._button(row, "Choose vision photo", self.test_vision).pack(side="left")
        self.fields["WEB_SEARCH_ENABLED"] = tk.StringVar(value=self.settings.get("WEB_SEARCH_ENABLED", "true"))
        ttk.Checkbutton(tab, text="Enable built-in web search", variable=self.fields["WEB_SEARCH_ENABLED"],
                        onvalue="true", offvalue="false").grid(row=10, column=0, columnspan=2, sticky="w", pady=(16, 6))
        self._entry(tab, 11, "Responses base URL", "AZURE_RESPONSES_BASE_URL",
                    self.settings.get("AZURE_RESPONSES_BASE_URL", ""), self.fields)
        ttk.Label(tab, text="Leave blank to use the default for your gateway. Search requires Responses / web_search access;\n"
                  "a working chat key alone does not confirm it. Use Recipe search to test your access.",
                  wraplength=690, style="Subtitle.TLabel").grid(row=12, column=0, columnspan=2, sticky="w", pady=8)

    def _search_tab(self, tab):
        ttk.Label(tab, text="Find recipes and ingredient information", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, text="Uses your GenAI provider's built-in web_search tool and the saved API key.\n"
                  "Results include source links when the gateway supports the tool. Each search uses API quota.",
                  wraplength=700, style="Subtitle.TLabel").pack(anchor="w", pady=(6, 14))
        ttk.Entry(tab, textvariable=self.search_question).pack(fill="x")
        self._button(tab, "Search online", self.search_recipes, True).pack(anchor="w", pady=12)
        self.search_result = scrolledtext.ScrolledText(tab, height=12, wrap="word", state="disabled",
                                                     font=("Segoe UI", 10))
        self.search_result.pack(fill="both", expand=True)
        ttk.Label(tab, text="You can also ask the browser coach: 'Search online for a tofu recipe'.\n"
                  "Online results do not replace the coach's saved step-by-step recipes.",
                  wraplength=700, style="Subtitle.TLabel").pack(anchor="w", pady=(12, 0))

    def _set_search_result(self, text, sources=()):
        view = self.search_result
        view.configure(state="normal")
        view.delete("1.0", "end")
        for tag in view.tag_names():
            if tag.startswith("source_"):
                view.tag_delete(tag)
        view.insert("end", text)
        if sources:
            view.insert("end", "\n\nWeb sources:\n")
        for number, source in enumerate(sources):
            url = source.get("url", "")
            link = urlparse(url)
            if link.scheme not in ("https", "http") or not link.hostname or link.username or link.password:
                continue
            tag = "source_" + str(number)
            view.insert("end", (source.get("title") or link.hostname) + "\n", tag)
            view.tag_configure(tag, foreground="#176859", underline=True)
            view.tag_bind(tag, "<Button-1>", lambda event, target=url: webbrowser.open(target))
        view.configure(state="disabled")

    def search_recipes(self):
        if self.busy:
            return
        query = self.search_question.get().strip()
        if not query:
            self.status.set("Enter a recipe or ingredient question first")
            return
        settings = self.save()
        if settings is None:
            return
        self.search_pending = True
        self._set_search_result("Searching online...")

        def work(log):
            result = search_online(settings, query)
            self.events.put(("search_result", result))
            return "Online search complete"
        self.task("Searching online (uses API quota)...", work)

    def _media_tab(self, tab):
        ttk.Label(tab, text="Show the coach what you're cooking", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, textvariable=self.media_status, style="Subtitle.TLabel",
                  wraplength=700).pack(anchor="w", pady=(6, 12))
        controls = ttk.Frame(tab)
        controls.pack(anchor="w", pady=(0, 8))
        ttk.Label(controls, text="Camera").pack(side="left", padx=(0, 6))
        ttk.Spinbox(controls, from_=0, to=9, width=3, textvariable=self.camera_index).pack(side="left", padx=(0, 8))
        self._button(controls, "Open camera", self.open_camera).pack(side="left", padx=(0, 6))
        self._button(controls, "Capture photo", self.capture_photo).pack(side="left", padx=(0, 6))
        self._button(controls, "Close camera", self.close_camera).pack(side="left")
        self._button(tab, "Upload photo", self.upload_photo).pack(anchor="w", pady=(0, 10))
        self.preview = tk.Canvas(tab, height=220, background="#182c28", highlightthickness=0)
        self.preview.pack(fill="x")
        self.preview.create_text(300, 110, text="Your camera or selected photo will appear here", fill="#e0ece7")
        ttk.Label(tab, text="Question about this photo").pack(anchor="w", pady=(12, 4))
        ttk.Entry(tab, textvariable=self.photo_question).pack(fill="x")
        self._button(tab, "Analyze photo", self.analyze_selected_photo, True).pack(anchor="w", pady=10)
        ttk.Label(tab, text="Preview stays on your computer. Analyze sends only the selected photo to your vision model.",
                  wraplength=700, style="Subtitle.TLabel").pack(anchor="w", pady=(0, 10))
        self.vision_result = scrolledtext.ScrolledText(tab, height=6, wrap="word", state="disabled", font=("Segoe UI", 10))
        self.vision_result.pack(fill="x")

    def _image_support(self):
        try:
            from PIL import Image, ImageTk  # noqa: F401
            return True
        except ImportError:
            messagebox.showinfo("Photo support", "Click Install / repair dependencies in Local app, then reopen the launcher.", parent=self)
            return False

    def _show_photo(self, data):
        from PIL import Image, ImageTk
        with Image.open(io.BytesIO(data)) as image:
            image.thumbnail((max(1, self.preview.winfo_width() - 12), 210), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image)
        self.preview.delete("all")
        self.preview.create_image(max(1, self.preview.winfo_width()) // 2, 110, image=self.preview_image)

    def _set_result(self, text):
        self.vision_result.configure(state="normal")
        self.vision_result.delete("1.0", "end")
        self.vision_result.insert("end", text)
        self.vision_result.configure(state="disabled")

    def open_camera(self):
        if not self._image_support():
            return
        if self.camera and self.camera.running:
            self.media_status.set("Camera is already open or closing. Wait a moment before reopening.")
            return
        try:
            index = int(self.camera_index.get())
            if not 0 <= index <= 9:
                raise ValueError
        except ValueError:
            messagebox.showerror("Camera number", "Choose a camera number from 0 to 9. The default is 0.", parent=self)
            return
        self.selected_photo = None
        self.camera_sequence = 0
        self._set_result("")
        self.preview.delete("all")
        self.media_status.set("Opening camera...")
        self.camera = CameraSession(index)
        self.camera.start()

    def close_camera(self):
        if self.camera:
            self.camera.stop()
        self.media_status.set("Camera closed. Capture or upload a photo to select an image.")

    def _select_photo(self, data, label):
        self.close_camera()
        self.selected_photo = data
        self._show_photo(data)
        self._set_result("")
        self.media_status.set(f"{label} selected · {len(data) // 1024} KB · ready to analyze")

    def upload_photo(self):
        if not self._image_support():
            return
        path = filedialog.askopenfilename(parent=self, title="Choose a cooking photo",
                                         filetypes=[("Photos", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            self._select_photo(prepare_photo(path), Path(path).name)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Cannot open photo", str(exc), parent=self)

    def capture_photo(self):
        if not self.camera or self.camera.stop_requested.is_set():
            messagebox.showinfo("Camera", "Open the camera and wait for its preview first.", parent=self)
            return
        _, data = self.camera.snapshot()
        if not data:
            self.media_status.set("Waiting for the first camera frame...")
            return
        try:
            self._select_photo(prepare_photo(data), "Camera photo")
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Camera photo", str(exc), parent=self)

    def analyze_selected_photo(self):
        if self.camera and not self.camera.stop_requested.is_set():
            self.capture_photo()
            if not self.camera.stop_requested.is_set():
                return
        if not self.selected_photo:
            messagebox.showinfo("Choose a photo", "Capture a camera frame or upload a photo first.", parent=self)
            return
        settings = self.save()
        if settings is None:
            return
        photo = self.selected_photo
        question = self.photo_question.get().strip() or "What food can you see?"
        self._set_result("Analyzing the selected photo...")

        def work(log):
            observation = analyze_photo(settings, photo, question)
            self.events.put(("vision_result", observation))
            return "Photo analysis complete"
        self.task("Analyzing photo (uses vision API quota)...", work)

    def _remote_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, textvariable=self.remote_status, style="Subtitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        try:
            saved = json.loads((ROOT / ".deployment" / "server.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        defaults = asdict(RemoteSettings())
        for row, (name, label) in enumerate([
            ("domain", "Public domain"), ("host", "Server IP / hostname"), ("user", "SSH username"),
            ("port", "SSH port"), ("identity", "SSH private-key file"), ("folder", "Folder in server home"),
        ], start=1):
            self._entry(tab, row, label, name, str(saved.get(name, defaults[name])), self.remote_fields)
        self._button(tab, "Browse", self.choose_identity).grid(row=5, column=2, padx=(8, 0))
        ttk.Label(tab, text="Needs Linux, Docker Engine 28+, Compose 2.33.1+, DNS and ports 80/443.\n"
                  "Use an SSH key / agent and trust the host in your SSH client first. No password is stored.",
                  style="Subtitle.TLabel", wraplength=710).grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 12))
        row = ttk.Frame(tab)
        row.grid(row=8, column=0, columnspan=3, sticky="w")
        self._button(row, "Check server", self.check_server).pack(side="left", padx=(0, 8))
        self._button(row, "Deploy to Lighthouse", self.deploy, True).pack(side="left", padx=(0, 8))
        self._button(row, "Prepare bundle", self.export_bundle).pack(side="left", padx=(0, 8))
        self._button(row, "Open site", self.open_remote).pack(side="left")
        ttk.Label(tab, text="Deploy uploads application code and the saved Azure settings to this server.\n"
                  "Existing conversation volumes are kept. It does not create a server or install Docker.",
                  style="Subtitle.TLabel", wraplength=700).grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _log(self, text):
        self.log_view.configure(state="normal")
        self.log_view.insert("end", redact(str(text), self.settings) + "\n")
        self.log_view.see("end")
        self.log_view.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._log(value)
                elif kind == "done":
                    self.busy = False
                    self.progress.stop()
                    for button in self.buttons:
                        button.configure(state="normal")
                    self.status.set(value)
                elif kind == "error":
                    self._log(value)
                    self.status.set("Action failed — see activity log")
                    if self.vision_result.get("1.0", "end").strip() == "Analyzing the selected photo...":
                        self._set_result(str(value))
                    if self.search_pending:
                        self.search_pending = False
                        self._set_search_result("Online search is unavailable. " + str(value))
                elif kind == "search_result":
                    self.search_pending = False
                    self._set_search_result(value["text"], value.get("sources", []))
                elif kind == "vision_result":
                    lines = [value.get("summary", "No description returned.")]
                    if value.get("items"):
                        lines.append("Visible: " + ", ".join(value["items"]))
                    if value.get("state"):
                        lines.append("Appearance: " + value["state"])
                    if value.get("hazards"):
                        lines.append("Safety: " + "; ".join(value["hazards"]))
                    self._set_result("\n\n".join(lines))
                elif kind == "open":
                    webbrowser.open(value)
        except queue.Empty:
            pass
        if self.camera:
            sequence, frame = self.camera.snapshot()
            if frame and sequence != self.camera_sequence and not self.camera.stop_requested.is_set():
                self.camera_sequence = sequence
                self._show_photo(frame)
                self.media_status.set("Live camera · capture a photo when you're ready")
            if not self.camera.running:
                if self.camera.error:
                    self.media_status.set(self.camera.error)
                    self._log(self.camera.error)
                self.camera = None
        if self.local.running:
            self.local_status.set("Running · " + self.local.url)
        elif not self.busy:
            self.local_status.set("Stopped · Runs on this computer")
        self.after(100, self._poll)

    def task(self, title, action):
        if self.busy:
            return
        self.busy = True
        for button in self.buttons:
            button.configure(state="disabled")
        self.status.set(title)
        self.progress.start(12)
        self._log(title)

        def worker():
            try:
                result = action(lambda text: self.events.put(("log", text)))
                self.events.put(("done", result or "Ready"))
            except Exception as exc:
                self.events.put(("error", str(exc)))
                self.events.put(("done", "Action failed — see activity log"))
        threading.Thread(target=worker, daemon=True).start()

    def save(self):
        try:
            updates = {name: variable.get().strip() for name, variable in self.fields.items()
                       if variable.get().strip() != self.settings.get(name, "")}
            # Preserve settings edited in a text editor while this window is open.
            candidate = {**model_settings(), **updates}
            validate_models(candidate)
            if updates or not (ROOT / "chatbot" / ".env").exists():
                save_env(ROOT / "chatbot" / ".env", updates or candidate)
            self.settings = candidate
            for name, variable in self.fields.items():
                variable.set(candidate.get(name, ""))
            self._log("Saved Azure settings to chatbot/.env. Restart a running coach to apply them.")
            self.status.set("Settings saved")
            return dict(candidate)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Check settings", str(exc), parent=self)
            return None

    def start_local(self):
        settings = self.save()
        if settings is None:
            self.tabs.select(self.settings_tab)
            return
        try:
            port = int(self.port.get())
        except ValueError:
            messagebox.showerror("Local port", "Enter a number between 1024 and 65535.", parent=self)
            return

        def start(log):
            url = self.local.start(settings, port, log)
            self.events.put(("open", url))
            return "Local coach ready · " + url
        self.task("Starting the local coach...", start)

    def stop_local(self):
        self.task("Stopping the local coach...", lambda log: self.local.stop() or "Local coach stopped")

    def open_local(self):
        if self.local.running:
            webbrowser.open(self.local.url)
        else:
            self.status.set("Start the local app first")

    def install(self):
        if self.local.running:
            messagebox.showinfo("Stop the coach", "Stop the local coach before installing dependencies.", parent=self)
            return
        self.task("Installing dependencies (internet required)...", lambda log: install_dependencies(log) or "Dependencies ready")

    def _test_azure(self, photo=None):
        settings = self.save()
        if settings is None:
            return
        env = dict(os.environ)
        for name in MODEL_KEYS:
            env.pop(name, None)
        env.update({key: value for key, value in settings.items() if key in MODEL_KEYS})
        args = [project_python(), str(ROOT / "scripts" / "check_setup.py"), "--live"]
        if photo:
            args += ["--image", photo]
        self.task("Testing Azure (uses API quota)...",
                  lambda log: run_command(args, lambda text: log(redact(text, settings)), env=env, timeout=180) or "Azure test passed")

    def test_chat(self):
        self._test_azure()

    def test_vision(self):
        self.tabs.select(self.media_tab)
        self.upload_photo()

    def choose_identity(self):
        path = filedialog.askopenfilename(parent=self, title="Choose your SSH private-key file")
        if path:
            self.remote_fields["identity"].set(path)

    def remote(self, require_ssh=True):
        try:
            values = {name: variable.get().strip() for name, variable in self.remote_fields.items()}
            values["port"] = int(values["port"])
            remote = RemoteSettings(**values)
            remote.validate(require_ssh=require_ssh)
            folder = ROOT / ".deployment"
            folder.mkdir(exist_ok=True)
            (folder / "server.json").write_text(json.dumps(asdict(remote), indent=2), encoding="utf-8")
            return remote
        except (OSError, ValueError) as exc:
            messagebox.showerror("Check server settings", str(exc), parent=self)
            return None

    def check_server(self):
        remote = self.remote()
        if remote:
            self.task("Checking SSH and Docker...", lambda log: check_remote(remote, log) or "Server prerequisites passed")

    def deploy(self):
        settings = self.save()
        if settings is None:
            return
        remote = self.remote()
        if remote is None:
            return

        def work(log):
            url = deploy_remote(settings, remote, log)
            return "Deployment healthy · " + url
        self.task("Deploying to " + remote.host + "...", work)

    def export_bundle(self):
        settings = self.save()
        if settings is None:
            return
        remote = self.remote(require_ssh=False)
        if remote is None:
            return

        def work(log):
            bundle = build_bundle(settings, remote)
            destination = ROOT / ".deployment" / "lh-cooking.tar.gz"
            destination.write_bytes(bundle)
            if os.name != "nt":
                destination.chmod(0o600)
            log("Prepared " + str(destination))
            log("Bundle includes your API key. Keep it private. On Linux: tar -xzf lh-cooking.tar.gz, then sh deploy.sh.")
            return "Deployment bundle ready (contains secret settings)"
        self.task("Preparing private deployment bundle...", work)

    def open_remote(self):
        remote = self.remote(require_ssh=False)
        if remote:
            webbrowser.open("https://" + remote.domain)

    def close(self):
        if self.busy:
            messagebox.showinfo("Operation in progress", "Wait for the current operation to finish before closing.", parent=self)
            return
        self.local.stop()
        self.close_camera()
        self.destroy()


def main():
    if "--help" in sys.argv:
        print("Cooking Coach desktop launcher\n\nRun: python app.py\nLocal service, Azure settings, and Lighthouse SSH deployment are available in the window.")
        return
    # A .pyw association may use system Python; use the installed project
    # interpreter so camera/photo packages are found after setup.
    interpreter = project_python()
    if (Path(interpreter).resolve() != Path(sys.executable).resolve()
            and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve()):
        subprocess.Popen([interpreter, str(ROOT / "app.py")], cwd=ROOT, **HIDDEN)
        return
    CookingApp().mainloop()


if __name__ == "__main__":
    main()
