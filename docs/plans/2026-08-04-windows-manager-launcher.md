# Windows Manager Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Windows-only graphical launcher that manages, repairs, updates, starts, and stops an AI Toolkit portable installation through the existing `manager` CLI.

**Architecture:** Keep `manager` as the source of truth. Add a portable runtime layout adapter so manager operations target `runtime/python`, `runtime/node`, and `runtime/ffmpeg`, then expose line-delimited JSON events for long operations. Build an unpackaged WPF launcher with a dependency-free core library that locates the portable root, invokes only whitelisted manager commands, monitors port 8675, and owns the launched process tree.

**Tech Stack:** Python 3.8+ stdlib, pytest, C#/.NET 8, WPF, System.Text.Json, PowerShell publishing.

---

### Task 1: Portable Runtime Layout

**Files:**
- Create: `testing/test_manager_portable_layout.py`
- Modify: `manager/util.py`
- Modify: `manager/nodejs.py`
- Modify: `manager/ffmpeg.py`
- Modify: `manager/uvbin.py`
- Modify: `manager/gitwin.py`
- Modify: `manager/launch.py`

**Steps:**
1. Write failing tests for automatic and explicit portable layout detection, direct `runtime/python/python.exe`, component directories, and launch PATH construction.
2. Run `python -m pytest testing/test_manager_portable_layout.py -q` and verify failures are caused by missing portable-layout APIs.
3. Implement a small layout API in `manager.util` and replace hard-coded `.node`, `.ffmpeg`, `.uv`, and `.mingit` roots.
4. Ensure standard `.venv/.node/.ffmpeg` behavior remains unchanged when portable mode is absent.
5. Run the focused tests and manager detect/version smoke tests.

### Task 2: Machine-Readable Manager Progress

**Files:**
- Create: `testing/test_manager_json_stream.py`
- Modify: `manager/util.py`
- Modify: `manager/__main__.py`
- Modify: `manager/README.md`

**Steps:**
1. Write failing tests for NDJSON info/warning/error/result events and streamed child-process output.
2. Run the focused test and verify the expected missing-interface failure.
3. Add global `--json-stream`, preserving existing `detect --json` and `check --json` contracts.
4. Forward child output as log events in stream mode and preserve the flag through update re-exec.
5. Run focused tests and CLI smoke tests.

### Task 3: Launcher Core

**Files:**
- Create: `launcher/AI.Toolkit.Launcher.Core/AI.Toolkit.Launcher.Core.csproj`
- Create: `launcher/AI.Toolkit.Launcher.Core/RepositoryLocator.cs`
- Create: `launcher/AI.Toolkit.Launcher.Core/ManagerCommand.cs`
- Create: `launcher/AI.Toolkit.Launcher.Core/ManagerClient.cs`
- Create: `launcher/AI.Toolkit.Launcher.Core/ManagerModels.cs`
- Create: `launcher/AI.Toolkit.Launcher.Core/UiSession.cs`
- Create: `launcher/AI.Toolkit.Launcher.Tests/AI.Toolkit.Launcher.Tests.csproj`
- Create: `launcher/AI.Toolkit.Launcher.Tests/Program.cs`

**Steps:**
1. Create a dependency-free console test harness with failing tests for root discovery, portable Python selection, command whitelisting, NDJSON parsing, and process-tree cancellation.
2. Run `dotnet run --project launcher/AI.Toolkit.Launcher.Tests` and verify it fails for missing implementations.
3. Implement the smallest core API needed by the tests.
4. Re-run the harness and build the core project with zero warnings.

### Task 4: WPF User Interface

**Files:**
- Create: `launcher/AI.Toolkit.Launcher/AI.Toolkit.Launcher.csproj`
- Create: `launcher/AI.Toolkit.Launcher/App.xaml`
- Create: `launcher/AI.Toolkit.Launcher/App.xaml.cs`
- Create: `launcher/AI.Toolkit.Launcher/MainWindow.xaml`
- Create: `launcher/AI.Toolkit.Launcher/MainWindow.xaml.cs`
- Create: `launcher/AI.Toolkit.Launcher/MainViewModel.cs`
- Create: `launcher/AI.Toolkit.Launcher/AsyncCommand.cs`

**Steps:**
1. Add view-model tests for command enablement and state transitions to the console harness; verify they fail.
2. Implement a Chinese, light/dark-aware operational UI with environment summary, install/update/repair controls, start/stop/open controls, progress, diagnostics, and live logs.
3. Keep all manager calls asynchronous and prevent conflicting operations.
4. Run the core tests and build the WPF project.

### Task 5: Packaging and Integration

**Files:**
- Create: `launcher/AI.Toolkit.Launcher.sln`
- Create: `scripts/build_windows_launcher.ps1`
- Modify: `manager/README.md`
- Modify: `README.md`

**Steps:**
1. Add a deterministic PowerShell publish script for `win-x64`, self-contained, single-file output.
2. Publish to `dist/windows-launcher` and verify exactly one launcher executable plus required metadata is produced.
3. Run the launcher against a fixture portable root, exercise detect/check/doctor and start/stop process behavior, then run it against the real portable layout without modifying its dependencies.
4. Inspect the window, verify controls do not overlap at 100%, 125%, and a compact window size, and verify port/process cleanup.
5. Run all focused Python tests, the C# harness, release build, and git diff/status review.
