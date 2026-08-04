using System.Diagnostics;
using AiToolkit.Launcher;
using AiToolkit.Launcher.Core;

internal static class Program
{
    private const string WpfSmokeArgument = "--wpf-smoke-child";
    private const string WpfSmokeMarker = "WPF_SMOKE_OK";

    private static async Task<int> Main(string[] args)
    {
        if (args.Contains(WpfSmokeArgument, StringComparer.Ordinal))
        {
            return RunWpfSmokeChild();
        }

        var tests = new (string Name, Func<Task> Run)[]
        {
            ("repository root discovery", TestRepositoryRootDiscovery),
            ("portable Python selection", TestPortablePythonSelection),
            ("direct Python preferred over py launcher", TestDirectPythonPreferred),
            ("manager command construction", TestManagerCommandConstruction),
            ("manager event parsing", TestManagerEventParsing),
            ("manager client output", TestManagerClientOutput),
            ("JSON document output capture", TestJsonDocumentOutputCapture),
            ("managed session stop", TestManagedSessionStop),
            ("toolkit status parsing", TestToolkitStatusParsing),
            ("UI readiness probe", TestUiReadinessProbe),
            ("view model maintenance state", TestViewModelMaintenanceState),
            ("view model shutdown waits for maintenance", TestViewModelShutdownWaitsForMaintenance),
            ("view model UI lifecycle", TestViewModelUiLifecycle),
            ("WPF window construction", TestWpfWindowConstruction),
        };

        var failures = new List<string>();
        foreach (var test in tests)
        {
            try
            {
                await test.Run();
                Console.WriteLine($"PASS {test.Name}");
            }
            catch (Exception ex)
            {
                failures.Add($"FAIL {test.Name}: {ex.Message}");
                Console.Error.WriteLine(failures[^1]);
            }
        }

        Console.WriteLine($"{tests.Length - failures.Count}/{tests.Length} tests passed");
        return failures.Count == 0 ? 0 : 1;
    }

    private static Task TestRepositoryRootDiscovery()
    {
        using var fixture = new TempDirectory();
        var root = fixture.Path;
        Directory.CreateDirectory(Path.Combine(root, "manager"));
        Directory.CreateDirectory(Path.Combine(root, "ui"));
        File.WriteAllText(Path.Combine(root, "manager", "__main__.py"), string.Empty);
        File.WriteAllText(Path.Combine(root, "ui", "package.json"), "{}");
        File.WriteAllText(Path.Combine(root, "version.py"), "VERSION = 'test'");
        var nested = Directory.CreateDirectory(Path.Combine(root, "launcher", "bin", "Debug"));

        Assert.Equal(Path.GetFullPath(root), RepositoryLocator.FindRoot(nested.FullName));
        return Task.CompletedTask;
    }

    private static Task TestPortablePythonSelection()
    {
        using var fixture = new TempDirectory();
        var python = Path.Combine(fixture.Path, "runtime", "python", "python.exe");
        Directory.CreateDirectory(Path.GetDirectoryName(python)!);
        File.WriteAllBytes(python, Array.Empty<byte>());

        var resolved = PythonLocator.Resolve(fixture.Path);

        Assert.Equal(python, resolved.FileName);
        Assert.True(resolved.IsPortable, "portable Python should be marked portable");
        Assert.Equal(0, resolved.PrefixArguments.Count);
        return Task.CompletedTask;
    }

    private static Task TestDirectPythonPreferred()
    {
        using var fixture = new TempDirectory();
        var tools = Path.Combine(fixture.Path, "tools");
        Directory.CreateDirectory(tools);
        var python = Path.Combine(tools, "python.exe");
        var py = Path.Combine(tools, "py.exe");
        File.WriteAllBytes(python, Array.Empty<byte>());
        File.WriteAllBytes(py, Array.Empty<byte>());
        var originalPath = Environment.GetEnvironmentVariable("PATH");
        try
        {
            Environment.SetEnvironmentVariable("PATH", tools);
            var resolved = PythonLocator.Resolve(fixture.Path);
            Assert.Equal(python, resolved.FileName);
            Assert.Equal(0, resolved.PrefixArguments.Count);
        }
        finally
        {
            Environment.SetEnvironmentVariable("PATH", originalPath);
        }
        return Task.CompletedTask;
    }

    private static Task TestManagerCommandConstruction()
    {
        using var fixture = new TempDirectory();
        var python = new PythonCommand("python.exe", Array.Empty<string>(), true);

        var invocation = ManagerCommand.Create(
            fixture.Path,
            python,
            ManagerAction.Sync,
            force: true
        );

        Assert.SequenceEqual(
            new[] { "-m", "manager", "--json-stream", "sync", "--force" },
            invocation.Arguments
        );
        Assert.Equal("portable", invocation.Environment["AITK_RUNTIME_LAYOUT"]);
        Assert.Equal("1", invocation.Environment["PYTHONUTF8"]);
        Assert.Equal("utf-8", invocation.Environment["PYTHONIOENCODING"]);
        Assert.Equal(ManagerOutputMode.JsonStream, invocation.OutputMode);
        return Task.CompletedTask;
    }

    private static Task TestManagerEventParsing()
    {
        var parsed = ManagerEventParser.Parse(
            "{\"type\":\"message\",\"level\":\"warning\",\"message\":\"请检查环境\"}"
        );
        var fallback = ManagerEventParser.Parse("raw process output");

        Assert.Equal("message", parsed.Type);
        Assert.Equal("warning", parsed.Level);
        Assert.Equal("请检查环境", parsed.Message);
        Assert.Equal("log", fallback.Type);
        Assert.Equal("raw process output", fallback.Message);
        return Task.CompletedTask;
    }

    private static async Task TestManagerClientOutput()
    {
        var invocation = new ManagerInvocation(
            "powershell.exe",
            new[]
            {
                "-NoProfile",
                "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                    + "Write-Output ''; "
                    + "Write-Output '{\"type\":\"message\",\"level\":\"info\",\"message\":\"hello\"}'",
            },
            Environment.CurrentDirectory,
            new Dictionary<string, string>(),
            ManagerOutputMode.JsonStream
        );
        var output = new List<ManagerEvent>();
        var client = new ManagerClient();

        var result = await client.RunAsync(invocation, output.Add, CancellationToken.None);

        Assert.Equal(0, result.ExitCode);
        Assert.True(result.Successful, "process should complete successfully");
        Assert.True(output.Any(item => item.Message == "hello"), "parsed output missing");
        Assert.True(
            output.All(item => !string.IsNullOrWhiteSpace(item.Message)),
            "blank process lines should not become log entries"
        );
    }

    private static async Task TestJsonDocumentOutputCapture()
    {
        var invocation = new ManagerInvocation(
            "powershell.exe",
            new[]
            {
                "-NoProfile",
                "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                    + "Write-Output '{'; Write-Output '\"value\": 1'; Write-Output '}'",
            },
            Environment.CurrentDirectory,
            new Dictionary<string, string>(),
            ManagerOutputMode.JsonDocument
        );
        var output = new List<ManagerEvent>();
        var result = await new ManagerClient().RunAsync(
            invocation,
            output.Add,
            CancellationToken.None
        );

        Assert.Equal(0, result.ExitCode);
        Assert.True(result.StandardOutput.Contains("\"value\": 1"), "JSON was not captured");
        Assert.Equal(0, output.Count);
    }

    private static async Task TestManagedSessionStop()
    {
        var invocation = new ManagerInvocation(
            "powershell.exe",
            new[] { "-NoProfile", "-Command", "Start-Sleep -Seconds 30" },
            Environment.CurrentDirectory,
            new Dictionary<string, string>(),
            ManagerOutputMode.Plain
        );
        var client = new ManagerClient();
        await using var session = client.StartSession(invocation, _ => { });

        Assert.True(session.ProcessId > 0, "session did not start");
        await session.StopAsync();
        var result = await session.Completion;
        Assert.True(result.WasStopped, "session should report an explicit stop");
        AssertProcessExited(session.ProcessId);
    }

    private static Task TestToolkitStatusParsing()
    {
        const string detectionJson = """
            {
              "os": "windows",
              "arch": "x86_64",
              "nvidia": {
                "gpus": [{"name": "RTX 5070 Ti", "memory": "16303 MiB", "compute_cap": "12.0"}],
                "driver": "610.88",
                "cuda_version": "13.3"
              },
              "spec": {"backend": "cu130"}
            }
            """;
        const string updateJson = """
            {
              "version": "1.18.0",
              "branch": "main",
              "dirty": false,
              "behind": 2,
              "venv": true,
              "deps_in_sync": false,
              "update_available": true,
              "backend": "cu130"
            }
            """;

        var hardware = ToolkitStatusParser.ParseDetection(detectionJson);
        var update = ToolkitStatusParser.ParseUpdateCheck(updateJson);

        Assert.Equal("RTX 5070 Ti", hardware.PrimaryGpuName);
        Assert.Equal("16303 MiB", hardware.PrimaryGpuMemory);
        Assert.Equal("13.3", hardware.CudaVersion);
        Assert.Equal("cu130", hardware.Backend);
        Assert.Equal("1.18.0", update.Version);
        Assert.Equal(2, update.Behind);
        Assert.True(update.UpdateAvailable, "update should be available");
        Assert.True(!update.DependenciesInSync, "dependency state should be preserved");
        return Task.CompletedTask;
    }

    private static async Task TestUiReadinessProbe()
    {
        using var listener = new System.Net.Sockets.TcpListener(
            System.Net.IPAddress.Loopback,
            0
        );
        listener.Start();
        var port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        var server = Task.Run(async () =>
        {
            using var client = await listener.AcceptTcpClientAsync();
            await using var stream = client.GetStream();
            var request = new byte[4096];
            var received = await stream.ReadAsync(request);
            Assert.True(received > 0, "probe connected without sending an HTTP request");
            var response = System.Text.Encoding.ASCII.GetBytes(
                "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
            );
            await stream.WriteAsync(response);
            await stream.FlushAsync();
        });

        var ready = await UiReadinessProbe.WaitAsync(
            new Uri($"http://127.0.0.1:{port}/"),
            TimeSpan.FromSeconds(5),
            CancellationToken.None
        );

        Assert.True(ready, "readiness probe did not observe the local server");
        await server;
    }

    private static async Task TestViewModelMaintenanceState()
    {
        var backend = new FakeLauncherBackend();
        var viewModel = new MainViewModel(backend, new ImmediateSynchronizationContext());
        await viewModel.InitializeAsync();

        var repair = viewModel.RepairAsync();
        await WaitUntilAsync(() => viewModel.IsBusy, TimeSpan.FromSeconds(2));
        Assert.True(!viewModel.CanStart, "start must be disabled during maintenance");
        Assert.Equal("正在修复环境", viewModel.StatusText);

        backend.CompleteMaintenance();
        await repair;

        Assert.True(!viewModel.IsBusy, "busy state was not cleared");
        Assert.True(viewModel.CanStart, "start should be enabled after maintenance");
        Assert.Equal("环境修复完成", viewModel.StatusText);
    }

    private static async Task TestViewModelUiLifecycle()
    {
        var backend = new FakeLauncherBackend();
        var viewModel = new MainViewModel(backend, new ImmediateSynchronizationContext());

        await viewModel.StartUiAsync();

        Assert.True(viewModel.IsUiRunning, "UI should be marked running after readiness");
        Assert.True(viewModel.CanStop, "stop should be enabled while UI runs");
        Assert.Equal("服务运行中", viewModel.ServiceStatus);

        await viewModel.StopUiAsync();

        Assert.True(!viewModel.IsUiRunning, "UI running state was not cleared");
        Assert.True(viewModel.CanStart, "start should be enabled after stop");
        Assert.Equal("服务已停止", viewModel.ServiceStatus);
    }

    private static async Task TestViewModelShutdownWaitsForMaintenance()
    {
        var backend = new FakeLauncherBackend();
        var viewModel = new MainViewModel(backend, new ImmediateSynchronizationContext());
        var repair = viewModel.RepairAsync();
        await WaitUntilAsync(() => viewModel.IsBusy, TimeSpan.FromSeconds(2));

        var dispose = viewModel.DisposeAsync().AsTask();
        await Task.Delay(50);
        Assert.True(!dispose.IsCompleted, "shutdown must wait for the active maintenance task");

        backend.CompleteMaintenance();
        await repair;
        await dispose;
    }

    private static async Task TestWpfWindowConstruction()
    {
        var executable = Environment.ProcessPath
            ?? throw new InvalidOperationException("Could not resolve the test executable.");
        var startInfo = new ProcessStartInfo
        {
            FileName = executable,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        startInfo.ArgumentList.Add(WpfSmokeArgument);

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Could not start the WPF smoke process.");
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        await process.WaitForExitAsync(timeout.Token);
        var capturedStdout = await stdout;
        var capturedStderr = await stderr;

        Assert.Equal(0, process.ExitCode);
        Assert.True(
            capturedStdout.Contains(WpfSmokeMarker, StringComparison.Ordinal),
            "WPF smoke child exited without its completion marker: " + capturedStderr
        );
    }

    private static int RunWpfSmokeChild()
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            var originalWindowsDirectory = Environment.GetEnvironmentVariable("WINDIR");
            try
            {
                Environment.SetEnvironmentVariable("WINDIR", null, EnvironmentVariableTarget.Process);
                var app = new App();
                app.InitializeComponent();
                var viewModel = new MainViewModel(
                    new FakeLauncherBackend(),
                    new ImmediateSynchronizationContext()
                );
                var window = new MainWindow(viewModel);
                window.Show();
                var dispatcher = System.Windows.Threading.Dispatcher.CurrentDispatcher;
                for (var index = 0; index < 8; index++)
                {
                    var messageIndex = index;
                    dispatcher.BeginInvoke(
                        new Action(
                            () =>
                                viewModel.Logs.Add(
                                    new LogEntry(
                                        DateTimeOffset.Now,
                                        "info",
                                        $"burst log {messageIndex}"
                                    )
                                )
                        )
                    );
                }
                dispatcher.Invoke(
                    () => { },
                    System.Windows.Threading.DispatcherPriority.ApplicationIdle
                );
                window.Measure(new System.Windows.Size(1180, 780));
                window.Arrange(new System.Windows.Rect(0, 0, 1180, 780));
                Console.WriteLine(WpfSmokeMarker);
                Console.Out.Flush();
                window.Hide();
            }
            catch (Exception ex)
            {
                failure = ex;
            }
            finally
            {
                Environment.SetEnvironmentVariable(
                    "WINDIR",
                    originalWindowsDirectory,
                    EnvironmentVariableTarget.Process
                );
            }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join(TimeSpan.FromSeconds(15));

        Assert.True(!thread.IsAlive, "WPF construction test timed out");
        if (failure is not null)
        {
            Console.Error.WriteLine(failure);
            return 1;
        }
        return 0;
    }

    private static async Task WaitUntilAsync(Func<bool> condition, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (!condition() && DateTime.UtcNow < deadline)
        {
            await Task.Delay(10);
        }
        Assert.True(condition(), "condition was not reached before timeout");
    }

    private static void AssertProcessExited(int processId)
    {
        try
        {
            using var process = Process.GetProcessById(processId);
            Assert.True(process.HasExited, "managed process is still running");
        }
        catch (ArgumentException)
        {
            // Process no longer exists.
        }
    }

    private sealed class TempDirectory : IDisposable
    {
        public TempDirectory()
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                "aitk-launcher-tests-" + Guid.NewGuid().ToString("N")
            );
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try
            {
                Directory.Delete(Path, recursive: true);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    private static class Assert
    {
        public static void Equal<T>(T expected, T actual)
        {
            if (!EqualityComparer<T>.Default.Equals(expected, actual))
            {
                throw new InvalidOperationException($"Expected '{expected}', got '{actual}'");
            }
        }

        public static void SequenceEqual<T>(IEnumerable<T> expected, IEnumerable<T> actual)
        {
            if (!expected.SequenceEqual(actual))
            {
                throw new InvalidOperationException(
                    $"Expected [{string.Join(", ", expected)}], got [{string.Join(", ", actual)}]"
                );
            }
        }

        public static void True(bool value, string message)
        {
            if (!value)
            {
                throw new InvalidOperationException(message);
            }
        }
    }

    private sealed class ImmediateSynchronizationContext : SynchronizationContext
    {
        public override void Post(SendOrPostCallback callback, object? state)
        {
            callback(state);
        }
    }

    private sealed class FakeLauncherBackend : ILauncherBackend
    {
        private readonly TaskCompletionSource<ManagerRunResult> _maintenance =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task<LauncherSnapshot> LoadSnapshotAsync(
            Action<ManagerEvent> onEvent,
            CancellationToken cancellationToken
        )
        {
            return Task.FromResult(
                new LauncherSnapshot(
                    "1.18.0",
                    new HardwareStatus(
                        "windows",
                        "x86_64",
                        "RTX 5070 Ti",
                        "16303 MiB",
                        "610.88",
                        "13.3",
                        "cu130"
                    ),
                    "便携运行时"
                )
            );
        }

        public Task<UpdateStatus> CheckUpdatesAsync(
            Action<ManagerEvent> onEvent,
            CancellationToken cancellationToken
        )
        {
            return Task.FromResult(
                new UpdateStatus("1.18.0", "main", false, 0, true, true, false, "cu130")
            );
        }

        public Task<ManagerRunResult> RunActionAsync(
            ManagerAction action,
            bool force,
            Action<ManagerEvent> onEvent,
            CancellationToken cancellationToken
        )
        {
            return _maintenance.Task;
        }

        public ILauncherUiSession StartUi(Action<ManagerEvent> onEvent)
        {
            return new FakeLauncherUiSession();
        }

        public void OpenUi() { }

        public void CompleteMaintenance()
        {
            _maintenance.TrySetResult(new ManagerRunResult(0, string.Empty, string.Empty, false));
        }
    }

    private sealed class FakeLauncherUiSession : ILauncherUiSession
    {
        private readonly TaskCompletionSource<ManagerRunResult> _completion =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public int ProcessId => 1234;

        public Task<ManagerRunResult> Completion => _completion.Task;

        public Task<bool> WaitUntilReadyAsync(CancellationToken cancellationToken)
        {
            return Task.FromResult(true);
        }

        public Task StopAsync()
        {
            _completion.TrySetResult(new ManagerRunResult(-1, string.Empty, string.Empty, true));
            return Task.CompletedTask;
        }

        public ValueTask DisposeAsync()
        {
            _completion.TrySetResult(new ManagerRunResult(-1, string.Empty, string.Empty, true));
            return ValueTask.CompletedTask;
        }
    }
}
