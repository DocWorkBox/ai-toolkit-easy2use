using System.Diagnostics;
using AiToolkit.Launcher.Core;

internal static class Program
{
    private static async Task<int> Main()
    {
        var tests = new (string Name, Func<Task> Run)[]
        {
            ("repository root discovery", TestRepositoryRootDiscovery),
            ("portable Python selection", TestPortablePythonSelection),
            ("manager command construction", TestManagerCommandConstruction),
            ("manager event parsing", TestManagerEventParsing),
            ("manager client output", TestManagerClientOutput),
            ("managed session stop", TestManagedSessionStop),
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
}
