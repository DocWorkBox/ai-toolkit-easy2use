using System.Diagnostics;
using System.Text;

namespace AiToolkit.Launcher.Core;

public sealed class ManagerClient
{
    public async Task<ManagerRunResult> RunAsync(
        ManagerInvocation invocation,
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        await using var session = StartSession(invocation, onEvent);
        using var registration = cancellationToken.Register(
            static state => _ = ((ManagedSession)state!).StopAsync(),
            session
        );
        var result = await session.Completion.ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    public ManagedSession StartSession(
        ManagerInvocation invocation,
        Action<ManagerEvent> onEvent
    )
    {
        ArgumentNullException.ThrowIfNull(invocation);
        ArgumentNullException.ThrowIfNull(onEvent);

        var startInfo = new ProcessStartInfo
        {
            FileName = invocation.FileName,
            WorkingDirectory = invocation.WorkingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var argument in invocation.Arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }
        foreach (var variable in invocation.Environment)
        {
            startInfo.Environment[variable.Key] = variable.Value;
        }

        var process = new Process { StartInfo = startInfo };
        if (!process.Start())
        {
            process.Dispose();
            throw new InvalidOperationException($"Could not start '{invocation.FileName}'.");
        }

        return new ManagedSession(
            process,
            invocation.OutputMode,
            onEvent,
            WindowsProcessJob.TryAssign(process)
        );
    }
}

public sealed class ManagedSession : IAsyncDisposable
{
    private readonly Process _process;
    private readonly object _eventLock = new();
    private readonly object _stopLock = new();
    private readonly Action<ManagerEvent> _onEvent;
    private readonly WindowsProcessJob? _processJob;
    private readonly CancellationTokenSource _outputCancellation = new();
    private Task? _stopTask;
    private int _stopRequested;
    private int _disposed;

    internal ManagedSession(
        Process process,
        ManagerOutputMode outputMode,
        Action<ManagerEvent> onEvent,
        WindowsProcessJob? processJob
    )
    {
        _process = process;
        _onEvent = onEvent;
        _processJob = processJob;
        ProcessId = process.Id;
        Completion = CompleteAsync(outputMode);
    }

    public int ProcessId { get; }

    public Task<ManagerRunResult> Completion { get; }

    public Task StopAsync()
    {
        lock (_stopLock)
        {
            return _stopTask ??= StopCoreAsync();
        }
    }

    private async Task StopCoreAsync()
    {
        Interlocked.Exchange(ref _stopRequested, 1);
        var jobTerminated = _processJob?.Terminate() == true;
        if (!jobTerminated)
        {
            await TryTaskKillAsync().ConfigureAwait(false);
            TryKillRoot();
        }

        try
        {
            await Completion.WaitAsync(TimeSpan.FromSeconds(15)).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            TryKillRoot();
            _outputCancellation.Cancel();
            await Completion.WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        if (!Completion.IsCompleted)
        {
            await StopAsync().ConfigureAwait(false);
        }
        else
        {
            await Completion.ConfigureAwait(false);
        }
        _outputCancellation.Dispose();
        _processJob?.Dispose();
        _process.Dispose();
    }

    private async Task<ManagerRunResult> CompleteAsync(ManagerOutputMode outputMode)
    {
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        var readStdout = ReadLinesAsync(
            _process.StandardOutput,
            stdout,
            outputMode == ManagerOutputMode.JsonStream,
            outputMode != ManagerOutputMode.JsonDocument
        );
        var readStderr = ReadLinesAsync(
            _process.StandardError,
            stderr,
            parseJson: false,
            emitEvents: true,
            fallbackLevel: "error"
        );

        await Task.WhenAll(readStdout, readStderr, _process.WaitForExitAsync()).ConfigureAwait(false);
        return new ManagerRunResult(
            _process.ExitCode,
            stdout.ToString(),
            stderr.ToString(),
            Volatile.Read(ref _stopRequested) != 0
        );
    }

    private async Task ReadLinesAsync(
        StreamReader reader,
        StringBuilder capture,
        bool parseJson,
        bool emitEvents,
        string? fallbackLevel = null
    )
    {
        try
        {
            while (await reader.ReadLineAsync(_outputCancellation.Token).ConfigureAwait(false)
                is { } line)
            {
                capture.AppendLine(line);
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }
                if (!emitEvents)
                {
                    continue;
                }
                var parsed = parseJson
                    ? ManagerEventParser.Parse(line)
                    : new ManagerEvent("log", fallbackLevel, line, line);
                lock (_eventLock)
                {
                    _onEvent(parsed);
                }
            }
        }
        catch (OperationCanceledException) when (_outputCancellation.IsCancellationRequested)
        {
            // A process outside the managed job retained a redirected pipe during shutdown.
        }
    }

    private async Task TryTaskKillAsync()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        try
        {
            var taskkillPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.System),
                "taskkill.exe"
            );
            var startInfo = new ProcessStartInfo
            {
                FileName = taskkillPath,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("/PID");
            startInfo.ArgumentList.Add(ProcessId.ToString());
            startInfo.ArgumentList.Add("/T");
            startInfo.ArgumentList.Add("/F");
            using var taskkill = Process.Start(startInfo);
            if (taskkill is null)
            {
                return;
            }
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            await taskkill.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (Exception error) when (
            error is InvalidOperationException
                or System.ComponentModel.Win32Exception
                or OperationCanceledException
        )
        {
            // Fall through to Process.Kill below.
        }
    }

    private void TryKillRoot()
    {
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (Exception error) when (
            error is InvalidOperationException or System.ComponentModel.Win32Exception
        )
        {
            // The process exited between the state check and Kill.
        }
    }
}
