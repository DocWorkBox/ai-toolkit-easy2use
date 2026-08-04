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

        return new ManagedSession(process, invocation.OutputMode, onEvent);
    }
}

public sealed class ManagedSession : IAsyncDisposable
{
    private readonly Process _process;
    private readonly object _eventLock = new();
    private readonly Action<ManagerEvent> _onEvent;
    private int _stopRequested;
    private int _disposed;

    internal ManagedSession(
        Process process,
        ManagerOutputMode outputMode,
        Action<ManagerEvent> onEvent
    )
    {
        _process = process;
        _onEvent = onEvent;
        ProcessId = process.Id;
        Completion = CompleteAsync(outputMode);
    }

    public int ProcessId { get; }

    public Task<ManagerRunResult> Completion { get; }

    public async Task StopAsync()
    {
        Interlocked.Exchange(ref _stopRequested, 1);
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
            // The process exited between the state check and Kill.
        }

        await Completion.ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        if (!_process.HasExited)
        {
            await StopAsync().ConfigureAwait(false);
        }
        else
        {
            await Completion.ConfigureAwait(false);
        }
        _process.Dispose();
    }

    private async Task<ManagerRunResult> CompleteAsync(ManagerOutputMode outputMode)
    {
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        var readStdout = ReadLinesAsync(
            _process.StandardOutput,
            stdout,
            outputMode == ManagerOutputMode.JsonStream
        );
        var readStderr = ReadLinesAsync(_process.StandardError, stderr, parseJson: false, "error");

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
        string? fallbackLevel = null
    )
    {
        while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            capture.AppendLine(line);
            var parsed = parseJson
                ? ManagerEventParser.Parse(line)
                : new ManagerEvent("log", fallbackLevel, line, line);
            lock (_eventLock)
            {
                _onEvent(parsed);
            }
        }
    }
}
