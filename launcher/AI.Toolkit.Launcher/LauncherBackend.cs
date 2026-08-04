using System.Diagnostics;
using AiToolkit.Launcher.Core;

namespace AiToolkit.Launcher;

public sealed record LauncherSnapshot(
    string Version,
    HardwareStatus Hardware,
    string RuntimeDescription
);

public interface ILauncherUiSession : IAsyncDisposable
{
    int ProcessId { get; }
    Task<ManagerRunResult> Completion { get; }
    Task<bool> WaitUntilReadyAsync(CancellationToken cancellationToken);
    Task StopAsync();
}

public interface ILauncherBackend
{
    Task<LauncherSnapshot> LoadSnapshotAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    );

    Task<UpdateStatus> CheckUpdatesAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    );

    Task<ManagerRunResult> RunActionAsync(
        ManagerAction action,
        bool force,
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    );

    ILauncherUiSession StartUi(Action<ManagerEvent> onEvent);

    void OpenUi();
}

public sealed class LauncherBackend : ILauncherBackend
{
    public static readonly Uri UiAddress = new("http://127.0.0.1:8675/");

    private readonly string _repositoryRoot;
    private readonly PythonCommand _python;
    private readonly ManagerClient _client;

    public LauncherBackend(string repositoryRoot, PythonCommand python, ManagerClient? client = null)
    {
        _repositoryRoot = repositoryRoot;
        _python = python;
        _client = client ?? new ManagerClient();
    }

    public string RepositoryRoot => _repositoryRoot;

    public static LauncherBackend CreateDefault()
    {
        var root = RepositoryLocator.FindRoot(AppContext.BaseDirectory);
        return new LauncherBackend(root, PythonLocator.Resolve(root));
    }

    public async Task<LauncherSnapshot> LoadSnapshotAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        var version = await RunCheckedAsync(ManagerAction.Version, false, onEvent, cancellationToken);
        var detection = await RunCheckedAsync(ManagerAction.Detect, false, onEvent, cancellationToken);
        return new LauncherSnapshot(
            version.StandardOutput.Trim(),
            ToolkitStatusParser.ParseDetection(detection.StandardOutput),
            _python.IsPortable ? "便携运行时" : "标准 Python 环境"
        );
    }

    public async Task<UpdateStatus> CheckUpdatesAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        var result = await RunCheckedAsync(
            ManagerAction.Check,
            false,
            onEvent,
            cancellationToken
        );
        return ToolkitStatusParser.ParseUpdateCheck(result.StandardOutput);
    }

    public Task<ManagerRunResult> RunActionAsync(
        ManagerAction action,
        bool force,
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        return RunCheckedAsync(action, force, onEvent, cancellationToken);
    }

    public ILauncherUiSession StartUi(Action<ManagerEvent> onEvent)
    {
        var invocation = ManagerCommand.Create(
            _repositoryRoot,
            _python,
            ManagerAction.Launch
        );
        return new LauncherUiSession(_client.StartSession(invocation, onEvent));
    }

    public void OpenUi()
    {
        Process.Start(new ProcessStartInfo(LauncherBackend.UiAddress.AbsoluteUri)
        {
            UseShellExecute = true,
        });
    }

    private async Task<ManagerRunResult> RunCheckedAsync(
        ManagerAction action,
        bool force,
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        var invocation = ManagerCommand.Create(_repositoryRoot, _python, action, force);
        var result = await _client.RunAsync(invocation, onEvent, cancellationToken);
        if (!result.Successful)
        {
            var detail = string.IsNullOrWhiteSpace(result.StandardError)
                ? $"manager {action.ToString().ToLowerInvariant()} exited with code {result.ExitCode}."
                : result.StandardError.Trim();
            throw new InvalidOperationException(detail);
        }
        return result;
    }
}

internal sealed class LauncherUiSession : ILauncherUiSession
{
    private readonly ManagedSession _session;

    public LauncherUiSession(ManagedSession session)
    {
        _session = session;
    }

    public int ProcessId => _session.ProcessId;

    public Task<ManagerRunResult> Completion => _session.Completion;

    public async Task<bool> WaitUntilReadyAsync(CancellationToken cancellationToken)
    {
        var ready = UiReadinessProbe.WaitAsync(
            LauncherBackend.UiAddress,
            TimeSpan.FromMinutes(5),
            cancellationToken
        );
        var completed = await Task.WhenAny(ready, Completion);
        return completed == ready && await ready;
    }

    public Task StopAsync()
    {
        return _session.StopAsync();
    }

    public ValueTask DisposeAsync()
    {
        return _session.DisposeAsync();
    }
}
