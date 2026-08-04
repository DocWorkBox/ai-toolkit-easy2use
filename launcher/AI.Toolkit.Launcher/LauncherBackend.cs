using System.Diagnostics;
using System.IO;
using AiToolkit.Launcher.Core;

namespace AiToolkit.Launcher;

public sealed record LauncherSnapshot(
    string Version,
    HardwareStatus Hardware,
    EnvironmentHealth Environment,
    string RepositoryRoot
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
    string RepositoryRoot { get; }

    Task<LauncherSnapshot> LoadSnapshotAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    );

    Task<UpdateStatus> CheckUpdatesAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    );

    Task<EnvironmentHealth> DiagnoseEnvironmentAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    );

    Task<ModelScanReport> ScanModelsAsync(
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

    void OpenModelsDirectory();

    void OpenUrl(string url);
}

public sealed class LauncherBackend : ILauncherBackend
{
    public static readonly Uri UiAddress = new("http://127.0.0.1:8675/");

    private readonly string _repositoryRoot;
    private readonly PythonCommand _python;
    private readonly ManagerClient _client;
    private readonly EnvironmentHealthCache _environmentCache;

    public LauncherBackend(
        string repositoryRoot,
        PythonCommand python,
        ManagerClient? client = null,
        EnvironmentHealthCache? environmentCache = null
    )
    {
        _repositoryRoot = repositoryRoot;
        _python = python;
        _client = client ?? new ManagerClient();
        _environmentCache = environmentCache ?? new EnvironmentHealthCache(repositoryRoot);
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
        var cachedEnvironment = _environmentCache.TryLoad();
        var environment = cachedEnvironment?.Health
            ?? await DiagnoseEnvironmentAsync(onEvent, cancellationToken);
        if (cachedEnvironment is not null)
        {
            var message = $"已读取上次环境诊断（{cachedEnvironment.CheckedAtUtc.ToLocalTime():yyyy-MM-dd HH:mm}）";
            onEvent(new ManagerEvent("log", "info", message, message));
        }
        return new LauncherSnapshot(
            version.StandardOutput.Trim(),
            ToolkitStatusParser.ParseDetection(detection.StandardOutput),
            environment,
            _repositoryRoot
        );
    }

    public async Task<EnvironmentHealth> DiagnoseEnvironmentAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        var result = await RunCheckedAsync(
            ManagerAction.Doctor,
            false,
            onEvent,
            cancellationToken
        );
        var health = ToolkitStatusParser.ParseEnvironmentHealth(result.StandardOutput);
        try
        {
            _environmentCache.Save(health, DateTimeOffset.UtcNow);
        }
        catch (Exception error) when (
            error is IOException
                or UnauthorizedAccessException
                or System.Security.SecurityException
                or NotSupportedException
        )
        {
            var message = $"环境诊断结果未能写入缓存：{error.Message}";
            onEvent(new ManagerEvent("log", "warning", message, message));
        }
        return health;
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

    public async Task<ModelScanReport> ScanModelsAsync(
        Action<ManagerEvent> onEvent,
        CancellationToken cancellationToken
    )
    {
        var result = await RunCheckedAsync(
            ManagerAction.Models,
            false,
            onEvent,
            cancellationToken
        );
        return ToolkitStatusParser.ParseModelScan(result.StandardOutput);
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
        OpenUrl(LauncherBackend.UiAddress.AbsoluteUri);
    }

    public void OpenModelsDirectory()
    {
        var modelsDirectory = Path.Combine(_repositoryRoot, "models");
        Directory.CreateDirectory(modelsDirectory);
        Process.Start(new ProcessStartInfo(modelsDirectory)
        {
            UseShellExecute = true,
        });
    }

    public void OpenUrl(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri)
            || uri.Scheme != Uri.UriSchemeHttps && uri.Scheme != Uri.UriSchemeHttp)
        {
            throw new ArgumentException("Only HTTP(S) addresses can be opened.", nameof(url));
        }

        Process.Start(new ProcessStartInfo(uri.AbsoluteUri)
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
