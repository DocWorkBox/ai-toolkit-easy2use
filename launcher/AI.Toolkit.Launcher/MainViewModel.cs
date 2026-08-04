using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using AiToolkit.Launcher.Core;

namespace AiToolkit.Launcher;

public sealed record LogEntry(DateTimeOffset Timestamp, string Level, string Message);

public sealed class MainViewModel : INotifyPropertyChanged, IAsyncDisposable
{
    private const int MaxLogEntries = 2000;

    private readonly ILauncherBackend _backend;
    private readonly SynchronizationContext _synchronizationContext;
    private readonly SemaphoreSlim _operationLock = new(1, 1);
    private ILauncherUiSession? _uiSession;
    private CancellationTokenSource? _operationCancellation;
    private string _versionText = "正在读取";
    private string _gpuName = "正在检测 GPU";
    private string _gpuDetail = "-";
    private string _environmentStatus = "正在检查环境";
    private string _updateStatus = "尚未检查";
    private string _serviceStatus = "服务已停止";
    private string _statusText = "准备就绪";
    private bool _isBusy;
    private bool _isUiRunning;
    private bool _isShuttingDown;

    public MainViewModel(
        ILauncherBackend backend,
        SynchronizationContext? synchronizationContext = null
    )
    {
        _backend = backend;
        _synchronizationContext = synchronizationContext
            ?? SynchronizationContext.Current
            ?? new SynchronizationContext();

        InstallCommand = new AsyncCommand(InstallAsync, CanRunMaintenance);
        CheckUpdatesCommand = new AsyncCommand(CheckUpdatesAsync, CanRunMaintenance);
        RepairCommand = new AsyncCommand(RepairAsync, CanRunMaintenance);
        UpdateCommand = new AsyncCommand(UpdateAsync, CanRunMaintenance);
        DoctorCommand = new AsyncCommand(DoctorAsync, CanRunMaintenance);
        StartCommand = new AsyncCommand(StartUiAsync, () => CanStart);
        StopCommand = new AsyncCommand(StopUiAsync, () => CanStop);
        OpenCommand = new RelayCommand(OpenUi, () => IsUiRunning);
        ClearLogsCommand = new RelayCommand(Logs.Clear, () => Logs.Count > 0);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public ObservableCollection<LogEntry> Logs { get; } = new();

    public AsyncCommand InstallCommand { get; }
    public AsyncCommand CheckUpdatesCommand { get; }
    public AsyncCommand RepairCommand { get; }
    public AsyncCommand UpdateCommand { get; }
    public AsyncCommand DoctorCommand { get; }
    public AsyncCommand StartCommand { get; }
    public AsyncCommand StopCommand { get; }
    public RelayCommand OpenCommand { get; }
    public RelayCommand ClearLogsCommand { get; }

    public string VersionText
    {
        get => _versionText;
        private set => SetField(ref _versionText, value);
    }

    public string GpuName
    {
        get => _gpuName;
        private set => SetField(ref _gpuName, value);
    }

    public string GpuDetail
    {
        get => _gpuDetail;
        private set => SetField(ref _gpuDetail, value);
    }

    public string EnvironmentStatus
    {
        get => _environmentStatus;
        private set => SetField(ref _environmentStatus, value);
    }

    public string UpdateStatus
    {
        get => _updateStatus;
        private set => SetField(ref _updateStatus, value);
    }

    public string ServiceStatus
    {
        get => _serviceStatus;
        private set => SetField(ref _serviceStatus, value);
    }

    public string StatusText
    {
        get => _statusText;
        private set => SetField(ref _statusText, value);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetField(ref _isBusy, value))
            {
                OnPropertyChanged(nameof(CanStart));
                RefreshCommands();
            }
        }
    }

    public bool IsUiRunning
    {
        get => _isUiRunning;
        private set
        {
            if (SetField(ref _isUiRunning, value))
            {
                OnPropertyChanged(nameof(CanStart));
                OnPropertyChanged(nameof(CanStop));
                RefreshCommands();
            }
        }
    }

    public bool CanStart => !_isShuttingDown && !IsBusy && _uiSession is null;

    public bool CanStop => !_isShuttingDown && _uiSession is not null;

    public Task InitializeAsync()
    {
        return RunExclusiveAsync(
            "正在检查环境",
            async cancellationToken =>
            {
                var snapshot = await _backend.LoadSnapshotAsync(AppendEvent, cancellationToken);
                VersionText = snapshot.Version;
                GpuName = snapshot.Hardware.PrimaryGpuName;
                GpuDetail = $"{snapshot.Hardware.PrimaryGpuMemory}  |  CUDA {snapshot.Hardware.CudaVersion}  |  {snapshot.Hardware.Backend}";
                EnvironmentStatus = snapshot.RuntimeDescription;
                StatusText = "环境信息已加载";
            }
        );
    }

    public Task InstallAsync()
    {
        return RunMaintenanceAsync(
            ManagerAction.Install,
            false,
            "正在安装环境",
            "环境安装完成"
        );
    }

    public Task RepairAsync()
    {
        return RunMaintenanceAsync(
            ManagerAction.Sync,
            true,
            "正在修复环境",
            "环境修复完成"
        );
    }

    public Task UpdateAsync()
    {
        return RunMaintenanceAsync(
            ManagerAction.Update,
            false,
            "正在更新程序",
            "程序更新完成"
        );
    }

    public Task DoctorAsync()
    {
        return RunMaintenanceAsync(
            ManagerAction.Doctor,
            false,
            "正在运行诊断",
            "环境诊断完成"
        );
    }

    public Task CheckUpdatesAsync()
    {
        return RunExclusiveAsync(
            "正在检查更新",
            async cancellationToken =>
            {
                var update = await _backend.CheckUpdatesAsync(AppendEvent, cancellationToken);
                EnvironmentStatus = update.DependenciesInSync
                    ? "依赖已同步"
                    : "依赖需要同步";
                UpdateStatus = update.UpdateAvailable
                    ? update.Behind is > 0
                        ? $"有 {update.Behind} 个代码更新"
                        : "依赖需要更新"
                    : "已是最新版本";
                StatusText = "更新检查完成";
            }
        );
    }

    public async Task StartUiAsync()
    {
        if (!await _operationLock.WaitAsync(0))
        {
            return;
        }

        _operationCancellation = new CancellationTokenSource();
        IsBusy = true;
        StatusText = "正在启动服务";
        ServiceStatus = "服务启动中";
        try
        {
            var session = _backend.StartUi(AppendEvent);
            _uiSession = session;
            OnPropertyChanged(nameof(CanStart));
            OnPropertyChanged(nameof(CanStop));
            RefreshCommands();

            var ready = await session.WaitUntilReadyAsync(_operationCancellation.Token);
            if (!ready)
            {
                var result = await session.Completion;
                throw new InvalidOperationException(
                    $"UI 服务在就绪前退出，代码 {result.ExitCode}。"
                );
            }

            IsUiRunning = true;
            ServiceStatus = "服务运行中";
            StatusText = "AI Toolkit 已启动";
            AppendLog("success", $"UI 服务已就绪，PID {session.ProcessId}");
            _backend.OpenUi();
            _ = ObserveUiExitAsync(session);
        }
        catch (OperationCanceledException)
        {
            StatusText = "启动已取消";
            ServiceStatus = "服务已停止";
        }
        catch (Exception ex)
        {
            AppendLog("error", ex.Message);
            StatusText = "启动失败";
            ServiceStatus = "服务启动失败";
            await ClearSessionAsync(_uiSession);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            IsBusy = false;
            _operationLock.Release();
        }
    }

    public async Task StopUiAsync()
    {
        var session = _uiSession;
        if (session is null)
        {
            return;
        }

        StatusText = "正在停止服务";
        ServiceStatus = "服务停止中";
        _operationCancellation?.Cancel();
        try
        {
            await session.StopAsync();
        }
        finally
        {
            await ClearSessionAsync(session);
            StatusText = "服务已停止";
            ServiceStatus = "服务已停止";
        }
    }

    public async Task ShutdownAsync()
    {
        _isShuttingDown = true;
        OnPropertyChanged(nameof(CanStart));
        OnPropertyChanged(nameof(CanStop));
        RefreshCommands();
        _operationCancellation?.Cancel();
        try
        {
            if (_uiSession is not null)
            {
                await StopUiAsync();
            }
        }
        finally
        {
            await _operationLock.WaitAsync();
            _operationLock.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        await ShutdownAsync();
        _operationCancellation?.Dispose();
        _operationLock.Dispose();
    }

    private Task RunMaintenanceAsync(
        ManagerAction action,
        bool force,
        string pendingStatus,
        string completedStatus
    )
    {
        return RunExclusiveAsync(
            pendingStatus,
            async cancellationToken =>
            {
                var result = await _backend.RunActionAsync(
                    action,
                    force,
                    AppendEvent,
                    cancellationToken
                );
                if (!result.Successful)
                {
                    throw new InvalidOperationException(
                        $"manager {action.ToString().ToLowerInvariant()} 失败，代码 {result.ExitCode}。"
                    );
                }
                StatusText = completedStatus;
                EnvironmentStatus = "环境已同步";
            }
        );
    }

    private async Task RunExclusiveAsync(
        string pendingStatus,
        Func<CancellationToken, Task> operation
    )
    {
        if (!await _operationLock.WaitAsync(0))
        {
            return;
        }

        using var cancellation = new CancellationTokenSource();
        _operationCancellation = cancellation;
        IsBusy = true;
        StatusText = pendingStatus;
        try
        {
            await operation(cancellation.Token);
        }
        catch (OperationCanceledException)
        {
            StatusText = "操作已取消";
        }
        catch (Exception ex)
        {
            AppendLog("error", ex.Message);
            StatusText = "操作失败";
        }
        finally
        {
            _operationCancellation = null;
            IsBusy = false;
            _operationLock.Release();
        }
    }

    private async Task ObserveUiExitAsync(ILauncherUiSession session)
    {
        var result = await session.Completion;
        _synchronizationContext.Post(
            async _ =>
            {
                if (!ReferenceEquals(_uiSession, session))
                {
                    return;
                }
                await ClearSessionAsync(session);
                ServiceStatus = result.WasStopped ? "服务已停止" : "服务意外退出";
                StatusText = result.WasStopped
                    ? "服务已停止"
                    : $"服务已退出，代码 {result.ExitCode}";
            },
            null
        );
    }

    private async Task ClearSessionAsync(ILauncherUiSession? session)
    {
        if (session is null)
        {
            return;
        }
        if (ReferenceEquals(_uiSession, session))
        {
            _uiSession = null;
        }
        await session.DisposeAsync();
        IsUiRunning = false;
        OnPropertyChanged(nameof(CanStart));
        OnPropertyChanged(nameof(CanStop));
        RefreshCommands();
    }

    private bool CanRunMaintenance()
    {
        return !_isShuttingDown && !IsBusy && _uiSession is null;
    }

    private void OpenUi()
    {
        try
        {
            _backend.OpenUi();
        }
        catch (Exception ex)
        {
            AppendLog("error", ex.Message);
            StatusText = "无法打开浏览器";
        }
    }

    private void AppendEvent(ManagerEvent managerEvent)
    {
        AppendLog(managerEvent.Level ?? managerEvent.Type, managerEvent.Message);
    }

    private void AppendLog(string level, string message)
    {
        _synchronizationContext.Post(
            _ =>
            {
                Logs.Add(new LogEntry(DateTimeOffset.Now, level, message));
                while (Logs.Count > MaxLogEntries)
                {
                    Logs.RemoveAt(0);
                }
                ClearLogsCommand.RaiseCanExecuteChanged();
            },
            null
        );
    }

    private void RefreshCommands()
    {
        InstallCommand.RaiseCanExecuteChanged();
        CheckUpdatesCommand.RaiseCanExecuteChanged();
        RepairCommand.RaiseCanExecuteChanged();
        UpdateCommand.RaiseCanExecuteChanged();
        DoctorCommand.RaiseCanExecuteChanged();
        StartCommand.RaiseCanExecuteChanged();
        StopCommand.RaiseCanExecuteChanged();
        OpenCommand.RaiseCanExecuteChanged();
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }
        field = value;
        OnPropertyChanged(name);
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? name = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
