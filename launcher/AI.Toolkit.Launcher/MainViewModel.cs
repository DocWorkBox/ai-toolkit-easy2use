using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
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
    private EnvironmentHealth? _environmentHealth;
    private string _versionText = "正在读取";
    private string _gpuName = "正在检测 GPU";
    private string _gpuDetail = "-";
    private string _environmentStatus = "正在检查环境";
    private string _environmentDetail = "等待完整依赖诊断";
    private string _managedRoot;
    private string _updateStatus = "尚未检查";
    private string _serviceStatus = "服务已停止";
    private string _statusText = "准备就绪";
    private string _modelsRoot;
    private string _modelSummaryText = "尚未扫描模型目录";
    private bool _isBusy;
    private bool _isUiRunning;
    private bool _isShuttingDown;

    public MainViewModel(
        ILauncherBackend backend,
        SynchronizationContext? synchronizationContext = null
    )
    {
        _backend = backend;
        _managedRoot = backend.RepositoryRoot;
        _modelsRoot = Path.Combine(backend.RepositoryRoot, "models");
        _synchronizationContext = synchronizationContext
            ?? SynchronizationContext.Current
            ?? new SynchronizationContext();

        InstallCommand = new AsyncCommand(InstallAsync, CanInstallEnvironment);
        CheckUpdatesCommand = new AsyncCommand(CheckUpdatesAsync, CanRunMaintenance);
        RepairCommand = new AsyncCommand(RepairAsync, CanRepairEnvironment);
        UpdateCommand = new AsyncCommand(UpdateAsync, CanRunMaintenance);
        DoctorCommand = new AsyncCommand(DoctorAsync, CanRunMaintenance);
        StartCommand = new AsyncCommand(StartUiAsync, () => CanStart);
        StopCommand = new AsyncCommand(StopUiAsync, () => CanStop);
        OpenCommand = new RelayCommand(OpenUi, () => IsUiRunning);
        ScanModelsCommand = new AsyncCommand(ScanModelsAsync, CanScanModels);
        OpenModelsDirectoryCommand = new RelayCommand(OpenModelsDirectory);
        OpenModelDownloadCommand = new RelayCommand<ModelStatusItem>(
            OpenModelDownload,
            model => !_isShuttingDown && model.CanDownload
        );
        ClearLogsCommand = new RelayCommand(Logs.Clear, () => Logs.Count > 0);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public ObservableCollection<LogEntry> Logs { get; } = new();
    public ObservableCollection<ModelStatusItem> Models { get; } = new();

    public AsyncCommand InstallCommand { get; }
    public AsyncCommand CheckUpdatesCommand { get; }
    public AsyncCommand RepairCommand { get; }
    public AsyncCommand UpdateCommand { get; }
    public AsyncCommand DoctorCommand { get; }
    public AsyncCommand StartCommand { get; }
    public AsyncCommand StopCommand { get; }
    public RelayCommand OpenCommand { get; }
    public AsyncCommand ScanModelsCommand { get; }
    public RelayCommand OpenModelsDirectoryCommand { get; }
    public RelayCommand<ModelStatusItem> OpenModelDownloadCommand { get; }
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

    public string EnvironmentDetail
    {
        get => _environmentDetail;
        private set => SetField(ref _environmentDetail, value);
    }

    public string ManagedRoot
    {
        get => _managedRoot;
        private set => SetField(ref _managedRoot, value);
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

    public string ModelsRoot
    {
        get => _modelsRoot;
        private set => SetField(ref _modelsRoot, value);
    }

    public string ModelSummaryText
    {
        get => _modelSummaryText;
        private set => SetField(ref _modelSummaryText, value);
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

    public bool CanStart =>
        !_isShuttingDown
        && !IsBusy
        && _uiSession is null
        && EnvironmentCanLaunch();

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
                ManagedRoot = snapshot.RepositoryRoot;
                ApplyEnvironmentHealth(snapshot.Environment);
                StatusText = snapshot.Environment.MeetsRequirements
                    ? "环境检查完成，可以启动"
                    : "环境检查完成，需要处理问题";
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
            ManagerAction.Repair,
            false,
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
        return RunExclusiveAsync(
            "正在运行诊断",
            async cancellationToken =>
            {
                var health = await _backend.DiagnoseEnvironmentAsync(
                    AppendEvent,
                    cancellationToken
                );
                ApplyEnvironmentHealth(health);
                StatusText = health.MeetsRequirements
                    ? "诊断完成，环境符合要求"
                    : "诊断完成，发现环境问题";
            }
        );
    }

    public Task CheckUpdatesAsync()
    {
        return RunExclusiveAsync(
            "正在检查更新",
            async cancellationToken =>
            {
                var update = await _backend.CheckUpdatesAsync(AppendEvent, cancellationToken);
                UpdateStatus = update.UpdateAvailable
                    ? update.Behind is > 0
                        ? $"有 {update.Behind} 个代码更新"
                        : "依赖需要更新"
                    : "已是最新版本";
                StatusText = "更新检查完成";
            }
        );
    }

    public Task ScanModelsAsync()
    {
        return RunExclusiveAsync(
            "正在扫描模型目录",
            async cancellationToken =>
            {
                var report = await _backend.ScanModelsAsync(AppendEvent, cancellationToken);
                ModelsRoot = report.ModelsRoot;
                Models.Clear();
                foreach (var model in report.Models)
                {
                    Models.Add(model);
                }

                var summary = report.Summary;
                ModelSummaryText = $"可用 {summary.Ready} · 文件问题 {summary.Issues} · 未安装 {summary.Missing} · 未识别 {summary.Unrecognized}";
                StatusText = "模型目录扫描完成";
                AppendLog(
                    summary.Issues > 0 ? "warning" : "success",
                    $"模型扫描：{ModelSummaryText}；目录：{report.ModelsRoot}"
                );
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
            OpenUiInBackground();
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
                var health = await _backend.DiagnoseEnvironmentAsync(
                    AppendEvent,
                    cancellationToken
                );
                ApplyEnvironmentHealth(health);
                StatusText = health.MeetsRequirements
                    ? $"{completedStatus}，环境符合要求"
                    : $"{completedStatus}，仍有环境问题";
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

    private void OpenUiInBackground()
    {
        _ = Task.Run(
            () =>
            {
                try
                {
                    _backend.OpenUi();
                }
                catch (Exception ex)
                {
                    _synchronizationContext.Post(
                        _ => AppendLog("warning", $"无法自动打开界面：{ex.Message}"),
                        null
                    );
                }
            }
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

    private bool CanScanModels()
    {
        return !_isShuttingDown && !IsBusy;
    }

    private bool CanInstallEnvironment()
    {
        return CanRunMaintenance() && _environmentHealth?.EnvironmentExists == false;
    }

    private bool CanRepairEnvironment()
    {
        return CanRunMaintenance()
            && _environmentHealth is
            {
                EnvironmentExists: true,
                MeetsRequirements: false,
            }
            && _environmentHealth.RepairableFailures.Count > 0;
    }

    private bool EnvironmentCanLaunch()
    {
        if (_environmentHealth is not { EnvironmentExists: true } health)
        {
            return false;
        }
        if (health.MeetsRequirements)
        {
            return true;
        }

        var launchChecks = new[] { "venv", "node", "ui_dependencies" };
        return launchChecks.All(
            key => health.Checks.Any(check => check.Key == key && check.Passed)
        );
    }

    private void ApplyEnvironmentHealth(EnvironmentHealth health)
    {
        _environmentHealth = health;
        if (!health.EnvironmentExists)
        {
            EnvironmentStatus = "环境未安装";
            EnvironmentDetail = "未找到 AI Toolkit Python 环境";
        }
        else if (health.MeetsRequirements)
        {
            EnvironmentStatus = "环境符合要求";
            EnvironmentDetail = $"{health.RequiredPassed}/{health.RequiredTotal} 项必需检查通过";
            if (health.Warnings.Count > 0)
            {
                EnvironmentDetail += $" · {health.Warnings.Count} 项建议";
            }
        }
        else
        {
            EnvironmentStatus = "环境不符合要求";
            var failed = health.Checks
                .Where(check => check.Required && !check.Passed)
                .Select(DescribeFailedCheck)
                .Distinct(StringComparer.Ordinal)
                .Take(2)
                .ToArray();
            var issueText = failed.Length > 0
                ? string.Join("；", failed)
                : string.Join("、", health.FailedRequired.Take(3));
            EnvironmentDetail = $"{health.RequiredPassed}/{health.RequiredTotal} 项通过";
            if (!string.IsNullOrWhiteSpace(issueText))
            {
                EnvironmentDetail += $" · {issueText}";
            }
        }

        AppendLog(
            health.MeetsRequirements ? "success" : "warning",
            $"环境诊断：{health.RequiredPassed}/{health.RequiredTotal} 项必需检查通过；目标目录：{ManagedRoot}"
        );
        foreach (var check in health.Checks.Where(check => check.Required && !check.Passed))
        {
            AppendLog(
                "error",
                $"{LocalizeCheckName(check.Key, check.Label)}：{check.Detail}"
            );
        }
        foreach (var check in health.Checks.Where(check => !check.Required && !check.Passed))
        {
            AppendLog(
                "warning",
                $"建议项 {LocalizeCheckName(check.Key, check.Label)}：{check.Detail}"
            );
        }

        OnPropertyChanged(nameof(CanStart));
        RefreshCommands();
    }

    private static string DescribeFailedCheck(EnvironmentCheckStatus check)
    {
        if (check.Key == "requirements" && !string.IsNullOrWhiteSpace(check.Detail))
        {
            return string.Join(
                "；",
                check.Detail
                    .Split("; ", StringSplitOptions.RemoveEmptyEntries)
                    .Take(2)
                    .Select(LocalizeRequirementProblem)
            );
        }

        var name = LocalizeCheckName(check.Key, check.Label);
        return string.IsNullOrWhiteSpace(check.Detail) ? name : $"{name}：{check.Detail}";
    }

    private static string LocalizeRequirementProblem(string problem)
    {
        const string versionMarker = " is installed, expected ";
        var versionIndex = problem.IndexOf(versionMarker, StringComparison.Ordinal);
        if (versionIndex > 0)
        {
            var installed = problem[..versionIndex];
            var expected = problem[(versionIndex + versionMarker.Length)..];
            return $"{installed}，要求 {expected}";
        }

        const string missingMarker = " is not installed";
        if (problem.EndsWith(missingMarker, StringComparison.Ordinal))
        {
            return $"{problem[..^missingMarker.Length]} 未安装";
        }

        return problem;
    }

    private static string LocalizeCheckName(string key, string fallback)
    {
        return key switch
        {
            "environment_spec" => "硬件环境规格",
            "git" => "Git",
            "gpu" => "GPU",
            "venv" => "Python 环境",
            "python" => "Python 版本",
            "torch" => "PyTorch",
            "torchvision" => "TorchVision",
            "torchaudio" => "TorchAudio",
            "torch_stack_pins" => "PyTorch 固定版本",
            "pip_check" => "Python 依赖完整性",
            "requirements" => "AI Toolkit 依赖",
            "torch_gpu" => "PyTorch GPU",
            "node" => "Node.js",
            "ui_dependencies" => "UI 依赖",
            "ffmpeg" => "FFmpeg",
            _ => fallback,
        };
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

    private void OpenModelsDirectory()
    {
        try
        {
            _backend.OpenModelsDirectory();
        }
        catch (Exception ex)
        {
            AppendLog("error", ex.Message);
            StatusText = "无法打开模型目录";
        }
    }

    private void OpenModelDownload(ModelStatusItem model)
    {
        try
        {
            _backend.OpenUrl(model.DownloadUrl);
            AppendLog("info", $"已打开 {model.Name} 的官方模型页面");
        }
        catch (Exception ex)
        {
            AppendLog("error", ex.Message);
            StatusText = "无法打开模型下载页面";
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
        ScanModelsCommand.RaiseCanExecuteChanged();
        OpenModelsDirectoryCommand.RaiseCanExecuteChanged();
        OpenModelDownloadCommand.RaiseCanExecuteChanged();
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
