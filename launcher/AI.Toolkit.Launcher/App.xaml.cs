using System.Windows;
using AiToolkit.Launcher.Core;

namespace AiToolkit.Launcher;

public partial class App : Application
{
    public App()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("WINDIR")))
        {
            var systemRoot = Environment.GetEnvironmentVariable("SystemRoot");
            if (!string.IsNullOrWhiteSpace(systemRoot))
            {
                Environment.SetEnvironmentVariable(
                    "WINDIR",
                    systemRoot,
                    EnvironmentVariableTarget.Process
                );
            }
        }
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        try
        {
            if (LauncherSelfUpdate.TryHandleHelperArguments(e.Args, out var exitCode))
            {
                Shutdown(exitCode);
                return;
            }

            var root = RepositoryLocator.FindRoot(AppContext.BaseDirectory);
            LauncherSelfUpdate.CleanupAppliedUpdate(root, e.Args);
            if (LauncherSelfUpdate.TryStartPendingUpdate(root))
            {
                Shutdown();
                return;
            }

            var backend = new LauncherBackend(root, PythonLocator.Resolve(root));
            var viewModel = new MainViewModel(backend);
            var window = new MainWindow(viewModel);
            MainWindow = window;
            window.Show();
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "无法启动 AI Toolkit 管理器。\n\n" + ex.Message,
                "AI Toolkit Easy2Use",
                MessageBoxButton.OK,
                MessageBoxImage.Error
            );
            Shutdown(1);
        }
    }
}
