using System.Collections.Specialized;
using System.ComponentModel;
using System.Windows;

namespace AiToolkit.Launcher;

public partial class MainWindow : Window
{
    private bool _allowClose;
    private bool _closeScheduled;
    private bool _logScrollPending;

    public MainWindow(MainViewModel viewModel)
    {
        InitializeComponent();
        ViewModel = viewModel;
        DataContext = viewModel;
        Loaded += OnLoaded;
        Closing += OnClosing;
        ViewModel.Logs.CollectionChanged += OnLogsChanged;
        ViewModel.RestartRequiredDetected += OnRestartRequiredDetected;
    }

    public MainViewModel ViewModel { get; }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        Loaded -= OnLoaded;
        await ViewModel.InitializeAsync();
        ScanOnFirstVisit();
    }

    private void OnModelsPageSelected(object sender, RoutedEventArgs e) => ScanOnFirstVisit();

    private void ScanOnFirstVisit()
    {
        if (IsLoaded && ModelsNav.IsChecked == true && ViewModel.ModelFilters.Count == 1
            && ViewModel.ScanModelsCommand.CanExecute(null))
            ViewModel.ScanModelsCommand.Execute(null);
    }

    private void OnLogsChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (_logScrollPending)
        {
            return;
        }

        _logScrollPending = true;
        Dispatcher.BeginInvoke(
            new Action(() =>
            {
                _logScrollPending = false;
                if (ViewModel.Logs.Count > 0)
                {
                    LogList.ScrollIntoView(ViewModel.Logs[^1]);
                }
            }),
            System.Windows.Threading.DispatcherPriority.ContextIdle
        );
    }

    private void OnRestartRequiredDetected(object? sender, EventArgs e)
    {
        Dispatcher.BeginInvoke(
            new Action(
                () =>
                    MessageBox.Show(
                        this,
                        "更新已完成。请关闭并重新打开启动器，以应用本次更新的新功能。",
                        "需要重启应用",
                        MessageBoxButton.OK,
                        MessageBoxImage.Information
                    )
            )
        );
    }

    private void OnClosing(object? sender, CancelEventArgs e)
    {
        if (_allowClose)
        {
            return;
        }

        e.Cancel = true;
        if (_closeScheduled)
        {
            return;
        }

        _closeScheduled = true;
        Dispatcher.BeginInvoke(
            new Action(async () =>
            {
                ViewModel.Logs.CollectionChanged -= OnLogsChanged;
                ViewModel.RestartRequiredDetected -= OnRestartRequiredDetected;
                try
                {
                    await ViewModel.DisposeAsync();
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine(ex);
                }
                finally
                {
                    _allowClose = true;
                    Close();
                }
            })
        );
    }
}
