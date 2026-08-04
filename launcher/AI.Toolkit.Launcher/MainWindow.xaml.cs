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
    }

    public MainViewModel ViewModel { get; }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        Loaded -= OnLoaded;
        await ViewModel.InitializeAsync();
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
