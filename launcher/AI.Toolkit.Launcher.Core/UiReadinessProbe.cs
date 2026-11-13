namespace AiToolkit.Launcher.Core;

public static class UiReadinessProbe
{
    public static async Task<Uri> WaitForEndpointAsync(string portFile, CancellationToken cancellationToken)
    {
        while (!File.Exists(portFile))
            await Task.Delay(150, cancellationToken).ConfigureAwait(false);
        var value = await File.ReadAllTextAsync(portFile, cancellationToken).ConfigureAwait(false);
        if (!ushort.TryParse(value, out var port) || port == 0)
            throw new InvalidDataException("Web 服务报告了无效端口。");
        var address = new Uri($"http://127.0.0.1:{port}/");
        if (!await WaitAsync(address, TimeSpan.FromMinutes(5), cancellationToken).ConfigureAwait(false))
            throw new TimeoutException($"Web 服务未就绪：{address}");
        return address;
    }

    public static async Task<bool> WaitAsync(
        Uri address,
        TimeSpan timeout,
        CancellationToken cancellationToken
    )
    {
        using var client = new HttpClient(new HttpClientHandler { UseProxy = false }) { Timeout = TimeSpan.FromSeconds(2) };
        var deadline = DateTimeOffset.UtcNow + timeout;

        while (DateTimeOffset.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                using var response = await client.GetAsync(address, cancellationToken)
                    .ConfigureAwait(false);
                if (response.IsSuccessStatusCode)
                {
                    return true;
                }
            }
            catch (HttpRequestException)
            {
                // The UI process is still starting.
            }
            catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                // A single probe timed out; continue until the overall deadline.
            }

            var remaining = deadline - DateTimeOffset.UtcNow;
            if (remaining <= TimeSpan.Zero)
            {
                break;
            }
            await Task.Delay(
                    remaining < TimeSpan.FromMilliseconds(500)
                        ? remaining
                        : TimeSpan.FromMilliseconds(500),
                    cancellationToken
                )
                .ConfigureAwait(false);
        }

        return false;
    }
}
