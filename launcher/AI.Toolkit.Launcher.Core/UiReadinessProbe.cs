namespace AiToolkit.Launcher.Core;

public static class UiReadinessProbe
{
    public static async Task<bool> WaitAsync(
        Uri address,
        TimeSpan timeout,
        CancellationToken cancellationToken
    )
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
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
