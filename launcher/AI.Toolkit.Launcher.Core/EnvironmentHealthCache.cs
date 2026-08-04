using System.Text;
using System.Text.Json;

namespace AiToolkit.Launcher.Core;

public sealed record EnvironmentHealthCacheEntry(
    EnvironmentHealth Health,
    DateTimeOffset CheckedAtUtc
);

public sealed class EnvironmentHealthCache
{
    private const int SchemaVersion = 1;
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
    };

    private readonly string _repositoryRoot;

    public EnvironmentHealthCache(string repositoryRoot)
    {
        _repositoryRoot = NormalizeRoot(repositoryRoot);
        CachePath = Path.Combine(
            _repositoryRoot,
            ".cache",
            "launcher",
            "environment-diagnosis.json"
        );
    }

    public string CachePath { get; }

    public EnvironmentHealthCacheEntry? TryLoad()
    {
        try
        {
            if (!File.Exists(CachePath))
            {
                return null;
            }

            var json = File.ReadAllText(CachePath, Encoding.UTF8);
            var document = JsonSerializer.Deserialize<CacheDocument>(json, SerializerOptions);
            if (document is null
                || document.SchemaVersion != SchemaVersion
                || !StringComparer.OrdinalIgnoreCase.Equals(
                    NormalizeRoot(document.RepositoryRoot),
                    _repositoryRoot
                )
                || !IsComplete(document.Health))
            {
                return null;
            }

            return new EnvironmentHealthCacheEntry(
                document.Health,
                document.CheckedAtUtc.ToUniversalTime()
            );
        }
        catch (Exception error) when (
            error is IOException
                or UnauthorizedAccessException
                or JsonException
                or ArgumentException
                or NotSupportedException
        )
        {
            return null;
        }
    }

    public void Save(EnvironmentHealth health, DateTimeOffset checkedAtUtc)
    {
        ArgumentNullException.ThrowIfNull(health);
        var cacheDirectory = Path.GetDirectoryName(CachePath)!;
        Directory.CreateDirectory(cacheDirectory);
        var temporaryPath = CachePath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            var document = new CacheDocument(
                SchemaVersion,
                _repositoryRoot,
                checkedAtUtc.ToUniversalTime(),
                health
            );
            var json = JsonSerializer.Serialize(document, SerializerOptions);
            File.WriteAllText(temporaryPath, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            File.Move(temporaryPath, CachePath, overwrite: true);
        }
        finally
        {
            try
            {
                File.Delete(temporaryPath);
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                // The completed cache file remains valid if temporary cleanup fails.
            }
        }
    }

    private static string NormalizeRoot(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));
    }

    private static bool IsComplete(EnvironmentHealth? health)
    {
        return health is not null
            && health.FailedRequired is not null
            && health.RepairableFailures is not null
            && health.Warnings is not null
            && health.Checks is not null;
    }

    private sealed record CacheDocument(
        int SchemaVersion,
        string RepositoryRoot,
        DateTimeOffset CheckedAtUtc,
        EnvironmentHealth Health
    );
}
