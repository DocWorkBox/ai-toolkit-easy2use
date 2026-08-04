using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;

namespace AiToolkit.Launcher;

public static class LauncherSelfUpdate
{
    public const string ApplyArgument = "--apply-portable-update";
    public const string CleanupArgument = "--cleanup-portable-update";

    private const string LauncherFileName = "AI Toolkit Launcher.exe";
    private const string PendingFileName = "AI Toolkit Launcher.update.exe";

    public static string PendingPath(string repositoryRoot)
    {
        return Path.Combine(
            Path.GetFullPath(repositoryRoot),
            ".cache",
            "portable-update",
            PendingFileName
        );
    }

    public static bool HasValidPendingUpdate(string repositoryRoot)
    {
        var path = PendingPath(repositoryRoot);
        try
        {
            using var stream = File.OpenRead(path);
            return stream.ReadByte() == 'M' && stream.ReadByte() == 'Z';
        }
        catch (Exception error) when (
            error is IOException
                or UnauthorizedAccessException
                or NotSupportedException
                or System.Security.SecurityException
        )
        {
            return false;
        }
    }

    public static bool TryHandleHelperArguments(
        IReadOnlyList<string> arguments,
        out int exitCode
    )
    {
        exitCode = 0;
        if (
            arguments.Count < 2
            || !string.Equals(arguments[0], ApplyArgument, StringComparison.Ordinal)
        )
        {
            return false;
        }

        exitCode = ApplyPendingUpdate(ParseProcessId(arguments[1]));
        return true;
    }

    public static void CleanupAppliedUpdate(
        string repositoryRoot,
        IReadOnlyList<string> arguments
    )
    {
        if (
            arguments.Count < 2
            || !string.Equals(arguments[0], CleanupArgument, StringComparison.Ordinal)
        )
        {
            return;
        }

        WaitForProcessExit(ParseProcessId(arguments[1]), TimeSpan.FromSeconds(15));
        TryDelete(PendingPath(repositoryRoot));
    }

    public static bool TryStartPendingUpdate(string repositoryRoot)
    {
        var root = Path.GetFullPath(repositoryRoot);
        var pendingPath = PendingPath(root);
        if (!HasValidPendingUpdate(root))
        {
            return false;
        }

        var installedPath = Path.Combine(root, LauncherFileName);
        if (File.Exists(installedPath) && FilesMatch(installedPath, pendingPath))
        {
            TryDelete(pendingPath);
            return false;
        }

        var startInfo = new ProcessStartInfo(pendingPath)
        {
            UseShellExecute = false,
            WorkingDirectory = root,
        };
        startInfo.ArgumentList.Add(ApplyArgument);
        startInfo.ArgumentList.Add(Environment.ProcessId.ToString());
        _ = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Could not start the launcher update helper.");
        return true;
    }

    private static int ApplyPendingUpdate(int previousProcessId)
    {
        var runningPath = Environment.ProcessPath;
        if (string.IsNullOrWhiteSpace(runningPath))
        {
            return 2;
        }

        var updateDirectory = Directory.GetParent(Path.GetFullPath(runningPath));
        var cacheDirectory = updateDirectory?.Parent;
        var rootDirectory = cacheDirectory?.Parent;
        if (rootDirectory is null)
        {
            return 3;
        }

        var root = rootDirectory.FullName;
        if (!PathsMatch(runningPath, PendingPath(root)))
        {
            return 4;
        }

        if (!WaitForProcessExit(previousProcessId, TimeSpan.FromMinutes(1)))
        {
            return 5;
        }

        var installedPath = Path.Combine(root, LauncherFileName);
        var replacementPath = Path.Combine(root, ".AI Toolkit Launcher.replace.exe");
        try
        {
            File.Copy(runningPath, replacementPath, true);
            File.Move(replacementPath, installedPath, true);

            var startInfo = new ProcessStartInfo(installedPath)
            {
                UseShellExecute = false,
                WorkingDirectory = root,
            };
            startInfo.ArgumentList.Add(CleanupArgument);
            startInfo.ArgumentList.Add(Environment.ProcessId.ToString());
            _ = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Could not restart the updated launcher.");
            return 0;
        }
        catch
        {
            TryDelete(replacementPath);
            return 6;
        }
    }

    private static int ParseProcessId(string value)
    {
        return int.TryParse(value, out var processId) && processId > 0 ? processId : -1;
    }

    private static bool WaitForProcessExit(int processId, TimeSpan timeout)
    {
        if (processId <= 0 || processId == Environment.ProcessId)
        {
            return processId != Environment.ProcessId;
        }

        try
        {
            using var process = Process.GetProcessById(processId);
            return process.WaitForExit((int)timeout.TotalMilliseconds);
        }
        catch (ArgumentException)
        {
            return true;
        }
        catch (InvalidOperationException)
        {
            return true;
        }
    }

    private static bool FilesMatch(string leftPath, string rightPath)
    {
        var left = new FileInfo(leftPath);
        var right = new FileInfo(rightPath);
        if (left.Length != right.Length)
        {
            return false;
        }

        using var leftStream = File.OpenRead(leftPath);
        using var rightStream = File.OpenRead(rightPath);
        return SHA256.HashData(leftStream).SequenceEqual(SHA256.HashData(rightStream));
    }

    private static bool PathsMatch(string leftPath, string rightPath)
    {
        return string.Equals(
            Path.GetFullPath(leftPath).TrimEnd(Path.DirectorySeparatorChar),
            Path.GetFullPath(rightPath).TrimEnd(Path.DirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase
        );
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception error) when (
            error is IOException
                or UnauthorizedAccessException
                or NotSupportedException
                or System.Security.SecurityException
        )
        {
            // A matching staged binary is ignored on the next launch.
        }
    }
}
