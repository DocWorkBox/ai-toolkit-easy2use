namespace AiToolkit.Launcher.Core;

public static class RepositoryLocator
{
    public static string FindRoot(string startPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(startPath);

        var fullPath = Path.GetFullPath(startPath);
        var directory = File.Exists(fullPath)
            ? new DirectoryInfo(Path.GetDirectoryName(fullPath)!)
            : new DirectoryInfo(fullPath);

        while (directory is not null)
        {
            if (IsRepositoryRoot(directory.FullName))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException(
            $"Could not find an AI Toolkit root above '{fullPath}'."
        );
    }

    public static bool IsRepositoryRoot(string path)
    {
        return File.Exists(Path.Combine(path, "manager", "__main__.py"))
            && File.Exists(Path.Combine(path, "ui", "package.json"))
            && File.Exists(Path.Combine(path, "version.py"));
    }
}
