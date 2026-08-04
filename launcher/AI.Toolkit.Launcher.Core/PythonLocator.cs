namespace AiToolkit.Launcher.Core;

public sealed record PythonCommand(
    string FileName,
    IReadOnlyList<string> PrefixArguments,
    bool IsPortable
);

public static class PythonLocator
{
    public static PythonCommand Resolve(string repositoryRoot)
    {
        var portable = Path.Combine(repositoryRoot, "runtime", "python", "python.exe");
        if (File.Exists(portable))
        {
            return new PythonCommand(portable, Array.Empty<string>(), true);
        }

        foreach (var relative in new[]
                 {
                     Path.Combine(".venv", "Scripts", "python.exe"),
                     Path.Combine("venv", "Scripts", "python.exe"),
                 })
        {
            var candidate = Path.Combine(repositoryRoot, relative);
            if (File.Exists(candidate))
            {
                return new PythonCommand(candidate, Array.Empty<string>(), false);
            }
        }

        var py = FindOnPath("py.exe");
        if (py is not null)
        {
            return new PythonCommand(py, new[] { "-3" }, false);
        }

        var python = FindOnPath("python.exe");
        if (python is not null)
        {
            return new PythonCommand(python, Array.Empty<string>(), false);
        }

        throw new FileNotFoundException(
            "No usable Python was found in runtime/python, .venv, venv, or PATH."
        );
    }

    private static string? FindOnPath(string executable)
    {
        var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (var directory in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                var candidate = Path.Combine(directory.Trim('"'), executable);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
            catch (ArgumentException)
            {
                // Ignore malformed PATH entries and continue searching.
            }
        }

        return null;
    }
}
