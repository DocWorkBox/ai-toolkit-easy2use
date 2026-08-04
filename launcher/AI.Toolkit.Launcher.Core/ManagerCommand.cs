namespace AiToolkit.Launcher.Core;

public enum ManagerAction
{
    Detect,
    Check,
    Install,
    Sync,
    Update,
    Launch,
    Doctor,
    Version,
}

public enum ManagerOutputMode
{
    Plain,
    JsonDocument,
    JsonStream,
}

public sealed record ManagerInvocation(
    string FileName,
    IReadOnlyList<string> Arguments,
    string WorkingDirectory,
    IReadOnlyDictionary<string, string> Environment,
    ManagerOutputMode OutputMode
);

public static class ManagerCommand
{
    public static ManagerInvocation Create(
        string repositoryRoot,
        PythonCommand python,
        ManagerAction action,
        bool force = false
    )
    {
        var arguments = new List<string>(python.PrefixArguments) { "-m", "manager" };
        var outputMode = ManagerOutputMode.Plain;

        switch (action)
        {
            case ManagerAction.Detect:
                arguments.AddRange(new[] { "detect", "--json" });
                outputMode = ManagerOutputMode.JsonDocument;
                break;
            case ManagerAction.Check:
                arguments.AddRange(new[] { "check", "--json" });
                outputMode = ManagerOutputMode.JsonDocument;
                break;
            case ManagerAction.Install:
            case ManagerAction.Sync:
            case ManagerAction.Update:
                arguments.Add("--json-stream");
                arguments.Add(action.ToString().ToLowerInvariant());
                if (force)
                {
                    arguments.Add("--force");
                }
                outputMode = ManagerOutputMode.JsonStream;
                break;
            case ManagerAction.Launch:
                arguments.AddRange(new[] { "launch", "--no-browser" });
                break;
            case ManagerAction.Doctor:
                arguments.Add("doctor");
                break;
            case ManagerAction.Version:
                arguments.Add("version");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(action), action, null);
        }

        var environment = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (python.IsPortable)
        {
            environment["AITK_RUNTIME_LAYOUT"] = "portable";
        }

        return new ManagerInvocation(
            python.FileName,
            arguments,
            repositoryRoot,
            environment,
            outputMode
        );
    }
}
