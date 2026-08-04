using System.Text.Json;

namespace AiToolkit.Launcher.Core;

public sealed record ManagerEvent(
    string Type,
    string? Level,
    string Message,
    string Raw,
    string? Command = null,
    bool? Ok = null,
    int? ExitCode = null
);

public static class ManagerEventParser
{
    public static ManagerEvent Parse(string line)
    {
        try
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("type", out var type))
            {
                return Log(line);
            }

            var eventType = type.GetString() ?? "log";
            var level = GetString(root, "level");
            var command = GetString(root, "command");
            var message = GetString(root, "message") ?? ResultMessage(root, command);
            bool? ok = root.TryGetProperty("ok", out var okElement)
                && okElement.ValueKind is JsonValueKind.True or JsonValueKind.False
                    ? okElement.GetBoolean()
                    : null;
            int? exitCode = root.TryGetProperty("exit_code", out var exitElement)
                && exitElement.TryGetInt32(out var parsedExit)
                    ? parsedExit
                    : null;
            return new ManagerEvent(eventType, level, message, line, command, ok, exitCode);
        }
        catch (JsonException)
        {
            return Log(line);
        }
    }

    private static ManagerEvent Log(string line)
    {
        return new ManagerEvent("log", null, line, line);
    }

    private static string? GetString(JsonElement root, string name)
    {
        return root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
    }

    private static string ResultMessage(JsonElement root, string? command)
    {
        if (root.TryGetProperty("ok", out var okElement) && okElement.ValueKind == JsonValueKind.True)
        {
            return string.IsNullOrWhiteSpace(command) ? "Completed" : $"{command} completed";
        }

        return string.IsNullOrWhiteSpace(command) ? "Operation failed" : $"{command} failed";
    }
}

public sealed record ManagerRunResult(
    int ExitCode,
    string StandardOutput,
    string StandardError,
    bool WasStopped
)
{
    public bool Successful => ExitCode == 0 && !WasStopped;
}
