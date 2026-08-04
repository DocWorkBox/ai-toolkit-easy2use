using System.Text.Json;

namespace AiToolkit.Launcher.Core;

public sealed record HardwareStatus(
    string OperatingSystem,
    string Architecture,
    string PrimaryGpuName,
    string PrimaryGpuMemory,
    string DriverVersion,
    string CudaVersion,
    string Backend
);

public sealed record UpdateStatus(
    string Version,
    string Branch,
    bool Dirty,
    int? Behind,
    bool EnvironmentExists,
    bool DependenciesInSync,
    bool UpdateAvailable,
    string Backend
);

public sealed record EnvironmentCheckStatus(
    string Key,
    string Label,
    bool Passed,
    bool Required,
    bool Repairable,
    string Detail
);

public sealed record EnvironmentHealth(
    bool MeetsRequirements,
    bool EnvironmentExists,
    int RequiredPassed,
    int RequiredTotal,
    IReadOnlyList<string> FailedRequired,
    IReadOnlyList<string> RepairableFailures,
    IReadOnlyList<string> Warnings,
    IReadOnlyList<EnvironmentCheckStatus> Checks
);

public static class ToolkitStatusParser
{
    public static HardwareStatus ParseDetection(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        var nvidia = ObjectOrDefault(root, "nvidia");
        var gpu = FirstArrayItem(nvidia, "gpus");
        var spec = ObjectOrDefault(root, "spec");
        return new HardwareStatus(
            String(root, "os"),
            String(root, "arch"),
            String(gpu, "name", "未检测到 NVIDIA GPU"),
            String(gpu, "memory", "未知"),
            String(nvidia, "driver", "未知"),
            String(nvidia, "cuda_version", "未知"),
            String(spec, "backend", "未知")
        );
    }

    public static UpdateStatus ParseUpdateCheck(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        return new UpdateStatus(
            String(root, "version", "未知"),
            String(root, "branch", "未知"),
            Boolean(root, "dirty"),
            NullableInteger(root, "behind"),
            Boolean(root, "venv"),
            Boolean(root, "deps_in_sync"),
            Boolean(root, "update_available"),
            String(root, "backend", "未知")
        );
    }

    public static EnvironmentHealth ParseEnvironmentHealth(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        var checks = new List<EnvironmentCheckStatus>();
        if (root.TryGetProperty("checks", out var checkArray)
            && checkArray.ValueKind == JsonValueKind.Array)
        {
            foreach (var check in checkArray.EnumerateArray())
            {
                checks.Add(
                    new EnvironmentCheckStatus(
                        String(check, "key", "unknown"),
                        String(check, "label", "unknown"),
                        Boolean(check, "passed"),
                        Boolean(check, "required"),
                        Boolean(check, "repairable"),
                        String(check, "detail")
                    )
                );
            }
        }

        return new EnvironmentHealth(
            Boolean(root, "ok"),
            Boolean(root, "environment_exists"),
            Integer(root, "required_passed"),
            Integer(root, "required_total"),
            StringArray(root, "failed_required"),
            StringArray(root, "repairable_failures"),
            StringArray(root, "warnings"),
            checks
        );
    }

    private static JsonElement ObjectOrDefault(JsonElement parent, string name)
    {
        return parent.ValueKind == JsonValueKind.Object
            && parent.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.Object
                ? value
                : default;
    }

    private static JsonElement FirstArrayItem(JsonElement parent, string name)
    {
        if (parent.ValueKind == JsonValueKind.Object
            && parent.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in value.EnumerateArray())
            {
                return item;
            }
        }
        return default;
    }

    private static string String(JsonElement parent, string name, string fallback = "")
    {
        return parent.ValueKind == JsonValueKind.Object
            && parent.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.String
                ? value.GetString() ?? fallback
                : fallback;
    }

    private static bool Boolean(JsonElement parent, string name)
    {
        return parent.TryGetProperty(name, out var value)
            && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            && value.GetBoolean();
    }

    private static int? NullableInteger(JsonElement parent, string name)
    {
        return parent.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.Number
            && value.TryGetInt32(out var result)
            ? result
            : null;
    }

    private static int Integer(JsonElement parent, string name)
    {
        return parent.TryGetProperty(name, out var value) && value.TryGetInt32(out var result)
            ? result
            : 0;
    }

    private static IReadOnlyList<string> StringArray(JsonElement parent, string name)
    {
        if (!parent.TryGetProperty(name, out var value)
            || value.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<string>();
        }

        return value.EnumerateArray()
            .Where(item => item.ValueKind == JsonValueKind.String)
            .Select(item => item.GetString() ?? string.Empty)
            .Where(item => item.Length > 0)
            .ToArray();
    }
}
