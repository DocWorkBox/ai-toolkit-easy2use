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
        return parent.TryGetProperty(name, out var value) && value.TryGetInt32(out var result)
            ? result
            : null;
    }
}
