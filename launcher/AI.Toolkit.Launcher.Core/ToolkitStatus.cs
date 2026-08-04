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

public sealed record ModelScanSummary(
    int Ready,
    int Issues,
    int Missing,
    int Unrecognized,
    int Total
);

public sealed record ModelStatusItem(
    string Id,
    string Name,
    string Category,
    string Status,
    string Path,
    string AbsolutePath,
    string Detail,
    string DownloadUrl,
    bool Special,
    string Family = ""
)
{
    public string FamilyCategoryLabel => string.IsNullOrWhiteSpace(Family)
        ? Category
        : $"{Family} · {Category}";

    public string StatusLabel => Status switch
    {
        "ready" => "可用",
        "missing" => "缺失",
        "incomplete" => "文件不完整",
        "misplaced" => "位置错误",
        "name_mismatch" => "名称不匹配",
        "unrecognized" => "未识别",
        _ => "未知",
    };

    public bool CanDownload =>
        Status is "missing" or "incomplete" or "misplaced" or "name_mismatch"
        && Uri.TryCreate(DownloadUrl, UriKind.Absolute, out var uri)
        && uri.Scheme == Uri.UriSchemeHttps;
}

public sealed record ModelScanReport(
    string ModelsRoot,
    string CatalogPath,
    ModelScanSummary Summary,
    IReadOnlyList<ModelStatusItem> Models,
    string ConfiguredModelsRoot = ""
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

    public static ModelScanReport ParseModelScan(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        var summary = ObjectOrDefault(root, "summary");
        var models = new List<ModelStatusItem>();
        if (root.TryGetProperty("models", out var modelArray)
            && modelArray.ValueKind == JsonValueKind.Array)
        {
            foreach (var model in modelArray.EnumerateArray())
            {
                models.Add(
                    new ModelStatusItem(
                        String(model, "id"),
                        String(model, "name"),
                        String(model, "category", "模型"),
                        String(model, "status", "unknown"),
                        String(model, "path"),
                        String(model, "absolute_path"),
                        String(model, "detail"),
                        String(model, "download_url"),
                        Boolean(model, "special"),
                        String(model, "family")
                    )
                );
            }
        }

        return new ModelScanReport(
            String(root, "models_root"),
            String(root, "catalog_path"),
            new ModelScanSummary(
                Integer(summary, "ready"),
                Integer(summary, "issues"),
                Integer(summary, "missing"),
                Integer(summary, "unrecognized"),
                Integer(summary, "total")
            ),
            models,
            String(root, "configured_models_root", String(root, "models_root"))
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
