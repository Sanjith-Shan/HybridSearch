using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace HybridSearch.Broker.Bench;

/// <summary>Writes the .meta.json sidecar every timing result needs (docs/ARCHITECTURE.md, Hardware labels).</summary>
public static class Meta
{
    public static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "proto"))) dir = dir.Parent;
            return dir?.FullName ?? throw new InvalidOperationException("repo root not found");
        }
    }

    public static string ResultsDir
    {
        get
        {
            var d = Path.Combine(RepoRoot, "results", "broker");
            Directory.CreateDirectory(d);
            return d;
        }
    }

    public static string Run(string file, params string[] args)
    {
        try
        {
            var psi = new ProcessStartInfo(file) { RedirectStandardOutput = true, RedirectStandardError = true };
            foreach (var a in args) psi.ArgumentList.Add(a);
            using var p = Process.Start(psi)!;
            var s = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit();
            return p.ExitCode == 0 ? s : "unknown";
        }
        catch (Exception) { return "unknown"; }
    }

    public static Dictionary<string, object?> Collect(string command, string loadAverageAtStart, Dictionary<string, object?>? extra = null)
    {
        var m = new Dictionary<string, object?>
        {
            ["label"] = OperatingSystem.IsMacOS() ? "dev-signal-only" : "unpinned",
            ["cpu"] = OperatingSystem.IsMacOS() ? Run("sysctl", "-n", "machdep.cpu.brand_string") : Run("sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
            ["logicalCores"] = Environment.ProcessorCount,
            ["os"] = RuntimeInformation.OSDescription,
            ["dotnet"] = RuntimeInformation.FrameworkDescription,
            ["arch"] = RuntimeInformation.ProcessArchitecture.ToString(),
            ["threadsPinned"] = false,
            ["loadAverageAtStart"] = loadAverageAtStart,
            ["loadAverageAtEnd"] = LoadAverage(),
            ["dateUtc"] = DateTimeOffset.UtcNow.ToString("O"),
            ["gitSha"] = Run("git", "-C", RepoRoot, "rev-parse", "HEAD"),
            ["command"] = command,
        };
        foreach (var (k, v) in extra ?? []) m[k] = v;
        return m;
    }

    public static string LoadAverage() => OperatingSystem.IsWindows() ? "n/a" : Run("sh", "-c", "uptime | sed 's/.*load average[s]*: //'");

    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public static void Write(string name, object result, Dictionary<string, object?> meta)
    {
        var path = Path.Combine(ResultsDir, name + ".json");
        File.WriteAllText(path, JsonSerializer.Serialize(result, Json));
        File.WriteAllText(Path.Combine(ResultsDir, name + ".meta.json"), JsonSerializer.Serialize(meta, Json));
        Console.WriteLine($"wrote {path} (+ .meta.json)");
    }
}
