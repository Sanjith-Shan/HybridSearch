```

BenchmarkDotNet v0.15.8, macOS Tahoe 26.5.1 (25F80) [Darwin 25.5.0]
Apple M3 Pro, 1 CPU, 12 logical and 12 physical cores
.NET SDK 10.0.401
  [Host]   : .NET 10.0.12 (10.0.12, 10.0.1226.42308), Arm64 RyuJIT armv8.0-a
  ShortRun : .NET 10.0.12 (10.0.12, 10.0.1226.42308), Arm64 RyuJIT armv8.0-a

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
| Method                | Mean     | Error     | StdDev   | Gen0   | Allocated |
|---------------------- |---------:|----------:|---------:|-------:|----------:|
| Lookup_k8             | 22.42 μs | 137.87 μs | 7.557 μs | 0.0500 |     557 B |
| NormalizeAndLookup_k8 | 23.31 μs | 127.93 μs | 7.012 μs | 0.0667 |     734 B |
