```

BenchmarkDotNet v0.15.8, macOS Tahoe 26.5.1 (25F80) [Darwin 25.5.0]
Apple M3 Pro, 1 CPU, 12 logical and 12 physical cores
.NET SDK 10.0.401
  [Host]   : .NET 10.0.12 (10.0.12, 10.0.1226.42308), Arm64 RyuJIT armv8.0-a
  ShortRun : .NET 10.0.12 (10.0.12, 10.0.1226.42308), Arm64 RyuJIT armv8.0-a

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
| Method    | K   | Mean     | Error       | StdDev    | Median   | Gen0   | Allocated |
|---------- |---- |---------:|------------:|----------:|---------:|-------:|----------:|
| **TeamDraft** | **10**  | **17.02 μs** |   **273.88 μs** |  **15.01 μs** | **21.27 μs** |      **-** |     **912 B** |
| **TeamDraft** | **100** | **81.91 μs** | **2,320.01 μs** | **127.17 μs** | **10.96 μs** | **0.9766** |    **8968 B** |
