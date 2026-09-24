```

BenchmarkDotNet v0.15.8, macOS Tahoe 26.5.1 (25F80) [Darwin 25.5.0]
Apple M3 Pro, 1 CPU, 12 logical and 12 physical cores
.NET SDK 10.0.401
  [Host]   : .NET 10.0.12 (10.0.12, 10.0.1226.42308), Arm64 RyuJIT armv8.0-a
  ShortRun : .NET 10.0.12 (10.0.12, 10.0.1226.42308), Arm64 RyuJIT armv8.0-a

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
| Method         | Depth | Mean        | Error         | StdDev       | Gen0    | Gen1   | Allocated |
|--------------- |------ |------------:|--------------:|-------------:|--------:|-------:|----------:|
| **Rrf**            | **100**   |    **72.99 μs** |     **435.72 μs** |    **23.883 μs** |  **2.3193** |      **-** |  **19.06 KB** |
| WeightedMinMax | 100   |    85.67 μs |     501.20 μs |    27.472 μs |  3.4180 |      - |  28.16 KB |
| WeightedZScore | 100   |   126.27 μs |     138.28 μs |     7.580 μs |  3.4180 |      - |  28.16 KB |
| **Rrf**            | **1000**  |   **814.11 μs** |   **4,161.67 μs** |   **228.115 μs** | **22.4609** | **3.9063** |  **190.3 KB** |
| WeightedMinMax | 1000  |   889.57 μs |   9,351.74 μs |   512.600 μs | 33.8135 | 8.4229 | 277.84 KB |
| WeightedZScore | 1000  | 7,524.07 μs | 133,076.24 μs | 7,294.358 μs | 31.2500 |      - | 277.97 KB |
