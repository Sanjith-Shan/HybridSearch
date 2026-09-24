using HybridSearch.Broker.Shards;

namespace HybridSearch.Broker.Tests;

public class PartitioningTests
{
    private static ShardTopology Topology(string partitioning, params int[] sliceIds) =>
        new(new TopologyOptions
        {
            Partitioning = partitioning,
            // Endpoints are never dialled here: SliceForDoc is pure routing.
            Slices = sliceIds.Select(id => new SliceOptions { Id = id, Replicas = [$"http://127.0.0.1:{59000 + id}"] }).ToList(),
        }, new HedgingOptions());

    [Theory]
    [InlineData(0UL, 0)]
    [InlineData(7067032UL, 0)]
    [InlineData(7067033UL, 1)]
    [InlineData(8841822UL, 2)]
    [InlineData(3UL, 3)]
    public void ModuloRoutesByDocIdModSliceCount(ulong docId, int expectedSlice)
    {
        using var t = Topology("modulo", 0, 1, 2, 3);
        Assert.Equal(expectedSlice, t.SliceForDoc(docId)!.Id);
    }

    [Fact]
    public void ModuloLooksUpSlicesByIdNotListPosition()
    {
        using var t = Topology("Modulo", 3, 1, 0, 2);
        for (ulong d = 0; d < 64; d++) Assert.Equal((int)(d % 4), t.SliceForDoc(d)!.Id);
    }

    [Fact]
    public void RangeIsTheDefaultAndUnknownBeforeHealthChecks()
    {
        // Range routing needs each slice's reported [first, last]; before any health
        // probe there is no range, so routing is unknown rather than guessed.
        using var t = Topology("range", 0, 1);
        Assert.Null(t.SliceForDoc(5));
        Assert.Equal("range", new TopologyOptions().Partitioning);
    }
}
