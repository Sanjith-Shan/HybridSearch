# syntax=docker/dockerfile:1.7
# Multi-stage build of the broker. The build context is broker/ (small); the shared proto
# directory is passed as a named context so the multi-GB data/ directory at the repo root is
# never sent to the daemon. From the repo root:
#
#   docker build -f broker/Dockerfile --build-context proto=proto -t hybridsearch-broker broker
#   docker run --rm -p 8080:8080 -v "$PWD/data:/data:ro" -v "$PWD/data/events:/data/events" \
#     -e Broker__Topology__Slices__0__Id=0 -e Broker__Topology__Slices__0__Replicas__0=http://host.docker.internal:50051 \
#     -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 hybridsearch-broker

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src/broker
COPY --from=proto . /src/proto/
COPY Directory.Build.props HybridSearch.sln ./
COPY HybridSearch.Contracts/HybridSearch.Contracts.csproj HybridSearch.Contracts/
COPY HybridSearch.Broker/HybridSearch.Broker.csproj HybridSearch.Broker/
RUN dotnet restore HybridSearch.Broker/HybridSearch.Broker.csproj
COPY HybridSearch.Contracts/ HybridSearch.Contracts/
COPY HybridSearch.Broker/ HybridSearch.Broker/
RUN dotnet publish HybridSearch.Broker/HybridSearch.Broker.csproj -c Release -o /app --no-restore /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app
COPY --from=build /app ./
ENV ASPNETCORE_URLS=http://+:8080 \
    Broker__DataRoot=/data \
    DOTNET_gcServer=1
# Writable event-log directory for the non-root user (a mounted volume must also be writable by UID 1654).
RUN mkdir -p /data/events && chown -R app:app /data
EXPOSE 8080
# The aspnet image ships a non-root "app" user (UID 1654).
USER app
# Health: GET /healthz (liveness) and /readyz (readiness) — probe them from compose / the orchestrator.
ENTRYPOINT ["dotnet", "HybridSearch.Broker.dll"]
