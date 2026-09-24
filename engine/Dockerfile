# Shard server image. Build context is the repo root (it needs proto/ too):
#   docker build -f engine/Dockerfile -t hybridsearch/shard .
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build g++-14 ca-certificates \
      libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY proto/ proto/
COPY engine/ engine/
# HS_NATIVE=OFF: the image must run on any x86-64/arm64 host, not just the build machine.
RUN CXX=g++-14 cmake -S engine -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DHS_NATIVE=OFF -DHS_BUILD_TESTS=OFF -DHS_BUILD_BENCH=OFF \
    && cmake --build /build --target hs_shard hs_lex_search hs_index_build

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgrpc++1.51t64 libprotobuf32t64 libzstd1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 shard
COPY --from=build /build/hs_shard /build/hs_lex_search /build/hs_index_build /usr/local/bin/
USER shard
EXPOSE 50051
ENTRYPOINT ["hs_shard"]
