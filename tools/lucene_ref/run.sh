#!/usr/bin/env bash
# Run the Lucene reference oracle. Downloads Lucene jars (the version Anserini master pins,
# see anserini/pom.xml <lucene.version>) into data/cache/lucene on first use.
# Usage: tools/lucene_ref/run.sh <mode> [args...]   (see LuceneRef.java)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
ver="${LUCENE_VERSION:-10.5.0}"
cache="$here/data/cache/lucene"
mkdir -p "$cache/classes-$ver"
for a in lucene-core lucene-analysis-common; do
  [ -f "$cache/$a-$ver.jar" ] || curl -sfL -o "$cache/$a-$ver.jar" \
    "https://repo1.maven.org/maven2/org/apache/lucene/$a/$ver/$a-$ver.jar"
done
java_home="${JAVA_HOME_21:-/opt/homebrew/opt/openjdk@21}"
cp="$cache/lucene-core-$ver.jar:$cache/lucene-analysis-common-$ver.jar"
src="$here/tools/lucene_ref/LuceneRef.java"
if [ ! -f "$cache/classes-$ver/LuceneRef.class" ] || [ "$src" -nt "$cache/classes-$ver/LuceneRef.class" ]; then
  "$java_home/bin/javac" -nowarn -cp "$cp" -d "$cache/classes-$ver" "$src"
fi
exec "$java_home/bin/java" -Xmx4g --add-modules jdk.incubator.vector -cp "$cp:$cache/classes-$ver" LuceneRef "$@" 2> >(grep -v "WARNING\|incubator" >&2)
