#!/usr/bin/env bash
set -euo pipefail

JARS_DIR="${JARS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/jars}"
mkdir -p "$JARS_DIR"

BASE="https://repo1.maven.org/maven2"

download() {
    local url="$1"
    local name
    name="$(basename "$url")"
    if [[ -f "$JARS_DIR/$name" ]]; then
        echo "= $name (уже есть)"
    else
        echo "+ $name"
        curl -fSL -o "$JARS_DIR/$name" "$url"
    fi
}

download "$BASE/org/apache/flink/flink-sql-connector-kafka/4.0.1-2.0/flink-sql-connector-kafka-4.0.1-2.0.jar"
download "$BASE/org/apache/flink/flink-connector-jdbc-core/4.0.0-2.0/flink-connector-jdbc-core-4.0.0-2.0.jar"
download "$BASE/org/apache/flink/flink-connector-jdbc-postgres/4.0.0-2.0/flink-connector-jdbc-postgres-4.0.0-2.0.jar"
download "$BASE/org/postgresql/postgresql/42.7.5/postgresql-42.7.5.jar"

echo
ls -1 "$JARS_DIR"
