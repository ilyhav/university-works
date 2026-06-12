import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflink.common import Configuration, Row, SimpleStringSchema, Types, WatermarkStrategy  # noqa: E402
from pyflink.datastream import StreamExecutionEnvironment  # noqa: E402
from pyflink.datastream.connectors.kafka import KafkaOffsetsInitializer, KafkaSource  # noqa: E402
from pyflink.table import DataTypes, Schema, StreamTableEnvironment  # noqa: E402

from src import config  # noqa: E402


def parse_event(raw: str):
    try:
        d = json.loads(raw)
        return Row(
            int(d["device_type_id"]),
            float(d["temperature"]),
            float(d["humidity"]),
            int(d["event_time"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


def main() -> None:
    conf = Configuration()
    conf.set_integer("rest.port", 8081)
    conf.set_string("execution.runtime-mode", "STREAMING")

    env = StreamExecutionEnvironment.get_execution_environment(conf)
    env.add_jars(*config.ALL_JARS)
    # Топик с 1 партицией: параллелизм 1, иначе простаивающие сабтаски источника
    # держат watermark и оконная агрегация не срабатывает.
    env.set_parallelism(1)

    tenv = StreamTableEnvironment.create(env)
    tenv.get_config().set("table.local-time-zone", config.TABLE_TZ)
    # Не держим watermark, если источник какое-то время простаивает.
    tenv.get_config().set("table.exec.source.idle-timeout", "5 s")

    # --- Kafka source (DataStream API) ---
    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(config.KAFKA_BOOTSTRAP)
        .set_topics(config.IOT_TOPIC)
        .set_value_only_deserializer(SimpleStringSchema())
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .build()
    )

    ds_events = (
        env.from_source(kafka_source, WatermarkStrategy.no_watermarks(), "kafka-iot-src")
        .map(
            parse_event,
            # output_type нужен для корректной сериализации python <-> java
            output_type=Types.ROW([Types.INT(), Types.DOUBLE(), Types.DOUBLE(), Types.LONG()]),
        )
        .filter(lambda r: r is not None)
    )

    # --- DataStream -> Table, объявление event time и watermark ---
    t_events = tenv.from_data_stream(
        ds_events,
        Schema.new_builder()
        .column("f0", DataTypes.INT())
        .column("f1", DataTypes.DOUBLE())
        .column("f2", DataTypes.DOUBLE())
        .column("f3", DataTypes.BIGINT())
        .column_by_expression("ts_event", "TO_TIMESTAMP_LTZ(f3, 3)")
        .watermark("ts_event", "ts_event - INTERVAL '5' SECOND")
        .build(),
    ).alias("device_type_id", "temperature", "humidity", "event_ms", "ts_event")

    tenv.create_temporary_view("events", t_events)

    # --- PostgreSQL source (Table/SQL API, JDBC) ---
    tenv.execute_sql(
        f"""
        CREATE TEMPORARY TABLE device_types (
            id        INT,
            type_name STRING
        ) WITH (
            'connector'  = 'jdbc',
            'url'        = '{config.PG_JDBC_URL}',
            'table-name' = '{config.PG_TABLE}',
            'username'   = '{config.PG_USER}',
            'password'   = '{config.PG_PASSWORD}'
        )
        """
    )

    # --- Окно 1 минута: AVG(температура) и PERCENTILE(влажность, 0.5) = медиана ---
    tenv.create_temporary_view(
        "minute_agg",
        tenv.sql_query(
            """
            SELECT
                window_start,
                device_type_id,
                AVG(temperature)          AS avg_temp,
                PERCENTILE(humidity, 0.5) AS median_humidity
            FROM TABLE(
                TUMBLE(TABLE events, DESCRIPTOR(ts_event), INTERVAL '1' MINUTE)
            )
            GROUP BY window_start, window_end, device_type_id
            """
        ),
    )

    # --- Lookup join со справочником Postgres (PROCTIME нужен для FOR SYSTEM_TIME AS OF) ---
    tenv.create_temporary_view(
        "minute_agg_pt",
        tenv.sql_query("SELECT *, PROCTIME() AS proc_time FROM minute_agg"),
    )

    result = tenv.sql_query(
        """
        SELECT
            DATE_FORMAT(a.window_start, 'HH:mm') AS event_time,
            d.type_name                          AS device_type,
            a.avg_temp                           AS avg_temp,
            a.median_humidity                    AS median_humidity
        FROM minute_agg_pt AS a
        JOIN device_types FOR SYSTEM_TIME AS OF a.proc_time AS d
            ON a.device_type_id = d.id
        """
    )

    # --- Kafka sink (Table/SQL API) ---
    tenv.execute_sql(
        f"""
        CREATE TEMPORARY TABLE kafka_sink (
            event_time      STRING,
            device_type     STRING,
            avg_temp        DOUBLE,
            median_humidity DOUBLE
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{config.OUTPUT_TOPIC}',
            'properties.bootstrap.servers' = '{config.KAFKA_BOOTSTRAP}',
            'format' = 'json'
        )
        """
    )

    print("[flink] джоба запущена. Web UI: http://localhost:8081")
    print(f"[flink] читаем '{config.IOT_TOPIC}', пишем результат в '{config.OUTPUT_TOPIC}'")
    result.execute_insert("kafka_sink").wait()


if __name__ == "__main__":
    main()
