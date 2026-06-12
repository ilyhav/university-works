import os


def _load_dotenv() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
IOT_TOPIC = os.getenv("IOT_TOPIC", "iot_events")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "iot_minute_stats")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "iot")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_TABLE = os.getenv("PG_TABLE", "device_types")
PG_JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"

DEVICE_TYPE_IDS = [1, 2, 3, 4, 5]

TABLE_TZ = os.getenv("TABLE_TZ", "Europe/Moscow")

JARS_DIR = os.path.abspath(
    os.getenv("JARS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "jars"))
)


def _jar(name: str) -> str:
    return "file://" + os.path.join(JARS_DIR, name)


KAFKA_CONNECTOR = _jar(os.getenv("KAFKA_JAR", "flink-sql-connector-kafka-4.0.1-2.0.jar"))
JDBC_CONNECTOR_CORE = _jar(os.getenv("JDBC_CORE_JAR", "flink-connector-jdbc-core-4.0.0-2.0.jar"))
JDBC_CONNECTOR_PG = _jar(os.getenv("JDBC_PG_JAR", "flink-connector-jdbc-postgres-4.0.0-2.0.jar"))
PG_DRIVER = _jar(os.getenv("PG_DRIVER_JAR", "postgresql-42.7.5.jar"))

ALL_JARS = [KAFKA_CONNECTOR, JDBC_CONNECTOR_CORE, JDBC_CONNECTOR_PG, PG_DRIVER]
