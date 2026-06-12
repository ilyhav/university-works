# Заключительный проект: потоковая аналитика IoT на Apache Flink (PyFlink)

Потоковый пайплайн на **Apache Flink 2.x (PyFlink)**, который в режиме **event time**
считает поминутную статистику по событиям IoT-устройств, обогащая их статичным
справочником типов из **PostgreSQL**, и публикует результат обратно в **Kafka**.

```
 ┌─────────────────┐   1 сообщение/сек    ┌──────────────────────────────────────────────┐
 │ Генератор IoT   │ ───────────────────▶ │ Kafka topic: iot_events                      │
 │ (Python)        │   JSON               │  {device_type_id, event_time, temperature,   │
 └─────────────────┘                      │   humidity}                                  │
                                          └───────────────────┬──────────────────────────┘
                                                              │  DataStream API
                                                              ▼
                                              from_data_stream  (DataStream ─▶ Table)
                                                              │  event time + watermark
                                                              ▼
 ┌─────────────────┐  ddl.sql / dml.sql     ┌──────────────────────────────────────────────┐
 │ PostgreSQL      │ ◀───  справочник  ───  │ Окно TUMBLE 1 минута (event time):           │
 │ device_types    │       (JDBC source)    │   AVG(temperature)        — средняя темп-ра   │
 │ (id, type_name) │ ─────────────────────▶ │   PERCENTILE(humidity,0.5)— медиана влаж-ти   │
 └─────────────────┘      lookup join        └───────────────────┬──────────────────────────┘
                          (обогащение типом)                     │  Table/SQL API (INSERT INTO)
                                                                 ▼
                                          ┌──────────────────────────────────────────────┐
                                          │ Kafka topic: iot_minute_stats (json)         │
                                          │  {event_time (hh:mm), device_type,           │
                                          │   avg_temp, median_humidity}                 │
                                          └──────────────────────────────────────────────┘
```

## Соответствие пунктам задания

| Пункт задания | Где реализовано |
|---|---|
| Генератор сообщений IoT (раз в секунду): тип устройства, время события, температура, влажность → в топик Kafka | [`iot_generator/generator.py`](iot_generator/generator.py) |
| DDL/DML для справочника типов IoT-устройств (Id, TypeName) | [`sql/ddl.sql`](sql/ddl.sql), [`sql/dml.sql`](sql/dml.sql) |
| Источник Kafka + источник PostgreSQL | `KafkaSource` (DataStream API) и `CREATE TABLE ... 'connector'='jdbc'` (Table API) в [`src/flink_job.py`](src/flink_job.py) |
| Соединение событий Kafka со статичным справочником из Postgres | `JOIN device_types FOR SYSTEM_TIME AS OF ...` (lookup join) |
| Окно и расчёт в каждой минуте: средняя температура и медиана влажности | `TUMBLE(... INTERVAL '1' MINUTE)` + `AVG` + `PERCENTILE(humidity, 0.5)` |
| Работа в event time | rowtime-атрибут `ts_event` + `WATERMARK` |
| Источник/приёмник на SQL/Table API | JDBC-источник Postgres и Kafka-приёмник (`CREATE TABLE` + `INSERT INTO`) |
| Переход между DataStream и SQL/Table API | `from_data_stream` (DataStream → Table); обратный переход `to_changelog_stream` показан в комментарии |
| Результат в Kafka: время (hh:mm), тип устройства (из pg), средняя температура, медиана влажности | топик `iot_minute_stats`, формат json |

> **Про порядок «join → window».** На схеме задания соединение стоит до окна.
> В основном варианте справочник (статичная размерность) присоединяется **после**
> окна — результат идентичен, но такой порядок устойчив к ограничению Flink
> «rowtime-атрибут нельзя проносить через regular join» и гарантированно даёт
> append-only поток для обычного Kafka-приёмника. Вариант «обогатить поток,
> затем окно» (точно по схеме, через lookup join, сохраняющий rowtime на Flink 2.x)
> приведён закомментированным блоком в конце `src/flink_job.py`.

## Структура репозитория

```
.
├── iot_generator/generator.py   # генератор событий IoT -> Kafka (1 msg/sec)
├── sql/ddl.sql                  # DDL справочника device_types
├── sql/dml.sql                  # DML наполнения справочника
├── src/config.py                # конфигурация (env / .env)
├── src/flink_job.py             # основной потоковый джоб Flink
├── scripts/download_jars.sh     # загрузка jar-коннекторов Flink
├── scripts/consume_output.py    # проверочный консьюмер выходного топика
├── docker-compose.yml           # Kafka (KRaft) + PostgreSQL для локального запуска
└── .env.example                 # пример переменных окружения
```

## Требования

- Python 3.11
- **Java 11+** (нужна PyFlink; проверено на OpenJDK 17)
- Docker (для локальных Kafka и PostgreSQL) — либо свои Kafka и PostgreSQL

## Установка (один раз)

```bash
# 1. Java (macOS, если ещё нет)
brew install openjdk@17

# 2. Зависимости Python (apache-flink, kafka-python, psycopg2-binary)
uv sync
# нет uv? — установить: pip install uv
# либо без uv:  python3.11 -m venv .venv && source .venv/bin/activate && pip install -e .

# 3. Jar-коннекторы Flink в ./jars
bash scripts/download_jars.sh
```

PyFlink-у нужна Java в окружении. В каждом терминале, где запускается генератор
или джоб, задайте (или один раз добавьте в `~/.zshrc`):

```bash
export JAVA_HOME=$(brew --prefix openjdk@17)
export PATH="$JAVA_HOME/bin:$PATH"
```

> Если порт **5432** на хосте занят (уже есть свой PostgreSQL), создайте файл
> `.env` со строкой `PG_PORT=5433` — этот порт подхватят и docker-compose, и Flink.

## Запуск

```bash
# 1. Kafka + PostgreSQL (Postgres сам применит ddl.sql и dml.sql)
docker compose up -d
source .venv/bin/activate
```

Далее — три терминала (в каждом сначала `export JAVA_HOME=...`, `source .venv/bin/activate`):

```bash
# Терминал 1 — генератор событий (1 msg/sec)
python -m iot_generator.generator

# Терминал 2 — потоковый джоб Flink. Web UI: http://localhost:8081
python -m src.flink_job

# Терминал 3 — результат из выходного топика (первые строки через ~1-2 мин)
python -m scripts.consume_output
```

Пример реального вывода консьюмера:

```
12:23  Temperature Sensor   avg_temp=23.84  median_humidity=64.72
12:23  Smart Meter          avg_temp=27.06  median_humidity=45.35
12:23  Thermostat           avg_temp=25.51  median_humidity=57.08
12:24  Weather Station      avg_temp=23.71  median_humidity=47.48
12:24  Humidity Sensor      avg_temp=25.28  median_humidity=60.96
```

## Запуск без Docker (своя инфраструктура)

1. Поднять Kafka и PostgreSQL.
2. Применить SQL-скрипты к своей БД:
   ```bash
   psql -h localhost -U postgres -d iot -f sql/ddl.sql
   psql -h localhost -U postgres -d iot -f sql/dml.sql
   ```
3. При других адресах/кредах — скопировать `.env.example` в `.env` и поправить,
   либо задать переменные окружения (`KAFKA_BOOTSTRAP`, `PG_*`, `JARS_DIR`, ...).
4. Если jar-ы уже есть локально (напр. `~/SD/jars`) — указать `JARS_DIR=~/SD/jars`
   вместо запуска `download_jars.sh`.

## Использованные jar-коннекторы (Apache Flink 2.x)

| Назначение | Файл |
|---|---|
| Kafka SQL-коннектор | `flink-sql-connector-kafka-4.0.1-2.0.jar` |
| JDBC-коннектор (ядро) | `flink-connector-jdbc-core-4.0.0-2.0.jar` |
| JDBC-диалект PostgreSQL | `flink-connector-jdbc-postgres-4.0.0-2.0.jar` |
| Драйвер PostgreSQL | `postgresql-42.7.5.jar` |

`PERCENTILE(x, 0.5)` — встроенная агрегатная функция Flink (медиана), доступная
с Flink 2.0 и предназначенная для оконного сценария.
