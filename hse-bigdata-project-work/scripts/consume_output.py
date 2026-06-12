import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka import KafkaConsumer  # noqa: E402

from src import config  # noqa: E402


def main() -> None:
    consumer = KafkaConsumer(
        config.OUTPUT_TOPIC,
        bootstrap_servers=[config.KAFKA_BOOTSTRAP],
        auto_offset_reset="earliest",
        value_deserializer=lambda b: b.decode("utf-8"),
    )
    print(f"[consumer] слушаем топик '{config.OUTPUT_TOPIC}' (Ctrl+C для выхода)\n")
    for msg in consumer:
        try:
            rec = json.loads(msg.value)
            print(
                f"{rec['event_time']}  {rec['device_type']:<20} "
                f"avg_temp={rec['avg_temp']:.2f}  median_humidity={rec['median_humidity']:.2f}"
            )
        except (ValueError, KeyError):
            print(msg.value)


if __name__ == "__main__":
    main()
