import json
import os
import random
import signal
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka import KafkaProducer  # noqa: E402

from src import config  # noqa: E402


def build_event() -> dict:
    return {
        "device_type_id": random.choice(config.DEVICE_TYPE_IDS),
        "event_time": int(time.time() * 1000),
        "temperature": round(random.uniform(15.0, 35.0), 2),
        "humidity": round(random.uniform(30.0, 90.0), 2),
    }


def main() -> None:
    producer = KafkaProducer(
        bootstrap_servers=[config.KAFKA_BOOTSTRAP],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(
        f"[generator] публикуем в топик '{config.IOT_TOPIC}' "
        f"на {config.KAFKA_BOOTSTRAP} (Ctrl+C для остановки)"
    )

    running = {"ok": True}

    def _stop(*_):
        running["ok"] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while running["ok"]:
            event = build_event()
            producer.send(config.IOT_TOPIC, value=event)
            ts = datetime.fromtimestamp(event["event_time"] / 1000).strftime("%H:%M:%S")
            print(f"[{ts}] -> {event}")
            time.sleep(1)
    finally:
        producer.flush()
        producer.close()
        print("\n[generator] остановлен")


if __name__ == "__main__":
    main()
