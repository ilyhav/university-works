"""Тонкая обёртка над GigaChat SDK с безопасной деградацией в офлайн.

Если пакет gigachat не установлен, ключ не задан или сеть недоступна — клиент
сообщает `available == False`, и вызывающий код переходит на детерминированный
шаблон. Демо и тесты остаются полностью работоспособными без ключа.
"""

from __future__ import annotations

from ..config import CONFIG, GigaChatConfig


class LLMUnavailable(RuntimeError):
    """LLM-слой недоступен (нет ключа/пакета/сети) — нужно использовать fallback."""


class GigaChatClient:
    def __init__(self, cfg: GigaChatConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.gigachat
        self._client = None  # ленивая инициализация SDK

    @property
    def available(self) -> bool:
        return self.cfg.enabled

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.cfg.enabled:
            raise LLMUnavailable("GigaChat отключён или не задан ключ (GIGACHAT_CREDENTIALS).")
        try:
            from gigachat import GigaChat  # импорт здесь, чтобы офлайн не падал
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("Пакет gigachat не установлен.") from e
        try:
            self._client = GigaChat(
                credentials=self.cfg.credentials,
                scope=self.cfg.scope,
                model=self.cfg.model,
                verify_ssl_certs=self.cfg.verify_ssl,
            )
        except Exception as e:  # pragma: no cover - сетевые/конфиг-ошибки
            raise LLMUnavailable(f"Не удалось инициализировать GigaChat: {e}") from e
        return self._client

    def chat(self, system: str, user: str) -> str:
        """Один запрос system+user → текст ответа. Бросает LLMUnavailable при сбое."""
        client = self._ensure_client()
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
        }
        try:
            resp = client.chat(payload)
            return resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover - сетевые ошибки/таймауты
            raise LLMUnavailable(f"Ошибка вызова GigaChat: {e}") from e
