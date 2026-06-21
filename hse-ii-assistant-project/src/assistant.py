"""Фасад ассистента: ядро → объяснение (LLM/шаблон) → журналирование.

Единая точка входа для веб-формы и демо-скриптов. Держит загруженный каталог и
LLM-клиент, чтобы не переинициализировать их на каждый запрос.
"""

from __future__ import annotations

from datetime import date

from .core.models import AssistantResponse, ClientRequest, Deposit
from .core.recommender import recommend
from .data_access import load_deposits
from .llm.explainer import explain
from .llm.gigachat_client import GigaChatClient
from .monitoring.logging_store import log_response


class Assistant:
    def __init__(
        self,
        deposits: list[Deposit] | None = None,
        client: GigaChatClient | None = None,
    ) -> None:
        self.deposits = deposits if deposits is not None else load_deposits()
        self.client = client if client is not None else GigaChatClient()

    @property
    def llm_available(self) -> bool:
        return self.client.available

    def ask(
        self,
        request: ClientRequest,
        today: date | None = None,
        use_llm: bool = True,
        log: bool = True,
    ) -> AssistantResponse:
        resp = recommend(request, self.deposits, today=today)
        explain(resp, self.client if use_llm else None)
        if log:
            log_response(resp)
        return resp
