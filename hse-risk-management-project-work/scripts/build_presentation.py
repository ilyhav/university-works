"""
Готовая презентация защиты — самодостаточный HTML-слайд-дек.

Запуск (после build_* и build_slides):
    python -m scripts.build_presentation
    -> outputs/slides/presentation.html

Открой двойным кликом в браузере, нажми F — полный экран. Навигация: стрелки
← → (или пробел), Home/End. Клавиша S — заметки докладчику. Все графики вшиты
в base64: файл открывается без интернета и пересылается одним файлом.

Дизайн: тёмный «финансовый research» — сериф Fraunces для заголовков, IBM Plex
для данных, акцент-золото, крупные числа. Числа берутся из outputs/* (синхронны
с FINAL_REPORT.md).
"""
from __future__ import annotations

import base64
import logging

from rm.config import OUTPUT_DIR
from scripts.build_slides import collect_kpis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_presentation")

O = OUTPUT_DIR


def data_uri(rel: str) -> str | None:
    p = O / rel
    if not p.exists():
        logger.warning("нет картинки %s — слайд без неё", rel)
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def esc_attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def slides_spec(k: dict) -> list[dict]:
    money = lambda x: f"{x/1e6:.2f}".rstrip("0").rstrip(".") + " млн"
    return [
        {
            "layout": "title",
            "kicker": "ФКН ВШЭ · Управление рисками · Проектная работа №2",
            "title": "Оценка рыночного риска портфеля",
            "lead": "VaR&nbsp;99% и ES&nbsp;97.5% · горизонты 1 и 10 дней · дата оценки 02.12.2025",
            "meta": "Портфель 260 млн ₽ — 5 ОФЗ · 10 акций · USD/EUR&nbsp;&nbsp;|&nbsp;&nbsp;"
                    "Данные: MOEX&nbsp;ISS + Банк России, 2021–2026",
            "team": "Чебаев Максим · Сиюхов Эльдар · Веселов Илья · "
                    "Аристархов Данила · Есипёнок Павел · Мадоян Александр",
            "notes": "Здравствуйте! Мы посчитали, сколько может потерять наш портфель на 260 миллионов — пять ОФЗ, десять акций и валюта. Меры две: VaR 99% и ES 97.5%, на день и на десять дней. И сразу важное: весь путь от сырых данных до бэктеста воспроизводим.",
        },
        {
            "layout": "split",
            "kicker": "Постановка задачи",
            "title": "Что именно мы измеряем",
            "lead": "Сколько портфель может потерять — и сколько теряет в среднем в худших сценариях.",
            "points": [
                "<b>VaR&nbsp;99%</b> — порог потерь, превышаемый лишь в 1% случаев.",
                "<b>ES&nbsp;97.5%</b> — средние потери в худшем хвосте (когерентная мера).",
                "Два горизонта: 1 день и 10 дней.",
                "Полный цикл: данные → факторы → модели → стоимость → риск → бэктест.",
            ],
            "imgs": [("slides/var_es_concept.png", "")],
            "notes": "Что такое VaR? Это порог: с вероятностью 99% за день мы не потеряем больше него. А ES заглядывает дальше в хвост и усредняет потери за этим порогом — поэтому он честнее VaR как мера риска. Считаем оба, на день и на десять дней.",
        },
        {
            "layout": "full",
            "kicker": "Карта проекта",
            "title": "Шесть этапов: от данных до проверки",
            "lead": "Каждый этап — отдельный пакет кода, результат предыдущего питает следующий.",
            "imgs": [("slides/pipeline.png", "Воспроизводимость: фиксированный seed, parquet-кэш данных, офлайн smoke-тесты")],
            "notes": "Проект собран по этапам. Данные с биржи и ЦБ кэшируются. Из них достаём риск-факторы, подгоняем модели по максимуму правдоподобия, считаем справедливую стоимость, затем VaR/ES методом Монте-Карло и в конце проверяем бэктестом за 2025-й.",
        },
        {
            "layout": "split",
            "kicker": "Портфель",
            "title": "260 млн ₽: где риск сидит ещё до расчётов",
            "lead": "<b>77% портфеля — это валюта.</b> Главный вывод можно сделать ещё до единого расчёта.",
            "points": [
                "5 ОФЗ × 10 млн = 50 млн (лесенка по срокам &lt;1…15 лет)",
                "10 акций × 1 млн = 10 млн (банк/нефтегаз/металлы/ритейл/телеком/техи)",
                "USD + EUR × 100 млн = 200 млн",
            ],
            "imgs": [("slides/portfolio_donut.png", "")],
            "notes": "Посмотрите на структуру — и половину выводов можно сделать сразу. 200 миллионов из 260 это валюта. Значит хвостовой риск почти наверняка будет валютным, просто из-за размера позиции. Облигации и акции по размеру намного меньше.",
        },
        {
            "layout": "split",
            "kicker": "Данные и источники",
            "title": "MOEX ISS + Банк России",
            "lead": "Прозрачная цепочка до первоисточника.",
            "points": [
                "Котировки акций/облигаций/индексов — MOEX&nbsp;ISS.",
                "КБД и курсы USD/EUR — Банк России.",
                "Купоны/погашения ОФЗ — расписания MOEX.",
                "⚠ <b>13.06.2024</b>: торги USD/EUR на MOEX остановлены, курс ЦБ стал внебиржевым.",
            ],
            "imgs": [("slides/data_timeline.png", "")],
            "notes": "Все данные — из открытых официальных источников. Отдельно подчеркну разрыв: с 13 июня 2024 года курс ЦБ считается по внебиржевым сделкам. Это смена источника прямо в середине нашей выборки, мы это учитываем при выборе окна оценки ковариаций.",
        },
        {
            "layout": "split",
            "kicker": "Риск-факторы · ставки",
            "title": "Кривая ставок = уровень + наклон + кривизна",
            "lead": "Три компоненты PCA объясняют <b>95.6%</b> дисперсии приращений кривой (Литтерман–Шейнкман).",
            "points": [
                "PC1 — уровень (параллельный сдвиг): 75.6%",
                "PC2 — наклон (короткий конец vs длинный): 12.8%",
                "PC3 — кривизна (середина vs концы): 7.2%",
                "Облигации переоцениваем от восстановленной по 3 PC кривой.",
            ],
            "imgs": [("factors/curve_loadings.png", "")],
            "notes": "По ставкам PCA работает идеально: три компоненты дают почти всю дисперсию и имеют классическую форму — уровень, наклон, кривизна. Это позволяет свести 12 сроков кривой к трём факторам без потери риска.",
        },
        {
            "layout": "split",
            "kicker": "Риск-факторы · акции",
            "title": "А вот акции почти не сжимаются",
            "lead": "Рыночный фактор PC1 — всего <b>54%</b>. Для 90% дисперсии нужно 8 из 10 компонент.",
            "points": [
                "Средняя парная корреляция бумаг ≈ 0.49 — портфель отраслево разнообразен.",
                "Сжать акции до 1–2 факторов = занизить их риск на ~35%.",
                "<b>Решение:</b> в риск-движок входят все 10 индивидуальных доходностей.",
                "PC1 оставлен только как описательный «рыночный фактор».",
            ],
            "imgs": [("factors/scree_equity.png", "")],
            "notes": "А вот тут самое интересное. С акциями PCA не сжимается: первая компонента тянет лишь половину дисперсии. Поэтому мы не стали их сжимать — в движок идут все десять бумаг с полной ковариацией. Если сжать акции до пары факторов, их риск был бы занижен примерно на треть.",
        },
        {
            "layout": "duo",
            "kicker": "Диагностика распределений",
            "title": "Нормальное распределение здесь не работает",
            "lead": "Тяжёлые хвосты (Жарка–Бера отвергает нормальность у 15/15) и кластеризация волатильности (ARCH у 14/15).",
            "points": [
                "Хвосты уходят от прямой нормали.",
                "t-Стьюдента: ν ≈ 3 — крайне тяжёлый хвост.",
                "Квадраты доходностей автокоррелированы.",
                "Тихие и бурные периоды группируются.",
            ],
            "imgs": [
                ("factors/qq_market_factor.png", "Хвосты против нормали и t"),
                ("factors/vol_clustering_market.png", "ACF квадратов: кластеризация волатильности"),
            ],
            "notes": "Дескриптивный анализ показывает две вещи: хвосты тяжелее нормальных у всех факторов, и волатильность кластеризуется. Это прямо диктует выбор моделей: нужна t-Стьюдента и условная волатильность, а не простая нормаль.",
        },
        {
            "layout": "duo",
            "kicker": "Стохастические модели · MLE",
            "title": "t-Стьюдента и условная волатильность",
            "lead": "t уверенно лучше нормали по AIC/BIC, <b>ν ≈ 4</b>; EWMA и GARCH ловят кластеризацию.",
            "points": [
                "Нормаль — база; t — тяжёлые хвосты (ν по EM).",
                "EWMA(λ=0.94) и GARCH(1,1)+CCC: α+β ≈ 0.99.",
                "Ставки на 1–10 дней ≈ случайное блуждание.",
                "Все параметры — по максимуму правдоподобия.",
            ],
            "imgs": [
                ("models/aic_normal_vs_t.png", "ΔAIC: t побеждает"),
                ("models/conditional_vol.png", "EWMA vs GARCH-волатильность"),
            ],
            "notes": "Мы оценили пять моделей методом максимального правдоподобия и сравнили по AIC и BIC. t-Стьюдента уверенно бьёт нормаль, степень свободы около четырёх — это очень тяжёлые хвосты. EWMA и GARCH описывают, как волатильность дышит во времени.",
        },
        {
            "layout": "split",
            "kicker": "Справедливая стоимость",
            "title": "Облигации — от кривой КБД",
            "lead": f"PV = Σ CFᵢ·DF(tᵢ). Средняя ошибка модель–рынок: <b>{k['pricing_mae']:.3f}%</b> номинала / <b>{k['ytm_bp']:.1f} б.п.</b> по доходности.",
            "points": [
                "Денежные потоки — из эмиссионного расписания; дисконт — из КБД.",
                "Годовая капитализация ближе к рынку, чем непрерывная (выбор обоснован ошибкой).",
                "Дюрация и выпуклость — sanity-check чувствительности к ставке.",
                "Точность проверена не на одну дату, а на всём 2025 году.",
            ],
            "imgs": [("pricing/pricing_error.png", "Ошибка по 5 ОФЗ на дату риска")],
            "notes": "Облигацию мы оцениваем так: сумма будущих потоков, дисконтированных по кривой ЦБ. Ошибка против рынка — доли процента номинала и пара десятков базисных пунктов по доходности, и это нормально. Кривая сглаженная, у каждого выпуска своя премия за ликвидность.",
        },
        {
            "layout": "full",
            "kicker": "Риск-движок",
            "title": "Как сценарий факторов превращается в потери",
            "lead": "VaR/ES считаются <b>не по факторам напрямую</b>, а через полную переоценку портфеля.",
            "imgs": [("slides/risk_engine_flow.png", "Ставки → сдвиг КБД → PV ОФЗ; акции/FX → exp(доходности)−1")],
            "notes": "Вот главная идея движка. Мы не считаем VaR факторов — на каждом сценарии переоцениваем весь портфель целиком. Сдвинули кривую — переоценили облигации. Сдвинули курс и цены акций — пересчитали через экспоненту. А из готового распределения P&L берём квантиль и хвост.",
        },
        {
            "layout": "split",
            "kicker": "Результаты",
            "title": "VaR 99% и ES 97.5%",
            "lead": "Самую консервативную оценку даёт историческая симуляция.",
            "points": [
                f"1 день: VaR <b>{money(k['var1'])} ₽</b>, ES {money(k['es1'])} ₽.",
                f"10 дней: VaR <b>{money(k['var10'])} ₽</b>, ES {money(k['es10'])} ₽.",
                "Сравниваем 5 моделей: historical, нормаль, t, EWMA+нормаль, GARCH+t.",
                "<span class='muted'>10 дней — buy-and-hold (без ежедневной ребалансировки): сознательное упрощение, эффект второго порядка.</span>",
            ],
            "imgs": [("risk/var_by_model.png", "")],
            "notes": "Однодневный VaR — около 7,4 миллиона, десятидневный — около 25. Историческая симуляция консервативнее параметрических моделей. И честно: десять дней мы считаем как buy-and-hold, без ежедневной ребалансировки. Это упрощение, но оно работает в запас.",
        },
        {
            "layout": "split",
            "kicker": "Откуда берётся риск",
            "title": "Почти весь хвост — валюта",
            "lead": "<b>≈99%</b> хвостового ES — валютные позиции. Не из-за волатильности, а из-за размера.",
            "points": [
                "Валюта — 200 из 260 млн номинала.",
                "Облигации чувствительны к параллельному сдвигу кривой, но их доля мала.",
                "Вывод для управления: хедж валютной экспозиции снял бы основную часть риска.",
            ],
            "imgs": [("slides/es_contribution.png", "")],
            "notes": "Декомпозиция ES показывает, что хвост почти полностью валютный. Это не потому что валюта самая волатильная, а потому что позиция самая большая. Практический вывод: если хеджировать валюту, общий риск резко падает.",
        },
        {
            "layout": "split",
            "kicker": "Бэктест за 2025",
            "title": "Проверка на 250 торговых днях",
            "lead": f"<b>{k['bt_exc']} пробоя</b> из {k['bt_obs']} (ожидалось 2.5), Basel <b>{k['bt_light']}</b>. Но пробои кластеризуются.",
            "points": [
                "Rolling historical VaR, окно 500 дней, сравнение с фактическим P&L.",
                "Ансамбль тестов: Kupiec (UC), Christoffersen (IND/CC), DQ Энгла–Манганелли.",
                "Подпортфели: облигации и валюта green; <b>акции — yellow</b> (7 пробоев).",
                "При 99% VaR за год мощность низкая → именно поэтому ансамбль тестов.",
            ],
            "imgs": [("backtest/var_backtest.png", "Фактический P&L против порога −VaR; красным — пробои")],
            "notes": "Прогнали rolling-VaR по всему 2025 году. Три пробоя из 250 при ожидаемых 2,5 — по частоте отлично, Kupiec не отвергает. Но Christoffersen видит, что пробои кластеризуются. По акциям отдельно — жёлтая зона, семь пробоев. Это честно называем слабым местом простой исторической модели.",
        },
        {
            "layout": "two",
            "kicker": "Критическое обсуждение",
            "title": "Границы модели и куда расти",
            "lead": "Сильная сторона работы — честный разбор ограничений, а не их замалчивание.",
            "colA": {
                "title": "Ограничения",
                "points": [
                    "Структурный разрыв 2022 → нестационарность режимов.",
                    "Смена методологии курса ЦБ с 13.06.2024.",
                    "Акции в бэктесте — yellow: historical VaR недо-консервативен.",
                    "10-дневный VaR — buy-and-hold, не путь-зависимый.",
                    "Годовой бэктест при 99% → низкая мощность тестов.",
                ],
            },
            "colB": {
                "title": "Направления развития",
                "points": [
                    "Путь-зависимый 10-дн. горизонт с ежедневной ребалансировкой.",
                    "GARCH-VaR / filtered HS для акций — поднять yellow до green.",
                    "Хедж валютной экспозиции — снимает основную часть риска.",
                    "Бонус: опционы на фьючерс (Блэк-76), встроенные опционы ОФЗ.",
                ],
            },
            "notes": "Слева — ограничения: разрыв 2022-го, смена источника курса в 2024-м, акции в жёлтой зоне, упрощённый десятидневный горизонт и низкая мощность годового бэктеста. Справа — что бы мы улучшили: путь-зависимый горизонт, GARCH-VaR для акций, хедж валюты и бонус с опционами.",
        },
        {
            "layout": "full",
            "kicker": "Итог",
            "title": "Рабочий риск-движок: от данных до бэктеста",
            "lead": "Воспроизводимый пайплайн, честная диагностика, понятный главный риск.",
            "imgs": [("slides/kpi_dashboard.png", "Главный риск — валюта · направления развития: путь-зависимый 10-дн. горизонт, GARCH-VaR для акций, бонус Блэк-76")],
            "notes": "Подытожим. Мы собрали рабочий риск-движок целиком — от данных до бэктеста. Главный риск портфеля валютный. Модель работает как базовая оценка со своими ограничениями. Спасибо, готовы к вопросам!",
        },
    ]


CSS = """
:root{
  --ink:#0c1320; --ink2:#101a2b; --panel:#13203447; --paper:#f3ede1;
  --muted:#93a4bd; --gold:#e2a44c; --teal:#62c2b0; --red:#e0796d;
  --line:rgba(243,237,225,.14);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:
    radial-gradient(1200px 700px at 78% -10%, #1b2c47 0%, transparent 55%),
    radial-gradient(900px 600px at 0% 110%, #16243b 0%, transparent 50%),
    var(--ink);
  color:var(--paper);
  font-family:"IBM Plex Sans","Helvetica Neue",system-ui,sans-serif;
  overflow:hidden;
}
body::before{ /* grain */
  content:"";position:fixed;inset:0;pointer-events:none;z-index:9;opacity:.05;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.deck{position:fixed;inset:0}
.slide{
  position:absolute;inset:0;display:none;
  padding:clamp(32px,5vw,80px) clamp(36px,7vw,120px);
  flex-direction:column;justify-content:center;
}
.slide.active{display:flex}
.kicker{
  font-family:"IBM Plex Mono",monospace;font-size:clamp(11px,1.05vw,14px);
  letter-spacing:.28em;text-transform:uppercase;color:var(--gold);
  margin-bottom:18px;
}
.title{
  font-family:"Fraunces","Georgia",serif;font-weight:600;
  font-size:clamp(28px,4.3vw,52px);line-height:1.05;letter-spacing:-.01em;
  max-width:24ch;
}
.lead{
  font-size:clamp(15px,1.5vw,20.5px);line-height:1.45;color:#e7ddcb;
  margin-top:15px;max-width:52ch;font-weight:400;
}
.lead b,.points b{color:var(--gold);font-weight:600}
.points{list-style:none;margin-top:6px;max-width:48ch}
.points li{
  position:relative;padding-left:24px;margin:11px 0;
  font-size:clamp(13.5px,1.3vw,17px);line-height:1.42;color:#d7cfc0;
}
.points li::before{
  content:"";position:absolute;left:0;top:.62em;width:9px;height:9px;
  background:var(--teal);transform:rotate(45deg);
}
.muted{color:var(--muted)}
b .muted,.muted{font-weight:400}

/* layouts */
.slide .head{max-width:92%}
.slide.split .body{display:flex;flex-direction:column;justify-content:center;
  gap:clamp(15px,2.2vw,28px)}
.slide.split .row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);
  gap:clamp(26px,3.4vw,54px);align-items:center}
.slide.split .row .points{margin-top:0}
.slide.split .row .points li{font-size:clamp(15px,1.55vw,20px);margin:13px 0;line-height:1.45}
.slide.full .body{display:flex;flex-direction:column;gap:14px;justify-content:center;align-items:center}
.slide.full .head{align-self:flex-start}
.slide.duo .body{display:flex;flex-direction:column;justify-content:center;gap:clamp(10px,1.5vw,18px)}
.slide.duo .points{columns:2;column-gap:clamp(30px,5vw,72px);max-width:100%;margin-top:2px}
.slide.duo .points li{break-inside:avoid;margin:8px 0}
.slide.duo .charts{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);
  gap:clamp(18px,2.2vw,30px)}
.slide.text .body{max-width:62ch}
.slide.two .body{display:flex;flex-direction:column;justify-content:center;gap:clamp(16px,2.2vw,28px)}
.slide.two .row2{display:grid;grid-template-columns:1fr 1fr;gap:clamp(30px,5vw,72px)}
.slide.two .coltitle{font-family:"IBM Plex Mono",monospace;font-size:12.5px;
  letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin-bottom:10px}
.slide.two .tcol.alt .coltitle{color:var(--teal)}
.slide.two .tcol.alt .points li::before{background:var(--gold)}
.slide.two .points{margin-top:0}
.slide.two .points li{font-size:clamp(14px,1.42vw,18.5px);margin:11px 0}
.charts{display:flex;flex-direction:column;gap:12px}
figure.chart{
  background:linear-gradient(180deg,#0f1a2bcc,#0c1422cc);
  border:1px solid var(--line);border-radius:14px;padding:12px 12px 9px;
  box-shadow:0 24px 60px -28px #000;backdrop-filter:blur(2px);
  display:flex;flex-direction:column;align-items:center;max-width:100%;
}
figure.chart img{display:block;max-width:100%;height:auto;border-radius:6px}
figure.chart figcaption{
  font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);
  margin-top:8px;line-height:1.4;letter-spacing:.01em;text-align:center;
}
.slide.split figure.chart img{max-height:50vh}
.slide.full figure.chart img{max-height:55vh}
.slide.duo figure.chart img{max-height:38vh}

/* title slide — отдельный layout-класс, чтобы не конфликтовать с h2.title */
.brandline{width:64px;height:3px;background:var(--gold)}
.slide.title-slide{text-align:center}
.slide.title-slide.active{display:grid;place-items:center}
.slide.title-slide .body{max-width:880px}
.slide.title-slide .brandline{margin:0 auto 22px}
.slide.title-slide .kicker{margin-bottom:18px}
.slide.title-slide .title{font-size:clamp(40px,6.2vw,82px);line-height:1.04}
.slide.title-slide .lead{font-size:clamp(16px,1.8vw,24px);margin:20px auto 0;color:#e7ddcb}
.t-divider{width:min(440px,60%);height:1px;background:var(--line);margin:34px auto 28px}
.t-meta{font-family:"IBM Plex Mono",monospace;font-size:clamp(11.5px,1.15vw,14px);
  color:#cdbfa6;letter-spacing:.03em;margin:0 auto;line-height:1.7}
.t-team{font-size:clamp(13px,1.3vw,16.5px);color:var(--muted);margin:18px auto 0;line-height:1.75}

/* chrome */
#bar{position:fixed;top:0;left:0;height:3px;background:var(--gold);width:0;
  z-index:12;transition:width .4s cubic-bezier(.4,0,.1,1)}
.snum{position:fixed;right:clamp(20px,4vw,56px);bottom:clamp(18px,3vw,40px);
  font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--muted);
  letter-spacing:.15em;z-index:12}
.snum b{color:var(--paper)}
.foot{position:fixed;left:clamp(20px,4vw,56px);bottom:clamp(18px,3vw,40px);
  font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);
  letter-spacing:.12em;z-index:12;text-transform:uppercase}
.nav{position:fixed;right:clamp(20px,4vw,56px);bottom:clamp(48px,7vw,84px);
  display:flex;gap:8px;z-index:13}
.nav button{
  width:38px;height:38px;border:1px solid var(--line);background:#0f1a2b88;
  color:var(--paper);border-radius:9px;cursor:pointer;font-size:16px;
  transition:.2s;backdrop-filter:blur(3px)}
.nav button:hover{border-color:var(--gold);color:var(--gold)}
#notes{position:fixed;left:0;right:0;bottom:0;z-index:20;display:none;
  background:#080d16f2;border-top:1px solid var(--line);
  padding:18px clamp(36px,7vw,120px);font-size:15px;line-height:1.5;color:#cdd6e4;
  max-height:34vh;overflow:auto}
body.notes-on #notes{display:block}
#notes .nt{font-family:"IBM Plex Mono",monospace;font-size:10.5px;
  letter-spacing:.2em;text-transform:uppercase;color:var(--gold);
  display:block;margin-bottom:8px}

/* reveal */
.slide.active .kicker{animation:rise .55s both}
.slide.active .title{animation:rise .6s .06s both}
.slide.active .lead{animation:rise .6s .14s both}
.slide.active .points li{animation:rise .55s both}
.slide.active .points li:nth-child(1){animation-delay:.20s}
.slide.active .points li:nth-child(2){animation-delay:.28s}
.slide.active .points li:nth-child(3){animation-delay:.36s}
.slide.active .points li:nth-child(4){animation-delay:.44s}
.slide.active .points li:nth-child(5){animation-delay:.52s}
.slide.active figure.chart{animation:fade .8s .2s both}
.slide.active .aside .stat{animation:rise .6s both}
.slide.active .aside .stat:nth-child(1){animation-delay:.30s}
.slide.active .aside .stat:nth-child(2){animation-delay:.40s}
.slide.active .aside .stat:nth-child(3){animation-delay:.50s}
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes fade{from{opacity:0;transform:scale(.985)}to{opacity:1;transform:none}}
@media (max-width:880px){
  .slide.split .body{grid-template-columns:1fr;gap:20px}
  .slide{padding:28px 28px 70px}
}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];
const bar=document.getElementById('bar');
const num=document.getElementById('num');
const notes=document.getElementById('notes');
let cur=0;
function show(i){
  cur=(i+slides.length)%slides.length;
  slides.forEach((s,k)=>s.classList.toggle('active',k===cur));
  bar.style.width=((cur+1)/slides.length*100)+'%';
  num.innerHTML='<b>'+String(cur+1).padStart(2,'0')+'</b> / '+String(slides.length).padStart(2,'0');
  notes.innerHTML='<span class="nt">Заметки докладчику · слайд '+(cur+1)+'</span>'+(slides[cur].dataset.note||'');
}
function next(){show(cur+1)} function prev(){show(cur-1)}
document.addEventListener('keydown',e=>{
  const t=e.key.toLowerCase();
  if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' '){next();e.preventDefault();}
  else if(e.key==='ArrowLeft'||e.key==='PageUp'){prev();}
  else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(slides.length-1);}
  else if(t==='s'){document.body.classList.toggle('notes-on');}
  else if(t==='f'){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen();}
});
document.getElementById('next').onclick=next;
document.getElementById('prev').onclick=prev;
show(0);
"""


def render_slide(s: dict) -> str:
    layout = s.get("layout", "split")
    kicker = f'<div class="kicker">{s["kicker"]}</div>' if s.get("kicker") else ""
    title = f'<h2 class="title">{s["title"]}</h2>' if s.get("title") else ""
    lead = f'<p class="lead">{s["lead"]}</p>' if s.get("lead") else ""
    points = ""
    if s.get("points"):
        points = '<ul class="points">' + "".join(f"<li>{p}</li>" for p in s["points"]) + "</ul>"

    charts = ""
    for rel, cap in s.get("imgs", []):
        uri = data_uri(rel)
        if not uri:
            continue
        cp = f'<figcaption>{cap}</figcaption>' if cap else ""
        charts += f'<figure class="chart"><img src="{uri}" alt="">{cp}</figure>'
    charts_block = f'<div class="charts">{charts}</div>' if charts else ""

    if layout == "two":  # шапка + две текстовые колонки с подзаголовками
        def col(c, alt=False):
            h = f'<div class="coltitle">{c.get("title","")}</div>' if c.get("title") else ""
            lis = "".join(f"<li>{p}</li>" for p in c.get("points", []))
            return f'<div class="tcol{" alt" if alt else ""}">{h}<ul class="points">{lis}</ul></div>'
        row = col(s.get("colA", {})) + col(s.get("colB", {}), alt=True)
        return (f'<section class="slide two" data-note="{esc_attr(s.get("notes",""))}">'
                f'<div class="body"><div class="head">{kicker}{title}{lead}</div>'
                f'<div class="row2">{row}</div></div></section>')

    if layout == "title":
        meta = f'<div class="t-divider"></div><p class="t-meta">{s["meta"]}</p>' if s.get("meta") else ""
        team = f'<p class="t-team">{s["team"]}</p>' if s.get("team") else ""
        inner = f'<div class="brandline"></div>{kicker}{title}{lead}{meta}{team}'
        return (f'<section class="slide title-slide" data-note="{esc_attr(s.get("notes",""))}">'
                f'<div class="body">{inner}</div></section>')
    elif layout == "text":
        body = f'<div class="body">{kicker}{title}{lead}{points}</div>'
    elif layout == "full":
        body = f'<div class="body"><div class="head">{kicker}{title}{lead}</div>{charts_block}{points}</div>'
    elif layout == "duo":  # шапка, компактные пункты в 2 колонки, 2 графика рядом
        body = (f'<div class="body"><div class="head">{kicker}{title}{lead}</div>'
                f'{points}{charts_block}</div>')
    else:  # split: шапка сверху, ниже ряд [пункты | график]
        body = (f'<div class="body"><div class="head">{kicker}{title}{lead}</div>'
                f'<div class="row">{points}{charts_block}</div></div>')

    note = esc_attr(s.get("notes", ""))
    return f'<section class="slide {layout}" data-note="{note}">{body}</section>'


def build() -> None:
    k = collect_kpis()
    k.setdefault("nu", 4.0)
    slides_html = "".join(render_slide(s) for s in slides_spec(k))

    html = (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Рыночный риск — защита проекта</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
        "<style>" + CSS + "</style></head><body>"
        '<div id="bar"></div>'
        '<div class="deck">' + slides_html + "</div>"
        '<div class="foot">Основы риск-менеджмента · VaR/ES · ФКН ВШЭ</div>'
        '<div class="snum" id="num"></div>'
        '<div class="nav"><button id="prev" title="Назад">‹</button>'
        '<button id="next" title="Вперёд">›</button></div>'
        '<div id="notes"></div>'
        "<script>" + JS + "</script></body></html>"
    )

    out = O / "slides" / "presentation.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("презентация: %s (%d слайдов, %.1f МБ)",
                out.relative_to(O.parent), len(slides_spec(k)), len(html) / 1e6)
    print("\n=== ПРЕЗЕНТАЦИЯ ГОТОВА ===")
    print(f"Открой в браузере: {out}")
    print("F — полный экран · ← → — слайды · S — заметки докладчику")


if __name__ == "__main__":
    build()
