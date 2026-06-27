"""
PCA по риск-факторам (через спектральное разложение ковариации).

Реализация на numpy.linalg.eigh — детерминированная (без рандома, в отличие от
рандомизированного SVD в некоторых режимах sklearn), с явным контролем знака
компонент и интерпретацией нагрузок. Это важно для воспроизводимости отчёта.

Два режима:
  * standardize=False — PCA по КОВАРИАЦИИ. Для кривой КБД: все сроки в одних
    единицах (доходность), и нам нужны компоненты в «натуральных» единицах
    ставки -> уровень / наклон / кривизна (Литтерман–Шейнкман).
  * standardize=True  — PCA по КОРРЕЛЯЦИИ (стандартизованные ряды). Для акций:
    бумаги с разной волатильностью вносят сопоставимый вклад, и PC1 выходит
    чистым «рыночным» фактором (а не просто самой волатильной бумагой).

Объект PCAResult умеет восстанавливать исходные ряды из компонент
(reconstruct) — это вход для риск-движка на следующем этапе.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    """Результат PCA. Нагрузки/счёты хранятся как pandas для читаемости."""

    loadings: pd.DataFrame              # переменные × компоненты (PC1..PCk)
    scores: pd.DataFrame                # даты × компоненты (значения факторов)
    explained_variance: np.ndarray      # собственные числа (все, по убыванию)
    explained_variance_ratio: np.ndarray  # доля дисперсии (все компоненты)
    mean: pd.Series                      # среднее по каждой переменной
    scale: pd.Series                     # делитель (std если standardize, иначе 1)
    standardized: bool

    @property
    def n_components(self) -> int:
        return self.scores.shape[1]

    @property
    def kept_ratio(self) -> np.ndarray:
        """Доля дисперсии оставленных компонент."""
        return self.explained_variance_ratio[: self.n_components]

    def cumulative_ratio(self) -> np.ndarray:
        """Накопленная доля дисперсии по всем компонентам."""
        return np.cumsum(self.explained_variance_ratio)

    def reconstruct(self, scores: pd.DataFrame | None = None) -> pd.DataFrame:
        """Восстановить исходные ряды (в оригинальных единицах) из счётов.

        X ≈ scores @ loadingsᵀ · scale + mean. Если standardize=True,
        умножение на scale возвращает ряды из стандартизованных в исходные —
        поэтому ковариация восстановленных рядов корректна и для риск-движка.
        """
        s = self.scores if scores is None else scores
        approx = s.to_numpy() @ self.loadings.to_numpy().T
        approx = approx * self.scale.to_numpy() + self.mean.to_numpy()
        return pd.DataFrame(approx, index=s.index, columns=self.loadings.index)


def fit_pca(
    X: pd.DataFrame,
    n_components: int | None = None,
    standardize: bool = False,
) -> PCAResult:
    """Подогнать PCA. X — даты × переменные, БЕЗ NaN (см. returns.clean_for_pca).

    n_components=None -> сохранить все. Знак каждой компоненты фиксируется так,
    чтобы нагрузка с максимальным модулем была положительной (детерминированно
    и воспроизводимо; дисперсия от знака не зависит).
    """
    if X.isna().any().any():
        raise ValueError("fit_pca получил NaN — прогони returns.clean_for_pca")
    if X.shape[0] <= X.shape[1]:
        raise ValueError(f"наблюдений ({X.shape[0]}) не больше числа переменных "
                         f"({X.shape[1]}) — ковариация вырождена")

    cols = list(X.columns)
    values = X.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1) if standardize else np.ones(values.shape[1])
    scale = np.where(scale == 0, 1.0, scale)  # защита от деления на ноль
    Z = (values - mean) / scale

    cov = np.cov(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)      # eigh: симметричная матрица
    order = np.argsort(eigvals)[::-1]           # по убыванию
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals_clipped = np.clip(eigvals, 0.0, None)
    total = eigvals_clipped.sum()
    ratio = eigvals_clipped / total if total > 0 else np.zeros_like(eigvals_clipped)

    k = eigvecs.shape[1] if n_components is None else min(n_components, eigvecs.shape[1])
    W = eigvecs[:, :k].copy()

    # Фиксация знака: максимальная по модулю нагрузка положительна.
    for j in range(k):
        idx = int(np.argmax(np.abs(W[:, j])))
        if W[idx, j] < 0:
            W[:, j] *= -1.0

    scores = Z @ W
    comp_names = [f"PC{i + 1}" for i in range(k)]

    return PCAResult(
        loadings=pd.DataFrame(W, index=cols, columns=comp_names),
        scores=pd.DataFrame(scores, index=X.index, columns=comp_names),
        explained_variance=eigvals,
        explained_variance_ratio=ratio,
        mean=pd.Series(mean, index=cols),
        scale=pd.Series(scale, index=cols),
        standardized=standardize,
    )


# --------------------------------------------------------------------------- #
# Интерпретация компонент кривой (уровень / наклон / кривизна)                 #
# --------------------------------------------------------------------------- #
def _sign_changes(vector: np.ndarray) -> int:
    """Сколько раз меняется знак вдоль вектора нагрузок (по упорядоченным срокам)."""
    signs = np.sign(vector)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def interpret_curve_components(loadings: pd.DataFrame) -> dict[str, str]:
    """Подписать компоненты кривой по форме нагрузок вдоль сроков.

    Эвристика Литтермана–Шейнкмана:
      0 смен знака -> «уровень»  (параллельный сдвиг кривой);
      1 смена      -> «наклон»   (короткий конец vs длинный);
      2 смены      -> «кривизна» (середина против концов).
    Нагрузки должны идти в порядке возрастания срока (см. curve_increments).
    """
    labels = {}
    names = {0: "уровень", 1: "наклон", 2: "кривизна"}
    for comp in loadings.columns:
        sc = _sign_changes(loadings[comp].to_numpy())
        labels[comp] = names.get(sc, f"высш. порядок ({sc} смен знака)")
    return labels
