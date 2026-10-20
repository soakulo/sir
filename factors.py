# factors.py
from pathlib import Path
import numpy as np
import pandas as pd

FACTORS_CSV_PATH = Path("regional_tb_data.csv")
MAX_CONTACTS_PER_YEAR = (
    90.0  # q1_max из статьи MDPI 2024 (макс. число контактов в год)
)
BASELINE_BETA = 0.35  # Средняя вероятность заражения по РФ (~35%)


def load_tb_factors_dataset(
    csv_path: Path = FACTORS_CSV_PATH,
) -> pd.DataFrame:
    """Загружает CSV с факторами и очищает строки округов."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Не найден файл с факторами: {csv_path.resolve()}"
        )

    df = pd.read_csv(csv_path)

    cols_to_numeric = [
        "RegionID",
        "median_hiv_tb_infected",
        "Плотность населения",
        "low_income",
        "people_per_doc",
        "unemployed",
    ]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Оставляем только субъекты РФ (где есть плотность населения)
    clean_df = df.dropna(
        subset=[
            "RegionID",
            "median_hiv_tb_infected",
            "Плотность населения",
            "people_per_doc",
        ]
    ).copy()
    clean_df["RegionID"] = clean_df["RegionID"].astype(int)

    if "low_income" in clean_df.columns:
        clean_df["low_income"] = clean_df["low_income"].fillna(
            clean_df["low_income"].median()
        )

    return clean_df.reset_index(drop=True)


def calculate_beta_and_alpha(
    region_id: int, factors_df: pd.DataFrame
) -> tuple[float, float]:
    """Считает:

    1. beta_prob in (0, 1) - вероятность заражения при контакте (через
    сигмоиду)
    2. alpha_val in (0, 90) - размерную интенсивность передачи в год
    """
    row = factors_df[factors_df["RegionID"] == region_id]

    if row.empty:
        # Для округов или отсутствующих регионов возвращаем среднее по стране
        return BASELINE_BETA, BASELINE_BETA * MAX_CONTACTS_PER_YEAR

    # 1. Извлекаем сырые показатели
    hiv = row["median_hiv_tb_infected"].values[0]
    density_log = np.log(max(row["Плотность населения"].values[0], 0.01))
    poverty = row["low_income"].values[0]
    doc_load = row["people_per_doc"].values[0]

    # 2. Z-нормализация по всей России
    densities_all_log = np.log(
        factors_df["Плотность населения"].clip(lower=0.01)
    )

    z_hiv = (hiv - factors_df["median_hiv_tb_infected"].mean()) / max(
        factors_df["median_hiv_tb_infected"].std(), 1e-6
    )
    z_density = (density_log - densities_all_log.mean()) / max(
        densities_all_log.std(), 1e-6
    )
    z_poverty = (poverty - factors_df["low_income"].mean()) / max(
        factors_df["low_income"].std(), 1e-6
    )
    z_doc = (doc_load - factors_df["people_per_doc"].mean()) / max(
        factors_df["people_per_doc"].std(), 1e-6
    )

    # 3. Веса факторов
    w_hiv = 0.40  # ВИЧ+ТБ
    w_density = 0.15  # Плотность контактов
    w_poverty = 0.30  # Бедность
    w_doc = 0.15  # Нагрузка на врачей

    # Базовый логит (смещение для среднего региона ~0.35)
    b0 = np.log(BASELINE_BETA / (1.0 - BASELINE_BETA))

    # Линейная комбинация
    logit = b0 + (
        w_hiv * z_hiv + w_density * z_density + w_poverty * z_poverty + w_doc * z_doc
    )

    # 4. Сигмоида: строго гарантирует значение в диапазоне (0.01, 0.99)
    beta_prob = 1.0 / (1.0 + np.exp(-logit))
    beta_prob = float(np.clip(beta_prob, 0.01, 0.99))

    # Годовая размерная скорость: alpha = beta * c (в диапазоне 1 .. 90 год^-1)
    alpha_val = float(beta_prob * MAX_CONTACTS_PER_YEAR)

    return beta_prob, alpha_val