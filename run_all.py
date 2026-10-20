# run_all.py

from calibration import fit_region
from config import MIN_OBSERVATIONS, OUTPUT_DIR, TARGET_PARAMETER_ID
from database import (
    get_connection,
    get_region_data_with_population,
    get_regions,
)
from factors import calculate_beta_and_alpha, load_tb_factors_dataset
import numpy as np
import pandas as pd


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_connection()

    try:
        # 1. Получаем список всех регионов из БД
        regions_df = get_regions(conn)
        total_regions = len(regions_df)
        print(f"Найдено регионов в базе данных: {total_regions}")

        # 2. Загружаем таблицу факторов (один раз перед циклом)
        factors_df = load_tb_factors_dataset()

        all_summaries = []

        # 3. Последовательный цикл по всем регионам
        for idx, row in regions_df.iterrows():
            region_id = int(row["RegionID"])
            region_name = str(row["RegionName"])

            print(
                f"\n[{idx + 1}/{total_regions}] Обработка: {region_name} (ID={region_id})...",
                end=" ",
            )

            try:
                # Загружаем заболеваемость и население для региона
                df = get_region_data_with_population(
                    conn=conn,
                    region_id=region_id,
                )

                if df.empty or len(df) < MIN_OBSERVATIONS:
                    print(
                        f"[ПРОПУСК] Недостаточно данных (наблюдений: {len(df)})"
                    )
                    continue

                years = df["Year"].to_numpy(dtype=int)
                A_data = df["A_data"].to_numpy(dtype=float)
                population_scale = float(df["population"].mean())

                if population_scale <= 0:
                    print("[ПРОПУСК] Нулевое население")
                    continue

                # Калибровка SIR-модели
                result = fit_region(
                    A_data=A_data,
                    population_scale=population_scale,
                    years=years,
                )

                # 4. Расчет R_median, beta_prob, alpha, B
                R_median = float(np.median(result["R_values"]))

                # Получаем строго нормализованную вероятность beta и размерную alpha
                beta_prob, alpha_val = calculate_beta_and_alpha(
                    region_id=region_id, factors_df=factors_df
                )

                # Итоговая размерная скорость передачи B (год^-1)
                B_val = R_median * alpha_val

                # Итоговая вероятность с учетом поведения населения (строго <= 1.0)
                beta_final = float(np.clip(R_median * beta_prob, 0.01, 0.99))
                # Сохраняем подробную динамику по годам для региона в отдельный CSV
                predictions_df = pd.DataFrame(
                    {
                        "RegionID": region_id,
                        "RegionName": region_name,
                        "Year": result["years"],
                        "A_data": result["A_data"],
                        "A_model": result["A_model"],
                        "R": np.round(result["R_values"], 4),
                    }
                )
                predictions_df.to_csv(
                    OUTPUT_DIR / f"sir_predictions_region_{region_id}.csv",
                    index=False,
                    sep=";",
                    encoding="utf-8-sig",
                )

                # Добавляем строку в общую сводную таблицу
                all_summaries.append(
                    {
                        "RegionID": region_id,
                        "RegionName": region_name,
                        "n_observations": len(A_data),
                        "success": result["success"],
                        "loss": round(result["loss"], 6),
                        "R_median": round(R_median, 4),
                        "alpha_factors": round(alpha_val, 4),
                        "B_total": round(B_val, 4),
                        "beta_prob": round(beta_final, 4),
                    }
                )

                print(
                    f"[OK] R={R_median:.4f} | alpha={alpha_val:.2f} | B={B_val:.2f} | Loss={result['loss']:.5f}"
                )

            except Exception as err:
                print(f"[ОШИБКА] Не удалось обработать: {err}")
                continue

        # 4. Сохраняем единую итоговую таблицу по всем регионам
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            summary_path = OUTPUT_DIR / "all_regions_summary.csv"
            summary_df.to_csv(
                summary_path,
                index=False,
                sep=";",
                encoding="utf-8-sig",
            )

            print("\n" + "=" * 65)
            print("ОБРАБОТКА ВСЕХ РЕГИОНОВ УСПЕШНО ЗАВЕРШЕНА!")
            print("=" * 65)
            print(
                f"Всего успешно рассчитано регионов: {len(summary_df)} из {total_regions}"
            )
            print(f"Итоговый файл: {summary_path.resolve()}")
            print("=" * 65)

            # Вывод первых 10 регионов для быстрой проверки
            print("\nПример первых 10 строк итоговой таблицы:")
            print(
                summary_df[
                    [
                        "RegionID",
                        "RegionName",
                        "R_median",
                        "alpha_factors",
                        "B_total",
                        "beta_prob",  # Строго от 0 до 1
                        "B_total" # Размерная скорость (год^-1, до 90)
                    ]
                ].head(10)
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()