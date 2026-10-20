import sqlite3
import pandas as pd

from config import (
    DB_PATH,
    REGIONS_TABLE,
    DATA_TABLE,
    REGION_ID_COL,
    REGION_NAME_COL,
    PARAMETER_ID_COL,
    YEAR_COL,
    VALUE_COL,
    TARGET_PARAMETER_ID,
    START_YEAR,
    END_YEAR,
    POPULATION_PARAMETER_ID
)


def quote_identifier(name: str) -> str:
    """
    Безопасное оформление имени таблицы или столбца SQLite.
    Имена нельзя передавать через SQL-параметры ?,
    поэтому предварительно заключаем их в двойные кавычки.
    """
    return '"' + name.replace('"', '""') + '"'


def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена база данных: {DB_PATH.resolve()}"
        )

    return sqlite3.connect(DB_PATH)


def get_regions(conn) -> pd.DataFrame:
    table = quote_identifier(REGIONS_TABLE)
    id_col = quote_identifier(REGION_ID_COL)
    name_col = quote_identifier(REGION_NAME_COL)

    query = f"""
        SELECT
            {id_col} AS RegionID,
            {name_col} AS RegionName
        FROM {table}
        ORDER BY {id_col}
    """

    return pd.read_sql_query(query, conn)
def get_region_data_with_population(conn, region_id):
    table = quote_identifier(DATA_TABLE)

    region_col = quote_identifier(REGION_ID_COL)
    parameter_col = quote_identifier(PARAMETER_ID_COL)
    year_col = quote_identifier(YEAR_COL)
    value_col = quote_identifier(VALUE_COL)

    query = f"""
        SELECT
            {year_col} AS Year,
            {parameter_col} AS ParameterID,
            {value_col} AS value
        FROM {table}
        WHERE {region_col} = ?
          AND {parameter_col} IN (?, ?)
          AND {year_col} BETWEEN ? AND ?
        ORDER BY {year_col}
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            region_id,
            TARGET_PARAMETER_ID,
            POPULATION_PARAMETER_ID,
            START_YEAR,
            END_YEAR,
        ),
    )

    if df.empty:
        return df

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Year", "value"]
    )

    df["Year"] = df["Year"].astype(int)

    # Преобразуем длинный формат в широкий:
    # Year | показатель 7 | показатель 28
    wide = df.pivot_table(
        index="Year",
        columns="ParameterID",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide = wide.rename(columns={
        TARGET_PARAMETER_ID: "A_data",
        POPULATION_PARAMETER_ID: "population_thousands",
    })

    required = [
        "A_data",
        "population_thousands",
    ]

    wide = wide.dropna(
        subset=required
    )

    # Население из тысяч человек переводим в человек
    wide["population"] = (
        wide["population_thousands"] * 1000
    )

    return wide.sort_values(
        "Year"
    ).reset_index(drop=True)

def get_region_series(
    conn,
    region_id: int,
    parameter_id: int = TARGET_PARAMETER_ID,
) -> pd.DataFrame:
    """
    Возвращает годовой ряд одного показателя одного региона.
    """

    table = quote_identifier(DATA_TABLE)
    region_col = quote_identifier(REGION_ID_COL)
    parameter_col = quote_identifier(PARAMETER_ID_COL)
    year_col = quote_identifier(YEAR_COL)
    value_col = quote_identifier(VALUE_COL)

    query = f"""
        SELECT
            {year_col} AS Year,
            {value_col} AS A_data
        FROM {table}
        WHERE {region_col} = ?
          AND {parameter_col} = ?
          AND {year_col} BETWEEN ? AND ?
        ORDER BY {year_col}
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            region_id,
            parameter_id,
            START_YEAR,
            END_YEAR,
        ),
    )

    if df.empty:
        return df

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce",
    )

    df["A_data"] = pd.to_numeric(
        df["A_data"],
        errors="coerce",
    )

    df = df.dropna(subset=["Year", "A_data"])
    df["Year"] = df["Year"].astype(int)

    # На случай дублирующихся записей за один год
    df = (
        df.groupby("Year", as_index=False)["A_data"]
        .sum()
        .sort_values("Year")
        .reset_index(drop=True)
    )

    return df


def get_table_columns(conn, table_name: str):
    """
    Вспомогательная функция для просмотра структуры таблицы.
    """
    table = quote_identifier(table_name)

    return conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()