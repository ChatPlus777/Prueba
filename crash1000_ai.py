# coding: utf-8
"""
Ejemplo basico de como analizar el indice Crash 1000 desde MT5 y
entrenar un modelo simple para intentar detectar caidas ("crash")
utilizando datos de 1 minuto.

Nota: Este codigo se proporciona unicamente con fines educativos y
no constituye consejo financiero. Operar con indices sinteticos implica
riesgos significativos.
"""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


@dataclass
class MT5Config:
    path: str
    login: int
    password: str
    server: str


def init_mt5(cfg: MT5Config) -> bool:
    """Inicializa la conexion a MetaTrader 5."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 library is not installed")
    if not mt5.initialize(path=cfg.path, login=cfg.login, password=cfg.password, server=cfg.server):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    return True


def fetch_data(symbol: str, bars: int) -> pd.DataFrame:
    """Descarga `bars` velas de 1 minuto para `symbol`."""
    utc_from = dt.datetime.now() - dt.timedelta(minutes=bars)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_from, bars)
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores tecnicos basicos (RSI, ADX)."""
    import pandas_ta as ta

    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14']
    return df


def label_crashes(df: pd.DataFrame, threshold: float = 0.015) -> pd.DataFrame:
    """Añade una columna 'crash' cuando la variacion negativa supera el `threshold`."""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['crash'] = (df['return'] <= -threshold).astype(int)
    return df


def analyze_statistics(df: pd.DataFrame) -> dict:
    """Calcula estadisticas sobre los crashes."""
    stats = {}
    crashes = df[df['crash'] == 1]
    if crashes.empty:
        return stats

    stats['avg_drop'] = crashes['return'].mean()
    stats['max_drop'] = crashes['return'].min()
    stats['count'] = len(crashes)
    stats['max_consecutive_green'] = (df['return'] > 0).astype(int).groupby((df['return'] <= 0).astype(int).cumsum()).cumsum().max()
    # Promedio de velas verdes antes de un crash
    before_crash = []
    count_green = 0
    for ret, crash in zip(df['return'], df['crash']):
        if crash:
            before_crash.append(count_green)
            count_green = 0
        elif ret > 0:
            count_green += 1
        else:
            count_green = 0
    stats['avg_green_before_crash'] = np.mean(before_crash) if before_crash else 0

    # Hora con mas crashes en un dia
    crashes_by_hour = crashes['time'].dt.hour.value_counts()
    if not crashes_by_hour.empty:
        stats['hour_most_crashes'] = int(crashes_by_hour.idxmax())

    # Minuto dentro de la hora con mas crashes
    crashes_by_minute = crashes['time'].dt.minute.value_counts()
    if not crashes_by_minute.empty:
        stats['minute_most_crashes'] = int(crashes_by_minute.idxmax())

    stats['max'] = df['high'].max()
    stats['min'] = df['low'].min()
    stats['crash_frequency'] = len(df) / stats['count'] if stats['count'] else None

    return stats


def train_model(df: pd.DataFrame):
    """Entrena un clasificador simple para predecir crashes."""
    features = df[['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'adx']].fillna(method='bfill')
    labels = df['crash']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Precision:', accuracy_score(y_test, preds))
    return clf


def main():
    cfg = MT5Config(path="/path/to/terminal64.exe", login=123456, password="password", server="Deriv-Server")
    init_mt5(cfg)
    data = fetch_data("Crash 1000 Index", 20000)
    data = compute_indicators(data)
    data = label_crashes(data)
    stats = analyze_statistics(data)
    for k, v in stats.items():
        print(f"{k}: {v}")
    model = train_model(data)
    # Aqui se podria entrar en un bucle para actualizar datos y nuevas predicciones


if __name__ == "__main__":
    main()
