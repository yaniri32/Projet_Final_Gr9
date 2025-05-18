import pandas as pd
from TP3 import create_labels_one_company, add_ta_features
import os

def compute_indicators(df, short_window=12, long_window=26, signal_window=9):
    """
    df : DataFrame contenant au moins la colonne 'Close'
    """
    close = df['Close']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=short_window, adjust=False).mean()
    ema26 = close.ewm(span=long_window, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    # MA/EMA Crossover
    ma50 = close.rolling(50).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    df["RSI"] = rsi
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MA50"] = ma50
    df["EMA20"] = ema20
    return df

def generate_signals(df):
    """
    Génère les signaux buy/sell/hold avec commentaires d’analyse technique.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {}

    # RSI
    rsi = latest["RSI"]
    if rsi > 70:
        signals["RSI"] = ("sell", "RSI > 70 : sur-achat, possible retournement baissier")
    elif rsi < 30:
        signals["RSI"] = ("buy", "RSI < 30 : sur-vente, possible retournement haussier")
    else:
        signals["RSI"] = ("hold", "RSI neutre")

    # MACD
    if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        signals["MACD"] = ("buy", "Croisement MACD haussier")
    elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        signals["MACD"] = ("sell", "Croisement MACD baissier")
    else:
        signals["MACD"] = ("hold", "Pas de croisement MACD")

    # MA/EMA
    if latest["EMA20"] > latest["MA50"] and prev["EMA20"] <= prev["MA50"]:
        signals["MA/EMA"] = ("buy", "Croisement EMA20 au-dessus de MA50")
    elif latest["EMA20"] < latest["MA50"] and prev["EMA20"] >= prev["MA50"]:
        signals["MA/EMA"] = ("sell", "Croisement EMA20 en-dessous de MA50")
    else:
        signals["MA/EMA"] = ("hold", "Pas de croisement")

    return signals

def technical_analysis(symbol, folder="historiques_entreprises"):
    csv_path = os.path.join(folder, f"{symbol}_historique.csv")
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    df = compute_indicators(df)
    signals = generate_signals(df)
    return signals

def get_last_features_for_classification(ticker):
    path = f"historiques_entreprises/{ticker}_historique.csv"
    df = create_labels_one_company(path)
    df = add_ta_features(df).dropna()

    drop_cols = ["Label", "Close_Horizon", "Horizon_Return", "Symbol"]
    if "Next Day Close" in df.columns:
        drop_cols.append("Next Day Close")
    X = df.drop(columns=drop_cols, errors="ignore")

    return X.iloc[[-1]]  # dernière ligne pour la prédiction