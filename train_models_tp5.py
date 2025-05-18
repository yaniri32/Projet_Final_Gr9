# train_models_tp5.py

from TP5 import compare_models

companies = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "005930.KS",
    "TCEHY", "BABA", "IBM", "INTC", "ORCL", "SONY", "ADBE",
    "NFLX", "AMD", "QCOM", "CSCO", "JPM", "GS", "V", "JNJ", "PFE",
    "XOM", "ASML.AS", "SAP.DE", "SIE.DE",  "MC.PA", "TTE.PA",
    "SHEL.L", "BIDU", "JD", "BYDDY", "1398.HK", "TM", "9984.T", "NTDOY",
    "HYMTF", "RELIANCE.NS", "TCS.NS"
]

print("🏁 Entraînement des modèles deep learning (TP5)...")

for symbol in companies:
    print(f"\n🚀 Entraînement pour : {symbol}")
    compare_models(symbol=symbol)

print("\n✅ Tous les modèles ont été entraînés et sauvegardés.")