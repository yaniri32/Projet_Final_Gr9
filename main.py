import TP1, TP2, TP6, TP8, TP_complementaire
from generate_pdf import generate_pdf_for_company
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, joblib, subprocess, json
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Partie 1 : Extraction des données\n")
TP1.scrape_ratios()
TP1.scrape_historical()
print(f"\nRécupération des données de marchés réalisée\n")

companies = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "005930.KS",
    "TCEHY", "BABA", "IBM", "INTC", "ORCL", "SONY", "ADBE",
    "NFLX", "AMD", "QCOM", "CSCO", "JPM", "GS", "V", "JNJ", "PFE",
    "XOM", "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "TTE.PA",
    "SHEL.L", "BIDU", "JD", "BYDDY", "1398.HK", "TM", "9984.T", "NTDOY",
    "HYMTF", "RELIANCE.NS", "TCS.NS"
]

print("Partie 2 : Clustering\n")
df_ratios = pd.read_csv("historiques_entreprises/ratios_financiers.csv")

# Clustering sur le risque
df_risk, X_risk, labels_risk = TP2.cluster_risk_profiles(df_ratios)
df_risk["Ticker"] = companies
df_risk["Cluster"] = labels_risk

# Clustering sur le rendement (corrélation)
labels_ret, corr_ret, dist_ret = TP2.cluster_return_correlations()
df_rendement = pd.DataFrame({
    "Ticker": companies,
    "Cluster": labels_ret
})

# Construction des dictionnaires de similarité

# Clustering risque
clustering_risque = {}
for cl in df_risk["Cluster"].unique():
    tickers = df_risk[df_risk["Cluster"] == cl]["Ticker"].tolist()
    for t in tickers:
        clustering_risque[t] = [x for x in tickers if x != t]

# Clustering rendement
clustering_rendement = {}
for cl in df_rendement["Cluster"].unique():
    tickers = df_rendement[df_rendement["Cluster"] == cl]["Ticker"].tolist()
    for t in tickers:
        clustering_rendement[t] = [x for x in tickers if x != t]

print("\nClustering risque et rendement générés avec succès.")

print("Partie 3 : Classification (achat - vente)\n")
model_classification_path = "models/best_model_classification.pkl"
scaler_classification_path = "models/scaler_classification.pkl"

if not os.path.exists(model_classification_path) or not os.path.exists(scaler_classification_path):
    print("Modèle ou scaler manquant. Entraînement en cours via train_models_tp3.py...")
    try:
        subprocess.run(["python", "train_models_tp3.py"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'entraînement TP3")
        print("STDOUT :", e.stdout)
        print("STDERR :", e.stderr)
        raise
else:
    print("Modèle et scaler trouvés.")

scaler_classification = joblib.load(scaler_classification_path)
model_classification = joblib.load(model_classification_path)
print(f"\nModèle de classification extrait\n")

print("Partie 4 : Régression\n")
companies = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "005930.KS",
    "TCEHY", "BABA", "IBM", "INTC", "ORCL", "SONY", "ADBE",
    "NFLX", "AMD", "QCOM", "CSCO", "JPM", "GS", "V", "JNJ", "PFE",
    "XOM", "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "TTE.PA",
    "SHEL.L", "BIDU", "JD", "BYDDY", "1398.HK", "TM", "9984.T", "NTDOY",
    "HYMTF", "RELIANCE.NS", "TCS.NS"
]

# Dictionnaires pour stocker les modèles et scalers
models_reg = {}
scalers_reg = {}

# Vérifie si tous les fichiers existent, sinon lance l'entraînement
missing_models = False
for company in companies:
    model_path = f"models/reg/best_model_{company}_reg.pkl"
    scaler_path = f"models/reg/scaler_{company}_reg.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Modèle/scaler manquant pour {company}")
        missing_models = True

if missing_models:
    print("Modèles manquants détectés. Entraînement en cours via train_models_tp4.py...")
    try:
        subprocess.run(["python", "train_models_tp4.py"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'entraînement TP4")
        print("STDOUT :", e.stdout)
        print("STDERR :", e.stderr)
        raise
else:
    print("Tous les modèles et scalers sont disponibles.")

# Chargement des modèles et scalers
for company in companies:
    try:
        model_path = f"models/reg/best_model_{company}_reg.pkl"
        scaler_path = f"models/reg/scaler_{company}_reg.pkl"
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        models_reg[company] = model
        scalers_reg[company] = scaler
    except Exception as e:
        print(f"Erreur lors du chargement pour {company} : {e}")

print(f"\n{len(models_reg)} modèles chargés avec succès")

print("Partie 5 : Réseaux de neurones (Deep Learning)\n")
# Dictionnaires pour stocker les modèles et scalers
models_dl = {}
scalers_dl = {}

# Vérifie si tous les fichiers existent, sinon lance l'entraînement
missing_models = False
for company in companies:
    model_path = f"models/dl/best_model_{company}_DL.h5"
    scaler_path = f"models/dl/scaler_{company}_DL.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Modèle/scaler manquant pour {company}")
        missing_models = True

if missing_models:
    print("Modèles manquants détectés. Entraînement en cours via train_models_tp5.py...")
    subprocess.run(["python", "train_models_tp5.py"], check=True)
else:
    print("Tous les modèles et scalers deep learning sont disponibles.")

# Chargement des modèles et scalers
for company in companies:
    try:
        model_path = f"models/dl/best_model_{company}_DL.h5"
        scaler_path = f"models/dl/scaler_{company}_DL.pkl"
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        models_dl[company] = model
        scalers_dl[company] = scaler
    except Exception as e:
        print(f"Erreur lors du chargement pour {company} : {e}")

print(f"\n{len(models_dl)} modèles DL chargés avec succès")

print("\nPartie 5bis : Comparaison TP4 vs TP5 (RMSE) et sélection du meilleur modèle\n")

rmse_path = "models/rmse_summary.csv"
best_models = {}

if not os.path.exists(rmse_path):
    print("Fichier rmse_summary.csv introuvable. Assure-toi que TP4 et TP5 ont bien exécuté leurs sauvegardes.")
else:
    df = pd.read_csv(rmse_path)

    for company in companies:
        subset = df[df["symbol"] == company]
        if subset.empty or subset["rmse"].isnull().any():
            print(f"Données incomplètes pour {company}")
            continue

        best_row = subset.loc[subset["rmse"].idxmin()]
        source = best_row["source"]

        if source == "TP4":
            model = models_reg.get(company)
            scaler = scalers_reg.get(company)
        elif source == "TP5":
            model = models_dl.get(company)
            scaler = scalers_dl.get(company)
        else:
            print(f"Source inconnue pour {company}")
            continue

        if model is None or scaler is None:
            print(f"Modèle ou scaler introuvable pour {company} ({source})")
            continue

        best_models[company] = {
            "source": source,
            "model": model,
            "scaler": scaler,
            "rmse": best_row["rmse"]
        }

print(f"\n{len(best_models)} modèles finaux sélectionnés")

print("Partie 6 : Scrapping des NEWS\n")
companies = [
    "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla",
    "NVIDIA", "Samsung", "Tencent","Alibaba", "IBM", "Intel",
    "Oracle", "Sony", "Adobe", "Netflix", "AMD", "Qualcomm",
    "Cisco", "JP Morgan", "Goldman Sachs", "Visa", "Johnson & Johnson",
    "Pfizer", "ExxonMobil", "ASML", "SAP", "Siemens", "LVMH",
    "TotalEnergies", "Shell", "Baidu", "JD.com", "BYD", "ICBC", "Toyota",
    "SoftBank", "Nintendo", "Hyundai", "Reliance Industries", "TCS"
]

for company in companies:
    try:
        TP6.get_news_by_date(company_name = company, api_key = "2046926da8c04dbaa796781be0bf1550")  # ou get_news_for_company
    except Exception as e:
        print(f"Erreur pour {company} : {e}")

print(f"\nRécupération des news réalisée\n")


print("\nPartie 7 : Analyse de sentiment LLM\n")

MODEL_DIR = "models/LLM"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_config.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# 1. Vérification et entraînement si besoin
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    print("Modèle LLM non trouvé. Lancement de l'entraînement via train_models_tp7.py...")
    subprocess.run(["python", "train_models_tp7.py"], check=True)
else:
    print("Modèle LLM fine-tuné détecté.")

# 2. Chargement du modèle et tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

print(f"\nRécupération du modèle LLM réalisée\n")

print("\nPartie 8 : Analyse de sentiment LLM\n")

sentiment_history = {}

for company in companies:
    try:
        file_path = f"news/{company}_news.json"
        if not os.path.exists(file_path):
            print(f"Fichier introuvable pour {company}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        news = []
        for date, articles in data.items():
            for article in articles:
                article["date"] = date
                news.append(article)

        today = datetime.today()
        cutoff_date = today - timedelta(days=30)
        last_news = [n for n in news if "date" in n and datetime.strptime(n["date"], "%Y-%m-%d") >= cutoff_date]

        # Si moins de 5 news dans les 30 derniers jours, compléter avec les plus récentes
        if len(last_news) < 5:
            # Trier toutes les news (même au-delà de 30 jours) par date décroissante
            news_sorted = sorted(news, key=lambda x: x["date"], reverse=True)
            # Compléter avec des articles plus anciens
            for n in news_sorted:
                if n not in last_news:
                    last_news.append(n)
                if len(last_news) >= 5:
                    break

        texts = [item["title"] for item in last_news if "title" in item]
        dates = [item["date"] for item in last_news if "date" in item]

        if not texts or len(texts) != len(dates):
            print(f"News invalides pour {company}")
            continue

        os.makedirs("sentiments", exist_ok=True)
        preds = TP8.analyze_sentiments_by_company({company: texts}, tokenizer, model)[company]
        sentiment_labels = [ ["Négatif", "Neutre", "Positif"][p] for p in preds ]

        df = pd.DataFrame({
            "date": dates,
            "headline": texts,
            "sentiment_code": preds,
            "sentiment_label": sentiment_labels
        })

        sentiment_history[company] = df

        # sauvegarde CSV
        df.to_csv(f"sentiments/sentiments_{company}.csv", index=False)

    except Exception as e:
        print(f"Erreur pour {company} : {e}")

print(f"\nRécupération des sentiments réalisée\n")


print("\nPartie Complémantaire : Analyse technique 📈\n")

tech_signals = {}
companies = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "005930.KS",
    "TCEHY", "BABA", "IBM", "INTC", "ORCL", "SONY", "ADBE",
    "NFLX", "AMD", "QCOM", "CSCO", "JPM", "GS", "V", "JNJ", "PFE",
    "XOM", "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "TTE.PA",
    "SHEL.L", "BIDU", "JD", "BYDDY", "1398.HK", "TM", "9984.T", "NTDOY",
    "HYMTF", "RELIANCE.NS", "TCS.NS"
]

for company in companies:
    try:
        signals = TP_complementaire.technical_analysis(company)
        tech_signals[company] = signals
    except Exception as e:
        print(f"Erreur analyse technique pour {company} : {e}")

print("\nAnalyse technique effectué\n")

print("\nGénération des rapports PDF\n")
for ticker in companies:
    try:
        generate_pdf_for_company(
            ticker=ticker,
            best_models=best_models,
            model_classification=model_classification,
            scaler_classification=scaler_classification,
            tech_signals=tech_signals,
            clustering_risque=clustering_risque,
            clustering_rendement=clustering_rendement
        )
    except Exception as e:
        print(f"Erreur pour {ticker} : {e}")
print("\nRapport Générés\n")
