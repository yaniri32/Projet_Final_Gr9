import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta


# TP1 – Extraction et traitement de données financières

# 1.1 Scraping des ratios financiers
# Pipeline :
# 1. Initialiser le dico des sociétés et le dico des ratios
# 2. Boucle sur chaque société pour récupérer les ratios via yfinance
# 3. Construire un df et save en CSV
def scrape_ratios():
    # 1. Initialisation du dico des sociétés et des ratios
    companies = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
        "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA",
        "NVIDIA": "NVDA", "Samsung": "005930.KS", "Tencent": "TCEHY",
        "Alibaba": "BABA", "IBM": "IBM", "Intel": "INTC",
        "Oracle": "ORCL", "Sony": "SONY", "Adobe": "ADBE",
        "Netflix": "NFLX", "AMD": "AMD", "Qualcomm": "QCOM",
        "Cisco": "CSCO", "JP Morgan": "JPM", "Goldman Sachs": "GS",
        "Visa": "V", "Johnson & Johnson": "JNJ", "Pfizer": "PFE",
        "ExxonMobil": "XOM", "ASML": "ASML.AS", "SAP": "SAP.DE",
        "Siemens": "SIE.DE", "LVMH": "MC.PA", "TotalEnergies": "TTE.PA",
        "Shell": "SHEL.L", "Baidu": "BIDU", "JD.com": "JD",
        "BYD": "BYDDY", "ICBC": "1398.HK", "Toyota": "TM",
        "SoftBank": "9984.T", "Nintendo": "NTDOY", "Hyundai": "HYMTF",
        "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS"
    }
    dict_ratios = {r: [] for r in [
        "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
        "trailingEps", "debtToEquity", "currentRatio", "quickRatio",
        "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
    ]}

    # 2. Boucle pour récup les ratios pour chaque société
    for nom, symb in companies.items():
        info = yf.Ticker(symb).info or {}

        for r in dict_ratios:
            dict_ratios[r].append(info.get(r))

    # 3. Construction du df et export en CSV
    os.makedirs("historiques_entreprises", exist_ok=True)
    df = pd.DataFrame(dict_ratios, index=companies.keys())
    df.index.name = "Société"
    fichier = "historiques_entreprises/ratios_financiers.csv"
    df.to_csv(fichier)
    print(f"\nRatios financiers sauvegardés dans {fichier}")


# 1.2 Scraping de l’historique des cours (5 ans)
# Pipeline :
# 1. Créer le dossier de sortie
# 2. Définir date_debut et date_fin (5 ans)
# 3. DL les données "Close" pour chaque société
# 4. Calculer "Close_Lendemain" et "Rendement" à partir de Series 1-D

def scrape_historical(histo_dir="historiques_entreprises"):
    # 1. Création du dossier de sortie
    companies = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
        "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA",
        "NVIDIA": "NVDA", "Samsung": "005930.KS", "Tencent": "TCEHY",
        "Alibaba": "BABA", "IBM": "IBM", "Intel": "INTC",
        "Oracle": "ORCL", "Sony": "SONY", "Adobe": "ADBE",
        "Netflix": "NFLX", "AMD": "AMD", "Qualcomm": "QCOM",
        "Cisco": "CSCO", "JP Morgan": "JPM", "Goldman Sachs": "GS",
        "Visa": "V", "Johnson & Johnson": "JNJ", "Pfizer": "PFE",
        "ExxonMobil": "XOM", "ASML": "ASML.AS", "SAP": "SAP.DE",
        "Siemens": "SIE.DE", "LVMH": "MC.PA", "TotalEnergies": "TTE.PA",
        "Shell": "SHEL.L", "Baidu": "BIDU", "JD.com": "JD",
        "BYD": "BYDDY", "ICBC": "1398.HK", "Toyota": "TM",
        "SoftBank": "9984.T", "Nintendo": "NTDOY", "Hyundai": "HYMTF",
        "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS"
    }

    os.makedirs(histo_dir, exist_ok=True)

    # 2. Définition de la période (aujourd'hui - 5 ans)
    date_fin = datetime.today()
    date_debut = date_fin - relativedelta(years=5)

    # 3. Téléchargement et calcul pour chaque société
    for nom, symb in companies.items():
        data = yf.download(
            symb,
            start=date_debut.strftime("%Y-%m-%d"),
            end=date_fin.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False
        )
        if data.empty:
            print(f"Aucune donnée pour {nom} ({symb})")
            continue

        # 4a. Récupération des Series 1-D de clôture
        close = data["Close"]
        close_next = close.shift(-1)
        rendement = (close_next - close) / close

        # 4b. Assemblage via concat et renommage des colonnes
        df_hist = pd.concat([close, close_next, rendement], axis=1)
        df_hist.columns = ["Close", "Close_Lendemain", "Rendement"]

        # 5. Sauvegarde du CSV
        chemin = os.path.join(histo_dir, f"{symb}_historique.csv")
        df_hist.to_csv(chemin)

    print(f"\nHistorique sauvegardé dans le dossier : {chemin}")
