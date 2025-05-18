# ───────────────────────── TP 3 – Classification ─────────────────────────
# Objectif : prédire une consigne buy / hold / sell sur 20j
# Pipeline proposé:
# 1.1 Création des labels
#   1.1.1 Charger l’historique
#   1.1.2 Calcul horizon de 20 jours
#   1.1.3 Définir Label {0,1,2}
#   1.1.4 Ajouter colonne Symbol
# 1.1.2 Ajout des TA
# 1.1.3 Construction du dataset multi-entreprises
# 1.1.4 Split train/test + StandardScaler
# 1.2 Entraînement des modèles
#   1.2.1 Random Forest
#   1.2.2 XGBoost
#   1.2.3 KNN
#   1.2.4 SVM linéaire
#   1.2.5 Régression logistique
# 1.2.6 Reporting classification_report + accuracy
# 1.3 Résumé des performances + sauvegarde

import glob, os, warnings, joblib
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 1.1 Création des labels pour chaque entreprise
def create_labels_one_company(csv_path):
    symb = os.path.basename(csv_path).split("_")[0]
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    df["Close_Horizon"] = df["Close"].shift(-20)
    df["Horizon_Return"] = (df["Close_Horizon"] - df["Close"]) / df["Close"]
    df["Label"] = np.where(df["Horizon_Return"] > 0.05, 2,
                    np.where(df["Horizon_Return"] < -0.05, 0, 1))
    df["Symbol"] = symb
    return df

# 1.1.2 Ajout des TA
def add_ta_features(df):
    close = df["Close"]
    df["SMA20"] = ta.trend.sma_indicator(close, window=20)
    df["EMA20"] = ta.trend.ema_indicator(close, window=20)
    df["RSI14"] = ta.momentum.rsi(close, window=14)
    df["MACD"] = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    boll = ta.volatility.BollingerBands(close, window=20)
    df["Bollinger_High"] = boll.bollinger_hband()
    df["Bollinger_Low"] = boll.bollinger_lband()
    df["Rolling_Vol20"] = close.pct_change().rolling(window=20).std()
    df["ROC10"] = ta.momentum.roc(close, window=10)
    return df

# 1.1.3 Construction du dataset multi-entreprises
def build_dataset(folder="historiques_entreprises"):
    frames = []
    for path in glob.glob(os.path.join(folder, "*.csv")):
        df = create_labels_one_company(path)
        df = add_ta_features(df)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).dropna()

    y = full["Label"].astype(int)
    drop_cols = ["Label", "Close_Horizon", "Horizon_Return", "Symbol"]
    if "Next Day Close" in full.columns:
        drop_cols.append("Next Day Close")
    X = full.drop(columns=drop_cols, errors="ignore")
    return X, y

# 1.1.4 Split train/test + StandardScaler
def train_test_scaled(X, y, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler

# 1.2.1 Random Forest
rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "max_features": ["sqrt"]
}

def rf_classifier(X_tr, y_tr):
    gs = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42),
                      rf_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# 1.2.2 XGBoost
xgb_grid = {
    "n_estimators": [300],
    "max_depth": [6],
    "learning_rate": [0.1]
}

def xgb_classifier(X_tr, y_tr):
    gs = GridSearchCV(XGBClassifier(num_class=3, subsample=0.8, colsample_bytree=0.8, random_state=42),
                      xgb_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# 1.2.3 KNN
knn_grid = {
    "n_neighbors": [5, 10, 15, 20],
    "weights": ["uniform", "distance"]
}

def knn_classifier(X_tr, y_tr):
    gs = GridSearchCV(KNeighborsClassifier(), knn_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# 1.2.4 SVM
svm_grid = {"C": [0.1, 1, 10]}

def svm_classifier(X_tr, y_tr):
    gs = GridSearchCV(SVC(kernel="linear", max_iter=10000, class_weight="balanced"),
                      svm_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# 1.2.5 LogReg
logreg_grid = {"C": [0.1, 1, 10]}

def logreg_classifier(X_tr, y_tr):
    gs = GridSearchCV(LogisticRegression(max_iter=1000, class_weight="balanced"),
                      logreg_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# 1.2.6 Reporting

def run_and_report(name, func, X_tr, X_te, y_tr, y_te):
    model, params, cv_score = func(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n──── {name} ────")
    if params: print("Meilleurs paramètres :", params)
    print(classification_report(y_te, y_pred, digits=3))
    print(f"Accur. test         : {acc:.3f}")
    if cv_score is not None:
        print(f"Accur. CV (approx.) : {cv_score:.3f}")

    return {"Modèle": name, "Acc_CV": cv_score, "Acc_test": acc, "model": model}

# 1.3 Pipeline principal
def main_pipeline(folder="historiques_entreprises", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    X, y = build_dataset(folder)
    X_tr, X_te, y_tr, y_te, scaler = train_test_scaled(X, y)

    algos = [
        ("Random Forest", rf_classifier),
        ("XGBoost", xgb_classifier),
        ("KNN", knn_classifier),
        ("SVM", svm_classifier),
        ("Régression log", logreg_classifier)
    ]

    results = []
    best_model_func = None
    best_cv_score = -np.inf

    for name, func in algos:
        result = run_and_report(name, func, X_tr, X_te, y_tr, y_te)
        results.append(result)

        if result["Acc_CV"] is not None and result["Acc_CV"] > best_cv_score:
            best_cv_score = result["Acc_CV"]
            best_model_func = func

    df_res = pd.DataFrame(results).set_index("Modèle")
    print("\n==== Résumé des performances ====")
    print(df_res.to_string())


    if best_model_func is not None:
        full_scaler = StandardScaler().fit(X)
        full_scaled = full_scaler.transform(X)
        retrained_model, _, _ = best_model_func(full_scaled, y)
        joblib.dump(retrained_model, os.path.join(output_dir, "best_model_classification.pkl"))
        joblib.dump(full_scaler, os.path.join(output_dir, "scaler_classification.pkl"))
        print(f"\nMeilleur modèle réentraîné sur l'ensemble du dataset → sauvegardé sous best_model_classification.pkl")
        print(f"\nScaler complet sauvegardé sous : scaler_classification.pkl")
    else:
        raise ValueError("\nLe meilleur modèle n'a pas pu être défini")

