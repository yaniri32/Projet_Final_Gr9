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
import ta # indicateurs techniques
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# 1.1 Création des labels pour chaque entreprise
def create_labels_one_company(csv_path):
    """
    1.1.1 Charger le CSV d'histo (colonnes : Close, etc)
    1.1.2 Crée Close_Horizon = Close.shift(-20) (horizon =20j)
    1.1.3 Calculer Horizon_Return = (Close_Horizon - Close)/Close
    1.1.4 Définir Label :
          2 = buy si > +5%
          1 = hold sinon,
          0 = sell si < -5%
    1.1.5 Ajouter colonne 'Symbol' = nom du ticker
    """
    symb = os.path.basename(csv_path).split("_")[0]
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    df["Close_Horizon"]  = df["Close"].shift(-20) # 1.1.2
    df["Horizon_Return"] = (df["Close_Horizon"] - df["Close"]) / df["Close"] # 1.1.3
    df["Label"] = np.where(df["Horizon_Return"] >  0.05, 2, # 1.1.4
                   np.where(df["Horizon_Return"] < -0.05, 0, 1))
    df["Symbol"] = symb # 1.1.5
    return df


# 1.1.2 Ajout des TA
def add_ta_features(df):
    """
    Pour chaque DF contenant 'Close' :
    - SMA20, EMA20
    - RSI14
    - MACD et MACD_Signal
    - Bollinger_High, Bollinger_Low
    - Rolling_Vol20 = volatilité réalisée sur 20j
    - ROC10 = Rate of Change sur 10j
    """
    close = df["Close"]
    df["SMA20"]          = ta.trend.sma_indicator(close, window=20)
    df["EMA20"]          = ta.trend.ema_indicator(close, window=20)
    df["RSI14"]          = ta.momentum.rsi(close, window=14)
    df["MACD"]           = ta.trend.macd(close)
    df["MACD_Signal"]    = ta.trend.macd_signal(close)
    boll                = ta.volatility.BollingerBands(close, window=20)
    df["Bollinger_High"] = boll.bollinger_hband()
    df["Bollinger_Low"]  = boll.bollinger_lband()
    df["Rolling_Vol20"]  = close.pct_change().rolling(window=20).std()
    df["ROC10"]          = ta.momentum.roc(close, window=10)
    return df


# 1.1.3 Construction du dataset multi-entreprises
def build_dataset(folder="historiques_entreprises"):
    """
    1. Parcourir tous les CSV du dossier
    2. Appliquer create_labels_one_company + add_ta_features
    3. Concaténer les DataFrames et supprimer les NaN
    4. Séparer en X (features TA) et y (Label)
    """
    frames = []
    for path in glob.glob(os.path.join(folder, "*.csv")): # 1.1.3.1
        df = create_labels_one_company(path)  # 1.1.3.2
        df = add_ta_features(df) # 1.1.3.2
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).dropna() # 1.1.3.3 & 1.1.3.4

    y = full["Label"].astype(int) # 1.1.3.4
    drop_cols = ["Label", "Close_Horizon", "Horizon_Return", "Symbol"]
    if "Next Day Close" in full.columns:
        drop_cols.append("Next Day Close")
    X = full.drop(columns=drop_cols, errors="ignore") # 1.1.3.4
    return X, y


# 1.1.4 Split train/test + StandardScaler
def train_test_scaled(X, y, test_size=0.2, random_state=42):
    """
    1. Séparation train/test
    2. Standardisation (fit sur train, transform sur test)
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler


# 1.2 Entraînement des modèles

# 1.2.1 Random Forest (RandomizedSearchCV)
def rf_classifier(X_tr, y_tr):
    """
    RandomizedSearchCV(n_iter=3, cv=2).
    """
    param_dist = {"n_estimators":[100,200], "max_depth":[10,20], "max_features":["sqrt"]}
    rs = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions=param_dist,
        n_iter=3,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42
    )
    rs.fit(X_tr, y_tr)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


# 1.2.2 XGBoost (GridSearchCV)
def xgb_classifier(X_tr, y_tr):
    """
    - GridSearchCV minimal sur n_estimators, max_depth, learning_rate
    - cv=2 pour rapidité car sinon bcp trop lent
    """
    grid = {"n_estimators":[300], "max_depth":[6], "learning_rate":[0.1]}
    gs = GridSearchCV(
        XGBClassifier(num_class=3, subsample=0.8, colsample_bytree=0.8, random_state=42),
        grid, cv=2, scoring="accuracy", n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# 1.2.3 KNN (paramètres fixes)
def knn_classifier(X_tr, y_tr):
    """
    - KNN avec k=15
    - Pas de recherche hyperparamètres car ça prend bcp trop de temps (+30m)
    """
    model = KNeighborsClassifier(n_neighbors=15, weights="distance")
    model.fit(X_tr, y_tr)
    return model, {}, None


# 1.2.4 SVM linéaire (LinearSVC)
def svm_classifier(X_tr, y_tr):
    """
    - SVM linéaire via LinearSVC pour la rapidité
    """
    model = LinearSVC(penalty="l2", loss="squared_hinge", dual=False,
                      C=1.0, max_iter=10000, random_state=42)
    model.fit(X_tr, y_tr)
    return model, {}, None


# 1.2.5 Régression logistique (par défaut)
def logreg_classifier(X_tr, y_tr):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    return model, {}, None


# 1.2.6 Reporting classification_report + accuracy
def run_and_report(name, func, X_tr, X_te, y_tr, y_te):
    """
    1. Entraînement du modèle via fct en question
    2. Prédiction sur X_te
    3. Affichage classification_report + accuracy
    4. Retour d’un dico de scores pour résumer
    """
    model, params, cv_score = func(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    print(f"\n──── {name} ────")
    if params: print("Meilleurs paramètres :", params)
    print(classification_report(y_te, y_pred, digits=3))
    if cv_score is not None:
        print(f"Accur. CV (approx.) : {cv_score:.3f}")
    print(f"Accur. test         : {acc:.3f}")

    return {"Modèle": name, "Acc_CV": cv_score, "Acc_test": acc}


# 1.3 Pipeline complet : appel des étapes et sauvegarde
def main_pipeline():
    # 1. Construction du dataset
    X, y = build_dataset() # 1.1 & 1.1.2 & 1.1.3
    # 2. Split + standardisation
    X_tr, X_te, y_tr, y_te, scaler = train_test_scaled(X, y) # 1.1.4

    # 3. Liste des algo à tester
    algos = [
        ("Random Forest", rf_classifier),
        ("XGBoost",       xgb_classifier),
        ("KNN",           knn_classifier),
        ("SVM",           svm_classifier),
        ("Régression log",logreg_classifier)
    ]

    # 4. Boucle train et reporting
    results = []
    for name, func in algos:
        results.append(run_and_report(name, func, X_tr, X_te, y_tr, y_te)) # 1.2 & 1.2.6

    # 5. Résumé des perfs
    df_res = pd.DataFrame(results).set_index("Modèle")
    print("\n==== Résumé des performances ====")
    display(df_res)

    # 6. Sauvegarde des meilleurs modèles
    for r in results:
        joblib.dump(r, f"best_{r['Modèle'].replace(' ', '_')}.pkl")

    return df_res


if __name__ == "__main__":
    df_perfs = main_pipeline() # 1.3

"""## Résumé des perfs des modeles

### 1. Random Forest
- **Hyperparamètres** :
  - n_estimators = 200
  - max_features = "sqrt"
  - max_depth = 20
- **Scores** :
  - Accuracy test : 0.672
  - Accuracy CV : 0.604 → écart de +0.068 (léger overfitting)
- **Par classe** :
  - Sell (0) : precision = 0.636, recall = 0.588, F1 = 0.611  
  - Hold (1) : precision = 0.688, recall = 0.746, F1 = 0.716  
  - Buy (2) : precision = 0.673, recall = 0.624, F1 = 0.647  
- **Interprétation** : modèle robuste avec un bon équilibre entre précision et rappel avec une légère variance.

---

### 2. XGBoost
- **Hyperparamètres** :
  - learning_rate = 0.1
  - max_depth = 6
  - n_estimators = 300
- **Scores** :
  - Accuracy test : 0.609
  - Accuracy CV : 0.576 → écart de +0.033
- **Par classe** :
  - Sell : precision = 0.636, recall = 0.379, F1 = 0.475  
  - Hold : precision = 0.589, recall = 0.823, F1 = 0.687  
  - Buy : precision = 0.650, recall = 0.457, F1 = 0.537  
- **Interprétation** : bon rappel sur la classe hold, mais déséquilibre important entre précision et rappel sur les classes sell et buy

---

### 3. KNN
- **Paramètres** :
  - k = 15, weights = "distance"
- **Accuracy test** : 0.475
- **Par classe** :
  - F1 “hold” ≃ 0.592  
  - F1 “sell” ≃ 0.290  
  - F1 “buy” ≃ 0.369  
- **Interprétation** : le modèle ne parvient pas à capturer les structures complexes du problème, en particulier avec un grand volume de données.

---

### 4. SVM Linéaire (LinearSVC)
- **Paramètres** :
  - C = 1, dual = False, max_iter = 10000
- **Accuracy test** : 0.483
- **Par classe** :
  - Très bon rappel sur “hold” (0.910)  
  - F1 très faibles sur “sell” et “buy” (≈ 0.17–0.19)  
- **Interprétation** : le séparateur linéaire n’est pas adapté à la complexité des classes

---

### 5. Régression Logistique
- **Paramètres** :
  - C = 1, class_weight = "balanced"
- **Accuracy test** : 0.463
- **Par classe** :
  - F1 “hold” ≃ 0.545  
  - F1 très faibles sur “sell” et “buy”  
- **Interprétation** : modèle linéaire simple, peu performant sur des relations non linéaires et des changements de tendance

---

## Synthèse

- Modèle le plus performant : Random Forest
- Classe la plus facilement identifiable : Hold
- Classes les plus difficiles à prédire : Sell et **Buy
"""