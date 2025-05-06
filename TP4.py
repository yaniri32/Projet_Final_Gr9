# ───────────────────────── TP 4 – Régression ─────────────────────────
# Objectif : prédire le Close à J+1 pour chaque corpo
#   1.1  Création du dataset
#        1.1.1  Charger le CSV histo du TP1 et ne prendre que Close
#        1.1.2  MinMaxScaler + split train/test
#        1.1.3  Création X, y sur les 30 derniers jours
#        1.1.4  Retourner X_train, X_test, y_train, y_test, scaler
#   1.2  Algorithmes de régression
#        1.2.1  XGBoost
#        1.2.2  Random Forest
#        1.2.3  KNN
#        1.2.4  Régression linéaire
#        Pour chaque modèle :
#            – entraîner, afficher MAE / RMSE
#            – tracer prédictions vs valeurs reelles
#   1.3  Tableau récapitulatif des perfs

import glob, os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────────────
# 1.1  Fonctions utilitaires – création X, y
# ────────────────────────────────────────────────────────────────────────
def create_target_features(arr, n=30):
    """
    ↳ 1.1.3  Fonction générique : transforme un vecteur 1-D (close)
      en matrice X (fenêtre de n valeurs précédentes) et vecteur y
      (close à J)
    """
    x, y = [], []
    for i in range(n, len(arr)):
        x.append(arr[i-n:i, 0])
        y.append(arr[i, 0])
    return np.array(x), np.array(y)


def build_dataset_reg(csv_path, window=30, test_size=0.2):
    """
    ↳ 1.1  Crée et retourne :
        X_train, X_test, y_train, y_test, scaler
    Étapes :
      1. Charger CSV et extraire col Close
      2. Reshape pour MinMaxScaler (fit sur train)
      3. Split train/test
      4. Appel create_target_features sur chaque split
    """
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    close = df["Close"].values.reshape(-1, 1)

    # 1.1.2  MinMaxScaler sur [0,1] (fit sur train uniquement)
    split_idx = int(len(close) * (1 - test_size))
    train_close, test_close = close[:split_idx], close[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_close)
    test_scaled  = scaler.transform(test_close)

    # 1.1.3  Fenêtre temporelle
    X_train, y_train = create_target_features(train_scaled, window)
    X_test,  y_test  = create_target_features(test_scaled,  window)

    # Reshape pour XGB / RF
    return X_train, X_test, y_train.ravel(), y_test.ravel(), scaler


# ────────────────────────────────────────────────────────────────────────
# 1.2  Modèles de régression
# ────────────────────────────────────────────────────────────────────────
def xgb_reg(X_tr, y_tr):
    """
    ↳ 1.2.1  XGBoostRegressor avec grid réduit (cv=2) pour rapiditer
    """
    grid = {
        "n_estimators": [300, 500],
        "max_depth":    [3, 6],
        "learning_rate": [0.05, 0.1]
    }
    gs = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        grid, cv=2, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def rf_reg(X_tr, y_tr):
    """
    ↳ 1.2.2  RandomForestRegressor (grid réduit, cv=2) meme raison
    """
    grid = {
        "n_estimators": [200, 400],
        "max_depth":    [None, 20]
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        grid, cv=2, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def knn_reg(X_tr, y_tr):
    """
    ↳ 1.2.3  KNeighborsRegressor (grid réduit, cv=2) meme raison
    """
    grid = {"n_neighbors": [5, 10, 15], "weights": ["uniform", "distance"]}
    gs = GridSearchCV(
        KNeighborsRegressor(),
        grid, cv=2, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def lin_reg(X_tr, y_tr):
    """
    ↳ 1.2.4  Régression linéaire
    """
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model, {}


# ────────────────────────────────────────────────────────────────────────
# 1.2.5  Fonction générique : entraînement + reporting
# ────────────────────────────────────────────────────────────────────────
def train_and_evaluate(name, trainer, X_tr, X_te, y_tr, y_te, scaler, close_series, window):
    model, params = trainer(X_tr, y_tr)

    # Prédiction et remise à l’échelle
    y_pred_scaled = model.predict(X_te)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n──── {name} ────")
    if params: print("Meilleurs paramètres :", params)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # tracé comparatif
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(close_series)), close_series, color="red", label="Valeurs réelles")
    ax.plot(
        range(len(y_tr)+window, len(y_tr)+window+len(y_pred)),
        y_pred, color="blue", label=f"{name} préd."
    )
    ax.set_title(f"{name} – Close réel vs prédiction")
    ax.legend()
    plt.show()

    return {"Modèle": name, "MAE": mae, "RMSE": rmse}


# ────────────────────────────────────────────────────────────────────────
# 1.3  Pipeline principal pour 1 entreprise
# ────────────────────────────────────────────────────────────────────────
def regression_pipeline_one_company(symbol="AAPL", window=30, folder="historiques_entreprises"):
    """
    #1.3
    1. Construit un dataset pour une entreprise (symbol)
    2. Entraîne XGB, RF, KNN, LR
    3. Affiche métriques + graphiques + retourne tableau
    """
    csv_path = os.path.join(folder, f"{symbol}_historique.csv")
    X_tr, X_te, y_tr, y_te, scaler = build_dataset_reg(csv_path, window)

    # Ferme le scaler inverse pour graphe complet
    close_full = pd.read_csv(csv_path, index_col=0)["Close"].values

    algos = [
        ("XGBoost",       xgb_reg),
        ("Random Forest", rf_reg),
        ("KNN",           knn_reg),
        ("Régression linéaire", lin_reg)
    ]

    results = []
    for name, trainer in algos:
        results.append(
            train_and_evaluate(name, trainer, X_tr, X_te, y_tr, y_te,
                               scaler, close_full, window)
        )

    # Tableau récapitulatif
    df_res = pd.DataFrame(results).set_index("Modèle")
    print("\n==== Récapitulatif –", symbol, "====")
    display(df_res)
    return df_res


# ────────────────────────────────────────────────────────────────────────
# Lancer la pipeline sur plusieurs grosses entreprises 
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ptit test
    companies_demo = ["AAPL", "MSFT", "AMZN"]
    for sym in companies_demo:
        regression_pipeline_one_company(sym)