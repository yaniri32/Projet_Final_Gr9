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
    test_scaled = scaler.transform(test_close)

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
    grid = {"n_neighbors": [5, 10, 15, 20], "weights": ["uniform", "distance"]}
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
def train_and_evaluate(name, trainer, X_tr, X_te, y_tr, y_te, scaler, close_series, window, show_plot=False):
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

    if show_plot:
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
def regression_pipeline_one_company(symbol="AAPL", window=30, folder="historiques_entreprises", output_dir="models/reg"):
    """
    1. Construit un dataset pour une entreprise (symbol)
    2. Entraîne XGB, RF, KNN, LR
    3. Affiche métriques + graphiques
    4. Sauvegarde du meilleur modèle (par RMSE)
    """

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(folder, f"{symbol}_historique.csv")
    X_tr, X_te, y_tr, y_te, scaler = build_dataset_reg(csv_path, window)

    close_full = pd.read_csv(csv_path, index_col=0)["Close"].values

    algos = [
        ("XGBoost",       xgb_reg),
        ("Random Forest", rf_reg),
        ("KNN",           knn_reg),
        ("Régression linéaire", lin_reg)
    ]

    results = []
    models = []

    for name, trainer in algos:
        model_result = train_and_evaluate(name, trainer, X_tr, X_te, y_tr, y_te,
                                          scaler, close_full, window)
        model, _ = trainer(X_tr, y_tr)  # réentraîne pour sauvegarder
        results.append(model_result)
        models.append((name, model))

    # Résumé
    df_res = pd.DataFrame(results).set_index("Modèle")
    print("\n==== Récapitulatif –", symbol, "====")
    print(df_res.to_string())

    # Sélection du meilleur modèle selon RMSE
    best_model_name = df_res["RMSE"].idxmin()
    best_trainer = dict(algos)[best_model_name]

    # Reprise des données complètes (sans split) pour réentraînement
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    close = df["Close"].shift(-1).dropna().values.reshape(-1, 1)
    scaler_full = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler_full.fit_transform(close)

    X_full, y_full = create_target_features(scaled_full, window)
    X_full, y_full = X_full, y_full.ravel()

    # Entraînement final sur toutes les données
    final_model, _ = best_trainer(X_full, y_full)

    # Prédiction finale sur tout le dataset
    y_pred_scaled = final_model.predict(X_full).reshape(-1, 1)
    y_pred = scaler_full.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_full.inverse_transform(y_full.reshape(-1, 1)).ravel()

    # Calcul de la RMSE finale
    rmse_final = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nRMSE finale sur l'ensemble du dataset : {rmse_final:.4f}")

    # Sauvegarde dans le fichier résumé
    summary_path = "models/rmse_summary.csv"
    entry = {
        "symbol": symbol,
        "source": "TP4",
        "model": best_model_name,
        "rmse": rmse_final
    }

    # Mise à jour du fichier résumé
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = df_summary[~((df_summary["symbol"] == symbol) & (df_summary["source"] == entry["source"]))]
        df_summary = pd.concat([df_summary, pd.DataFrame([entry])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([entry])

    df_summary.to_csv(summary_path, index=False)
    print(f"Résumé mis à jour dans {summary_path}")

    # Sauvegarde
    model_path = os.path.join(output_dir, f"best_model_{symbol}_reg.pkl")
    scaler_path = os.path.join(output_dir, f"scaler_{symbol}_reg.pkl")
    joblib.dump(final_model, model_path)
    joblib.dump(scaler_full, scaler_path)

    print(f"\nMeilleur modèle ({best_model_name}) réentraîné sur tout le dataset → sauvegardé sous {model_path}")
    print(f"\nScaler complet sauvegardé sous : {scaler_path}")
