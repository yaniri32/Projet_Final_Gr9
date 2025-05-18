# ───────────────────────── TP 5 – Réseaux de neurones ─────────────────────────
# Objectif : prédire le cours d'action à J+1 avec MLP, RNN et  LSTM
#
#  1.1  Création du dataset  (on reprend qq fonctions du TP4)
#  1.2  Modèles de deep-learning
#       1.2.1  build_mlp_model
#       1.2.1  build_rnn_model
#       1.2.1  build_lstm_model
#       1.2.2  train_model
#       1.2.3  predict_model
#       1.2.4  compare_models
# ------------------------------------------------------------------------------

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ────────────────────────────────────────────────────────────────────────
# 1.1  Création du dataset pour la régression (repris du TP4)
# ────────────────────────────────────────────────────────────────────────
def _window_Xy(arr, window=30):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i, 0])
        y.append(arr[i, 0])
    return np.array(X), np.array(y)

def build_dataset_reg(csv_path, window=30, test_size=0.2):
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    close = df["Close"].values.reshape(-1, 1)

    split = int(len(close) * (1 - test_size))
    close_train, close_test = close[:split], close[split:]

    scaler = MinMaxScaler()
    close_train = scaler.fit_transform(close_train)
    close_test  = scaler.transform(close_test)

    X_train, y_train = _window_Xy(close_train, window)
    X_test,  y_test  = _window_Xy(close_test,  window)
    return X_train, X_test, y_train.ravel(), y_test.ravel(), scaler

# ───────────────────────────────────────────────────────────────────────
# 1.2  Modèles de deep-learning
# ───────────────────────────────────────────────────────────────────────

def build_mlp_model(input_shape, hidden_dims=[64, 32], dropout_rate=0.1, activation="relu", optimizer="adam", learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        if dropout_rate:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}}), loss="mse")
    return model

def build_rnn_model(input_shape, units=50, dropout_rate=0.1, activation="tanh", optimizer="adam", learning_rate=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(units, activation=activation, dropout=dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}}), loss="mse")
    return model

def build_lstm_model(input_shape, units=50, dropout_rate=0.1, activation="tanh", optimizer="adam", learning_rate=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(units, activation=activation, dropout=dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}}), loss="mse")
    return model

def train_model(model_type, X_train, y_train, epochs=10, batch_size=32, **params):
    if model_type == "MLP":
        model = build_mlp_model(input_shape=(X_train.shape[1],), **params)
        X_tr = X_train
    elif model_type == "RNN":
        model = build_rnn_model(input_shape=(X_train.shape[1], 1), **params)
        X_tr = X_train[..., np.newaxis]
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape=(X_train.shape[1], 1), **params)
        X_tr = X_train[..., np.newaxis]
    else:
        raise ValueError("model_type doit être MLP ou RNN ou LSTM")

    model.fit(X_tr, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_model(model, model_type, X_test, y_test, scaler, show_plot=False):
    X_te = X_test if model_type == "MLP" else X_test[..., np.newaxis]
    y_pred_scaled = model.predict(X_te, verbose=0).ravel()

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n──── {model_type} – Évaluation ────")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("\nPremières prédictions :")
    for i in range(10):
        print(f"{i+1:2d}  préd = {y_pred[i]:.2f}   réel = {y_true[i]:.2f}")

    if show_plot:
        plt.figure(figsize=(10,4))
        plt.plot(y_true, label="Réel", color="red")
        plt.plot(y_pred, label="Préd.", color="blue")
        plt.title(f"{model_type} – Close réel vs prédiction")
        plt.legend()
        plt.show()

    return mae

def compare_models(symbol="AAPL", window=30, test_size=0.2,
                   epochs=8, batch_size=32):
    """
    * Construit le dataset
    * Définit une grille pour MLP, RNN, LSTM
    * Garde le meilleur (+ petit MAE) pour chaque type
    * Affiche les résultats + tableau de synthèse
    * Sauvegarde le meilleur modèle (tous types confondus) et le scaler
    """
    path = f"historiques_entreprises/{symbol}_historique.csv"
    X_tr, X_te, y_tr, y_te, scaler = build_dataset_reg(path, window, test_size)

    configs = {
        "MLP":  [
            {"hidden_dims":[64,32],  "dropout_rate":0.1, "learning_rate":1e-3},
            {"hidden_dims":[128,64], "dropout_rate":0.2, "learning_rate":5e-4}
        ],
        "RNN": [
            {"units":50,  "dropout_rate":0.1, "learning_rate":1e-3},
            {"units":100, "dropout_rate":0.2, "learning_rate":5e-4}
        ],
        "LSTM":[
            {"units":50,  "dropout_rate":0.1, "learning_rate":1e-3},
            {"units":100, "dropout_rate":0.2, "learning_rate":5e-4}
        ]
    }

    best = {}  # model_type : (model, params, mae)
    for mtype, param_list in configs.items():
        best_mae = np.inf
        best_mod, best_par = None, None
        for p in param_list:
            print(f"\n=== {mtype} – entraînement avec {p} ===")
            mdl = train_model(mtype, X_tr, y_tr,
                              epochs=epochs, batch_size=batch_size, **p)
            mae = predict_model(mdl, mtype, X_te, y_te, scaler)
            if mae < best_mae:
                best_mae, best_mod, best_par = mae, mdl, p
        best[mtype] = (best_mod, best_par, best_mae)

    # Sélection du meilleur modèle tous types confondus
    best_model_type = min(best, key=lambda k: best[k][2])
    best_model, best_params, best_mae = best[best_model_type]

    # Sauvegarde du modèle
    output_dir = os.path.abspath("models/dl")
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"models/best_model_{symbol}_DL.h5"
    scaler_path = f"models/scaler_{symbol}_DL.pkl"
    best_model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Prédiction finale sur tout le dataset (DL)
    X_full = X_tr if best_model_type == "MLP" else X_tr[..., np.newaxis]
    y_pred_scaled = best_model.predict(X_full, verbose=0).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_tr.reshape(-1, 1)).ravel()

    # Calcul de la RMSE finale
    rmse_final = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nRMSE finale (DL) sur l'ensemble du dataset : {rmse_final:.4f}")

    # Sauvegarde dans le fichier résumé
    summary_path = "models/rmse_summary.csv"
    entry = {
        "symbol": symbol,
        "source": "TP5",
        "model": best_model_type,
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

    print(f"\nMeilleur modèle ({best_model_type}, MAE = {best_mae:.4f}) sauvegardé sous : {model_path}")
    print(f"\nScaler sauvegardé sous : {scaler_path}")
