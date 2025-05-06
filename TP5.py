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

import os, glob, warnings
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
    """Transforme un vecteur 1-D (close) en X et y (valeur cible)."""
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i, 0])
        y.append(arr[i, 0])
    return np.array(X), np.array(y)

def build_dataset_reg(csv_path, window=30, test_size=0.2):
    """
    1.1  Pipeline dataset (exactement comme TP4) :
    """
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


# ────────────────────────────────────────────────────────────────────────
# 1.2  Modèles de deep-learning
# ───────────────────────────────────────────────────────────────────────

# ── 1.2.1  Création des modèles ──────────────────────────────────────────────
def build_mlp_model(input_shape,
                    hidden_dims=[64, 32],
                    dropout_rate=0.1,
                    activation="relu",
                    optimizer="adam",
                    learning_rate=1e-3):
    """ MLP """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        if dropout_rate:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.get(
                    {"class_name": optimizer, "config": {"learning_rate": learning_rate}}),
                  loss="mse")
    return model

def build_rnn_model(input_shape,
                    units=50,
                    dropout_rate=0.1,
                    activation="tanh",
                    optimizer="adam",
                    learning_rate=1e-3):
    """ RNN """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(units, activation=activation, dropout=dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get(
                    {"class_name": optimizer, "config": {"learning_rate": learning_rate}}),
                  loss="mse")
    return model

def build_lstm_model(input_shape,
                     units=50,
                     dropout_rate=0.1,
                     activation="tanh",
                     optimizer="adam",
                     learning_rate=1e-3):
    """ LSTM """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(units, activation=activation, dropout=dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get(
                    {"class_name": optimizer, "config": {"learning_rate": learning_rate}}),
                  loss="mse")
    return model

# ── 1.2.2  Entraînement des modeles  ───────────────────────────────────────────
def train_model(model_type, X_train, y_train, epochs=10, batch_size=32, **params):
    """
    Sélectionne MLP ou RNN ou LSTM, adapte X_train selon
  ce dernier, entraîne et retourne le model
    """
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

# ── 1.2.3  Prédiction & métriques ───────────────────────────────────────────
def predict_model(model, model_type, X_test, y_test, scaler, window):
    """Calcule MAE/RMSE, affiche 10 premières prédictions et trace le graphique"""
    X_te = X_test if model_type == "MLP" else X_test[..., np.newaxis]
    y_pred_scaled = model.predict(X_te, verbose=0).ravel()

    # inversion de l’échelle
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

    # Graphique
    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="Réel", color="red")
    plt.plot(y_pred, label="Préd.", color="blue")
    plt.title(f"{model_type} – Close réel vs prédiction")
    plt.legend()
    plt.show()

    return mae


# ── 1.2.4  Comparaison ──────────────────────────────────────────────────────
def compare_models(symbol="AAPL", window=30, test_size=0.2,
                   epochs=8, batch_size=32):
    """
    * Construit le dataset
    * Définit une grille pour MLP, RNN, LSTM
    * Garde le meilleur (+ petit MAE) pour chaque type
    * Affiche les res + tableau de synthèse
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
            mae = predict_model(mdl, mtype, X_te, y_te, scaler, window)
            if mae < best_mae:
                best_mae, best_mod, best_par = mae, mdl, p
        best[mtype] = (best_mod, best_par, best_mae)

    # Tableau comparatif
    df = pd.DataFrame(
        [{"Modèle":k, "MAE":v[2]} for k,v in best.items()]
    ).set_index("Modèle")
    print(f"\n==== Synthèse – {symbol} (fenêtre {window}j) ====")
    display(df)


# ────────────────────────────────────────────────────────────────────────
# 1.3  Test
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ticker in ["AAPL", "MSFT", "AMZN"]:
        compare_models(ticker, window=30, test_size=0.2,
                       epochs=5, batch_size=32)