# ───────────────────────── TP8 – Classification de sentiment ─────────────────────────
# Objectif : utiliser le meilleur modèle fine-tuné (depuis TP7) pour prédire
# le sentiment (positif, neutre, négatif) des news par entreprise.
#
# Étapes :
#   1. Charger le modèle/tokenizer finetuné
#   2. Prédire les sentiments pour un ensemble de news
#   3. Renvoyer les résultats par entreprise
#
# ================================================================================
import torch
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# 1. Chargement du modèle/tokenizer fine-tuné (le meilleur selon TP7)
# ────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────
# 2. Prédiction du sentiment pour une liste de textes
# ────────────────────────────────────────────────────────────────────────
def predict_sentiment(texts, tokenizer, model, return_probs=False):
    """
    Prédit le sentiment pour une liste de textes.
    return_probs=True permet de renvoyer les probabilités au lieu des classes.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    if return_probs:
        return preds, probs
    return preds

# ────────────────────────────────────────────────────────────────────────
# 3. Application à un dictionnaire d'entreprise : {ticker: [news list]}
# ────────────────────────────────────────────────────────────────────────
def analyze_sentiments_by_company(news_dict, tokenizer, model):
    """
    news_dict : {"AAPL": ["news1", "news2", ...], "MSFT": [...], ...}
    Retourne un dict {ticker: [sentiment1, sentiment2, ...]}
    """
    results = {}
    for company, news_list in news_dict.items():
        if not news_list:
            results[company] = []
            continue
        sentiments = predict_sentiment(news_list, tokenizer, model)
        results[company] = sentiments.tolist()
    return results
