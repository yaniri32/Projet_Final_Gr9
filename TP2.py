# ───────────────────────── TP 2 – Clustering ─────────────────────────

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


# ───────────────────────────
# 1) Profils financiers – K-Means
# ───────────────────────────
# Pipeline :
# 1.1 Charger ratios_financiers.csv
# 1.2 Sélectionner colonnes de performance
# 1.3 Supprimer colonnes entièrement vides
# 1.4 Imputer NaN par moyenne de chaque colonne
# 1.5 Supprimer colonnes à variance nulle
# 1.6 Standardiser
# 1.7 Méthode du coude pour choisir K
# 1.8 Appliquer KMeans et ajouter labels au DataFrame
# 1.9 Visualiser avec t-SNE

def cluster_financial_profiles(df_ratios, k=4, show_plots=False):
    # 1.1 & 1.2
    cols_perf = [
        "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
        "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
    ]
    df_fin = df_ratios[cols_perf].copy()

    # 1.3
    df_fin.dropna(axis=1, how="all", inplace=True)

    # 1.4
    df_fin.fillna(df_fin.mean(), inplace=True)

    # 1.5
    df_fin = df_fin.loc[:, df_fin.nunique() > 1]

    # 1.6
    scaler = StandardScaler()
    X = scaler.fit_transform(df_fin)

    # 1.7
    if show_plots:
        inertias = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(1, 9)]
        plt.plot(range(1, 9), inertias, "o-")
        plt.title("Méthode du coude – Profils financiers")
        plt.show()

    # 1.8
    model = KMeans(n_clusters=k, random_state=42).fit(X)
    df_fin["cluster_financier"] = model.labels_

    # 1.9
    if show_plots:
        vis = TSNE(perplexity=10, random_state=42).fit_transform(X)
        for c in range(k):
            plt.scatter(vis[model.labels_ == c, 0], vis[model.labels_ == c, 1], label=f"Cluster {c}")
        plt.title("t-SNE – Profils financiers")
        plt.legend()
        plt.show()

    return df_fin, X, model.labels_

# ───────────────────────────
# 2) Profils de risque – Hiérarchique
# ───────────────────────────
# Pipeline :
# 2.1 Sélection des ratios de risque
# 2.2 Imputer NaN par moyenne de la colonne
# 2.3 Supprimer colonnes à variance nulle
# 2.4 Standardiser
# 2.5 Appliquer AgglomerativeClustering
# 2.6 Tracer dendrogramme

def cluster_risk_profiles(df_ratios, n_clusters=3, show_plots=False):
    # 2.1 & 2.2
    cols_risk = ["debtToEquity", "currentRatio", "quickRatio"]
    df_risk = df_ratios[cols_risk].copy()
    df_risk.fillna(df_risk.mean(), inplace=True)

    # 2.3
    df_risk = df_risk.loc[:, df_risk.nunique() > 1]

    # 2.4
    X = StandardScaler().fit_transform(df_risk)

    # 2.5
    model = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    df_risk["cluster_risque"] = model.labels_

    # 2.6
    if show_plots:
        linked = linkage(X, method="ward")
        plt.figure(figsize=(10, 4))
        dendrogram(linked, labels=df_risk.index)
        plt.title("Dendrogramme – Profils de risque")
        plt.tight_layout()
        plt.show()

    return df_risk, X, model.labels_


# ───────────────────────────
# 3) Corrélations de rendements – Hiérarchique
# ───────────────────────────
# Pipeline :
# 3.1 Charger historiques_entreprises/*.csv
# 3.2 Construire DataFrame des rendements journaliers
# 3.3 Imputer NaN par moyenne de chaque série
# 3.4 Calculer matrice 1−corrélation
# 3.5 Appliquer clustering hiérarchique et tracer dendrogramme
# 3.6 Extraire labels de clusters

def cluster_return_correlations(histo_dir="historiques_entreprises", n_clusters=4, show_plots=False):
    # 3.1 & 3.2
    data = {}
    for path in glob.glob(f"{histo_dir}/*.csv"):
        name = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path, index_col=0)
        if "Rendement" in df.columns:
            data[name] = df["Rendement"]

    # 3.3
    df_ret = pd.DataFrame(data).apply(lambda s: s.fillna(s.mean()))

    # 3.4
    corr = df_ret.corr()
    dist = 1 - corr

    # 3.5
    linked = linkage(squareform(dist.values), method="average")

    if show_plots:
        plt.figure(figsize=(12, 4))
        dendrogram(linked, labels=corr.columns)
        plt.title("Dendrogramme – Corrélations rendements")
        plt.tight_layout()
        plt.show()

    # 3.6
    labels = fcluster(linked, t=n_clusters, criterion="maxclust") - 1
    clusters = pd.Series(labels, index=corr.columns, name="cluster_rendements")

    return clusters, corr, dist


# ───────────────────────────
# 4) Évaluation et sauvegarde
# ───────────────────────────
# Pipeline :
# 4.1 Calculer silhouette pour :
#     – 4.1.1 K-Means (profils financiers)
#     – 4.1.2 Hiérarchique (profils de risque)
#     – 4.1.3 DBSCAN (profils financiers) si ≥2 clusters
# 4.2 Afficher les scores

def evaluate_clustering(X, labels, name=""):
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette {name}: {score:.3f}")
        return score
    else:
        print(f"Silhouette {name}: N/A (1 cluster)")
        return np.nan

from sklearn.cluster import DBSCAN

def evaluate_all_clusterings(X_fin, labels_fin, X_risk, labels_risk):
    print("Évaluation des scores de silhouette :")

    # 4.1.1
    sil_fin = evaluate_clustering(X_fin, labels_fin, name="KMeans (finance)")

    # 4.1.2
    sil_risk = evaluate_clustering(X_risk, labels_risk, name="Hiérarchique (risque)")

    print(f"Silhouette K-Means (finance)       : {sil_fin:.3f}")
    print(f"Silhouette Hiérarchique (risque)  : {sil_risk:.3f}")

    # 4.1.3
    db = DBSCAN(eps=1.5, min_samples=3).fit(X_fin)
    labels_db = db.labels_
    mask = labels_db != -1
    labels_eff = labels_db[mask]

    # 4.2
    if len(np.unique(labels_eff)) > 1:
        sil_db = silhouette_score(X_fin[mask], labels_eff)
        print(f"Silhouette DBSCAN (finance)       : {sil_db:.3f}")
    else:
        sil_db = np.nan
        print("Silhouette DBSCAN (finance)       : N/A (moins de 2 clusters)")

    return sil_fin, sil_risk, sil_db


"""## 1. Clustering des profils financiers (K-Means)

On a fait le clustering des entreprises selon leurs profils financiers via K-Means.
On a imputé 9 ratios de valorisation et de performance qu'on a standerdisé.

Le choix du nombre optimal de clusters s’est appuyé sur la méthode du coude. Celle-ci révèle une diminution régulière de l’inertie jusqu’à K = 8, sans coude très marqué. Le choix de K = 4 a été retenu comme un bon compromis.

Le score de silhouette obtenu (environ 0,26) traduit une séparation modérée entre les groupes.

Les profils identifiés sont les suivants :

- Value stocks : entreprises avec de faibles forward PE et des marges modérées.

- Sociétés à valorisation médiane : profil relativement stable et équilibré.

- Growth high-tech : entreprises à forte valorisation (PE élevé) et fortes marges.

- Groupe à part : ensemble hétérogène composé de valeurs extrêmes ou de sociétés financières.

La visualisation t-SNE a permis de confirmer la présence de ces quatre regroupements dans un espace de dimension réduite.

## 2. Clustering des profils de risque (Hiérarchique – méthode de Ward)

Cette analyse hiérarchique s’est concentrée sur des indicateurs de risque, les ratios d’endettement (dettes/fonds propres) et les ratios de liquidité (current ratio et quick ratio). Les données ont été pré-traitées par imputation des moyennes et suppression des variables sans dispersion.

Le score de silhouette obtenu (environ 0,75) indique une très bonne séparation entre les groupes formés.

Le dendrogramme fait apparaître trois grandes catégories d’entreprises :

- Sociétés très endettées : souvent issues des secteurs financier et énergétique.

- Sociétés moyennement endettées : majoritairement des entreprises industrielles matures.

- Sociétés peu endettées et bien liquides : notamment dans les technologies ou les secteurs de croissance.

## 3. Clustering des corrélations de rendements (Hiérarchique)

Une analyse hiérarchique a également été conduite sur les corrélations des rendements boursiers, calculés à partir de données journalières sur cinq ans. Les valeurs manquantes ont été imputées par la moyenne.

Le dendrogramme révèle une structuration claire selon les secteurs d’activité, regroupant par exemple les entreprises de l’énergie, de la finance, de la tech (US vs Asie), ou encore de l’automobile.

Une extraction en quatre groupes permet de mettre en évidence des familles d’entreprises ayant des comportements de marché similaires.

## 4. DBSCAN sur les profils financiers

L’algorithme DBSCAN sur les profils financiers n’a abouti qu’à un seul cluster principal, le reste des observations étant considérées comme du bruit.

## CONCLUSION

- Le clustering des profils de risque s’est avéré le plus discriminant, avec un score de silhouette élevé (environ 0,75).

- Le clustering des profils financiers est beaucoup moins discriminant (environ 0,26), suggérant une structure moins marquée.

- L’analyse des corrélations de rendements valide des regroupements sectoriels cohérents.

- L’algorithme DBSCAN n'est pas exploitable

"""
