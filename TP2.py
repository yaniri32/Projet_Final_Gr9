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

# 0) Vérification des données du TP1
# Pipeline :
# 0.1 Vérifier existence de ratios_financiers.csv
# 0.2 Vérifier existence du dossier historiques_entreprises/*.csv
# 0.3 Si manquant, appeler scrape_ratios() et scrape_historical()
if not os.path.isfile("ratios_financiers.csv"):
    print("ratios_financiers.csv introuvable")
    scrape_ratios()
if len(glob.glob("historiques_entreprises/*.csv")) == 0:
    print("historiques_entreprises manquant")
    scrape_historical()

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

# 1.1 & 1.2
df_ratios = pd.read_csv("ratios_financiers.csv", index_col=0)
cols_perf = [
    "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
    "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
]
df_fin = df_ratios[cols_perf].copy()

# 1.3
vides = df_fin.columns[df_fin.isna().all()]
if len(vides) > 0:
    df_fin.drop(columns=vides, inplace=True)
    print(f"> Colonnes vides supprimées : {list(vides)}")

# 1.4
df_fin.fillna(df_fin.mean(), inplace=True)

# 1.5
constantes = df_fin.columns[df_fin.nunique() <= 1]
if len(constantes) > 0:
    df_fin.drop(columns=constantes, inplace=True)
    print(f"> Colonnes constantes supprimées : {list(constantes)}")

# 1.6
scaler_fin = StandardScaler()
X_fin = scaler_fin.fit_transform(df_fin.values)

# 1.7
inerties = []
ks = range(1, 9)
for k in ks:
    inerties.append(KMeans(n_clusters=k, random_state=42).fit(X_fin).inertia_)
plt.figure(figsize=(6,4))
plt.plot(ks, inerties, "o-")
plt.title("Méthode du coude – Profils financiers")
plt.xlabel("K")
plt.ylabel("Inertie")
plt.show()

# 1.8
k_fin = 4  # après verificaiton c'est le bon k
km_fin = KMeans(n_clusters=k_fin, random_state=42).fit(X_fin)
df_fin["cluster_financier"] = km_fin.labels_

# 1.9
vis_fin = TSNE(perplexity=10, random_state=42).fit_transform(X_fin)
plt.figure(figsize=(6,5))
for c in range(k_fin):
    mask = (km_fin.labels_ == c)
    plt.scatter(vis_fin[mask,0], vis_fin[mask,1], label=f"Cluster {c}")
plt.title("t-SNE – Profils financiers")
plt.legend()
plt.show()


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

# 2.1 & 2.2
cols_risk = ["debtToEquity", "currentRatio", "quickRatio"]
df_risk = df_ratios[cols_risk].copy()
df_risk.fillna(df_risk.mean(), inplace=True)

# 2.3
const_risk = df_risk.columns[df_risk.nunique() <= 1]
if len(const_risk) > 0:
    df_risk.drop(columns=const_risk, inplace=True)
    print(f"> Colonnes constantes risque supprimées : {list(const_risk)}")

# 2.4
X_risk = StandardScaler().fit_transform(df_risk.values)

# 2.5
n_risk = 3
hc = AgglomerativeClustering(n_clusters=n_risk, linkage="ward").fit(X_risk)
df_risk["cluster_risque"] = hc.labels_

# 2.6
linked_risk = linkage(X_risk, method="ward")
plt.figure(figsize=(10,4))
dendrogram(linked_risk, labels=df_risk.index, leaf_rotation=90)
plt.title("Dendrogramme – Profils de risque")
plt.tight_layout()
plt.show()


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

# 3.1 & 3.2
d = {}
for f in glob.glob("historiques_entreprises/*.csv"):
    symb = os.path.basename(f).split("_")[0]
    tmp = pd.read_csv(f, index_col=0)
    if "Rendement" in tmp.columns:
        d[symb] = tmp["Rendement"]
df_ret = pd.DataFrame(d)

# 3.3
df_ret = df_ret.apply(lambda s: s.fillna(s.mean()), axis=0)

# 3.4
corr = df_ret.corr()
dist = 1 - corr

# 3.5
linked_ret = linkage(squareform(dist.values), method="average")
plt.figure(figsize=(12,4))
dendrogram(linked_ret, labels=corr.columns, leaf_rotation=90)
plt.title("Dendrogramme – Corrélations rendements")
plt.tight_layout()
plt.show()

# 3.6
clusters_ret = fcluster(linked_ret, t=4, criterion="maxclust") - 1
ser_ret = pd.Series(clusters_ret, index=corr.columns, name="cluster_rendements")


# ───────────────────────────
# 4) Évaluation et sauvegarde
# ───────────────────────────
# Pipeline :
# 4.1 Calculer silhouette pour :
#     – 4.1.1 K-Means (profils financiers)
#     – 4.1.2 Hiérarchique (profils de risque)
#     – 4.1.3 DBSCAN (profils financiers) si ≥2 clusters
# 4.2 Afficher les scores
# 4.3 Sauvegarder DataFrames et Series

# 4.1.1
sil_fin = silhouette_score(X_fin, km_fin.labels_)

# 4.1.2
sil_risk = silhouette_score(X_risk, hc.labels_)

# 4.1.3
db = DBSCAN(eps=1.5, min_samples=3).fit(X_fin)
labels_db = db.labels_
mask_db = labels_db != -1
labels_eff = labels_db[mask_db]
if len(np.unique(labels_eff)) > 1:
    sil_db = silhouette_score(X_fin[mask_db], labels_eff)
else:
    sil_db = np.nan
    print("DBSCAN n’a pas formé ≥2 clusters : silhouette non calculée")

# 4.2
print(f"Silhouette K-Means (finance)       : {sil_fin:.3f}")
print(f"Silhouette Hiérarchique (risque)  : {sil_risk:.3f}")
print(f"Silhouette DBSCAN (finance)       : {sil_db if not np.isnan(sil_db) else 'N/A'}")

# 4.3
df_fin.to_csv("clusters_financiers.csv")
df_risk.to_csv("clusters_risque.csv")
ser_ret.to_frame().to_csv("clusters_rendements.csv")
print("Fichiers sauvegardés → clusters_financiers.csv, clusters_risque.csv, clusters_rendements.csv")

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