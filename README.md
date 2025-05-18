# Projet Final Groupe 9 – Pipeline Data Science pour Recommandations Boursières

## Présentation
Ce projet implémente un pipeline automatisé (exécuté chaque jour) qui :

1. Exécute les travaux pratiques de Data Science :  
   - Clustering d’entreprises  
   - Classification « Buy / Sell / Hold »  
   - Prédiction de rendement à J+1  
   - Analyse de sentiment sur les news financières  
2. Agrège ces différents signaux pour fournir un **conseil d’investissement**.
3. Génère, pour chaque titre, un rapport d’activité quotidien (liste d’entreprises similaires, prévision de rendement, sentiment global, recommandation).

Le code Python est organisé en modules, et un script `main.py` orchestre l’ensemble.

---

## 📁 Structure du dépôt

├── TP1.py # Scraping ratios financiers + Scraping variations de cours historiques
├── TP2.py # Clustering d’entreprises
├── TP3.py # Classification Buy/Sell/Hold
├── TP4.py # Régression de rendement J+1
├── TP5.py # Réseaux de Neurones pour la prédiction
├── TP6.py # Scraping de news financières
├── TP7.py # Fine-tuning BERT pour l’analyse de sentiment
├── TP8.py # Classification de news et impact sur les variations de cours
├── main.py # Script principal 
├── report/ # Rapport final en PDF 
└── outputs/ # Fichiers générés (CSV, logs, graphiques… à enlever eventuellement ?)

