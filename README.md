# Neobank Analytics

Dashboard interactif d'analyse d'une neobank sur la période **janvier 2018 – mai 2019**.
Le projet couvre l'ensemble de la chaîne analytique : exploration des données, visualisation et modèles de machine learning.

**Par Arthur & Matthieu**

🚀 **[Voir le dashboard en ligne]([https://fc8kupji4o4ap5cwgweovj.streamlit.app/](https://neobank-analytics-dnbsrjfvdxuvpsycgvshvg.streamlit.app/))**

---

## Modules

### 1 — Base Clients
Analyse de la base utilisateurs : KPIs clés, évolution des inscriptions, répartition des plans (Standard, Premium, Metal), appareils et comportement des notifications.

### 2 — Transactions
Exploration des flux financiers : volumes, types de transactions (paiement carte, virement, échange), devises utilisées, comportements en ligne vs physique.

### 3 — Machine Learning
Trois modèles déployés :
- **Clustering K-Means** — segmentation en 3 profils clients (Engagés, Réguliers, À Risque)
- **Random Forest** — prédiction du churn avec optimisation du recall (seuil 0.25)
- **Isolation Forest** — détection d'anomalies et transactions suspectes

---

## Stack

`Python` `Streamlit` `Scikit-learn` `Plotly` `Pandas` `NumPy`

---

## Notebooks

| Module | Lien |
|--------|------|
| Module 1 — Base Clients | [Ouvrir dans Colab](https://colab.research.google.com/drive/1D5Fw8hr_07lUND5rfeiTs1YZ_5dTASvz?usp=sharing) |
| Module 2 — Transactions | [Ouvrir dans Colab](https://colab.research.google.com/drive/12pkpm4XfVTnCwsZwDzsgGTvUMdtFKEXp?usp=sharing) |
| Module 3 — Machine Learning | [Ouvrir dans Colab](https://colab.research.google.com/drive/1wKdNAIQrlbLxdMrm8OPAd0G-BzuG_aV9?usp=sharing) |
