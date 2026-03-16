# 💳 Neobank Analytics

Projet d'analyse de données et de machine learning sur un dataset de néobanque.
Réalisé dans le cadre d'une formation Data Analyst / Data Scientist.

---

## 👥 Équipe

| Membre | Rôle |
|---|---|
| [Arthur/Matthieu] | Modules 1 & 2 — EDA & Visualisation |
| [Arthur/Matthieu] | Module 3 — Machine Learning |

---

## 📊 Dataset

| Table | Lignes | Description |
|---|---|---|
| `users.csv` | 19 430 | Profil des utilisateurs |
| `transactions.csv` | 2 740 075 | Historique des transactions |
| `notifications.csv` | 121 813 | Notifications envoyées |
| `devices.csv` | 19 431 | Appareils utilisés |

**Période couverte :** Janvier 2018 — Mai 2019

---

## 🗂️ Structure du projet
```
neobank-analytics/
├── app.py                  ← Application Streamlit principale
├── requirements.txt        ← Dépendances Python
├── README.md               ← Ce fichier
├── data/
│   ├── users.csv
│   ├── transactions.csv
│   ├── notifications.csv
│   └── devices.csv
└── modules/
    ├── __init__.py
    ├── accueil.py          ← Page d'accueil
    ├── module1.py          ← Customer Base & Notifications
    ├── module2.py          ← Transaction Analysis
    └── module3.py          ← ML & Cross-Module Insights
```

---

## 📦 Modules

### 📍 Module 1 — Customer Base & Notifications
> *"Qui sont nos utilisateurs et comment les engage-t-on ?"*

- Répartition géographique (carte Europe)
- Distribution démographique
- Croissance des inscriptions
- Funnel Standard → Premium → Metal
- Adoption crypto par plan
- Délivrabilité des notifications par canal
- Opt-in push & email

### 💳 Module 2 — Transaction Analysis
> *"Quels sont les patterns transactionnels ?"*

- Timeline mensuelle — croissance ×53 en 15 mois
- Taux d'échec par type de transaction
- Usage digital vs physique (81% en ligne)
- Top catégories de dépenses (MCC)
- Impact des campagnes de notifications
- Répartition des devises

### 🤖 Module 3 — Machine Learning
> *"Comment segmenter et prédire le comportement des clients ?"*

**Modèle 1 — Clustering K-Means**
- Segmentation en 3 clusters : Engagés / Réguliers / À Risque
- Visualisation PCA 2D
- Heatmap des profils

**Modèle 2 — Churn Prediction (Random Forest)**
- Prédiction du risque de churn par utilisateur
- Accuracy : 84% | Recall churned : 67%
- Feature importance

**Modèle 3 — Détection de Fraude (Isolation Forest)**
- Détection d'anomalies sur 2.7M transactions
- 54 605 transactions suspectes détectées (2%)
- Validation via proxy REVERTED

---

## 🚀 Installation & Lancement

### En local
```bash
# Cloner le repository
git clone https://github.com/[username]/neobank-analytics.git
cd neobank-analytics

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

### Sur Streamlit Cloud

L'application est déployée ici → **[lien Streamlit Cloud]**

---

## 🔧 Technologies utilisées

| Outil | Usage |
|---|---|
| Python 3.12 | Langage principal |
| Pandas | Manipulation des données |
| NumPy | Calculs mathématiques |
| Plotly | Visualisations interactives |
| Scikit-learn | Machine Learning |
| Streamlit | Application web |
| Pycountry | Conversion codes pays |

---

## 💡 Insights clés

- 🇬🇧 **32%** des users viennent du Royaume-Uni
- 📈 Croissance **×53** en 15 mois (6K → 334K transactions/mois)
- 💳 **81%** des paiements sont en ligne → profil digital-first
- 📬 Users notifiés font **×2** plus de transactions
- ⚠️ SMS : **34%** de délivrabilité → canal à abandonner
- 🔴 **23%** des users sont à risque de churn

---

## ⚠️ Data Quality

- Transactions TRANSFER aberrantes détectées jusqu'à **$85 milliards**
- Toutes `DECLINED` → erreurs techniques ou transactions de test
- Exclues via filtre P99 sur les montants TRANSFER

---

## 📁 Notebooks

Les notebooks d'analyse sont disponibles sur Google Colab :

| Notebook | Lien |
|---|---|
| Module 1 & 2 — EDA | [Ouvrir dans Colab] |
| Module 3 — ML | [Ouvrir dans Colab] |
