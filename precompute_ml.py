"""
Script à lancer UNE FOIS en local pour pré-calculer les résultats ML.
Les fichiers sont sauvegardés dans data/ml/ et committés sur GitHub.
Streamlit Cloud charge ces fichiers sans avoir besoin de recalculer.
"""

import pandas as pd
import numpy as np
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

os.makedirs('data/ml', exist_ok=True)

print("Chargement des données...")
df_tx = pd.read_parquet('data/transactions.parquet')
df_users = pd.read_csv('data/users.csv')
df_tx['created_date'] = pd.to_datetime(df_tx['created_date'], utc=True, format='mixed')
df_users['created_date'] = pd.to_datetime(df_users['created_date'], utc=True, format='mixed')

p99 = df_tx.loc[df_tx['transactions_type'] == 'TRANSFER', 'amount_usd'].quantile(0.99)
df_tx_clean = df_tx[~((df_tx['transactions_type'] == 'TRANSFER') & (df_tx['amount_usd'] > p99))].copy()

REF_DATE = pd.Timestamp('2019-05-16', tz='UTC')
cap_montant = df_tx_clean['amount_usd'].quantile(0.99)

print("Feature engineering...")
user_features = (
    df_tx_clean.groupby('user_id').agg(
        nb_tx         = ('transaction_id', 'count'),
        montant_moyen = ('amount_usd', lambda x: x[x <= cap_montant].mean()),
        nb_devises    = ('transactions_currency', 'nunique'),
        pct_online    = ('ea_cardholderpresence',
                         lambda x: (x == 'FALSE').sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan),
        premiere_tx   = ('created_date', 'min'),
        derniere_tx   = ('created_date', 'max'),
    ).reset_index()
)

user_features['premiere_tx'] = pd.to_datetime(user_features['premiere_tx'], utc=True)
user_features['derniere_tx'] = pd.to_datetime(user_features['derniere_tx'], utc=True)
user_features['nb_mois_actifs'] = ((user_features['derniere_tx'] - user_features['premiere_tx']).dt.days / 30).clip(lower=1)
user_features['frequence'] = user_features['nb_tx'] / user_features['nb_mois_actifs']

user_features = user_features.merge(
    df_users[['user_id', 'plan', 'user_settings_crypto_unlocked', 'created_date']],
    on='user_id', how='left'
)
user_features['created_date'] = pd.to_datetime(user_features['created_date'], utc=True)
user_features['anciennete'] = (REF_DATE - user_features['created_date']).dt.days

plan_map = {'STANDARD': 0, 'PREMIUM': 1, 'METAL': 2, 'METAL_FREE': 2, 'PREMIUM_FREE': 1, 'PREMIUM_OFFER': 1}
user_features['plan_ordinal'] = user_features['plan'].map(plan_map).fillna(0).astype(int)
user_features['crypto'] = user_features['user_settings_crypto_unlocked'].astype(int)
user_features['pct_online'] = user_features['pct_online'].fillna(-1)
user_features['montant_moyen'] = user_features['montant_moyen'].fillna(0)

last_tx = df_tx_clean.groupby('user_id')['created_date'].max().reset_index().rename(columns={'created_date': 'last_tx_date'})
last_tx['days_inactive'] = (REF_DATE - last_tx['last_tx_date']).dt.days.clip(lower=0)
last_tx['churn'] = (last_tx['days_inactive'] > 90).astype(int)
user_features = user_features.merge(last_tx[['user_id', 'churn']], on='user_id', how='left')

FEATURES = ['frequence', 'montant_moyen', 'nb_devises', 'pct_online', 'plan_ordinal', 'crypto', 'anciennete']

# ── CLUSTERING ───────────────────────────────────────────────────
print("Clustering...")
df_clust = user_features[['user_id'] + FEATURES].dropna()
scaler = StandardScaler()
X = scaler.fit_transform(df_clust[FEATURES])

inertias, silhouettes = [], []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clust['cluster'] = km_final.fit_predict(X)

profil = df_clust.groupby('cluster')[FEATURES].mean()
freq_order = profil['frequence'].sort_values(ascending=False).index.tolist()
names = {freq_order[0]: 'Engages', freq_order[1]: 'Reguliers', freq_order[2]: 'A Risque'}
df_clust['segment'] = df_clust['cluster'].map(names)

df_clust.to_parquet('data/ml/cluster.parquet', index=False)
json.dump({
    'inertias': inertias,
    'silhouettes': silhouettes,
    'profil': profil.to_dict(),
    'names': {str(k): v for k, v in names.items()},
    'freq_order': freq_order
}, open('data/ml/cluster_meta.json', 'w'))
print("  Clustering sauvegardé")

# ── CHURN ────────────────────────────────────────────────────────
print("Churn...")
df_ch = user_features[FEATURES + ['churn', 'user_id']].dropna()
X_ch = df_ch[FEATURES]
y_ch = df_ch['churn']

X_train, X_test, y_train, y_test = train_test_split(X_ch, y_ch, test_size=0.2, random_state=42, stratify=y_ch)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_proba = rf.predict_proba(X_test)[:, 1]

best_t = 0.25
best_f1 = 0
results = []
for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    yp = (y_proba >= t).astype(int)
    p, r, f = precision_score(y_test, yp), recall_score(y_test, yp), f1_score(y_test, yp)
    results.append({'seuil': t, 'precision': p, 'recall': r, 'f1': f})
    if t == best_t:
        best_f1 = f

y_pred = (y_proba >= best_t).astype(int)
cm = confusion_matrix(y_test, y_pred)
importances = pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_}).sort_values('importance', ascending=True)

df_ch = df_ch.copy()
df_ch['churn_proba'] = rf.predict_proba(df_ch[FEATURES])[:, 1]

df_ch[['user_id', 'churn', 'churn_proba']].to_parquet('data/ml/churn.parquet', index=False)
importances.to_parquet('data/ml/churn_importances.parquet', index=False)
pd.DataFrame({'y_test': y_test.values, 'y_pred': y_pred, 'y_proba': y_proba}).to_parquet('data/ml/churn_test.parquet', index=False)
json.dump({
    'best_t': best_t,
    'best_f1': best_f1,
    'results': results,
    'cm': cm.tolist()
}, open('data/ml/churn_meta.json', 'w'))
print("  Churn sauvegardé")

# ── FRAUDE ───────────────────────────────────────────────────────
print("Fraude...")
df_f = df_tx_clean.copy()
df_f['hour'] = df_f['created_date'].dt.hour
df_f = df_f.sort_values(['user_id', 'created_date'])

def _is_new_country(group):
    seen = set()
    result = []
    for country in group:
        if pd.isna(country):
            result.append(0)
        else:
            result.append(int(country not in seen))
            seen.add(country)
    return pd.Series(result, index=group.index)

df_f['is_new_country'] = df_f.groupby('user_id')['ea_merchant_country'].apply(_is_new_country).reset_index(level=0, drop=True)

profile = df_f.groupby('user_id').agg(
    mean_amount=('amount_usd', 'mean'), std_amount=('amount_usd', 'std'),
    median_hour=('hour', 'median'), std_hour=('hour', 'std'),
    n_txn=('transaction_id', 'count'),
    pct_online=('ea_cardholderpresence', lambda x: (x == 'FALSE').sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan),
).reset_index()
profile['std_amount'] = profile['std_amount'].fillna(0)
profile['std_hour'] = profile['std_hour'].fillna(0)
profile['pct_online'] = profile['pct_online'].fillna(0.5)
profile = profile[profile['n_txn'] >= 5]

df_f = df_f[df_f['user_id'].isin(profile['user_id'])].copy()
df_f = df_f.merge(profile, on='user_id', how='left', suffixes=('', '_profile'))

df_f['z_score_montant'] = np.where(df_f['std_amount'] > 0, (df_f['amount_usd'] - df_f['mean_amount']) / df_f['std_amount'], 0)
df_f['z_score_heure'] = np.where(df_f['std_hour'] > 0, (df_f['hour'] - df_f['median_hour']) / df_f['std_hour'], 0)
df_f['time_since_last'] = df_f.groupby('user_id')['created_date'].diff().dt.total_seconds()
df_f['time_since_last'] = df_f['time_since_last'].fillna(df_f['time_since_last'].median())
df_f['time_since_last_log'] = np.log1p(df_f['time_since_last'].clip(lower=0))
df_f['is_online'] = (df_f['ea_cardholderpresence'] == 'FALSE').astype(float)
df_f['is_online'] = df_f['is_online'].where(df_f['ea_cardholderpresence'].notna(), np.nan)
df_f['card_presence_flip'] = np.where(df_f['is_online'].isna(), 0, (df_f['is_online'] - df_f['pct_online']).abs())

FRAUD_FEATURES = ['z_score_montant', 'is_new_country', 'z_score_heure', 'time_since_last_log', 'card_presence_flip']
scaler2 = StandardScaler()
X_f = scaler2.fit_transform(df_f[FRAUD_FEATURES])

iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
iso.fit(X_f)
df_f['anomaly_score'] = iso.decision_function(X_f)
df_f['is_suspect'] = iso.predict(X_f)
df_f['is_reverted'] = (df_f['transactions_state'] == 'REVERTED').astype(int)

# Stats fraude
n_suspect = (df_f['is_suspect'] == -1).sum()
rev_global = df_f['is_reverted'].mean()
rev_suspect = df_f.loc[df_f['is_suspect'] == -1, 'is_reverted'].mean()
rev_normal = df_f.loc[df_f['is_suspect'] == 1, 'is_reverted'].mean()
enrichment = rev_suspect / rev_global if rev_global > 0 else 0

# Top 30 users suspects
top_suspect_users = (
    df_f[df_f['is_suspect'] == -1]
    .groupby('user_id').size()
    .sort_values(ascending=False)
    .head(30)
    .reset_index()
)
top_suspect_users.columns = ['user_id', 'nb_suspects']
top_users_list = top_suspect_users['user_id'].tolist()

# Scatter sample
scatter_cols = ['z_score_montant', 'z_score_heure', 'is_suspect']
scatter_sample = df_f[scatter_cols].sample(n=min(50000, len(df_f)), random_state=42)
scatter_sample.to_parquet('data/ml/fraud_scatter.parquet', index=False)

# Histogram
hist_cols = ['anomaly_score', 'is_reverted']
df_f[hist_cols].to_parquet('data/ml/fraud_histogram.parquet', index=False)

# Timeline top users suspects
timeline_cols = ['user_id', 'created_date', 'amount_usd', 'is_suspect', 'transactions_type', 'ea_merchant_country', 'z_score_montant']
df_f[df_f['user_id'].isin(top_users_list)][timeline_cols].to_parquet('data/ml/fraud_timeline.parquet', index=False)

json.dump({
    'n_suspect': int(n_suspect),
    'n_total': int(len(df_f)),
    'rev_global': float(rev_global),
    'rev_suspect': float(rev_suspect),
    'rev_normal': float(rev_normal),
    'enrichment': float(enrichment),
    'top_suspect_users': top_suspect_users.to_dict('records')
}, open('data/ml/fraud_meta.json', 'w'))
print("  Fraude sauvegardé")

print("\nTerminé ! Fichiers dans data/ml/")
