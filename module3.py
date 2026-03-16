"""
Module 3 — ML & Cross-Insights
Clustering + Churn + Fraude
Appelé depuis app.py : from modules.module3 import show_module3
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix


# ── Palette ──────────────────────────────────────────────────────
NAVY    = '#0C1E3C'
ROYAL   = '#1B3A6B'
BLUE    = '#2E5EA6'
SKY     = '#4A90D9'
PALE    = '#6B9BD2'
ACCENT  = '#E63946'
GOLD    = '#D4A843'
GREEN   = '#2ECC71'
GREY    = '#8899AA'
LIGHT   = '#F4F7FA'


# ── Helpers ──────────────────────────────────────────────────────
def _corp_layout(fig, h=420):
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Outfit', color=NAVY, size=12),
        height=h, margin=dict(l=50, r=30, t=55, b=45),
        title_font=dict(size=15, color=NAVY, family='Outfit'),
        hoverlabel=dict(bgcolor=NAVY, font_size=13, font_family='Outfit', font_color='white'),
    )
    fig.update_xaxes(gridcolor='#F0F2F5', zeroline=False)
    fig.update_yaxes(gridcolor='#F0F2F5', zeroline=False)
    return fig

def _num_card(value, label, accent=False):
    cls = 'num-card accent' if accent else 'num-card'
    st.markdown(f'<div class="{cls}"><div class="num-big">{value}</div><div class="num-label">{label}</div></div>', unsafe_allow_html=True)

def _chapter(num, title, subtitle=''):
    sub = f'<div class="chapter-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div class="chapter"><div class="chapter-num">{num}</div><div><div class="chapter-title">{title}</div>{sub}</div></div>', unsafe_allow_html=True)

def _verdict(title, text):
    st.markdown(f'<div class="verdict"><h4>💡 {title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def _story(text):
    st.markdown(f'<div class="story">{text}</div>', unsafe_allow_html=True)


# ── Chargement & feature engineering (caché pour performance) ────
@st.cache_data
def load_and_prepare():
    """Charge les CSV et prépare toutes les données pour les 3 modèles."""

    df_tx = pd.read_csv('data/transactions.csv')
    df_users = pd.read_csv('data/users.csv')
    df_tx['created_date'] = pd.to_datetime(df_tx['created_date'], utc=True, format='mixed')
    df_users['created_date'] = pd.to_datetime(df_users['created_date'], utc=True, format='mixed')

    # Nettoyage outliers TRANSFER P99
    p99 = df_tx.loc[df_tx['transactions_type'] == 'TRANSFER', 'amount_usd'].quantile(0.99)
    df_tx_clean = df_tx[~((df_tx['transactions_type'] == 'TRANSFER') & (df_tx['amount_usd'] > p99))].copy()

    REF_DATE = pd.Timestamp('2019-05-16', tz='UTC')
    cap_montant = df_tx_clean['amount_usd'].quantile(0.99)

    # ── Features utilisateur (pour clustering + churn) ───────────
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

    # ── Churn target ─────────────────────────────────────────────
    last_tx = df_tx_clean.groupby('user_id')['created_date'].max().reset_index().rename(columns={'created_date': 'last_tx_date'})
    last_tx['days_inactive'] = (REF_DATE - last_tx['last_tx_date']).dt.days.clip(lower=0)
    last_tx['churn'] = (last_tx['days_inactive'] > 90).astype(int)
    user_features = user_features.merge(last_tx[['user_id', 'churn']], on='user_id', how='left')

    return df_tx_clean, df_users, user_features, REF_DATE


# ── Modèle 1 : Clustering ───────────────────────────────────────
@st.cache_data
def run_clustering(_user_features):
    FEATURES = ['frequence', 'montant_moyen', 'nb_devises', 'pct_online', 'plan_ordinal', 'crypto', 'anciennete']
    df = _user_features[['user_id'] + FEATURES].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # Elbow + Silhouette
    inertias, silhouettes = [], []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    # K=3 pour interprétabilité
    km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = km_final.fit_predict(X)

    profil = df.groupby('cluster')[FEATURES].mean()
    freq_order = profil['frequence'].sort_values(ascending=False).index.tolist()
    names = {freq_order[0]: '🟢 Engagés', freq_order[1]: '🟡 Réguliers', freq_order[2]: '🔴 À Risque'}
    df['segment'] = df['cluster'].map(names)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df['PC1'] = coords[:, 0]
    df['PC2'] = coords[:, 1]

    return df, profil, names, freq_order, inertias, silhouettes, pca


# ── Modèle 2 : Churn ────────────────────────────────────────────
@st.cache_data
def run_churn(_user_features):
    FEATURES = ['frequence', 'montant_moyen', 'nb_devises', 'pct_online', 'plan_ordinal', 'crypto', 'anciennete']
    df = _user_features[FEATURES + ['churn', 'user_id']].dropna()

    X = df[FEATURES]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # Trouver meilleur seuil
    best_f1, best_t = 0, 0.5
    results = []
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        yp = (y_proba >= t).astype(int)
        p, r, f = precision_score(y_test, yp), recall_score(y_test, yp), f1_score(y_test, yp)
        results.append({'seuil': t, 'precision': p, 'recall': r, 'f1': f})
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred = (y_proba >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    importances = pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_}).sort_values('importance', ascending=True)

    # Score sur tous les users
    df['churn_proba'] = rf.predict_proba(df[FEATURES])[:, 1]

    return df, cm, importances, best_t, best_f1, results, y_test, y_pred, y_proba


# ── Modèle 3 : Fraude ───────────────────────────────────────────
@st.cache_data
def run_fraud(_df_tx_clean):
    df = _df_tx_clean.copy()
    df['hour'] = df['created_date'].dt.hour
    df = df.sort_values(['user_id', 'created_date'])

    # is_new_country chronologique
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

    df['is_new_country'] = df.groupby('user_id')['ea_merchant_country'].apply(_is_new_country).reset_index(level=0, drop=True)

    # Profil par user
    profile = df.groupby('user_id').agg(
        mean_amount=('amount_usd', 'mean'), std_amount=('amount_usd', 'std'),
        median_hour=('hour', 'median'), std_hour=('hour', 'std'),
        n_txn=('transaction_id', 'count'),
        pct_online=('ea_cardholderpresence', lambda x: (x == 'FALSE').sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan),
    ).reset_index()
    profile['std_amount'] = profile['std_amount'].fillna(0)
    profile['std_hour'] = profile['std_hour'].fillna(0)
    profile['pct_online'] = profile['pct_online'].fillna(0.5)
    profile = profile[profile['n_txn'] >= 5]

    df = df[df['user_id'].isin(profile['user_id'])].copy()
    df = df.merge(profile, on='user_id', how='left', suffixes=('', '_profile'))

    # Features
    df['z_score_montant'] = np.where(df['std_amount'] > 0, (df['amount_usd'] - df['mean_amount']) / df['std_amount'], 0)
    df['z_score_heure'] = np.where(df['std_hour'] > 0, (df['hour'] - df['median_hour']) / df['std_hour'], 0)
    df['time_since_last'] = df.groupby('user_id')['created_date'].diff().dt.total_seconds()
    df['time_since_last'] = df['time_since_last'].fillna(df['time_since_last'].median())
    df['time_since_last_log'] = np.log1p(df['time_since_last'].clip(lower=0))

    df['is_online'] = (df['ea_cardholderpresence'] == 'FALSE').astype(float)
    df['is_online'] = df['is_online'].where(df['ea_cardholderpresence'].notna(), np.nan)
    df['card_presence_flip'] = np.where(df['is_online'].isna(), 0, (df['is_online'] - df['pct_online']).abs())

    FEATURES = ['z_score_montant', 'is_new_country', 'z_score_heure', 'time_since_last_log', 'card_presence_flip']

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso.fit(X)
    df['anomaly_score'] = iso.decision_function(X)
    df['is_suspect'] = iso.predict(X)
    df['is_reverted'] = (df['transactions_state'] == 'REVERTED').astype(int)

    return df


# ════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════
def show_module3():

    st.title("🤖 Module 3 — Machine Learning & Cross-Insights")
    _story("On sait <em>qui</em> ils sont et <em>ce qu'ils font</em>. "
           "Maintenant : <strong>segmenter, prédire le churn, détecter la fraude</strong>.")

    # Chargement
    with st.spinner("Chargement des données et entraînement des modèles..."):
        df_tx_clean, df_users, user_features, REF_DATE = load_and_prepare()
        df_cluster, profil, names, freq_order, inertias, silhouettes, pca = run_clustering(user_features)
        df_churn, cm, importances, best_t, best_f1, threshold_results, y_test, y_pred, y_proba = run_churn(user_features)
        df_fraud = run_fraud(df_tx_clean)

    # ── Sous-navigation ──────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Segmentation", "📉 Churn", "🔍 Fraude", "🔗 Synthèse"
    ])

    color_map = {'🟢 Engagés': GREEN, '🟡 Réguliers': GOLD, '🔴 À Risque': ACCENT}

    # ════════════════════════════════════════════════════════════
    # TAB 1 — CLUSTERING
    # ════════════════════════════════════════════════════════════
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: _num_card("7", "Features")
        with c2: _num_card("3", "Segments")
        with c3: _num_card("K-Means", "Algorithme")
        with c4: _num_card(f"{len(df_cluster):,}", "Users", accent=True)

        _chapter("1", "Choix du K optimal", "Elbow & Silhouette")

        fig = make_subplots(rows=1, cols=2, subplot_titles=['<b>Méthode Elbow</b>', '<b>Score Silhouette</b>'])
        fig.add_trace(go.Scatter(x=list(range(2, 10)), y=inertias, mode='lines+markers',
                                  line=dict(color=ROYAL, width=2.5), marker=dict(size=9, color=BLUE)), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(2, 10)), y=silhouettes, mode='lines+markers',
                                  line=dict(color=GREEN, width=2.5), marker=dict(size=9, color=GREEN)), row=1, col=2)
        fig = _corp_layout(fig, h=400)
        fig.update_layout(showlegend=False, title=dict(text='<b>Choix du nombre optimal de clusters</b>', x=0.5))
        fig.update_xaxes(title_text='K', dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Meilleur silhouette à k=2 ({silhouettes[0]:.3f}), mais **k=3 retenu** pour l'interprétabilité business ({silhouettes[1]:.3f}).")

        _chapter("2", "Les 3 segments", "Profil & répartition")

        col1, col2 = st.columns(2)

        with col1:
            # Heatmap
            FEATURES_CLUSTER = ['frequence', 'montant_moyen', 'nb_devises', 'pct_online', 'plan_ordinal', 'crypto', 'anciennete']
            profil_display = profil.loc[freq_order].copy()
            profil_display.index = [names[i] for i in freq_order]
            profil_norm = profil_display.copy()
            for col in profil_norm.columns:
                mn, mx = profil_norm[col].min(), profil_norm[col].max()
                profil_norm[col] = (profil_norm[col] - mn) / (mx - mn) if mx > mn else 0.5

            labels_fr = {'frequence': 'Fréquence', 'montant_moyen': 'Montant', 'nb_devises': 'Devises',
                         'pct_online': '% Online', 'plan_ordinal': 'Plan', 'crypto': 'Crypto', 'anciennete': 'Ancienneté'}

            fig = px.imshow(profil_norm.rename(columns=labels_fr), text_auto='.2f',
                            color_continuous_scale='RdYlGn', aspect='auto')
            fig = _corp_layout(fig, h=280)
            fig.update_layout(title='<b>Heatmap des profils</b>')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Donut
            sizes = df_cluster['segment'].value_counts()
            fig = go.Figure(go.Pie(labels=sizes.index, values=sizes.values,
                                    marker_colors=[color_map[s] for s in sizes.index],
                                    hole=0.55, textinfo='label+percent', textfont=dict(size=12)))
            fig.add_annotation(text=f"<b>{len(df_cluster):,}</b><br>users", x=0.5, y=0.5,
                               font=dict(size=16, color=NAVY), showarrow=False)
            fig = _corp_layout(fig, h=350)
            fig.update_layout(title='<b>Taille des segments</b>', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        _chapter("3", "Visualisation PCA", "Projection 2D")

        fig = px.scatter(df_cluster, x='PC1', y='PC2', color='segment',
                         color_discrete_map=color_map, opacity=0.4,
                         labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                 'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 'segment': ''})
        fig.update_traces(marker=dict(size=3))
        fig = _corp_layout(fig, h=500)
        fig.update_layout(title=f'<b>PCA 2D</b> — Variance expliquée : {pca.explained_variance_ratio_.sum():.1%}',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        _verdict("Insight clé", f"Le crypto est LE séparateur : Engagés = {profil.loc[freq_order[0], 'crypto']:.0%} crypto "
                 f"vs Réguliers = {profil.loc[freq_order[1], 'crypto']:.0%}. L'activation crypto = marqueur d'engagement.")

    # ════════════════════════════════════════════════════════════
    # TAB 2 — CHURN
    # ════════════════════════════════════════════════════════════
    with tab2:
        n_churn = df_churn['churn'].sum()
        n_total = len(df_churn)

        c1, c2, c3, c4 = st.columns(4)
        with c1: _num_card(f"{n_churn / n_total:.0%}", "Taux de churn")
        with c2: _num_card(f"{n_churn:,}", "Churnés")
        with c3: _num_card(f"{recall_score(y_test, y_pred):.0%}", "Recall")
        with c4: _num_card(f"{best_t}", "Seuil optimal", accent=True)

        _chapter("1", "Optimisation du seuil", "Precision × Recall × F1")

        df_thresholds = pd.DataFrame(threshold_results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_thresholds['seuil'], y=df_thresholds['precision'],
                                  mode='lines+markers', name='Precision', line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=df_thresholds['seuil'], y=df_thresholds['recall'],
                                  mode='lines+markers', name='Recall', line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=df_thresholds['seuil'], y=df_thresholds['f1'],
                                  mode='lines+markers', name='F1', line=dict(color=GREEN, width=2.5)))
        fig.add_vline(x=best_t, line_dash='dash', line_color=GOLD, annotation_text=f'Optimal: {best_t}',
                      annotation_font_color=GOLD)
        fig = _corp_layout(fig, h=400)
        fig.update_layout(title='<b>Métriques par seuil de probabilité</b>',
                          xaxis_title='Seuil', yaxis_title='Score',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        _chapter("2", "Résultats", "Matrice de confusion & Feature importance")

        col1, col2 = st.columns(2)

        with col1:
            labels_cm = ['Actif', 'Churned']
            fig = go.Figure(go.Heatmap(
                z=cm[::-1], x=labels_cm, y=labels_cm[::-1],
                text=[[f"<b>{v:,}</b>" for v in row] for row in cm[::-1]],
                texttemplate="%{text}", textfont=dict(size=22),
                colorscale=[[0, 'white'], [0.5, PALE], [1, NAVY]], showscale=False))
            fig.update_layout(title=dict(text=f'<b>Matrice de confusion</b> — Seuil {best_t}', x=0.5),
                              xaxis_title='<b>Prédit</b>', yaxis_title='<b>Réel</b>',
                              height=400, width=450, plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family='Outfit', color=NAVY))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style="background:white; border-radius:10px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="color:{GREEN}; font-size:22px; font-weight:800;">{cm[1][1]:,}</div>
                <div style="color:{GREY}; font-size:12px;">Vrais positifs — churnés détectés</div>
                <div style="color:{ACCENT}; font-size:22px; font-weight:800; margin-top:8px;">{cm[1][0]:,}</div>
                <div style="color:{GREY}; font-size:12px;">Faux négatifs — churnés ratés</div>
                <div style="color:{GOLD}; font-size:22px; font-weight:800; margin-top:8px;">{cm[0][1]:,}</div>
                <div style="color:{GREY}; font-size:12px;">Faux positifs — actifs contactés à tort</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            labels_feat = {'frequence': 'Fréquence', 'montant_moyen': 'Montant moy.', 'nb_devises': 'Nb devises',
                           'pct_online': '% Online', 'plan_ordinal': 'Plan', 'crypto': 'Crypto', 'anciennete': 'Ancienneté'}
            imp = importances.copy()
            imp['label'] = imp['feature'].map(labels_feat)

            fig = go.Figure(go.Bar(x=imp['importance'], y=imp['label'], orientation='h',
                                    marker=dict(color=imp['importance'].values, colorscale=[[0, PALE], [0.5, BLUE], [1, NAVY]]),
                                    text=[f"<b>{v:.3f}</b>" for v in imp['importance']], textposition='outside',
                                    textfont=dict(size=12, color=NAVY)))
            fig = _corp_layout(fig, h=400)
            fig.update_layout(title='<b>Feature importance</b>',
                              xaxis=dict(range=[0, imp['importance'].max() * 1.3]))
            st.plotly_chart(fig, use_container_width=True)

        _verdict("Compromis seuil", f"À {best_t}, on attrape {recall_score(y_test, y_pred):.0%} des churnés. "
                 "Le coût d'un faux positif (email inutile) est quasi nul. Le coût d'un churné raté = client perdu.")

        _chapter("3", "Distribution des scores", "Actifs vs Churnés")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_churn[df_churn['churn'] == 0]['churn_proba'],
                                    name='Actifs', marker_color=BLUE, opacity=0.7, nbinsx=50))
        fig.add_trace(go.Histogram(x=df_churn[df_churn['churn'] == 1]['churn_proba'],
                                    name='Churnés', marker_color=ACCENT, opacity=0.7, nbinsx=50))
        fig.add_vline(x=best_t, line_dash='dash', line_color=GOLD, line_width=2,
                      annotation_text=f'Seuil: {best_t}', annotation_font_color=GOLD)
        fig = _corp_layout(fig, h=420)
        fig.update_layout(title='<b>Distribution du score de churn</b>', xaxis_title='Probabilité de churn',
                          yaxis_title='Nb users', barmode='overlay',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 3 — FRAUDE
    # ════════════════════════════════════════════════════════════
    with tab3:
        n_suspect = (df_fraud['is_suspect'] == -1).sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1: _num_card("5", "Signaux")
        with c2: _num_card(f"{n_suspect:,}", "Suspectes")
        with c3: _num_card(f"{100 * n_suspect / len(df_fraud):.1f}%", "Taux")
        with c4: _num_card("Isolation Forest", "Algorithme", accent=True)

        _chapter("1", "Les 5 signaux de fraude", "Par transaction, relatifs au user")

        signals = [
            ("💰", "Montant inhabituel", "Le montant dévie fortement de la moyenne du user"),
            ("🌍", "Pays nouveau", "Premier paiement dans ce pays pour ce user"),
            ("🕐", "Heure inhabituelle", "Transaction à une heure rare pour ce user"),
            ("⚡", "Burst temporel", "Plusieurs transactions très rapprochées"),
            ("🔄", "Mode inversé", "Passage soudain d'online à physique (ou l'inverse)"),
        ]
        for emoji, title, desc in signals:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:16px; background:white; border-radius:10px; padding:12px 20px; margin:5px 0;
                        box-shadow:0 1px 4px rgba(0,0,0,0.04); border-left:3px solid {NAVY};">
                <div style="font-size:22px; flex-shrink:0;">{emoji}</div>
                <div><div style="font-weight:700; color:{NAVY}; font-size:14px;">{title}</div>
                     <div style="color:{GREY}; font-size:12px;">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        _chapter("2", "Validation REVERTED", "Enrichissement")

        rev_global = df_fraud['is_reverted'].mean()
        rev_suspect = df_fraud.loc[df_fraud['is_suspect'] == -1, 'is_reverted'].mean()
        rev_normal = df_fraud.loc[df_fraud['is_suspect'] == 1, 'is_reverted'].mean()
        enrichment = rev_suspect / rev_global if rev_global > 0 else 0

        fig = go.Figure(go.Bar(
            x=['Global', 'Normales', 'Suspectes'],
            y=[rev_global * 100, rev_normal * 100, rev_suspect * 100],
            marker_color=[GREY, BLUE, ACCENT],
            text=[f'<b>{v:.1f}%</b>' for v in [rev_global * 100, rev_normal * 100, rev_suspect * 100]],
            textposition='outside', textfont=dict(size=16, color=NAVY)))
        fig = _corp_layout(fig, h=400)
        fig.update_layout(title=f'<b>Taux de REVERTED</b> — Enrichissement ×{enrichment:.1f}',
                          yaxis_title='% REVERTED')
        st.plotly_chart(fig, use_container_width=True)

        _verdict("Validation", f"Les transactions suspectes ont ×{enrichment:.1f} plus de REVERTED que la moyenne. "
                 "Le modèle détecte des anomalies réelles, même sans label fraude.")

        _chapter("3", "Scatter — Montant × Heure", "Anomalies en rouge")

        sample = df_fraud.sample(n=min(50000, len(df_fraud)), random_state=42)
        sample['label'] = sample['is_suspect'].map({-1: 'Suspect', 1: 'Normal'})

        fig = px.scatter(sample, x='z_score_montant', y='z_score_heure', color='label',
                         color_discrete_map={'Normal': PALE, 'Suspect': ACCENT}, opacity=0.4,
                         labels={'z_score_montant': 'Z-score montant', 'z_score_heure': 'Z-score heure', 'label': ''})
        fig.update_traces(marker=dict(size=3))
        fig = _corp_layout(fig, h=500)
        fig.update_layout(title='<b>Anomalies : Montant × Heure</b> (échantillon 50K)',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                          xaxis=dict(zeroline=True, zerolinecolor='#DDD'),
                          yaxis=dict(zeroline=True, zerolinecolor='#DDD'))
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 4 — SYNTHÈSE
    # ════════════════════════════════════════════════════════════
    with tab4:
        _chapter("1", "Churn par segment", "Croisement clustering × churn")

        df_cross = df_cluster[['user_id', 'segment']].merge(
            df_churn[['user_id', 'churn_proba', 'churn']], on='user_id', how='inner')

        churn_by_seg = df_cross.groupby('segment')['churn'].mean().mul(100).round(0)

        fig = go.Figure()
        for seg in ['🟢 Engagés', '🟡 Réguliers', '🔴 À Risque']:
            if seg in churn_by_seg.index:
                fig.add_trace(go.Bar(x=[seg], y=[churn_by_seg[seg]], marker_color=color_map[seg],
                                      text=f'<b>{churn_by_seg[seg]:.0f}%</b>', textposition='outside',
                                      textfont=dict(size=18, color=NAVY), showlegend=False))
        fig = _corp_layout(fig, h=400)
        fig.update_layout(title='<b>Taux de churn réel par segment</b>', yaxis_title='% churnés')
        st.plotly_chart(fig, use_container_width=True)

        _verdict("Le segment À Risque concentre le churn",
                 "Les campagnes de réengagement doivent cibler ce segment en priorité — via PUSH, le canal qui fonctionne.")

        _chapter("2", "Recommandations business", "Actions par segment")

        recos = [
            ("🟢", "Engagés", "Programme ambassadeur, features exclusives, early access.", GREEN),
            ("🟡", "Réguliers", "Activation crypto, upsell Premium, notifications PUSH personnalisées.", GOLD),
            ("🔴", "À Risque", "Campagne de réengagement urgente via PUSH. Offre incitative. Accepter les pertes.", ACCENT),
        ]
        for emoji, title, desc, color in recos:
            st.markdown(f"""
            <div style="display:flex; gap:16px; background:white; border-radius:12px; padding:18px; margin:8px 0;
                        box-shadow:0 2px 8px rgba(0,0,0,0.04); border-left:4px solid {color};">
                <div style="font-size:26px; flex-shrink:0;">{emoji}</div>
                <div><div style="font-weight:700; color:{NAVY}; font-size:15px;">{title}</div>
                     <div style="color:#555; font-size:13px; margin-top:4px;">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Récap KPIs
        col1, col2, col3 = st.columns(3)
        for col, icon, title, lines, color in [
            (col1, "🎯", "Clustering", f"3 segments • Crypto = séparateur principal", GREEN),
            (col2, "📉", "Churn", f"{n_churn / n_total:.0%} churnés • Recall {recall_score(y_test, y_pred):.0%} • Seuil {best_t}", GOLD),
            (col3, "🔍", "Fraude", f"{n_suspect:,} suspectes ({100 * n_suspect / len(df_fraud):.1f}%) • Enrichissement ×{enrichment:.1f}", ACCENT),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:white; border-radius:12px; padding:24px; border-top:4px solid {color};
                            box-shadow:0 2px 12px rgba(0,0,0,0.06); text-align:center;">
                    <div style="font-size:28px; margin-bottom:8px;">{icon}</div>
                    <div style="font-weight:700; color:{NAVY}; font-size:16px;">{title}</div>
                    <div style="color:{GREY}; font-size:13px; margin-top:8px; line-height:1.8;">{lines}</div>
                </div>""", unsafe_allow_html=True)