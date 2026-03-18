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
import json

from sklearn.metrics import recall_score

from modules.utils import corp_layout, NAVY, ROYAL, BLUE, SKY, PALE, ACCENT, GOLD, GREEN, GREY, RED, AMBER, BORDER


def _num_card(value, label, accent=False):
    cls = 'num-card accent' if accent else 'num-card'
    st.markdown(f'<div class="{cls}"><div class="num-big">{value}</div><div class="num-label">{label}</div></div>', unsafe_allow_html=True)

def _chapter(num, title, subtitle=''):
    sub = f'<div class="chapter-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div class="chapter"><div class="chapter-num">{num}</div><div><div class="chapter-title">{title}</div>{sub}</div></div>', unsafe_allow_html=True)

def _verdict(title, text):
    st.markdown(f'<div class="verdict"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

def _story(text):
    st.markdown(f'<div class="story">{text}</div>', unsafe_allow_html=True)


# ── Chargement & feature engineering (caché pour performance) ────
@st.cache_data
def load_and_prepare():
    """Charge les CSV et prépare toutes les données pour les 3 modèles."""

    df_tx = pd.read_parquet('data/transactions.parquet')
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


# ── Chargement résultats pré-calculés ───────────────────────────
@st.cache_data
def run_clustering(_unused):
    df = pd.read_parquet('data/ml/cluster.parquet')
    meta = json.load(open('data/ml/cluster_meta.json'))
    profil = pd.DataFrame(meta['profil'])
    names = {int(k): v for k, v in meta['names'].items()}
    freq_order = [int(x) for x in meta['freq_order']]
    return df, profil, names, freq_order, meta['inertias'], meta['silhouettes']


@st.cache_data
def run_churn(_unused):
    df = pd.read_parquet('data/ml/churn.parquet')
    importances = pd.read_parquet('data/ml/churn_importances.parquet')
    test_data = pd.read_parquet('data/ml/churn_test.parquet')
    meta = json.load(open('data/ml/churn_meta.json'))
    cm = np.array(meta['cm'])
    y_test = test_data['y_test']
    y_pred = test_data['y_pred'].values
    y_proba = test_data['y_proba'].values
    return df, cm, importances, meta['best_t'], meta['best_f1'], meta['results'], y_test, y_pred, y_proba


@st.cache_data
def load_fraud_data():
    meta = json.load(open('data/ml/fraud_meta.json'))
    scatter = pd.read_parquet('data/ml/fraud_scatter.parquet')
    histogram = pd.read_parquet('data/ml/fraud_histogram.parquet')
    timeline = pd.read_parquet('data/ml/fraud_timeline.parquet')
    top_users = pd.DataFrame(meta['top_suspect_users'])
    return meta, scatter, histogram, timeline, top_users


# ════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════
def show_module3():

    st.markdown("""
    <style>
    /* ── Chapter headers ─────────────────── */
    .chapter {
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 36px 0 20px 0;
        padding: 18px 24px;
        background: white;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    }
    .chapter-num {
        font-size: 28px;
        font-weight: 800;
        color: #3B82F6;
        min-width: 36px;
        line-height: 1;
    }
    .chapter-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.3;
    }
    .chapter-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 3px;
        font-weight: 400;
    }

    /* ── Num cards ───────────────────────── */
    .num-card {
        background: white;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05);
        margin-bottom: 12px;
    }
    .num-card.accent {
        border-top: 3px solid #3B82F6;
    }
    .num-big {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .num-label {
        font-size: 11px;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }

    /* ── Verdict boxes ───────────────────── */
    .verdict {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 20px 0;
    }
    .verdict h4 {
        color: #1E3A5F !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 6px !important;
    }
    .verdict p {
        color: #334155 !important;
        font-size: 14px !important;
        margin: 0 !important;
        line-height: 1.6 !important;
    }

    /* ── Selectbox ───────────────────────── */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    div[data-baseweb="select"] span {
        color: #0F172A !important;
        font-weight: 500 !important;
    }

    /* ── Story intro ─────────────────────── */
    .story {
        font-size: 15px;
        color: #475569;
        line-height: 1.7;
        padding: 16px 0 8px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Module 3 — Machine Learning & Synthèse")
    _story("On sait <em>qui</em> ils sont et <em>ce qu'ils font</em>. "
           "Maintenant : <strong>segmenter, prédire le churn, détecter la fraude</strong>.")

    # Chargement
    with st.spinner("Chargement des données..."):
        df_cluster, profil, names, freq_order, inertias, silhouettes = run_clustering(None)
        df_churn, cm, importances, best_t, best_f1, threshold_results, y_test, y_pred, y_proba = run_churn(None)
        fraud_meta, fraud_scatter, fraud_histogram, fraud_timeline, top_suspect_users = load_fraud_data()
        n_suspect = fraud_meta['n_suspect']
        enrichment = fraud_meta['enrichment']

    # ── Sous-navigation ──────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Segmentation", "Churn", "Fraude", "Synthese"
    ])

    color_map = {'Engages': GREEN, 'Reguliers': AMBER, 'A Risque': RED}

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

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(2, 10)), y=inertias, mode='lines+markers',
            name='Inertie (Elbow)', line=dict(color=BLUE, width=2.5),
            marker=dict(size=9, color=BLUE), yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(2, 10)), y=silhouettes, mode='lines+markers',
            name='Silhouette', line=dict(color=GREEN, width=2.5),
            marker=dict(size=9, color=GREEN), yaxis='y2'
        ))
        fig = corp_layout(fig, h=400)
        fig.update_layout(
            xaxis=dict(title='K', dtick=1),
            yaxis=dict(title=dict(text='Inertie', font=dict(color=BLUE)), tickfont=dict(color=BLUE)),
            yaxis2=dict(title=dict(text='Score Silhouette', font=dict(color=GREEN)), tickfont=dict(color=GREEN),
                        overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Meilleur silhouette à k=2 ({silhouettes[0]:.3f}), mais **k=3 retenu** pour l'interprétabilité business ({silhouettes[1]:.3f}).")

        _chapter("2", "Les 3 segments", "Répartition")

        sizes = df_cluster['segment'].value_counts()
        fig = go.Figure(go.Pie(
            labels=sizes.index, values=sizes.values,
            marker_colors=[color_map[s] for s in sizes.index],
            hole=0.55, textinfo='label+percent', textfont=dict(size=13),
            hovertemplate='<b>%{label}</b><br>%{value:,.0f} users<extra></extra>'
        ))
        fig.add_annotation(text=f"<b>{len(df_cluster):,}</b><br>users", x=0.5, y=0.5,
                           font=dict(size=18, color='#0F172A'), showarrow=False)
        fig = corp_layout(fig, h=420)
        fig.update_layout(showlegend=False)
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
                                  mode='lines+markers', name='Recall', line=dict(color=RED, width=2)))
        fig.add_trace(go.Scatter(x=df_thresholds['seuil'], y=df_thresholds['f1'],
                                  mode='lines+markers', name='F1', line=dict(color=GREEN, width=2.5)))
        fig.add_vline(x=best_t, line_dash='dash', line_color=AMBER, annotation_text=f'Optimal: {best_t}',
                      annotation_font_color=AMBER)
        fig = corp_layout(fig, h=400)
        fig.update_layout(xaxis_title='Seuil', yaxis_title='Score',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        _chapter("2", "Résultats", "Matrice de confusion & Feature importance")

        col1, col2 = st.columns(2)

        with col1:
            labels_cm = ['Actif', 'Churned']
            fig = go.Figure(go.Heatmap(
                z=cm[::-1], x=labels_cm, y=labels_cm[::-1],
                text=[[f"<b>{v:,}</b>" for v in row] for row in cm[::-1]],
                texttemplate="%{text}", textfont=dict(size=22, color='white'),
                colorscale=[[0, '#DBEAFE'], [1, '#1E3A5F']], showscale=False))
            fig.update_layout(
                height=400, width=450, plot_bgcolor='white', paper_bgcolor='white',
                font=dict(family='Inter', color='#0F172A'),
                xaxis=dict(title=dict(text='<b>Prédit</b>', font=dict(color='#0F172A')),
                           tickfont=dict(color='#0F172A')),
                yaxis=dict(title=dict(text='<b>Réel</b>', font=dict(color='#0F172A')),
                           tickfont=dict(color='#0F172A'))
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style="background:white; border-radius:10px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="color:{GREEN}; font-size:22px; font-weight:800;">{cm[1][1]:,}</div>
                <div style="color:{GREY}; font-size:12px;">Vrais positifs — churnés détectés</div>
                <div style="color:{RED}; font-size:22px; font-weight:800; margin-top:8px;">{cm[1][0]:,}</div>
                <div style="color:{GREY}; font-size:12px;">Faux négatifs — churnés ratés</div>
                <div style="color:{AMBER}; font-size:22px; font-weight:800; margin-top:8px;">{cm[0][1]:,}</div>
                <div style="color:{GREY}; font-size:12px;">Faux positifs — actifs contactés à tort</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            labels_feat = {'frequence': 'Fréquence', 'montant_moyen': 'Montant moy.', 'nb_devises': 'Nb devises',
                           'pct_online': '% Online', 'plan_ordinal': 'Plan', 'crypto': 'Crypto', 'anciennete': 'Ancienneté'}
            imp = importances.copy()
            imp['label'] = imp['feature'].map(labels_feat)

            fig = go.Figure(go.Bar(x=imp['importance'], y=imp['label'], orientation='h',
                                    marker=dict(color=imp['importance'].values, colorscale=[[0, '#DBEAFE'], [0.5, BLUE], [1, '#1E3A5F']]),
                                    text=[f"<b>{v:.3f}</b>" for v in imp['importance']], textposition='outside',
                                    textfont=dict(size=12, color='#1E293B')))
            fig = corp_layout(fig, h=400)
            fig.update_layout(xaxis=dict(range=[0, imp['importance'].max() * 1.3]))
            st.plotly_chart(fig, use_container_width=True)

        _verdict("Compromis seuil", f"À {best_t}, on attrape {recall_score(y_test, y_pred):.0%} des churnés. "
                 "Le coût d'un faux positif (email inutile) est quasi nul. Le coût d'un churné raté = client perdu.")

        _chapter("3", "Distribution des scores", "Actifs vs Churnés")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_churn[df_churn['churn'] == 0]['churn_proba'],
                                    name='Actifs', marker_color=BLUE, opacity=0.7, nbinsx=50))
        fig.add_trace(go.Histogram(x=df_churn[df_churn['churn'] == 1]['churn_proba'],
                                    name='Churnés', marker_color=RED, opacity=0.7, nbinsx=50))
        fig.add_vline(x=best_t, line_dash='dash', line_color=AMBER, line_width=2,
                      annotation_text=f'Seuil: {best_t}', annotation_font_color=AMBER)
        fig = corp_layout(fig, h=420)
        fig.update_layout(xaxis_title='Probabilité de churn',
                          yaxis_title='Nb users', barmode='overlay',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 3 — FRAUDE
    # ════════════════════════════════════════════════════════════
    with tab3:
        c1, c2, c3, c4 = st.columns(4)
        with c1: _num_card("5", "Signaux")
        with c2: _num_card(f"{n_suspect:,}", "Suspectes")
        with c3: _num_card(f"{100 * n_suspect / fraud_meta['n_total']:.1f}%", "Taux")
        with c4: _num_card("Isolation Forest", "Algorithme", accent=True)

        _chapter("1", "Les 5 signaux de fraude", "Par transaction, relatifs au user")

        signals = [
            ("Montant inhabituel", "Le montant dévie fortement de la moyenne du user"),
            ("Pays nouveau", "Premier paiement dans ce pays pour ce user"),
            ("Heure inhabituelle", "Transaction à une heure rare pour ce user"),
            ("Burst temporel", "Plusieurs transactions très rapprochées"),
            ("Mode inversé", "Passage soudain d'online à physique (ou l'inverse)"),
        ]
        for title, desc in signals:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:16px; background:white; border-radius:10px; padding:12px 20px; margin:5px 0;
                        box-shadow:0 1px 4px rgba(0,0,0,0.04); border-left:3px solid #3B82F6;">
                <div><div style="font-weight:700; color:#0F172A; font-size:14px;">{title}</div>
                     <div style="color:{GREY}; font-size:12px;">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        _chapter("2", "Validation REVERTED", "Enrichissement")

        rev_global  = fraud_meta['rev_global']
        rev_suspect = fraud_meta['rev_suspect']
        rev_normal  = fraud_meta['rev_normal']

        fig = go.Figure(go.Bar(
            x=['Global', 'Normales', 'Suspectes'],
            y=[rev_global * 100, rev_normal * 100, rev_suspect * 100],
            marker_color=[GREY, BLUE, RED],
            text=[f'<b>{v:.1f}%</b>' for v in [rev_global * 100, rev_normal * 100, rev_suspect * 100]],
            textposition='outside', textfont=dict(size=16, color='#1E293B')))
        fig = corp_layout(fig, h=400)
        fig.update_layout(yaxis_title='% REVERTED')
        st.plotly_chart(fig, use_container_width=True)

        _verdict("Validation", f"Les transactions suspectes ont ×{enrichment:.1f} plus de REVERTED que la moyenne. "
                 "Le modèle détecte des anomalies réelles, même sans label fraude.")

        _chapter("3", "Scatter — Montant × Heure", "Anomalies en rouge")

        sample = fraud_scatter.copy()
        sample['label'] = sample['is_suspect'].map({-1: 'Suspect', 1: 'Normal'})

        fig = px.scatter(sample, x='z_score_montant', y='z_score_heure', color='label',
                         color_discrete_map={'Normal': '#DBEAFE', 'Suspect': RED}, opacity=0.4,
                         labels={'z_score_montant': 'Z-score montant', 'z_score_heure': 'Z-score heure', 'label': ''})
        fig.update_traces(marker=dict(size=3))
        fig = corp_layout(fig, h=500)
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                          xaxis=dict(zeroline=True, zerolinecolor='#DDD'),
                          yaxis=dict(zeroline=True, zerolinecolor='#DDD'))
        st.plotly_chart(fig, use_container_width=True)

        _chapter("4", "Distribution du score d'anomalie", "REVERTED vs Non-REVERTED")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=fraud_histogram.loc[fraud_histogram['is_reverted'] == 0, 'anomaly_score'],
            name='Non-REVERTED', marker_color=BLUE, opacity=0.7, nbinsx=100
        ))
        fig.add_trace(go.Histogram(
            x=fraud_histogram.loc[fraud_histogram['is_reverted'] == 1, 'anomaly_score'],
            name='REVERTED', marker_color=RED, opacity=0.7, nbinsx=100
        ))
        fig = corp_layout(fig, h=420)
        fig.update_layout(
            xaxis_title="Score d'anomalie (plus négatif = plus suspect)",
            yaxis_title='Nb transactions', barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

        _chapter("5", "Timeline d'un user suspect", "Sélectionne un user pour inspecter ses transactions")

        top_suspect_users['label'] = top_suspect_users.apply(
            lambda r: f"{r['user_id']}  ({r['nb_suspects']} suspectes)", axis=1
        )

        selected_label = st.selectbox(
            "Choisir un utilisateur :",
            options=top_suspect_users['label'].tolist(),
            index=0
        )
        selected_user = top_suspect_users.loc[
            top_suspect_users['label'] == selected_label, 'user_id'
        ].values[0]

        user_timeline = fraud_timeline[fraud_timeline['user_id'] == selected_user].copy()
        user_timeline['label'] = user_timeline['is_suspect'].map({-1: 'Suspect', 1: 'Normal'})
        n_total    = len(user_timeline)
        n_suspects = (user_timeline['is_suspect'] == -1).sum()

        fig = px.scatter(
            user_timeline, x='created_date', y='amount_usd',
            color='label',
            color_discrete_map={'Normal': BLUE, 'Suspect': RED},
            size=user_timeline['is_suspect'].map({-1: 12, 1: 4}).tolist(),
            hover_data=['transactions_type', 'ea_merchant_country', 'z_score_montant'],
            labels={'created_date': '', 'amount_usd': 'Montant ($)', 'label': ''}
        )
        fig = corp_layout(fig, h=450)
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            title=dict(
                text=f'{selected_user} — {n_total} transactions dont {n_suspects} suspectes',
                x=0.5, font_size=14
            ),
            xaxis=dict(gridcolor='#F0F2F5'),
            yaxis=dict(gridcolor='#F0F2F5')
        )
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 4 — SYNTHÈSE
    # ════════════════════════════════════════════════════════════
    with tab4:
        _chapter("1", "Churn par segment", "Croisement clustering × churn")

        df_cross = df_cluster[['user_id', 'segment']].merge(
            df_churn[['user_id', 'churn_proba', 'churn']], on='user_id', how='inner')

        fig = go.Figure()
        for seg, color in color_map.items():
            subset = df_cross[df_cross['segment'] == seg]['churn_proba']
            fig.add_trace(go.Box(
                y=subset, name=seg, marker_color=color, boxmean=True,
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
        fig = corp_layout(fig, h=450)
        fig.update_layout(yaxis_title='Probabilité de churn', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        _verdict("Le segment À Risque concentre le churn",
                 "Les campagnes de réengagement doivent cibler ce segment en priorité — via PUSH, le canal qui fonctionne.")

        _chapter("2", "Recommandations business", "Actions par segment")

        recos = [
            ("Engages", "Programme ambassadeur, features exclusives, early access.", GREEN),
            ("Reguliers", "Activation crypto, upsell Premium, notifications PUSH personnalisées.", AMBER),
            ("A Risque", "Campagne de réengagement urgente via PUSH. Offre incitative. Accepter les pertes.", RED),
        ]
        for title, desc, color in recos:
            st.markdown(f"""
            <div style="display:flex; gap:16px; background:white; border-radius:12px; padding:18px; margin:8px 0;
                        box-shadow:0 1px 3px rgba(15,23,42,0.06); border-left:4px solid {color}; border:1px solid #E2E8F0;">
                <div><div style="font-weight:700; color:#0F172A; font-size:15px;">{title}</div>
                     <div style="color:#64748B; font-size:13px; margin-top:4px;">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Récap KPIs
        col1, col2, col3 = st.columns(3)
        for col, title, lines, color in [
            (col1, "Clustering", f"3 segments • Crypto = séparateur principal", GREEN),
            (col2, "Churn", f"{n_churn / n_total:.0%} churnés • Recall {recall_score(y_test, y_pred):.0%} • Seuil {best_t}", AMBER),
            (col3, "Fraude", f"{n_suspect:,} suspectes ({100 * n_suspect / fraud_meta['n_total']:.1f}%) • Enrichissement ×{enrichment:.1f}", RED),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:white; border-radius:12px; padding:24px; border-top:4px solid {color};
                            box-shadow:0 1px 3px rgba(15,23,42,0.06); border:1px solid #E2E8F0; text-align:center;">
                    <div style="font-weight:700; color:#0F172A; font-size:16px;">{title}</div>
                    <div style="color:#64748B; font-size:13px; margin-top:8px; line-height:1.8;">{lines}</div>
                </div>""", unsafe_allow_html=True)
