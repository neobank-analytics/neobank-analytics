"""
Module 2 — Transaction Analysis & Campaign Performance
Appelé depuis app.py : from modules.module2 import show_module2
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Palette ──────────────────────────────────────────────────────
NAVY = '#0C1E3C'
ROYAL = '#1B3A6B'
BLUE = '#2E5EA6'
SKY = '#4A90D9'
PALE = '#6B9BD2'
ACCENT = '#E63946'
GOLD = '#D4A843'
GREEN = '#2ECC71'
GREY = '#8899AA'


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
    st.markdown(f'<div class="{cls}"><div class="num-big">{value}</div><div class="num-label">{label}</div></div>',
                unsafe_allow_html=True)


def _chapter(num, title, subtitle=''):
    sub = f'<div class="chapter-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div class="chapter"><div class="chapter-num">{num}</div>'
                f'<div><div class="chapter-title">{title}</div>{sub}</div></div>',
                unsafe_allow_html=True)


def _verdict(title, text):
    st.markdown(f'<div class="verdict"><h4>💡 {title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def _story(text):
    st.markdown(f'<div class="story">{text}</div>', unsafe_allow_html=True)


# ── Point d'entrée ──────────────────────────────────────────────
def show_module2():

    # ── Chargement données ──────────────────────────────────────
    df_tx = pd.read_csv('data/transactions.csv')
    df_tx['created_date'] = pd.to_datetime(df_tx['created_date'], utc=True, format='mixed')
    df_notif = pd.read_csv('data/notifications.csv')
    df_users = pd.read_csv('data/users.csv')

    # Nettoyage outliers TRANSFER P99
    p99 = df_tx.loc[df_tx['transactions_type'] == 'TRANSFER', 'amount_usd'].quantile(0.99)
    df_tx = df_tx[~((df_tx['transactions_type'] == 'TRANSFER') & (df_tx['amount_usd'] > p99))]

    # ── Header ──────────────────────────────────────────────────
    st.title("💳 Module 2 — Analyse des transactions")
    _story("2.7 millions de transactions en 17 mois, une croissance spectaculaire, "
           "mais aussi des <strong>points de friction révélateurs</strong>.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: _num_card(f"{len(df_tx):,.0f}", "Transactions")
    with c2: _num_card(f"{df_tx['transactions_currency'].nunique()}", "Devises")
    with c3: _num_card(f"{df_tx['ea_merchant_country'].nunique()}", "Pays marchands")
    with c4: _num_card("×53", "Croissance 15 mois", accent=True)

    st.divider()

    # ════════════════════════════════════════════════════════════
    # CHAPITRE 1 — Croissance
    # ════════════════════════════════════════════════════════════
    _chapter("1", "Une croissance fulgurante", "Volume mensuel & users actifs")

    df_tx['month'] = df_tx['created_date'].dt.to_period('M').astype(str)
    monthly = df_tx.groupby('month').agg(
        txn_count=('transaction_id', 'count'),
        active_users=('user_id', 'nunique')
    ).reset_index()
    # Exclure mai 2019 (tronqué)
    monthly = monthly[monthly['month'] != '2019-05']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly['month'], y=monthly['txn_count'], name='Transactions',
        marker_color=ROYAL, opacity=0.75,
        hovertemplate='<b>%{x}</b><br>%{y:,.0f} txn<extra></extra>'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['active_users'], name='Users actifs',
        line=dict(color=ACCENT, width=3), marker=dict(size=6, color=ACCENT,
        line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>%{y:,.0f} users<extra></extra>'
    ), secondary_y=True)
    fig = _corp_layout(fig, h=450)
    fig.update_layout(title="Évolution mensuelle — Transactions & Users actifs",
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text='Transactions', secondary_y=False)
    fig.update_yaxes(title_text='Users actifs', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    _verdict("Croissance ×53", "De 6 248 txn/mois (jan 2018) à 334 710 (mars 2019). "
             "Le nombre d'users actifs plafonne à ~11 500 début 2019.")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # CHAPITRE 2 — Typologie & taux d'échec
    # ════════════════════════════════════════════════════════════
    _chapter("2", "Radiographie des transactions", "Types & taux d'échec")

    col1, col2 = st.columns(2)

    # --- Volume par type ---
    with col1:
        type_vol = df_tx['transactions_type'].value_counts().reset_index()
        type_vol.columns = ['type', 'count']

        fig = go.Figure(go.Bar(
            x=type_vol['count'][::-1].values,
            y=type_vol['type'][::-1].values,
            orientation='h',
            marker=dict(color=type_vol['count'][::-1].values,
                        colorscale=[[0, PALE], [0.5, BLUE], [1, NAVY]]),
            text=[f'{c/1e6:.1f}M' if c > 100000 else f'{c/1e3:.0f}K'
                  for c in type_vol['count'][::-1].values],
            textposition='outside', textfont=dict(size=10, color=GREY),
            hovertemplate='<b>%{y}</b><br>%{x:,.0f} txn<extra></extra>'
        ))
        fig = _corp_layout(fig)
        fig.update_layout(title="Volume par type de transaction")
        st.plotly_chart(fig, use_container_width=True)

    # --- Taux d'échec avec gradient ---
    with col2:
        # Compter les états d'échec
        fail_states = ['DECLINED', 'FAILED', 'REVERTED', 'CANCELLED']
        df_tx['is_failed'] = df_tx['transactions_state'].isin(fail_states).astype(int)

        fail_rate = (
            df_tx.groupby('transactions_type')['is_failed']
            .mean().mul(100).round(1)
            .sort_values(ascending=False)
            .reset_index()
        )
        fail_rate.columns = ['type', 'fail_pct']

        fig = px.bar(
            fail_rate, x='fail_pct', y='type', orientation='h',
            color='fail_pct',
            color_continuous_scale=[[0, GREEN], [0.5, GOLD], [1, ACCENT]],
            text='fail_pct',
            labels={'fail_pct': "Taux d'échec (%)", 'type': ''}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig = _corp_layout(fig)
        fig.update_layout(title="Taux d'échec par type (%)",
                          coloraxis_colorbar=dict(title='%'))
        st.plotly_chart(fig, use_container_width=True)

    _verdict("ATM = friction majeure",
             "Les retraits ATM et les TOPUP ont les taux d'échec les plus élevés. "
             "Le taux de déclin ATM (15.5%) est 2.7× supérieur aux paiements carte.")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # CHAPITRE 3 — Digital-first & géographie
    # ════════════════════════════════════════════════════════════
    _chapter("3", "Une banque digital-first", "Online vs physique & géographie marchands")

    col3, col4 = st.columns(2)

    # --- Donut card-present ---
    with col3:
        cp = df_tx['ea_cardholderpresence'].value_counts()
        online = cp.get('FALSE', 0)
        physical = cp.get('TRUE', 0)
        unknown = cp.get('UNKNOWN', 0)
        total_cp = online + physical + unknown

        fig = go.Figure(go.Pie(
            labels=[f'En ligne ({online/total_cp:.0%})', f'Physique ({physical/total_cp:.0%})', 'Inconnu'],
            values=[online, physical, unknown],
            marker_colors=[ROYAL, PALE, '#DDD'], hole=0.6,
            textinfo='label', textfont=dict(size=12, color=NAVY),
            hovertemplate='<b>%{label}</b><br>%{value:,.0f} paiements<extra></extra>'
        ))
        fig = _corp_layout(fig)
        fig.update_layout(title="Répartition des paiements", showlegend=False)
        fig.add_annotation(text=f"<b>{online/total_cp:.0%}</b><br>en ligne",
                           x=0.5, y=0.5, font=dict(size=22, color=NAVY), showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Top 10 pays marchands ---
    with col4:
        top_countries = (
            df_tx['ea_merchant_country'].value_counts().head(10).reset_index()
        )
        top_countries.columns = ['country', 'count']

        fig = go.Figure(go.Bar(
            x=top_countries['count'][::-1].values,
            y=top_countries['country'][::-1].values,
            orientation='h',
            marker=dict(color=top_countries['count'][::-1].values,
                        colorscale=[[0, PALE], [1, NAVY]]),
            hovertemplate='<b>%{y}</b><br>%{x:,.0f} txn<extra></extra>'
        ))
        fig = _corp_layout(fig)
        fig.update_layout(title="Top 10 pays marchands")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════
    # CHAPITRE 4 — Top catégories MCC
    # ════════════════════════════════════════════════════════════
    _chapter("4", "Où va l'argent ?", "Top catégories de dépenses")

    mcc_map = {
        5812: 'Restaurants', 5411: 'Épiceries', 5814: 'Fast food',
        6011: 'ATM / Cash', 4121: 'Taxis', 5499: 'Alimentation',
        4111: 'Transport public', 5813: 'Bars', 5541: 'Stations-service', 7011: 'Hôtels'
    }

    top_mcc = (
        df_tx['ea_merchant_mcc'].value_counts().head(10).reset_index()
    )
    top_mcc.columns = ['mcc', 'count']
    top_mcc['label'] = top_mcc['mcc'].map(mcc_map).fillna(top_mcc['mcc'].astype(str))

    fig = go.Figure(go.Bar(
        x=top_mcc['count'][::-1].values,
        y=top_mcc['label'][::-1].values,
        orientation='h',
        marker=dict(color=top_mcc['count'][::-1].values,
                    colorscale=[[0, PALE], [0.5, BLUE], [1, NAVY]]),
        hovertemplate='<b>%{y}</b><br>%{x:,.0f} txn<extra></extra>'
    ))
    fig = _corp_layout(fig)
    fig.update_layout(title="Top 10 catégories marchands (MCC)")
    st.plotly_chart(fig, use_container_width=True)

    _verdict("Alimentation = 35% des paiements",
             "Restaurants + épiceries + fast food dominent. "
             "Potentiel de partenariats commerciaux identifiable.")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # CHAPITRE 5 — Impact campagnes
    # ════════════════════════════════════════════════════════════
    _chapter("5", "Les notifications font-elles la différence ?", "Impact campagnes × transactions")

    col5, col6 = st.columns(2)

    # Calculer txn par user notifié vs non notifié
    notified_users = set(df_notif['user_id'].unique())
    txn_per_user = df_tx.groupby('user_id').size().reset_index(name='txn_count')
    txn_per_user['notified'] = txn_per_user['user_id'].isin(notified_users)
    avg_notified = txn_per_user[txn_per_user['notified']]['txn_count'].mean()
    avg_not_notified = txn_per_user[~txn_per_user['notified']]['txn_count'].mean()

    with col5:
        fig = go.Figure(go.Bar(
            x=['Notifiés', 'Non notifiés'],
            y=[avg_notified, avg_not_notified],
            marker_color=[ROYAL, PALE],
            text=[f'<b>{avg_notified:.0f}</b>', f'<b>{avg_not_notified:.0f}</b>'],
            textposition='outside',
            textfont=dict(size=18, color=NAVY, family='Outfit'),
            hovertemplate='<b>%{x}</b><br>%{y:.1f} txn en moyenne<extra></extra>'
        ))
        ratio = avg_notified / avg_not_notified
        fig.add_annotation(x=0.5, y=avg_notified * 0.85,
                           text=f"<b>×{ratio:.1f}</b>", showarrow=False,
                           font=dict(size=28, color=ACCENT, family='Outfit'))
        fig = _corp_layout(fig, h=400)
        fig.update_layout(title="Transactions moyennes par user")
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("<br><br>", unsafe_allow_html=True)
        _verdict("Corrélation forte",
                 f"Les utilisateurs notifiés transactent ×{ratio:.1f} plus que les non-notifiés. "
                 "Corrélation ou causalité ? Une analyse temporelle avant/après notification "
                 "permettrait de trancher — mais le signal est fort.")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # CHAPITRE 6 — Direction des flux
    # ════════════════════════════════════════════════════════════
    _chapter("6", "Direction des flux", "Inbound vs Outbound")

    direction = df_tx['direction'].value_counts()

    fig = go.Figure(go.Pie(
        labels=direction.index, values=direction.values,
        marker_colors=[ROYAL, PALE], hole=0.55,
        textinfo='label+percent', textfont=dict(size=14, color=NAVY)
    ))
    fig = _corp_layout(fig, h=380)
    fig.update_layout(title="Répartition Inbound / Outbound", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    _verdict("80/20 sortant",
             "80% des transactions sont sortantes (paiements, achats). "
             "20% sont entrantes (virements reçus, recharges).")