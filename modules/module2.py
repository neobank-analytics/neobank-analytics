import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from modules.utils import corp_layout, NAVY, ROYAL, BLUE, SKY, PALE, ACCENT, GOLD, GREEN, GREY, RED, AMBER, BORDER


@st.cache_data
def load_data():
    df_tx    = pd.read_parquet('data/transactions.parquet')
    df_notif = pd.read_csv('data/notifications.csv')
    df_users = pd.read_csv('data/users.csv')

    df_tx['created_date'] = pd.to_datetime(df_tx['created_date'], utc=True, format='mixed')

    p99 = df_tx.loc[df_tx['transactions_type'] == 'TRANSFER', 'amount_usd'].quantile(0.99)
    df_tx = df_tx[~((df_tx['transactions_type'] == 'TRANSFER') & (df_tx['amount_usd'] > p99))].copy()

    fail_states = ['DECLINED', 'FAILED', 'REVERTED', 'CANCELLED']
    df_tx['is_failed'] = df_tx['transactions_state'].isin(fail_states).astype(int)
    df_tx['month'] = df_tx['created_date'].dt.to_period('M').astype(str)

    return df_tx, df_notif, df_users

def show_module2():
    st.title("Module 2 — Analyse des Transactions")
    st.markdown("#### *Quels sont les patterns transactionnels et l'impact des campagnes ?*")
    st.markdown("---")

    df_tx, df_notif, df_users = load_data()

    # ── KPIs ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{len(df_tx):,}")
    col2.metric("Devises", f"{df_tx['transactions_currency'].nunique()}")
    col3.metric("Pays marchands", f"{df_tx['ea_merchant_country'].nunique()}")
    col4.metric("Croissance", "×53")

    st.markdown("---")

    # ── Timeline mensuelle ────────────────────────────────────────
    st.markdown("### Timeline mensuelle")
    monthly = df_tx.groupby('month').agg(
        txn_count=('transaction_id', 'count'),
        active_users=('user_id', 'nunique')
    ).reset_index()
    monthly = monthly[monthly['month'] != '2019-05']

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Bar(
        x=monthly['month'], y=monthly['txn_count'],
        name='Transactions', marker_color=BLUE, opacity=0.75,
        hovertemplate='<b>%{x}</b><br>%{y:,.0f} txn<extra></extra>'
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['active_users'],
        name='Users actifs', line=dict(color=RED, width=3),
        marker=dict(size=6, color=RED, line=dict(width=2, color='white')),
        hovertemplate='<b>%{x}</b><br>%{y:,.0f} users<extra></extra>'
    ), secondary_y=True)
    fig1 = corp_layout(fig1, h=450)
    fig1.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified', xaxis=dict(tickangle=45)
    )
    fig1.update_yaxes(title_text='Transactions', secondary_y=False, color=GREY)
    fig1.update_yaxes(title_text='Users actifs', secondary_y=True, color=GREY)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**De 6 248 txn/mois (jan 2018) à 334 710 (mars 2019) — croissance ×53**")

    st.markdown("---")

    # ── Volume par type + Taux d'échec ────────────────────────────
    st.markdown("### Radiographie des transactions")
    col1, col2 = st.columns(2)

    with col1:
        type_vol = df_tx['transactions_type'].value_counts().reset_index()
        type_vol.columns = ['type', 'count']
        fig2 = go.Figure(go.Bar(
            x=type_vol['count'][::-1].values,
            y=type_vol['type'][::-1].values,
            orientation='h',
            marker=dict(color=type_vol['count'][::-1].values,
                        colorscale=[[0, '#DBEAFE'], [0.5, BLUE], [1, '#1E3A5F']]),
            text=[f'{c/1e6:.1f}M' if c > 100000 else f'{c/1e3:.0f}K' for c in type_vol['count'][::-1].values],
            textposition='outside', textfont=dict(size=10, color=GREY),
            hovertemplate='<b>%{y}</b><br>%{x:,.0f} txn<extra></extra>'
        ))
        fig2 = corp_layout(fig2)
        max_count = type_vol['count'].max()
        fig2.update_layout(xaxis=dict(range=[0, max_count * 1.25]))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fail_rate = (
            df_tx.groupby('transactions_type')['is_failed']
            .mean().mul(100).round(1)
            .sort_values(ascending=False).reset_index()
        )
        fail_rate.columns = ['type', 'fail_pct']
        fig3 = px.bar(
            fail_rate, x='fail_pct', y='type', orientation='h',
            color='fail_pct',
            color_continuous_scale=[[0, GREEN], [0.5, AMBER], [1, RED]],
            text='fail_pct', labels={'fail_pct': "Taux d'échec (%)", 'type': ''}
        )
        fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3 = corp_layout(fig3)
        fig3.update_layout(coloraxis_colorbar=dict(title='%'))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**ATM = 15.5% d'échec → principal point de friction**")

    st.markdown("---")

    # ── Sunburst ──────────────────────────────────────────────────
    st.markdown("### Mix transactionnel — Type → État")
    tx_sunburst = (
        df_tx.groupby(['transactions_type', 'transactions_state'])
        .size().reset_index(name='count')
    )
    fig_sun = px.sunburst(
        tx_sunburst,
        path=['transactions_type', 'transactions_state'],
        values='count',
        color='transactions_type',
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_sun.update_layout(
        height=520,
        paper_bgcolor='white', font=dict(color='#1E293B')
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("---")

    # ── Digital vs physique + Top pays ───────────────────────────
    st.markdown("### Digital vs physique & Géographie")
    col3, col4 = st.columns(2)

    with col3:
        card_presence = (
            df_tx
            .dropna(subset=['ea_cardholderpresence'])
            .query("ea_cardholderpresence != 'UNKNOWN'")
            .groupby('ea_cardholderpresence')
            .size().reset_index(name='count')
        )
        card_presence['label'] = card_presence['ea_cardholderpresence'].map({
            'FALSE': 'Card-not-present (en ligne)',
            'TRUE':  'Card-present (physique)'
        })
        card_presence['pct'] = (card_presence['count'] / card_presence['count'].sum() * 100).round(1)

        fig4 = px.pie(
            card_presence, names='label', values='count', hole=0.5,
            color_discrete_sequence=[BLUE, GREEN],
            labels={'label': ''}
        )
        fig4.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
        fig4 = corp_layout(fig4, h=420)
        st.plotly_chart(fig4, use_container_width=True)
        pct_online = card_presence[card_presence['ea_cardholderpresence']=='FALSE']['pct'].values[0]
        st.markdown(f"**{pct_online}% des paiements sont en ligne → profil digital-first**")

    with col4:
        top_countries = df_tx['ea_merchant_country'].value_counts().head(10).reset_index()
        top_countries.columns = ['country', 'count']
        fig5 = go.Figure(go.Bar(
            x=top_countries['count'][::-1].values,
            y=top_countries['country'][::-1].values,
            orientation='h',
            marker=dict(color=top_countries['count'][::-1].values,
                        colorscale=[[0, '#DBEAFE'], [1, '#1E3A5F']]),
            hovertemplate='<b>%{y}</b><br>%{x:,.0f} txn<extra></extra>'
        ))
        fig5 = corp_layout(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ── Carte marchands ───────────────────────────────────────────
    st.markdown("### Carte des transactions par pays marchand")
    merchant_country = (
        df_tx.groupby('ea_merchant_country')
        .size().reset_index(name='nb_transactions')
    )
    fig_map = px.choropleth(
        merchant_country,
        locations='ea_merchant_country', locationmode='ISO-3',
        color='nb_transactions',
        color_continuous_scale='Blues', scope='europe',
        hover_name='ea_merchant_country',
        hover_data={'nb_transactions': ':,'},
        labels={'nb_transactions': 'Transactions'}
    )
    fig_map.update_layout(
        height=480, paper_bgcolor='white',
        font=dict(color=NAVY),
        margin=dict(l=0, r=140, t=20, b=0),
        coloraxis_colorbar=dict(
            tickfont=dict(color=GREY, size=12),
            title=dict(text='Transactions', font=dict(color=GREY, size=12)),
            thickness=16, len=0.6, x=1.0
        ),
        geo=dict(
            scope='europe', showframe=False,
            projection_scale=2, center=dict(lat=52, lon=10),
            bgcolor='white', lakecolor='#EFF6FF',
            landcolor='#EFF6FF', subunitcolor='rgba(15,23,42,0.12)'
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # ── Top MCC ───────────────────────────────────────────────────
    st.markdown("### Top catégories de dépenses (MCC)")
    mcc_map = {
        5812: 'Restaurants', 5411: 'Épiceries', 5814: 'Fast food',
        6011: 'ATM / Cash', 4121: 'Taxis', 5499: 'Alimentation',
        4111: 'Transport public', 5813: 'Bars', 5541: 'Stations-service', 7011: 'Hôtels'
    }
    top_mcc = df_tx['ea_merchant_mcc'].value_counts().head(10).reset_index()
    top_mcc.columns = ['mcc', 'count']
    top_mcc['label'] = top_mcc['mcc'].map(mcc_map).fillna(top_mcc['mcc'].astype(str))

    fig6 = go.Figure(go.Bar(
        x=top_mcc['count'][::-1].values,
        y=top_mcc['label'][::-1].values,
        orientation='h',
        marker=dict(color=top_mcc['count'][::-1].values,
                    colorscale=[[0, '#DBEAFE'], [0.5, BLUE], [1, '#1E3A5F']]),
        hovertemplate='<b>%{y}</b><br>%{x:,.0f} txn<extra></extra>'
    ))
    fig6 = corp_layout(fig6)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("**Alimentation + restauration = ~35% des paiements**")

    st.markdown("---")

    # ── Impact campagnes ──────────────────────────────────────────
    st.markdown("### Impact des campagnes de notifications")
    col5, col6 = st.columns(2)

    notified_users = set(df_notif['user_id'].unique())
    txn_per_user = df_tx.groupby('user_id').size().reset_index(name='txn_count')
    txn_per_user['notified'] = txn_per_user['user_id'].isin(notified_users)
    avg_notified     = txn_per_user[txn_per_user['notified']]['txn_count'].mean()
    avg_not_notified = txn_per_user[~txn_per_user['notified']]['txn_count'].mean()

    with col5:
        fig7 = go.Figure(go.Bar(
            x=['Notifiés', 'Non notifiés'],
            y=[avg_notified, avg_not_notified],
            marker_color=[BLUE, GREY],
            text=[f'<b>{avg_notified:.0f}</b>', f'<b>{avg_not_notified:.0f}</b>'],
            textposition='outside',
            textfont=dict(size=18, color='#1E293B'),
            hovertemplate='<b>%{x}</b><br>%{y:.1f} txn en moyenne<extra></extra>'
        ))
        ratio = avg_notified / avg_not_notified
        fig7.add_annotation(
            x=0.5, y=avg_notified * 0.85,
            text=f"<b>×{ratio:.1f}</b>", showarrow=False,
            font=dict(size=28, color=RED)
        )
        fig7 = corp_layout(fig7, h=400)
        st.plotly_chart(fig7, use_container_width=True)

    with col6:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info(f"Les users notifiés transactent **×{ratio:.1f}** plus que les non-notifiés.")
        st.info(f"Notifiés : **{avg_notified:.0f} txn** en moyenne")
        st.info(f"Non notifiés : **{avg_not_notified:.0f} txn** en moyenne")

    st.markdown("---")

    # ── Top 10 transactions par montant ──────────────────────────
    st.markdown("### Top 10 transactions COMPLETED")
    top10_tx = (
        df_tx[df_tx['transactions_state'] == 'COMPLETED']
        .nlargest(10, 'amount_usd')
        [['transaction_id', 'user_id', 'transactions_type', 'amount_usd', 'transactions_state']]
        .copy()
    )
    top10_tx['user_short'] = 'user …' + top10_tx['user_id'].str[-4:]
    top10_tx['label'] = top10_tx['transaction_id'].str[-6:] + ' — ' + top10_tx['transactions_type']
    top10_tx = top10_tx.sort_values(['user_short', 'amount_usd'])

    fig_top10 = px.bar(
        top10_tx, x='amount_usd', y='user_short', color='label',
        barmode='stack', orientation='h', text='amount_usd',
        labels={'amount_usd': 'Montant (USD)', 'user_short': 'Utilisateur', 'label': 'Transaction'},
        color_discrete_sequence=[BLUE, ACCENT, GOLD, GREEN, RED, GREY]
    )
    fig_top10.update_traces(
        texttemplate='$%{text:,.0f}',
        textposition='inside',
        insidetextanchor='middle',
        textangle=0
    )
    fig_top10 = corp_layout(fig_top10, h=450)
    fig_top10.update_layout(
        xaxis=dict(range=[0, top10_tx.groupby('user_short')['amount_usd'].sum().max() * 1.2])
    )
    st.plotly_chart(fig_top10, use_container_width=True)

    st.markdown("---")

    # ── Devises ───────────────────────────────────────────────────
    st.markdown("### Répartition des devises")

    crypto    = ['BTC', 'XRP', 'ETH', 'LTC', 'BCH']
    euro_zone = ['EUR', 'GBP', 'PLN', 'RON', 'CZK', 'CHF', 'SEK', 'NOK', 'DKK', 'HUF', 'HRK', 'BGN']

    def categorize(c):
        if c in crypto:    return 'Crypto'
        if c in euro_zone: return 'Europe'
        return 'International'

    currency_all = (
        df_tx['transactions_currency']
        .value_counts().reset_index()
    )
    currency_all.columns = ['currency', 'count']
    currency_all['categorie'] = currency_all['currency'].apply(categorize)

    top5  = currency_all.head(5).copy()
    reste = currency_all.iloc[5:].copy()

    autres_europe   = reste[reste['categorie'] == 'Europe']['count'].sum()
    autres_internat = reste[reste['categorie'] == 'International']['count'].sum()
    autres_crypto   = reste[reste['categorie'] == 'Crypto']['count'].sum()

    extra = pd.DataFrame({
        'currency': ['Autres Europe', 'Autres International', 'Crypto'],
        'count':    [autres_europe, autres_internat, autres_crypto]
    })
    donut_data = pd.concat([
        top5[['currency', 'count']],
        extra
    ], ignore_index=True)

    color_devise_map = {
        'EUR':                  BLUE,
        'GBP':                  ROYAL,
        'PLN':                  ACCENT,
        'RON':                  GREEN,
        'USD':                  SKY,
        'Autres Europe':        GOLD,
        'Autres International': GREY,
        'Crypto':               RED
    }

    # Donut volume USD
    currency_vol = (
        df_tx.groupby('transactions_currency')['amount_usd']
        .sum().sort_values(ascending=False).reset_index()
    )
    currency_vol.columns = ['currency', 'vol']
    currency_vol['categorie'] = currency_vol['currency'].apply(categorize)
    top5_vol  = currency_vol.head(5).copy()
    reste_vol = currency_vol.iloc[5:].copy()
    extra_vol = pd.DataFrame({
        'currency': ['Autres Europe', 'Autres International', 'Crypto'],
        'vol': [
            reste_vol[reste_vol['categorie'] == 'Europe']['vol'].sum(),
            reste_vol[reste_vol['categorie'] == 'International']['vol'].sum(),
            reste_vol[reste_vol['categorie'] == 'Crypto']['vol'].sum(),
        ]
    })
    donut_data_vol = pd.concat([top5_vol[['currency', 'vol']], extra_vol], ignore_index=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Par nombre de transactions**")
        fig8 = px.pie(
            donut_data, names='currency', values='count', hole=0.5,
            color_discrete_map=color_devise_map
        )
        fig8.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
        fig8 = corp_layout(fig8, h=450)
        fig8.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
        st.plotly_chart(fig8, use_container_width=True)

    with col_d2:
        st.markdown("**Par volume total (USD)**")
        fig8b = px.pie(
            donut_data_vol, names='currency', values='vol', hole=0.5,
            color_discrete_map=color_devise_map
        )
        fig8b.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
        fig8b = corp_layout(fig8b, h=450)
        fig8b.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
        st.plotly_chart(fig8b, use_container_width=True)

    st.markdown(f"**EUR + GBP = {currency_all.head(2)['count'].sum() / currency_all['count'].sum():.0%} des transactions**")

    st.markdown("---")

    # ── Insights clés ─────────────────────────────────────────────
    st.markdown("### Insights clés")
    col1, col2 = st.columns(2)
    with col1:
        st.success("Croissance ×53 en 15 mois")
        st.success("81% des paiements en ligne → digital-first")
        st.info(f"Notifiés = ×{ratio:.1f} transactions")
    with col2:
        st.warning("ATM : 15.5% d'échec → point de friction")
        st.warning("Transactions aberrantes jusqu'à $85Md → exclues")
        st.info("EUR + GBP = 75% du volume")
