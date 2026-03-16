import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
from datetime import datetime

@st.cache_data
def load_data():
    df_users         = pd.read_csv('data/users.csv')
    df_notifications = pd.read_csv('data/notifications.csv')
    df_devices       = pd.read_csv('data/devices.csv')

    # Features dérivées
    df_users['age'] = datetime.now().year - df_users['birth_year']
    df_users['age_group'] = pd.cut(
        df_users['age'],
        bins=[0, 25, 35, 45, 60, 100],
        labels=['18-25', '26-35', '36-45', '46-60', '60+']
    )
    df_users['created_date'] = pd.to_datetime(df_users['created_date'], utc=True)
    df_users['signup_month'] = df_users['created_date'].dt.to_period('M').astype(str)
    df_users['is_premium']   = df_users['plan'].isin(['PREMIUM', 'METAL', 'METAL_FREE', 'PREMIUM_FREE', 'PREMIUM_OFFER'])
    df_users['user_settings_crypto_unlocked'] = df_users['user_settings_crypto_unlocked'].astype(bool)

    def alpha2_to_alpha3(code):
        try:    return pycountry.countries.get(alpha_2=code).alpha_3
        except: return None

    df_users['country_3'] = df_users['country'].apply(alpha2_to_alpha3)

    return df_users, df_notifications, df_devices

def show():
    st.title("👥 Module 1 — Customer Base & Notifications")
    st.markdown("#### *Qui sont nos utilisateurs et comment les engage-t-on ?*")
    st.markdown("---")

    df_users, df_notifications, df_devices = load_data()

    # ── KPIs ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👤 Users", f"{len(df_users):,}")
    col2.metric("💎 Premium+", f"{df_users['is_premium'].sum():,}")
    col3.metric("₿ Crypto", f"{df_users['user_settings_crypto_unlocked'].sum():,}")
    col4.metric("📬 Notifications", f"{len(df_notifications):,}")

    st.markdown("---")

# ── Carte Europe ─────────────────────────────────────────────
    st.markdown("### 🌍 Répartition géographique")

    country_counts = (
        df_users.groupby(['country', 'country_3'])
        .size()
        .reset_index(name='nb_users')
    )

    fig = px.choropleth(
        country_counts,
        locations='country_3',
        locationmode='ISO-3',
        color='nb_users',
        color_continuous_scale='Blues',
        scope='europe',
        title='Nombre d\'utilisateurs par pays',
        hover_name='country',
        hover_data={'nb_users': ':,', 'country_3': False},
        labels={'nb_users': 'Utilisateurs'}
    )
    fig.update_layout(
        height=600,
        title_x=0.5,
        geo=dict(
            scope='europe',
            showframe=False,
            projection_scale=2,
            center=dict(lat=52, lon=10)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 pays
    top10_pays = country_counts.nlargest(10, 'nb_users')
    st.markdown(f"**🇬🇧 GB = {top10_pays.iloc[0]['nb_users']:,} users ({top10_pays.iloc[0]['nb_users']/len(df_users):.1%} de la base)**")

# ── Croissance inscriptions ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Croissance des inscriptions")

    monthly = (
        df_users.groupby('signup_month')
        .size()
        .reset_index(name='nb_users')
    )
    monthly['cumul'] = monthly['nb_users'].cumsum()

    fig2 = make_subplots(specs=[[{'secondary_y': True}]])

    fig2.add_trace(go.Bar(
        x=monthly['signup_month'],
        y=monthly['nb_users'],
        name='Nouveaux users',
        marker_color='#4C9BE8',
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Nouveaux : %{y:,}<extra></extra>'
    ), secondary_y=False)

    fig2.add_trace(go.Scatter(
        x=monthly['signup_month'],
        y=monthly['cumul'],
        name='Total cumulé',
        line=dict(color='#00e5be', width=3),
        mode='lines+markers',
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>Total : %{y:,}<extra></extra>'
    ), secondary_y=True)

    fig2.update_layout(
        height=420,
        title_x=0.5,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(tickangle=45)
    )
    fig2.update_yaxes(title_text='Nouveaux users', secondary_y=False)
    fig2.update_yaxes(title_text='Total cumulé', secondary_y=True)

    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"**💡 Pic : Décembre 2018 — {monthly.loc[monthly['nb_users'].idxmax(), 'nb_users']:,} inscriptions**")

# ── Funnel plans ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔺 Funnel Standard → Premium → Metal")

    plan_counts = df_users['plan'].value_counts().reset_index()
    plan_counts.columns = ['plan', 'count']

    # Regroupement
    plan_groups = {
        'STANDARD': 'Standard',
        'PREMIUM': 'Premium', 'PREMIUM_FREE': 'Premium', 'PREMIUM_OFFER': 'Premium',
        'METAL': 'Metal', 'METAL_FREE': 'Metal'
    }
    df_users['plan_group'] = df_users['plan'].map(plan_groups).fillna('Standard')
    funnel_data = df_users['plan_group'].value_counts().reindex(['Standard', 'Premium', 'Metal'])

    fig3 = go.Figure(go.Funnel(
        y=funnel_data.index.tolist(),
        x=funnel_data.values.tolist(),
        textinfo='value+percent initial',
        marker=dict(color=['#4C9BE8', '#7b61ff', '#00e5be'])
    ))
    fig3.update_layout(height=400, title_x=0.5)
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Standard", f"{funnel_data['Standard']:,}", f"{funnel_data['Standard']/len(df_users):.1%}")
    col2.metric("Premium", f"{funnel_data['Premium']:,}", f"{funnel_data['Premium']/len(df_users):.1%}")
    col3.metric("Metal", f"{funnel_data['Metal']:,}", f"{funnel_data['Metal']/len(df_users):.1%}")

# ── Adoption crypto ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ₿ Adoption crypto par plan")

    crypto_by_plan = (
        df_users[df_users['plan_group'].isin(['Standard', 'Premium', 'Metal'])]
        .groupby('plan_group')['user_settings_crypto_unlocked']
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    crypto_by_plan.columns = ['plan', 'pct_crypto']

    fig4 = px.bar(
        crypto_by_plan,
        x='plan',
        y='pct_crypto',
        color='plan',
        text='pct_crypto',
        title='% Adoption crypto par plan',
        labels={'pct_crypto': '% Crypto activé', 'plan': ''},
        color_discrete_map={
            'Standard': '#4C9BE8',
            'Premium':  '#7b61ff',
            'Metal':    '#00e5be'
        }
    )
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4.update_layout(
        height=400, title_x=0.5,
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(f"**💡 Metal = {crypto_by_plan[crypto_by_plan['plan']=='Metal']['pct_crypto'].values[0]:.1f}% d'adoption crypto**")

    # ── Délivrabilité notifications ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📬 Délivrabilité des notifications par canal")

    notif_stats = (
        df_notifications
        .groupby(['channel', 'status'])
        .size()
        .reset_index(name='count')
    )
    notif_total = notif_stats.groupby('channel')['count'].sum().reset_index(name='total')
    notif_sent  = notif_stats[notif_stats['status'] == 'SENT'].groupby('channel')['count'].sum().reset_index(name='sent')
    notif_rate  = notif_total.merge(notif_sent, on='channel', how='left').fillna(0)
    notif_rate['taux'] = (notif_rate['sent'] / notif_rate['total'] * 100).round(1)
    notif_rate = notif_rate.sort_values('taux', ascending=True)

    fig5 = px.bar(
        notif_rate,
        x='taux',
        y='channel',
        orientation='h',
        text='taux',
        color='taux',
        color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
        title='Taux de délivrabilité par canal (%)',
        labels={'taux': 'Taux (%)', 'channel': ''}
    )
    fig5.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig5.update_layout(
        height=350, title_x=0.5,
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 120])
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**💡 SMS = canal mort | PUSH = canal prioritaire**")

# ── Opt-in push & email ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔔 Opt-in Push & Email")

    col1, col2 = st.columns(2)

    # Push
    push_counts = df_users['attributes_notifications_marketing_push'].value_counts()
    push_data = pd.DataFrame({
        'label': ['Opt-in Push', 'Opt-out Push'],
        'count': [
            push_counts.get(True, 0) + push_counts.get(1, 0),
            push_counts.get(False, 0) + push_counts.get(0, 0)
        ]
    })

    fig6 = px.pie(
        push_data,
        names='label',
        values='count',
        hole=0.5,
        title='Notifications Push',
        color_discrete_map={
            'Opt-in Push': '#4C9BE8',
            'Opt-out Push': '#e74c3c'
        }
    )
    fig6.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
    fig6.update_layout(height=350, title_x=0.5, showlegend=False)

    with col1:
        st.plotly_chart(fig6, use_container_width=True)

    # Email
    email_counts = df_users['attributes_notifications_marketing_email'].value_counts()
    email_data = pd.DataFrame({
        'label': ['Opt-in Email', 'Opt-out Email'],
        'count': [
            email_counts.get(True, 0) + email_counts.get(1, 0),
            email_counts.get(False, 0) + email_counts.get(0, 0)
        ]
    })

    fig7 = px.pie(
        email_data,
        names='label',
        values='count',
        hole=0.5,
        title='Notifications Email',
        color_discrete_map={
            'Opt-in Email': '#00e5be',
            'Opt-out Email': '#e74c3c'
        }
    )
    fig7.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
    fig7.update_layout(height=350, title_x=0.5, showlegend=False)

    with col2:
        st.plotly_chart(fig7, use_container_width=True)

    # ── Synthèse finale ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Insights clés")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🇬🇧 GB = 32% de la base — forte concentration UK")
        st.info("📈 Pic d'inscriptions Décembre 2018")
        st.info("₿ Metal = 59.6% d'adoption crypto")
    with col2:
        st.warning("⚠️ SMS : 34% de délivrabilité → canal mort")
        st.success("✅ PUSH : 78.5% de délivrabilité → canal prioritaire")
        st.warning("⚠️ 92.6% en Standard → levier d'upsell énorme")