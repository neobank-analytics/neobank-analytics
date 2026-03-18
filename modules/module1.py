import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
from datetime import datetime

from modules.utils import corp_layout, NAVY, SKY, BLUE, ACCENT, GOLD, GREEN, GREY, PALE, RED, AMBER, ROYAL, BORDER


@st.cache_data
def load_data():
    df_users         = pd.read_csv('data/users.csv')
    df_notifications = pd.read_csv('data/notifications.csv')
    df_devices       = pd.read_csv('data/devices.csv')

    df_users['age'] = datetime.now().year - df_users['birth_year']
    df_users['age_group'] = pd.cut(
        df_users['age'], bins=[0,25,35,45,60,100],
        labels=['18-25','26-35','36-45','46-60','60+']
    )
    df_users['created_date'] = pd.to_datetime(df_users['created_date'], utc=True)
    df_users['signup_month'] = df_users['created_date'].dt.to_period('M').astype(str)
    df_users['is_premium']   = df_users['plan'].isin(['PREMIUM','METAL','METAL_FREE','PREMIUM_FREE','PREMIUM_OFFER'])
    df_users['user_settings_crypto_unlocked'] = df_users['user_settings_crypto_unlocked'].astype(bool)

    def alpha2_to_alpha3(code):
        try:    return pycountry.countries.get(alpha_2=code).alpha_3
        except: return None
    df_users['country_3'] = df_users['country'].apply(alpha2_to_alpha3)

    plan_groups = {
        'STANDARD':'Standard','PREMIUM':'Premium','PREMIUM_FREE':'Premium',
        'PREMIUM_OFFER':'Premium','METAL':'Metal','METAL_FREE':'Metal'
    }
    df_users['plan_group'] = df_users['plan'].map(plan_groups).fillna('Standard')

    return df_users, df_notifications, df_devices

def show_module1():
    st.title("Module 1 — Base Clients & Notifications")
    st.markdown("#### *Qui sont nos utilisateurs et comment les engage-t-on ?*")
    st.markdown("---")

    df_users, df_notifications, df_devices = load_data()

    # ── KPIs ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Utilisateurs", f"{len(df_users):,}")
    col2.metric("Premium+", f"{df_users['is_premium'].sum():,}")
    col3.metric("Crypto activé", f"{df_users['user_settings_crypto_unlocked'].sum():,}")
    col4.metric("Notifications", f"{len(df_notifications):,}")

    st.markdown("---")

    # ── Carte Europe ─────────────────────────────────────────────
    st.markdown("### Répartition géographique")
    country_counts = (
        df_users.groupby(['country','country_3'])
        .size().reset_index(name='nb_users')
    )
    fig = px.choropleth(
        country_counts, locations='country_3', locationmode='ISO-3',
        color='nb_users', color_continuous_scale='Blues', scope='europe',
        hover_name='country',
        hover_data={'nb_users':':,','country_3':False},
        labels={'nb_users':'Utilisateurs'}
    )
    fig.update_layout(
        height=550, paper_bgcolor='white',
        font=dict(color=NAVY),
        margin=dict(l=0, r=140, t=20, b=0),
        coloraxis_colorbar=dict(
            tickfont=dict(color=GREY, size=12),
            title=dict(text='Utilisateurs', font=dict(color=GREY, size=12)),
            thickness=16,
            len=0.6,
            x=1.0,
        ),
        geo=dict(
            scope='europe', showframe=False,
            projection_scale=2, center=dict(lat=52, lon=10),
            bgcolor='white', lakecolor='#EFF6FF',
            landcolor='#EFF6FF', subunitcolor='rgba(15,23,42,0.12)'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    top10_pays = country_counts.nlargest(10, 'nb_users')
    st.markdown(f"**GB = {top10_pays.iloc[0]['nb_users']:,} users ({top10_pays.iloc[0]['nb_users']/len(df_users):.1%} de la base)**")

    # ── Top 10 pays + Top 10 villes ──────────────────────────────
    st.markdown("---")
    st.markdown("### Top 10 pays & villes")
    top10_villes = df_users['city'].value_counts().head(10).reset_index()
    top10_villes.columns = ['city', 'count']
    top10_villes = top10_villes.sort_values('count')

    fig_geo = make_subplots(rows=1, cols=2, subplot_titles=['Top 10 pays', 'Top 10 villes'])
    fig_geo.add_trace(go.Bar(
        x=top10_pays.sort_values('nb_users')['nb_users'].values,
        y=top10_pays.sort_values('nb_users')['country'].values,
        orientation='h',
        marker_color=BLUE,
        text=top10_pays.sort_values('nb_users')['nb_users'].apply(lambda x: f'{x:,}').values,
        textposition='outside', showlegend=False,
        hovertemplate='<b>%{y}</b> : %{x:,}<extra></extra>'
    ), row=1, col=1)
    fig_geo.add_trace(go.Bar(
        x=top10_villes['count'].values,
        y=top10_villes['city'].values,
        orientation='h',
        marker_color=GREEN,
        text=top10_villes['count'].apply(lambda x: f'{x:,}').values,
        textposition='outside', showlegend=False,
        hovertemplate='<b>%{y}</b> : %{x:,}<extra></extra>'
    ), row=1, col=2)
    fig_geo = corp_layout(fig_geo, h=430)
    st.plotly_chart(fig_geo, use_container_width=True)

    # ── Croissance inscriptions ───────────────────────────────────
    st.markdown("---")
    st.markdown("### Croissance des inscriptions")
    monthly = df_users.groupby('signup_month').size().reset_index(name='nb_users')
    monthly = monthly[monthly['signup_month'] != '2019-01']
    monthly['cumul'] = monthly['nb_users'].cumsum()

    fig2 = make_subplots(specs=[[{'secondary_y':True}]])
    fig2.add_trace(go.Bar(
        x=monthly['signup_month'], y=monthly['nb_users'],
        name='Nouveaux users', marker_color=BLUE, opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Nouveaux : %{y:,}<extra></extra>'
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=monthly['signup_month'], y=monthly['cumul'],
        name='Total cumulé', line=dict(color=RED, width=3),
        mode='lines+markers', marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>Total : %{y:,}<extra></extra>'
    ), secondary_y=True)
    fig2 = corp_layout(fig2, h=420)
    fig2.update_layout(
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(tickangle=45)
    )
    fig2.update_yaxes(title_text='Nouveaux users', secondary_y=False, color=GREY)
    fig2.update_yaxes(title_text='Total cumulé', secondary_y=True, color=GREY)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"**Pic : Décembre 2018 — {monthly.loc[monthly['nb_users'].idxmax(),'nb_users']:,} inscriptions**")

    # ── Distribution âge ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Distribution des âges")

    fig_age = px.histogram(
        df_users,
        x='age',
        nbins=30,
        marginal='box',
        color_discrete_sequence=[BLUE],
        labels={'age': 'Âge', 'count': "Nombre d'utilisateurs"},
        opacity=0.85
    )
    fig_age.update_traces(marker_line_width=1, marker_line_color='white')
    fig_age = corp_layout(fig_age, h=420)
    fig_age.update_layout(bargap=0.05)
    st.plotly_chart(fig_age, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Âge médian", f"{df_users['age'].median():.0f} ans")
    col2.metric("Âge moyen", f"{df_users['age'].mean():.0f} ans")
    col3.metric("Tranche dominante", f"{df_users['age_group'].value_counts().index[0]}")

    # ── Répartition des plans ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Répartition des utilisateurs par plan")
    plans_order = ['Standard', 'Premium', 'Metal']
    plan_counts = df_users['plan_group'].value_counts().reindex(plans_order).reset_index()
    plan_counts.columns = ['plan', 'count']
    plan_counts['pct'] = (plan_counts['count'] / plan_counts['count'].sum() * 100).round(1)
    plan_counts['label'] = plan_counts.apply(lambda r: f"{r['count']:,}  ({r['pct']}%)", axis=1)

    fig3 = go.Figure(go.Bar(
        x=plan_counts['count'].values,
        y=plan_counts['plan'].values,
        orientation='h',
        text=plan_counts['label'].values,
        textposition='outside',
        marker=dict(color=[BLUE, ACCENT, GREEN], line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>%{x:,} users<extra></extra>'
    ))
    fig3 = corp_layout(fig3, h=320)
    fig3.update_layout(
        xaxis=dict(range=[0, plan_counts['count'].max() * 1.35]),
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for col, plan in zip([col1, col2, col3], plans_order):
        row = plan_counts[plan_counts['plan'] == plan]
        col.metric(plan, f"{row['count'].values[0]:,}", f"{row['pct'].values[0]:.1f}%")

    # ── Adoption crypto ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Adoption crypto par plan")
    crypto_by_plan = (
        df_users[df_users['plan_group'].isin(['Standard','Premium','Metal'])]
        .groupby('plan_group')['user_settings_crypto_unlocked']
        .mean().mul(100).round(1).reset_index()
    )
    crypto_by_plan.columns = ['plan','pct_crypto']

    fig4 = px.bar(
        crypto_by_plan, x='plan', y='pct_crypto', color='plan', text='pct_crypto',
        labels={'pct_crypto':'% Crypto activé','plan':''},
        color_discrete_map={'Standard':BLUE,'Premium':ACCENT,'Metal':GREEN}
    )
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4 = corp_layout(fig4, h=400)
    fig4.update_layout(showlegend=False, yaxis=dict(range=[0,110]))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(f"**Metal = {crypto_by_plan[crypto_by_plan['plan']=='Metal']['pct_crypto'].values[0]:.1f}% d'adoption crypto**")

    # ── Devices ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### iOS vs Android")

    df_ud = df_users.merge(df_devices, on='user_id', how='left')
    df_ud = df_ud[df_ud['device_type'].isin(['Apple', 'Android'])]

    fig_dev = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Répartition globale', 'Par plan'],
        specs=[[{'type':'pie'},{'type':'bar'}]]
    )

    device_counts = df_ud['device_type'].value_counts().reset_index()
    device_counts.columns = ['device','count']

    fig_dev.add_trace(go.Pie(
        labels=device_counts['device'],
        values=device_counts['count'],
        hole=0.5,
        marker_colors=['#a2aaad','#3ddc84'],
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b> : %{value:,} users (%{percent})<extra></extra>'
    ), row=1, col=1)

    plans_focus = ['STANDARD','PREMIUM','METAL']
    device_plan = (
        df_ud[df_ud['plan'].isin(plans_focus)]
        .groupby(['plan','device_type'])
        .size().reset_index(name='count')
    )
    device_plan['pct'] = (
        device_plan['count'] /
        device_plan.groupby('plan')['count'].transform('sum') * 100
    ).round(1)

    for device, color in [('Apple','#a2aaad'),('Android','#3ddc84')]:
        d = device_plan[device_plan['device_type'] == device]
        fig_dev.add_trace(go.Bar(
            name=device, x=d['plan'], y=d['pct'],
            marker_color=color,
            text=d['pct'].round(0).astype(str)+'%',
            textposition='inside',
            hovertemplate=f'<b>{device}</b> — %{{x}} : %{{y:.1f}}%<extra></extra>'
        ), row=1, col=2)

    fig_dev = corp_layout(fig_dev, h=430)
    fig_dev.update_layout(
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.04)
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    apple_metal = df_ud[df_ud['plan']=='METAL']['device_type'].value_counts(normalize=True).get('Apple',0)
    st.markdown(f"**Les users METAL sont Apple à {apple_metal:.0%}**")
    st.caption(f"Base appareils : {len(df_devices):,} lignes — ratio 1 appareil / utilisateur")

    # ── Délivrabilité notifications ───────────────────────────────
    st.markdown("---")
    st.markdown("### Performance des canaux de notification")
    canal_stats = (
        df_notifications.groupby('channel').agg(
            envoyes=('status', 'count'),
            reussis=('status', lambda x: (x == 'SENT').sum()),
            echoues=('status', lambda x: (x == 'FAILED').sum())
        ).reset_index()
    )
    canal_stats['taux_delivrabilite'] = (canal_stats['reussis'] / canal_stats['envoyes'] * 100).round(1)

    fig5 = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Volume par canal (SENT vs FAILED)', 'Taux de délivrabilité (%)']
    )
    for status, color, label in [('reussis', GREEN, 'SENT'), ('echoues', RED, 'FAILED')]:
        fig5.add_trace(go.Bar(
            name=label, x=canal_stats['channel'], y=canal_stats[status],
            marker_color=color,
            text=canal_stats[status].apply(lambda x: f'{x:,}'),
            textposition='inside',
            hovertemplate=f'<b>{label}</b> — %{{x}} : %{{y:,}}<extra></extra>'
        ), row=1, col=1)
    colors_canal = [GREEN if v >= 70 else AMBER if v >= 50 else RED
                    for v in canal_stats['taux_delivrabilite']]
    fig5.add_trace(go.Bar(
        x=canal_stats['channel'], y=canal_stats['taux_delivrabilite'],
        marker_color=colors_canal,
        text=canal_stats['taux_delivrabilite'].astype(str) + '%',
        textposition='outside', showlegend=False,
        hovertemplate='<b>%{x}</b> : %{y:.1f}%<extra></extra>'
    ), row=1, col=2)
    fig5.add_hline(y=70, line_dash='dash', line_color=AMBER,
                   annotation_text='Seuil 70%', annotation_font_color=AMBER,
                   row=1, col=2)
    fig5 = corp_layout(fig5, h=430)
    fig5.update_layout(barmode='stack', legend=dict(orientation='h', yanchor='bottom', y=1.04))
    fig5.update_yaxes(range=[0, 115], row=1, col=2)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**SMS = canal mort | PUSH = canal prioritaire**")

    # ── Top 10 raisons de notification ───────────────────────────
    st.markdown("---")
    st.markdown("### Top 10 raisons de notification")
    top_reasons = df_notifications['reason'].value_counts().head(10).reset_index()
    top_reasons.columns = ['reason', 'count']
    top_reasons['label'] = top_reasons['reason'].str.replace('_', ' ').str.title()
    top_reasons = top_reasons.sort_values('count')

    fig_reasons = go.Figure(go.Bar(
        x=top_reasons['count'].values,
        y=top_reasons['label'].values,
        orientation='h',
        marker=dict(color=top_reasons['count'].values,
                    colorscale=[[0, '#DBEAFE'], [1, '#1E3A5F']]),
        text=top_reasons['count'].apply(lambda x: f'{x:,}').values,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>%{x:,} envois<extra></extra>'
    ))
    fig_reasons = corp_layout(fig_reasons, h=430)
    fig_reasons.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(range=[0, top_reasons['count'].max() * 1.15])
    )
    st.plotly_chart(fig_reasons, use_container_width=True)

    # ── Opt-in push & email ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### Opt-in Push & Email")
    col1, col2 = st.columns(2)

    push_counts = df_users['attributes_notifications_marketing_push'].value_counts()
    push_data = pd.DataFrame({
        'label':['Opt-in Push','Opt-out Push'],
        'count':[push_counts.get(True,0)+push_counts.get(1,0),
                 push_counts.get(False,0)+push_counts.get(0,0)]
    })
    fig6 = px.pie(push_data, names='label', values='count', hole=0.5,
                  color_discrete_map={'Opt-in Push':BLUE,'Opt-out Push':RED})
    fig6.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
    fig6 = corp_layout(fig6, h=350)
    fig6.update_layout(showlegend=False)
    with col1:
        st.plotly_chart(fig6, use_container_width=True)

    email_counts = df_users['attributes_notifications_marketing_email'].value_counts()
    email_data = pd.DataFrame({
        'label':['Opt-in Email','Opt-out Email'],
        'count':[email_counts.get(True,0)+email_counts.get(1,0),
                 email_counts.get(False,0)+email_counts.get(0,0)]
    })
    fig7 = px.pie(email_data, names='label', values='count', hole=0.5,
                  color_discrete_map={'Opt-in Email':GREEN,'Opt-out Email':RED})
    fig7.update_traces(texttemplate='%{label}<br>%{percent:.1%}')
    fig7 = corp_layout(fig7, h=350)
    fig7.update_layout(showlegend=False)
    with col2:
        st.plotly_chart(fig7, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Insights clés")
    col1, col2 = st.columns(2)
    with col1:
        st.info("GB = 32% de la base — forte concentration UK")
        st.info("Pic d'inscriptions Décembre 2018")
        st.info("Metal = 59.6% d'adoption crypto")
    with col2:
        st.warning("SMS : 34% de délivrabilité → canal mort")
        st.success("PUSH : 78.5% de délivrabilité → canal prioritaire")
        st.warning("92.6% en Standard → levier d'upsell énorme")
