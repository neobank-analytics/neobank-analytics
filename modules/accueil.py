import streamlit as st

def show_accueil():
    # Header
    st.markdown("""
    <div style="padding: 40px 0 32px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 32px;">
        <div style="font-size: 11px; font-weight: 600; color: #3B82F6; letter-spacing: 2px;
                    text-transform: uppercase; margin-bottom: 10px;">
            Vue d'ensemble · Jan. 2018 – Mai 2019
        </div>
        <div style="font-size: 30px; font-weight: 800; color: #0F172A; letter-spacing: -0.8px;
                    margin-bottom: 8px;">
            Neobank Analytics Dashboard
        </div>
        <div style="font-size: 15px; color: #64748B; font-weight: 400; max-width: 560px;">
            Synthèse exécutive de la croissance utilisateurs, des performances transactionnelles
            et des insights machine learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Utilisateurs",          "19 430")
    col2.metric("Transactions",          "2,74M")
    col3.metric("Notifications envoyées","121 813")
    col4.metric("Appareils suivis",      "19 430")

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    # Section title
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 600; color: #3B82F6; letter-spacing: 2px;
                    text-transform: uppercase; margin-bottom: 6px;">Modules d'analyse</div>
        <div style="font-size: 18px; font-weight: 700; color: #0F172A;">Sélectionnez un module</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    modules = [
        (col1, "Base Clients", "Segmentation & performance des notifications", "#3B82F6", [
            ("32%",    "des utilisateurs basés au Royaume-Uni"),
            ("92,6%",  "sur le plan Standard — fort potentiel d'upsell"),
            ("18,1%",  "taux d'adoption crypto"),
            ("34%",    "taux de délivrabilité SMS — canal déprécié"),
        ], "/Customer_Base"),
        (col2, "Transactions", "Croissance du volume & attribution des campagnes", "#10B981", [
            ("×53",    "croissance du volume de transactions en 15 mois"),
            ("81%",    "des paiements traités en ligne"),
            ("16%",    "taux d'échec ATM — principal point de friction"),
            ("×2",     "transactions supplémentaires pour les utilisateurs notifiés"),
        ], "/Transactions"),
        (col3, "Machine Learning", "Segmentation, prédiction du churn & détection de fraude", "#8B5CF6", [
            ("3",       "segments clients distincts identifiés"),
            ("23%",     "des utilisateurs à risque élevé de churn"),
            ("54 000",  "transactions suspectes détectées"),
            ("×3",      "enrichissement fraude vs. la moyenne globale"),
        ], "/Machine_Learning"),
    ]

    for col, title, subtitle, color, stats, href in modules:
        stats_html = ""
        for value, label in stats:
            stats_html += f"""
            <div style="display:flex; align-items:baseline; gap:10px; padding: 10px 0;
                        border-bottom: 1px solid #F1F5F9;">
                <div style="font-size:18px; font-weight:800; color:{color}; min-width:56px;">{value}</div>
                <div style="font-size:12px; color:#64748B; line-height:1.4;">{label}</div>
            </div>"""

        with col:
            st.markdown(f"""
            <a href="{href}" target="_self" style="text-decoration:none;">
            <div style="background:white; border-radius:14px; padding:28px;
                        border:1px solid #E2E8F0; cursor:pointer;
                        box-shadow:0 1px 3px rgba(15,23,42,0.06);">
                <div style="width:36px; height:4px; background:{color}; border-radius:2px; margin-bottom:20px;"></div>
                <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:4px;">{title}</div>
                <div style="font-size:12px; color:#64748B; margin-bottom:20px;">{subtitle}</div>
                {stats_html}
                <div style="margin-top:16px; font-size:12px; font-weight:600; color:{color};">
                    Voir l'analyse →
                </div>
            </div>
            </a>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 48px; padding-top: 24px; border-top: 1px solid #E2E8F0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#94A3B8; font-size:12px; text-align:center; letter-spacing:0.5px;'>
    Développé avec Python · Plotly · Streamlit · Scikit-learn &nbsp;|&nbsp; Données : 2018–2019
    </p>
    """, unsafe_allow_html=True)
