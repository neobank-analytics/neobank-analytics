import streamlit as st

def show():
    st.title("💳 Neobank Analytics")
    st.markdown("#### Analyse complète d'une néobanque — Jan 2018 → Mai 2019")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👤 Utilisateurs", "19 430")
    col2.metric("💳 Transactions", "2.74M")
    col3.metric("📬 Notifications", "121 813")
    col4.metric("📱 Devices", "19 431")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👥 Module 1")
        st.markdown("**Customer Base & Notifications**")
        st.markdown("""
        - 🇬🇧 GB = 32% de la base
        - 📈 Pic inscriptions Déc 2018
        - 💎 92.6% Standard
        - 📬 SMS mort (34% délivrabilité)
        - ₿ 18.1% adoption crypto
        """)

    with col2:
        st.markdown("### 💳 Module 2")
        st.markdown("**Transaction Analysis**")
        st.markdown("""
        - 📈 Croissance ×53 en 15 mois
        - 🌐 81% paiements en ligne
        - ⚠️ ATM : 16% taux d'échec
        - 📬 Notifiés = ×2 transactions
        - 💱 EUR + GBP = 75% du volume
        """)

    with col3:
        st.markdown("### 🤖 Module 3")
        st.markdown("**Machine Learning**")
        st.markdown("""
        - 🎯 3 segments identifiés
        - 📉 23% users à risque churn
        - 🔍 54K transactions suspectes
        - ✅ Recall churn : 67%
        - ×3 enrichissement fraude
        """)

    st.markdown("---")
    st.markdown("*Projet réalisé avec Python, Plotly & Streamlit*")