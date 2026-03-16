import streamlit as st

st.set_page_config(
    page_title="Neobank Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("💳 Neobank Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil",
     "👥 Module 1 — Customer Base",
     "💳 Module 2 — Transactions",
     "🤖 Module 3 — Machine Learning"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("🗓️ Jan 2018 — Mai 2019")
st.sidebar.markdown("👤 19 430 utilisateurs")
st.sidebar.markdown("💳 2.74M transactions")

if page == "🏠 Accueil":
    st.title("💳 Neobank Analytics")
    st.markdown("### Bienvenue sur le dashboard d'analyse")
    
elif page == "👥 Module 1 — Customer Base":
    st.title("👥 Module 1 — Customer Base & Notifications")
    st.markdown("### En construction...")

elif page == "💳 Module 2 — Transactions":
    st.title("💳 Module 2 — Transaction Analysis")
    st.markdown("### En construction...")

elif page == "🤖 Module 3 — Machine Learning":
    st.title("🤖 Module 3 — Machine Learning")
    st.markdown("### En construction...")
