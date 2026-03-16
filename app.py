import streamlit as st
from modules.module1 import show_module1
from modules.module2 import show_module2
from modules.module3 import show_module3

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
    show_module1()

elif page == "💳 Module 2 — Transactions":
    show_module2()

elif page == "🤖 Module 3 — Machine Learning":
    show_module3()