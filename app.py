import streamlit as st

st.set_page_config(
    page_title="Neobank Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS custom ───────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ───────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=80)
st.sidebar.title("Neobank Analytics")
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

# ── Routing ──────────────────────────────────────────────────────
if page == "🏠 Accueil":
    from modules.accueil import show
    show()
elif page == "👥 Module 1 — Customer Base":
    from modules.module1 import show
    show()
elif page == "💳 Module 2 — Transactions":
    from modules.module2 import show
    show()
elif page == "🤖 Module 3 — Machine Learning":
    from modules.module3 import show
    show()