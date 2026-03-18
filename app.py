import streamlit as st
from modules.utils import apply_global_css, render_nav, render_module_nav
from modules.accueil import show_accueil

st.set_page_config(page_title="Neobank Analytics", layout="wide", page_icon=None, initial_sidebar_state="collapsed")
apply_global_css()
render_nav("home")
show_accueil()
render_module_nav("home")
