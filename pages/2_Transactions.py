import streamlit as st
from modules.utils import apply_global_css, render_nav, render_module_nav
from modules.module2 import show_module2

st.set_page_config(page_title="Transactions — Neobank Analytics", layout="wide", page_icon=None, initial_sidebar_state="collapsed")

apply_global_css()
render_nav("transactions")
show_module2()
render_module_nav("transactions")
