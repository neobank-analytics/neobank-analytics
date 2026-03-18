import streamlit as st
from modules.utils import apply_global_css, render_nav, render_module_nav
from modules.module1 import show_module1

st.set_page_config(page_title="Base Clients — Neobank Analytics", layout="wide", page_icon=None, initial_sidebar_state="collapsed")
apply_global_css()
render_nav("customer")
show_module1()
render_module_nav("customer")
