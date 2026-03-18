import streamlit as st
from modules.utils import apply_global_css, render_nav, render_module_nav
from modules.module3 import show_module3

st.set_page_config(page_title="Machine Learning — Neobank Analytics", layout="wide", page_icon=None, initial_sidebar_state="collapsed")

apply_global_css()
render_nav("ml")
show_module3()
render_module_nav("ml")
