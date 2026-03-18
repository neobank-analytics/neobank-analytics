import plotly.io as pio
import streamlit as st

# ── Executive palette ──────────────────────────────────────────
NAVY    = '#0F172A'    # primary dark
ROYAL   = '#1E3A5F'   # secondary dark
BLUE    = '#3B82F6'   # primary accent
LBLUE   = '#EFF6FF'   # light blue
SKY     = '#60A5FA'   # lighter blue
PALE    = '#DBEAFE'   # very light blue
GREEN   = '#10B981'   # success / positive
LGREEN  = '#ECFDF5'   # light green bg
RED     = '#EF4444'   # danger / negative
AMBER   = '#F59E0B'   # warning
GREY    = '#64748B'   # muted text
BORDER  = '#E2E8F0'   # borders
ACCENT  = '#8B5CF6'   # purple accent for ML module
GOLD    = '#F59E0B'

# ── Template Plotly partagé ───────────────────────────────────────
pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.font.color = '#1E293B'
pio.templates.default = "custom"


def get_theme():
    return st.query_params.get('theme', 'light')


# ── Fonction de mise en page corporate partagée ───────────────────
def corp_layout(fig, h=420):
    theme = get_theme()
    bg         = '#1E293B' if theme == 'dark' else 'white'
    text_color = '#E2E8F0' if theme == 'dark' else '#1E293B'
    grid_color = '#334155' if theme == 'dark' else '#F1F5F9'
    axis_color = '#94A3B8' if theme == 'dark' else '#64748B'
    border_col = '#475569' if theme == 'dark' else '#E2E8F0'
    legend_bg  = '#1E293B' if theme == 'dark' else 'white'

    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(family='Inter, DM Sans, sans-serif', color=text_color, size=12),
        height=h,
        margin=dict(l=48, r=24, t=24, b=40),
        hoverlabel=dict(bgcolor='#0F172A', font_size=13, font_color='white', bordercolor='#0F172A'),
        legend=dict(
            bgcolor=legend_bg,
            font=dict(color=text_color, size=12),
            bordercolor=border_col,
            borderwidth=1
        ),
        title_text=''
    )
    fig.update_xaxes(
        gridcolor=grid_color,
        zeroline=False,
        color=axis_color,
        tickfont=dict(color=axis_color, size=11),
        linecolor=border_col,
        showline=True
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zeroline=False,
        color=axis_color,
        tickfont=dict(color=axis_color, size=11),
        linecolor=border_col,
        showline=False
    )
    fig.update_annotations(font=dict(color=text_color, size=12))
    return fig


# ── CSS global partagé ────────────────────────────────────────────
def apply_global_css():
    theme = get_theme()

    dark_overrides = ""
    if theme == 'dark':
        dark_overrides = """
        .stApp { background-color: #0F172A !important; }
        [data-testid="stMetric"] {
            background: #1E293B !important;
            border-color: #334155 !important;
        }
        [data-testid="stMetricValue"] { color: #F1F5F9 !important; }
        [data-testid="stMetricLabel"] p { color: #94A3B8 !important; }
        [data-testid="stMetricDelta"] { color: #94A3B8 !important; }
        [data-testid="stPlotlyChart"] {
            background: #1E293B !important;
            border-color: #334155 !important;
        }
        h1 { color: #F1F5F9 !important; }
        h2 { color: #F1F5F9 !important; }
        h3 { color: #F1F5F9 !important; }
        h4 { color: #60A5FA !important; }
        p, span { color: #CBD5E1 !important; }
        .stMarkdown p { color: #CBD5E1 !important; }
        strong { color: #F1F5F9 !important; }
        hr { border-top-color: #334155 !important; }
        [data-testid="stInfo"]    { background: #1E3A5F !important; border-left-color: #3B82F6 !important; }
        [data-testid="stWarning"] { background: #431407 !important; border-left-color: #F59E0B !important; }
        [data-testid="stSuccess"] { background: #022C22 !important; border-left-color: #10B981 !important; }
        [data-testid="stInfo"] p, [data-testid="stWarning"] p, [data-testid="stSuccess"] p { color: #E2E8F0 !important; }
        .stTabs [data-baseweb="tab-list"] { background: #1E293B !important; }
        .stTabs [data-baseweb="tab"] { color: #94A3B8 !important; }
        .stTabs [aria-selected="true"] { background: #0F172A !important; color: #F1F5F9 !important; }
        .modnav-bar { border-top-color: #334155 !important; }
        .modnav-btn { background: #1E293B !important; color: #CBD5E1 !important; border-color: #334155 !important; }
        .modnav-btn:hover { background: #334155 !important; color: #F1F5F9 !important; }
        """

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Reset & base ────────────────────────────── */
    * {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background-color: #F8FAFC; }}
    .block-container {{
        padding-top: 0 !important;
        padding-bottom: 3rem;
        max-width: 1280px;
    }}

    /* ── Hide sidebar completely ─────────────────── */
    section[data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="collapsedControl"]  {{ display: none !important; }}

    /* ── Typography ──────────────────────────────── */
    h1 {{ color: #0F172A !important; font-size: 24px !important; font-weight: 800 !important; letter-spacing: -0.5px !important; margin-bottom: 4px !important; }}
    h2 {{ color: #0F172A !important; font-size: 20px !important; font-weight: 700 !important; }}
    h3 {{ color: #0F172A !important; font-size: 16px !important; font-weight: 700 !important; letter-spacing: -0.2px !important; margin-top: 0 !important; }}
    h4 {{ color: #3B82F6 !important; font-size: 13px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }}
    p, span, div {{ color: #334155; }}
    strong {{ color: #0F172A !important; font-weight: 600 !important; }}
    .stMarkdown p {{ color: #334155 !important; font-size: 14px !important; line-height: 1.6 !important; }}

    /* ── Metric cards ────────────────────────────── */
    [data-testid="stMetric"] {{
        background: white !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04) !important;
        border: 1px solid #E2E8F0 !important;
    }}
    [data-testid="stMetricLabel"] p {{ color: #64748B !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.6px !important; }}
    [data-testid="stMetricValue"] {{ color: #0F172A !important; font-size: 28px !important; font-weight: 800 !important; letter-spacing: -0.5px !important; }}
    [data-testid="stMetricDelta"] {{ font-size: 13px !important; font-weight: 500 !important; }}

    /* ── Charts ──────────────────────────────────── */
    [data-testid="stPlotlyChart"] {{
        background: white !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04) !important;
        border: 1px solid #E2E8F0 !important;
    }}

    /* ── Dividers ────────────────────────────────── */
    hr {{ border: none !important; border-top: 1px solid #E2E8F0 !important; margin: 2rem 0 !important; }}

    /* ── Alerts ──────────────────────────────────── */
    [data-testid="stInfo"]    {{ background: #EFF6FF !important; border-left: 3px solid #3B82F6 !important; border-radius: 8px !important; }}
    [data-testid="stWarning"] {{ background: #FFFBEB !important; border-left: 3px solid #F59E0B !important; border-radius: 8px !important; }}
    [data-testid="stSuccess"] {{ background: #ECFDF5 !important; border-left: 3px solid #10B981 !important; border-radius: 8px !important; }}
    [data-testid="stInfo"] p, [data-testid="stWarning"] p, [data-testid="stSuccess"] p {{ font-size: 13px !important; }}

    /* ── Tabs ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: #F1F5F9 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 2px !important;
        border: none !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #64748B !important;
        border: none !important;
        padding: 7px 18px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: white !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.1) !important;
    }}

    /* ── Spinner ─────────────────────────────────── */
    .stSpinner > div {{ border-top-color: #3B82F6 !important; }}

    /* ── Columns gap ─────────────────────────────── */
    [data-testid="stHorizontalBlock"] {{ gap: 16px !important; }}

    /* ── Module navigation bar ───────────────────── */
    .modnav-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid #E2E8F0;
    }}
    .modnav-btn {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: white;
        color: #334155;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 13px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.15s;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }}
    .modnav-btn:hover {{
        background: #F8FAFC;
        border-color: #CBD5E1;
        color: #0F172A;
        box-shadow: 0 2px 6px rgba(15,23,42,0.08);
    }}
    .modnav-next {{
        background: #3B82F6;
        color: white !important;
        border-color: #3B82F6;
    }}
    .modnav-next:hover {{
        background: #2563EB;
        border-color: #2563EB;
        color: white !important;
    }}

    {dark_overrides}

    </style>
    """, unsafe_allow_html=True)


# ── Top navigation bar ────────────────────────────────────────────
def render_nav(active="home"):
    theme = get_theme()
    new_theme    = 'dark' if theme == 'light' else 'light'
    toggle_label = "Mode sombre" if theme == 'light' else "Mode clair"

    st.markdown("""
    <style>
    .topnav-wrapper {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #0F172A;
        padding: 0 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 56px;
        margin: -2rem -2rem 2rem -2rem;
        box-shadow: 0 1px 0 rgba(255,255,255,0.06);
    }
    .topnav-brand {
        color: white;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .topnav-brand span {
        width: 28px; height: 28px;
        background: #3B82F6;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 800;
        color: white;
    }
    .topnav-links {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .topnav-links a {
        color: rgba(255,255,255,0.6);
        text-decoration: none;
        font-size: 13px;
        font-weight: 500;
        padding: 6px 14px;
        border-radius: 6px;
        transition: all 0.15s;
    }
    .topnav-links a:hover {
        color: white;
        background: rgba(255,255,255,0.08);
    }
    .topnav-links a.active {
        color: white;
        background: rgba(59,130,246,0.2);
        border: 1px solid rgba(59,130,246,0.3);
    }
    .topnav-links a.theme-toggle {
        color: rgba(255,255,255,0.5);
        border: 1px solid rgba(255,255,255,0.15);
        margin-left: 8px;
    }
    .topnav-links a.theme-toggle:hover {
        color: white;
        border-color: rgba(255,255,255,0.35);
        background: rgba(255,255,255,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    pages = [
        ("home",         "Vue d'ensemble", "/"),
        ("customer",     "Base Clients",   "/Customer_Base"),
        ("transactions", "Transactions",   "/Transactions"),
        ("ml",           "Machine Learning", "/Machine_Learning"),
    ]

    links_html = ""
    for key, label, href in pages:
        sep    = "?" if "?" not in href else "&"
        href_t = f"{href}{sep}theme={theme}"
        cls    = "active" if active == key else ""
        links_html += f'<a href="{href_t}" class="{cls}" target="_self">{label}</a>'

    links_html += f'<a href="?theme={new_theme}" class="theme-toggle" target="_self">{toggle_label}</a>'

    st.markdown(f"""
    <div class="topnav-wrapper">
        <a class="topnav-brand" href="/?theme={theme}" target="_self">
            <span>N</span>
            Neobank Analytics
        </a>
        <div class="topnav-links">
            {links_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Navigation prev / next entre modules ─────────────────────────
def render_module_nav(current):
    theme = get_theme()

    pages_order = [
        ("home",         "Vue d'ensemble",   "/"),
        ("customer",     "Base Clients",      "/Customer_Base"),
        ("transactions", "Transactions",      "/Transactions"),
        ("ml",           "Machine Learning",  "/Machine_Learning"),
    ]

    current_idx = next(i for i, (key, _, _) in enumerate(pages_order) if key == current)
    prev_page   = pages_order[current_idx - 1] if current_idx > 0 else None
    next_page   = pages_order[current_idx + 1] if current_idx < len(pages_order) - 1 else None

    def href_t(page):
        sep = "?" if "?" not in page[2] else "&"
        return f"{page[2]}{sep}theme={theme}"

    prev_html = (
        f'<a href="{href_t(prev_page)}" class="modnav-btn" target="_self">← {prev_page[1]}</a>'
        if prev_page else '<span></span>'
    )
    next_html = (
        f'<a href="{href_t(next_page)}" class="modnav-btn modnav-next" target="_self">{next_page[1]} →</a>'
        if next_page else '<span></span>'
    )

    st.markdown(f"""
    <div class="modnav-bar">
        {prev_html}
        {next_html}
    </div>
    """, unsafe_allow_html=True)
