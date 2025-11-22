import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors
from datetime import datetime
import numpy.random as npr

# ========== CONFIG PAGE (UNE SEULE FOIS, EN PREMIER) ==========
st.set_page_config(
    page_title="Nintendo Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

# ========== SESSION STATE GLOBAL (UNE SEULE FOIS) ==========
if "show_daisy_page" not in st.session_state:
    st.session_state["show_daisy_page"] = False

# ========== CSS : FOND D'ÉCRAN ==========
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://wallpaper.forfun.com/fetch/16/16b882fa988ab528cbe12f8ae188c25c.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

# ========== CSS : CURSEUR ÉTOILE ==========
st.markdown("""
    <style>
    * {
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"><path fill="%23FFD700" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>') 16 16, auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# ========== CSS : CARTES UNIFORMES ==========
st.markdown("""
<style>
    .main { background-color: transparent; }

    .custom-card {
        background-color: rgba(255, 255, 255, 0.85); 
        backdrop-filter: blur(10px); 
        -webkit-backdrop-filter: blur(10px); 
        border-radius: 12px;
        padding: 15px;
        margin: 10px auto; 
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 280px; 
    }

    .card-img {
        width: 70px; 
        margin-bottom: 8px;
    }
    
    .custom-card h3 {
        font-size: 1.1em; 
        margin: 8px 0;
    }
    
    .custom-card p {
        font-size: 0.9em; 
        margin: 5px 0;
    }

    .placeholder-box {
        background-color: rgba(94, 82, 64, 0.05);
        border: 2px dashed rgba(94, 82, 64, 0.3);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("<h1 style='text-align: center;'>Dashboard for Nintendo's Investors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 40px;'>Sélectionne une section pour explorer les modules.</p>", unsafe_allow_html=True)

# ========== GRID LAYOUT : CARTES ==========
# On affiche les cartes SEULEMENT si la page Daisy n'est PAS ouverte
if not st.session_state["show_daisy_page"]:
    
    col1, col2 = st.columns(2)

# ---------- PARTIE 1 : DAISY ----------
    with col1:
        st.markdown("""
        <div class="custom-card">
            <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/2ad3a5c2b5b8309627236c3eb193e4bd0b5b54fea0c8950a1b8c2dcb.png" class="card-img">
            <h3>Financial Forecasting</h3>
            <p style="opacity: 0.6;">Daisy fait fleurir vos profits ! 🌼💰</p>
            <p style="opacity: 0.8;">Module de prévision des tendances financières.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Ouvrir le module Daisy", key="open_daisy"):
            st.session_state["show_daisy_page"] = True
            st.rerun()

# ---------- PARTIE 2 : PEACH ----------
with col2:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/60b3f7c1d2a16cffef93fcf29e0af2b4da2ff4482a5c9a1db9b1d85e.png" class="card-img">
        <h3>Portfolio Optimization</h3>
        <p style="opacity: 0.6;">Peach your assets! 🍑💼</p>
        <p style="opacity: 0.8;">Optimisation du portefeuille.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Voir les détails et intégrer le code"):
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-text">Section à compléter par Peach</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- LIGNE 2 ----------
col3, col4 = st.columns(2)

# ---------- PARTIE 3 : BIRDO ----------
with col3:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/9bc8e27736eeeb46bd8af86f6956c3294355ea99b12f9b33751a6361.png" class="card-img">
        <h3>Algorithmic Trading</h3>
        <p style="opacity: 0.6;">Vos trades, pondus et gérés par Birdo 🥚📈</p>
        <p style="opacity: 0.8;">Stratégies automatisées et backtesting.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Voir les détails et intégrer le code"):
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-text">Section à compléter par Birdo</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- PARTIE 4 : BOWSER ----------
with col4:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/828f7ec3955d9049a1295309226e2c0696daadf60c3202fdedac0992.png" class="card-img">
        <h3>Option Pricing</h3>
        <p style="opacity: 0.6;">Ne vous brûlez pas seul : Bowser hedge vos positions 🐢💼</p>
        <p style="opacity: 0.8;">Modélisation et valorisation des options.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Voir les détails et intégrer le code"):
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-text">Section à compléter par Bowser</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- LIGNE 3 : LUIGI (CENTRÉ) ----------
col5, col6, col7 = st.columns([1, 2, 1])

with col6:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/63f4fbcbf84bd8532d9e041b3f6671c611706eb9ecc792f6fb74499a.png" class="card-img">
        <h3>Risk management</h3>
        <p style="opacity: 0.6;">Ne laissez pas vos risques vous hanter : Luigi est là 👻💸</p>
        <p style="opacity: 0.8;">Analyse des risques financiers.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Voir les détails et intégrer le code"):
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-text">Section à compléter par Luigi</div>
        </div>
        """, unsafe_allow_html=True)

# ====================== PAGE DAISY FULL WIDTH ======================
if st.session_state["show_daisy_page"]:

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center; margin-top:10px;'>🌼 Daisy – Nintendo Financial Forecasting</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; opacity:0.85;'>Vue analyste complète : états financiers, performance boursière, simulations Monte Carlo et scénarios.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Retour au dashboard principal", key="close_daisy"):
        st.session_state["show_daisy_page"] = False
        st.rerun()

    # ---------- PARAMÈTRES GÉNÉRAUX ----------
    start = "2015-09-30"
    end = "2025-09-30"

    companies = {
        "NTDOY": "Nintendo Co., Ltd.",
        "SONY": "Sony Group Corporation",
        "MSFT": "Microsoft Corporation",
        "EA": "Electronic Arts Inc.",
        "TCEHY": "Tencent Holdings Corporation"
    }

    st.markdown("### 📁 Paramètres d'analyse")
    st.write(f"Période analysée : **{start} → {end}**")
    st.code(
        'start = "2015-09-30"\nend   = "2025-09-30"\ncompanies = {"NTDOY": "Nintendo", ...}',
        language="python"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- LIGNE 1 : ÉTATS FINANCIERS & PRIX ----------
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown("### 📊 États financiers – Nintendo")
        ntd = yf.Ticker("NTDOY")
        balance_sheet = ntd.balance_sheet
        income_stmt = ntd.income_stmt
        cashflow_stmt = ntd.cashflow

        st.markdown("**📘 Bilan**")
        st.dataframe(balance_sheet)

        st.markdown("**📗 Compte de résultat**")
        st.dataframe(income_stmt)

        st.markdown("**📙 Tableau de flux de trésorerie**")
        st.dataframe(cashflow_stmt)

    with col_right:
        st.markdown("### 📈 Performance boursière comparée")

        tickers = list(companies.keys())
        prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]

        def base100(df):
            return df / df.iloc[0] * 100

        px_norm = base100(prices)
        px_norm.columns = [companies[c] for c in px_norm.columns]

        fig_prices = go.Figure()
        for col_name in px_norm.columns:
            fig_prices.add_trace(
                go.Scatter(
                    x=px_norm.index,
                    y=px_norm[col_name],
                    mode="lines",
                    name=col_name
                )
            )

        fig_prices.update_layout(
            title="Performance normalisée (Base 100)",
            xaxis_title="Date",
            yaxis_title="Indice (Base 100)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prices, use_container_width=True)

    st.markdown("---")

    # ---------- LIGNE 2 : MONTE CARLO & FORECAST ----------
    col_mc, col_fc = st.columns(2)

    with col_mc:
        st.markdown("### 🎲 Simulation Monte Carlo – NTDOY")

        returns = prices["NTDOY"].pct_change().dropna()
        r = returns.mean()
        sigma = returns.std()

        T = 5
        M = 100
        dt = T / M
        I = 500

        S = np.zeros((M + 1, I))
        S0 = prices["NTDOY"].iloc[-1]
        S[0] = S0

        for t in range(1, M + 1):
            S[t] = S[t - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * npr.randn(I)
            )

        fig_mc = go.Figure()
        for i in range(80):
            fig_mc.add_trace(
                go.Scatter(
                    x=list(range(M + 1)),
                    y=S[:, i],
                    mode="lines",
                    line=dict(width=1, color="rgba(255, 215, 0, 0.25)"),
                    showlegend=False
                )
            )

        fig_mc.add_trace(
            go.Scatter(
                x=list(range(M + 1)),
                y=S.mean(axis=1),
                mode="lines",
                name="Trajectoire moyenne",
                line=dict(width=3, color="#FFD700")
            )
        )

        fig_mc.update_layout(
            title="Monte Carlo – Distribution future du cours NTDOY",
            xaxis_title="Pas de temps",
            yaxis_title="Prix simulé",
            height=380,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_fc:
        st.markdown("### 🔮 Projection de revenus (simulation)")

        metric = "Total Revenue"
        years = np.arange(2025, 2031)
        base_value = income_stmt.loc["Total Revenue"].mean()
        growth = np.linspace(1.00, 1.25, len(years))

        forecast = pd.DataFrame({
            "Year": years,
            "Simulated Forecast": base_value * growth
        })

        st.dataframe(forecast)

        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(
                x=forecast["Year"],
                y=forecast["Simulated Forecast"],
                mode="lines+markers",
                line=dict(width=3, color="#FF7F0E"),
                name="Revenus simulés"
            )
        )
        fig_fc.update_layout(
            title="Projection simulée – Total Revenue",
            xaxis_title="Année",
            yaxis_title="JPY (simulé)",
            height=380,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # ---------- LIGNE 3 : SCÉNARIOS KPI ----------
    st.markdown("### 🧪 Scénarios de résultat opérationnel")

    scenario_factors = {"Pessimistic": 0.85, "Central": 1.00, "Optimistic": 1.15}
    metric = "Operating Income"
    base_value = income_stmt.loc["Operating Income"].mean()

    df_scen = pd.DataFrame({
        "Scenario": list(scenario_factors.keys()),
        "Value": [base_value * f for f in scenario_factors.values()]
    })

    col_tab, col_bar = st.columns([1, 2])

    with col_tab:
        st.dataframe(df_scen)

    with col_bar:
        fig_scen = go.Figure()
        fig_scen.add_bar(
            x=df_scen["Scenario"],
            y=df_scen["Value"],
            marker_color=["#E15759", "#4E79A7", "#59A14F"]
        )
        fig_scen.update_layout(
            title="Scénarios sur l'Operating Income",
            yaxis_title="JPY (simulé)",
            height=360,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_scen, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Module Daisy : outil de support à la décision pour les investisseurs Nintendo.")

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### Navigation")

    if st.button("Partie 1 - Daisy"):
        st.image("images/Daisy.png", width=120)

    if st.button("Partie 2 - Peach"):
        st.image("images/Peach.png", width=120)

    if st.button("Partie 3 - Birdo"):
        st.image("images/Birdo.png", width=120)

    if st.button("Partie 4 - Bowser"):
        st.image("images/Bowser.png", width=120)

    if st.button("Partie 5 - Luigi"):
        st.image("images/Luigi.png", width=120)
