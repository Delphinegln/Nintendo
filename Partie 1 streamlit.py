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

# ========== CSS : CARTES UNIFORMES + AMÉLIORATIONS ==========
st.markdown("""
<style>
    .main { background-color: transparent !important; }
    
    [data-testid="stMainBlockContainer"] {
        background-color: transparent !important;
    }

    .custom-card {
        background-color: rgba(255, 255, 255, 0.35) !important; 
        backdrop-filter: blur(20px) !important; 
        -webkit-backdrop-filter: blur(20px) !important; 
        border-radius: 16px !important;
        padding: 30px !important;
        margin: 15px auto !important; 
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
        text-align: center !important;
        max-width: 450px !important;
        width: 100% !important;
        display: block !important;
    }

    .card-img {
        width: 100px !important; 
        margin-bottom: 15px !important;
    }
    
    .custom-card h3 {
        font-size: 1.4em !important; 
        margin: 15px 0 !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
    }
    
    .custom-card p {
        font-size: 1.05em !important; 
        margin: 10px 0 !important;
        color: #333 !important;
    }

    .placeholder-box {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 2px dashed rgba(94, 82, 64, 0.5) !important;
        border-radius: 12px !important;
        padding: 30px !important;
        text-align: center !important;
        min-height: 140px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Style pour la page Daisy */
    .daisy-container {
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-radius: 20px !important;
        padding: 35px !important;
        margin: 25px 0 !important;
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    
    .chart-container {
        background-color: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        margin: 15px 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    }
    
    .intro-text {
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-radius: 14px !important;
        padding: 30px !important;
        margin: 25px 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        line-height: 1.9 !important;
        font-size: 1.08em !important;
        color: #2c3e50 !important;
    }
    
    .intro-text p {
        text-align: justify !important;
        margin-bottom: 15px !important;
    }
    
    .intro-text strong {
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("<h1 style='text-align: center; color: #1a1a1a;'>🎮 Dashboard for Nintendo's Investors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 40px; color: #2c3e50;'>Sélectionne une section pour explorer les modules.</p>", unsafe_allow_html=True)

# ========== GRID LAYOUT : CARTES ==========
if not st.session_state["show_daisy_page"]:
    
    col1, col2 = st.columns(2)

    # ---------- PARTIE 1 : DAISY ----------
    with col1:
        st.markdown("""
        <div class="custom-card">
            <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/2ad3a5c2b5b8309627236c3eb193e4bd0b5b54fea0c8950a1b8c2dcb.png" class="card-img">
            <h3>Financial Forecasting</h3>
            <p style="opacity: 0.7;">Daisy fait fleurir vos profits ! 🌼💰</p>
            <p style="opacity: 0.85;">Module de prévision des tendances financières.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Ouvrir le module Daisy", key="open_daisy", use_container_width=True):
            st.session_state["show_daisy_page"] = True
            st.rerun()

    # ---------- PARTIE 2 : PEACH ----------
    with col2:
        st.markdown("""
        <div class="custom-card">
            <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/60b3f7c1d2a16cffef93fcf29e0af2b4da2ff4482a5c9a1db9b1d85e.png" class="card-img">
            <h3>Portfolio Optimization</h3>
            <p style="opacity: 0.7;">Peach your assets! 🍑💼</p>
            <p style="opacity: 0.85;">Optimisation du portefeuille.</p>
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
            <p style="opacity: 0.7;">Vos trades, pondus et gérés par Birdo 🥚📈</p>
            <p style="opacity: 0.85;">Stratégies automatisées et backtesting.</p>
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
            <p style="opacity: 0.7;">Ne vous brûlez pas seul : Bowser hedge vos positions 🐢💼</p>
            <p style="opacity: 0.85;">Modélisation et valorisation des options.</p>
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
            <h3>Risk Management</h3>
            <p style="opacity: 0.7;">Ne laissez pas vos risques vous hanter : Luigi est là 👻💸</p>
            <p style="opacity: 0.85;">Analyse des risques financiers.</p>
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

    if st.button("⬅️ Retour au dashboard principal", key="close_daisy", use_container_width=True):
        st.session_state["show_daisy_page"] = False
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; font-size: 2.8em; color: #1a1a1a;'>🌼 Module Daisy – Financial Forecasting</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Texte d'introduction
    st.markdown("""
    <div class="intro-text">
        <p>
            <strong>Bienvenue dans le module Daisy de prévision financière.</strong> Cet outil d'aide à la décision vous permet d'analyser 
            en profondeur la performance financière de Nintendo Co., Ltd. et de ses principaux concurrents du secteur gaming.
        </p>
        <p>
            L'analyse couvre une <strong>période de 10 ans (30 septembre 2015 - 30 septembre 2025)</strong> et inclut une comparaison 
            avec Sony Group Corporation, Microsoft Corporation, Electronic Arts Inc. et Tencent Holdings Corporation.
        </p>
        <p>
            Les simulations Monte Carlo et projections de revenus vous offrent une vision probabiliste des performances futures, 
            permettant d'évaluer différents scénarios d'investissement avec une approche quantitative rigoureuse.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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

    # ---------- LIGNE 1 : ÉTATS FINANCIERS & PRIX ----------
    st.markdown("<div class='daisy-container'>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("### 📊 États financiers – Nintendo")
        
        ntd = yf.Ticker("NTDOY")
        balance_sheet = ntd.balance_sheet
        income_stmt = ntd.income_stmt
        cashflow_stmt = ntd.cashflow

        tab1, tab2, tab3 = st.tabs(["📘 Bilan", "📗 Compte de résultat", "📙 Flux de trésorerie"])
        
        with tab1:
            st.dataframe(balance_sheet, use_container_width=True, height=400)
        
        with tab2:
            st.dataframe(income_stmt, use_container_width=True, height=400)
        
        with tab3:
            st.dataframe(cashflow_stmt, use_container_width=True, height=400)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("### 📈 Performance boursière comparée")

        tickers = list(companies.keys())
        prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]

        def base100(df):
            return df / df.iloc[0] * 100

        px_norm = base100(prices)
        px_norm.columns = [companies[c] for c in px_norm.columns]

        fig_prices = go.Figure()
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        
        for idx, col_name in enumerate(px_norm.columns):
            fig_prices.add_trace(
                go.Scatter(
                    x=px_norm.index,
                    y=px_norm[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(width=3, color=colors[idx % len(colors)])
                )
            )

        fig_prices.update_layout(
            title={
                'text': "Performance normalisée (Base 100)",
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Date",
            yaxis_title="Indice (Base 100)",
            height=550,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        fig_prices.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        fig_prices.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        
        st.plotly_chart(fig_prices, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- LIGNE 2 : MONTE CARLO & FORECAST ----------
    st.markdown("<div class='daisy-container'>", unsafe_allow_html=True)
    
    col_mc, col_fc = st.columns(2)

    with col_mc:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("### 🎲 Simulation Monte Carlo – NTDOY")
        st.markdown("*Projection à 5 ans basée sur 500 trajectoires simulées*")

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
        
        for i in range(100):
            fig_mc.add_trace(
                go.Scatter(
                    x=list(range(M + 1)),
                    y=S[:, i],
                    mode="lines",
                    line=dict(width=0.8, color="rgba(255, 215, 0, 0.15)"),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

        fig_mc.add_trace(
            go.Scatter(
                x=list(range(M + 1)),
                y=S.mean(axis=1),
                mode="lines",
                name="Trajectoire moyenne",
                line=dict(width=4, color="#FFD700")
            )
        )
        
        fig_mc.add_trace(
            go.Scatter(
                x=list(range(M + 1)),
                y=np.percentile(S, 90, axis=1),
                mode="lines",
                name="90e percentile",
                line=dict(width=2, color="rgba(46, 204, 113, 0.7)", dash='dash')
            )
        )
        
        fig_mc.add_trace(
            go.Scatter(
                x=list(range(M + 1)),
                y=np.percentile(S, 10, axis=1),
                mode="lines",
                name="10e percentile",
                line=dict(width=2, color="rgba(231, 76, 60, 0.7)", dash='dash')
            )
        )

        fig_mc.update_layout(
            title={
                'text': "Distribution future du cours NTDOY",
                'font': {'size': 16}
            },
            xaxis_title="Pas de temps",
            yaxis_title="Prix simulé (USD)",
            height=520,
            margin=dict(l=60, r=30, t=70, b=50),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.9)"
            )
        )
        
        fig_mc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        fig_mc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        
        st.plotly_chart(fig_mc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_fc:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("### 🔮 Projection de revenus")
        st.markdown("*Scénario de croissance simulée 2025-2030*")

        metric = "Total Revenue"
        years = np.arange(2025, 2031)
        base_value = income_stmt.loc["Total Revenue"].mean()
        growth = np.linspace(1.00, 1.25, len(years))

        forecast = pd.DataFrame({
            "Année": years,
            "Prévision (JPY)": base_value * growth
        })

        forecast["Prévision (Milliards JPY)"] = (forecast["Prévision (JPY)"] / 1e9).round(2)
        
        st.dataframe(
            forecast[["Année", "Prévision (Milliards JPY)"]], 
            use_container_width=True,
            hide_index=True
        )

        fig_fc = go.Figure()
        
        fig_fc.add_trace(
            go.Scatter(
                x=forecast["Année"],
                y=forecast["Prévision (JPY)"],
                mode="lines+markers",
                line=dict(width=4, color="#FF7F0E"),
                marker=dict(size=12, color="#FF7F0E", line=dict(width=2, color='white')),
                name="Revenus simulés",
                fill='tozeroy',
                fillcolor='rgba(255, 127, 14, 0.2)'
            )
        )
        
        fig_fc.update_layout(
            title={
                'text': "Projection Total Revenue",
                'font': {'size': 16}
            },
            xaxis_title="Année",
            yaxis_title="Revenus (JPY)",
            height=520,
            margin=dict(l=60, r=30, t=70, b=50),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        fig_fc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        fig_fc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        
        st.plotly_chart(fig_fc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- LIGNE 3 : SCÉNARIOS KPI ----------
    st.markdown("<div class='daisy-container'>", unsafe_allow_html=True)
    st.markdown("### 🧪 Analyse de scénarios – Résultat opérationnel")
    st.markdown("*Évaluation sous trois hypothèses de performance*")
    
    st.markdown("<br>", unsafe_allow_html=True)

    scenario_factors = {"Pessimiste": 0.85, "Central": 1.00, "Optimiste": 1.15}
    metric = "Operating Income"
    base_value = income_stmt.loc["Operating Income"].mean()

    df_scen = pd.DataFrame({
        "Scénario": list(scenario_factors.keys()),
        "Valeur (JPY)": [base_value * f for f in scenario_factors.values()]
    })
    
    df_scen["Valeur (Milliards JPY)"] = (df_scen["Valeur (JPY)"] / 1e9).round(2)

    col_tab, col_bar = st.columns([1, 2])

    with col_tab:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.dataframe(
            df_scen[["Scénario", "Valeur (Milliards JPY)"]], 
            use_container_width=True,
            hide_index=True,
            height=250
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("**Hypothèses:**\n\n- Pessimiste: -15%\n- Central: Baseline\n- Optimiste: +15%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bar:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        
        fig_scen = go.Figure()
        
        fig_scen.add_bar(
            x=df_scen["Scénario"],
            y=df_scen["Valeur (JPY)"],
            marker_color=["#E15759", "#4E79A7", "#59A14F"],
            text=df_scen["Valeur (Milliards JPY)"],
            texttemplate='%{text:.2f}B JPY',
            textposition='outside',
            textfont=dict(size=14, color='black')
        )
        
        fig_scen.update_layout(
            title={
                'text': "Operating Income par scénario",
                'font': {'size': 16}
            },
            yaxis_title="Revenus opérationnels (JPY)",
            height=450,
            margin=dict(l=60, r=30, t=70, b=50),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        fig_scen.update_xaxes(showgrid=False)
        fig_scen.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.4)')
        
        st.plotly_chart(fig_scen, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: rgba(255, 255, 255, 0.5); 
                backdrop-filter: blur(15px); border-radius: 14px; margin: 20px 0; border: 1px solid rgba(255, 255, 255, 0.5);'>
        <p style='margin: 0; opacity: 0.8; font-size: 0.95em; color: #2c3e50;'>
            📊 <strong>Module Daisy</strong> – Outil de support à la décision pour investisseurs institutionnels et retail
        </p>
        <p style='margin: 5px 0 0 0; opacity: 0.6; font-size: 0.85em; color: #34495e;'>
            Données fournies par Yahoo Finance – À des fins éducatives uniquement
        </p>
    </div>
    """, unsafe_allow_html=True)


# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🎮 Navigation")
    st.markdown("---")

    if st.button("🌼 Partie 1 - Daisy", use_container_width=True):
        pass

    if st.button("🍑 Partie 2 - Peach", use_container_width=True):
        pass

    if st.button("🥚 Partie 3 - Birdo", use_container_width=True):
        pass

    if st.button("🐢 Partie 4 - Bowser", use_container_width=True):
        pass

    if st.button("👻 Partie 5 - Luigi", use_container_width=True):
        pass
