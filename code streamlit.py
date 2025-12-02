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
import warnings
from dataclasses import dataclass
from typing import List
import scipy.stats as stats
import base64
from pathlib import Path
from scipy.stats import norm
from plotly.subplots import make_subplots
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

IMG = Path.cwd() / "images"


# HRP
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

warnings.filterwarnings("ignore")

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

if "show_peach_page" not in st.session_state:
    st.session_state["show_peach_page"] = False

if "show_luigi_page" not in st.session_state:
    st.session_state["show_luigi_page"] = False

if "show_bowser_page" not in st.session_state:
    st.session_state["show_bowser_page"] = False

if "show_birdo_page" not in st.session_state:
    st.session_state["show_birdo_page"] = False
    
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

# ========== CSS : CARTES AVEC GRILLE 2+2+1 ==========
st.markdown("""
<style>
    .main { background-color: transparent; }

    .row-custom-cards {
        display: flex;
        gap: 38px;
        justify-content: center;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }

    .card-glass {
        background: rgba(255,255,255,0.7);
        border-radius: 18px;
        padding: 24px 22px 22px 22px;
        box-shadow: 0 4px 24px rgba(50, 50, 93, 0.11), 0 1.5px 2.5px rgba(0,0,0,0.07);
        backdrop-filter: blur(8px);
        width: 240px;
        min-width: 220px;
        max-width: 260px;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: box-shadow 0.2s;
        margin: 0 auto;
    }

    .card-glass:hover { 
        box-shadow: 0 6px 30px rgba(0,0,0,0.18);
    }

    .card-glass img { 
        margin-bottom: 18px; 
        border-radius: 12px; 
    }

    .card-glass h3 { 
        margin-bottom: 8px; 
        font-size: 1.18em;
    }

    .card-glass .sous-titre { 
        opacity: 0.7; 
        font-size: 1em; 
        margin-bottom: 8px;
    }

    .card-glass .desc { 
        font-size: 1em; 
        opacity: 0.88; 
        margin-bottom: 15px; 
        text-align: center;
    }

    @media (max-width: 900px) {
        .row-custom-cards { 
            flex-direction: column; 
            align-items: center; 
        }
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
    
    /* Conteneurs pour graphiques Daisy */
    .chart-container {
        background-color: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .intro-box {
        background-color: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        line-height: 1.8;
        font-size: 1.05em;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("<h1 style='text-align: center;'>Dashboard for Nintendo's Investors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 40px;'>Sélectionne une section pour explorer les modules.</p>", unsafe_allow_html=True)

def card_with_button(img_path, title, subtitle, desc, btn_label, key):

    # ✅ Conversion de l’image en base64 pour l’intégrer en HTML
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    img_html = f"""
    <div class="card-glass">
        <img src="data:image/png;base64,{data}" width="70">
        <h3>{title}</h3>
        <div class="sous-titre">{subtitle}</div>
        <div class="desc">{desc}</div>
    </div>
    """

    st.markdown(img_html, unsafe_allow_html=True)

    clicked = st.button(btn_label, key=key)
    return clicked


# ========== GRID LAYOUT : CARTES AVEC DISPOSITION 2+2+1 ==========
if not (st.session_state["show_daisy_page"] or st.session_state["show_peach_page"] or st.session_state["show_luigi_page"] or st.session_state["show_bowser_page"] or st.session_state["show_birdo_page"]):




    # ===== LIGNE 1 : DAISY + PEACH (CENTRÉES EN HAUT) =====
    col1, col2, col3, col4, col5 = st.columns([1, 2, 0.5, 2, 1])

    with col2:
        if card_with_button(
            IMG / "Daisy.png",
            "Financial Forecasting",
            "Daisy fait fleurir vos profits 🌼💰",
            "Prévision des tendances financières.",
            "🔍 Ouvrir le module Daisy",
            "open_daisy"
        ):
            st.session_state["show_daisy_page"] = True
            st.rerun()

    with col4:
        if card_with_button(
            IMG / "Peach.png",
            "Portfolio Optimization",
            "Peach your assets 🍑💼",
            "Optimisation du portefeuille.",
            "🔍 Ouvrir le module Peach",
            "open_peach"
        ):
            st.session_state["show_peach_page"] = True
            st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)

    # ===== LIGNE 2 : BIRDO + BOWSER =====
    col1, col2, col3, col4, col5 = st.columns([1, 2, 0.5, 2, 1])

    with col2:
       if card_with_button(
            IMG / "Birdo.png",
            "Algorithmic Trading",
            "Birdo gère tes trades 🥚📈",
            "Stratégies automatisées & backtesting.",
            "🔍 Module Birdo",
            "open_birdo"
        ):
            st.session_state["show_birdo_page"] = True
            st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)

    with col4:
        if card_with_button(
            IMG / "Bowser.png",
            "Option Pricing",
            "Bowser hedge vos positions 🐢🔥",
            "Modélisation des options.",
            "🔍 Module Bowser",
            "open_bowser"
        ):
            st.session_state["show_bowser_page"] = True
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ===== LIGNE 3 : LUIGI SEUL PARFAITEMENT CENTRÉ =====
    col1, col2, col3 = st.columns([1.5, 2, 1.5])

    with col2:
        if card_with_button(
            IMG / "Luigi.png",
            "Risk Management",
            "Luigi protège vos investissements 👻💸",
            "Analyse avancée des risques financiers.",
            "🔍 Ouvrir le module Luigi",
            "open_luigi"
        ):
            st.session_state["show_luigi_page"] = True
            st.rerun()

# ====================== PAGE DAISY FULL WIDTH ======================================================================================================
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

    # ---------- TEXTE DESCRIPTIF AU LIEU DU CODE ----------
    st.markdown("""
    <div class="intro-box">
        <p style='text-align: justify; font-size: 1.1em; line-height: 1.8;'>
            Nous avons analysé le titre <strong>Nintendo Co., Ltd.</strong> sur une période de <strong>10 ans</strong>, 
            du <strong>30 septembre 2015</strong> au <strong>30 septembre 2025</strong>. Cette analyse comparative 
            inclut également les performances de <strong>Sony Group Corporation</strong>, <strong>Microsoft Corporation</strong>, 
            <strong>Electronic Arts Inc.</strong> et <strong>Tencent Holdings Corporation</strong>, permettant une 
            vision holistique du secteur du gaming et du divertissement interactif.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- GRAPHIQUE 1 : ÉTATS FINANCIERS (PLEINE LARGEUR) ----------
    st.markdown("### 📊 États financiers – Nintendo")
    
    ntd = yf.Ticker("NTDOY")
    balance_sheet = ntd.balance_sheet
    income_stmt = ntd.income_stmt
    cashflow_stmt = ntd.cashflow

    tab1, tab2, tab3 = st.tabs(["📘 Bilan", "📗 Compte de résultat", "📙 Flux de trésorerie"])
    
    with tab1:
        st.dataframe(balance_sheet, use_container_width=True, height=500)
    
    with tab2:
        st.dataframe(income_stmt, use_container_width=True, height=500)
    
    with tab3:
        st.dataframe(cashflow_stmt, use_container_width=True, height=500)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- GRAPHIQUE 2 : PERFORMANCE BOURSIÈRE (PLEINE LARGEUR) ----------
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
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis_title="Date",
        yaxis_title="Indice (Base 100)",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13)
    )
    
    fig_prices.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    fig_prices.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    st.plotly_chart(fig_prices, use_container_width=True)
    st.markdown("""
<div class="intro-box">
    <p style='text-align: justify; font-size: 1.08em; line-height: 1.8;'>
        La performance boursière comparée met en lumière la solidité du titre <strong>Nintendo</strong> 
        au cours des dix dernières années. Le titre suit globalement une trajectoire ascendante tout en 
        affichant une volatilité maîtrisée. <strong>Microsoft</strong> reste l’acteur le plus performant 
        du panel, soutenu par une diversification forte et une croissance structurelle du cloud. 
        <strong>Tencent</strong> présente une évolution dynamique mais irrégulière, affectée par 
        les régulations chinoises récentes.  
        <br><br>
        Dans ce contexte, Nintendo occupe une position intermédiaire : une croissance régulière, 
        peu de drawdowns sévères et une capacité de résilience élevée. Cela confirme la robustesse 
        du modèle économique basé sur les franchises propriétaires et un pipeline de produits très stable.
    </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- GRAPHIQUE 3 : MONTE CARLO (PLEINE LARGEUR) ----------
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
    
    for i in range(120):
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
            line=dict(width=5, color="#FFD700")
        )
    )
    
    fig_mc.add_trace(
        go.Scatter(
            x=list(range(M + 1)),
            y=np.percentile(S, 90, axis=1),
            mode="lines",
            name="90e percentile",
            line=dict(width=3, color="rgba(46, 204, 113, 0.8)", dash='dash')
        )
    )
    
    fig_mc.add_trace(
        go.Scatter(
            x=list(range(M + 1)),
            y=np.percentile(S, 10, axis=1),
            mode="lines",
            name="10e percentile",
            line=dict(width=3, color="rgba(231, 76, 60, 0.8)", dash='dash')
        )
    )

    fig_mc.update_layout(
        title={
            'text': "Distribution future du cours NTDOY",
            'font': {'size': 20}
        },
        xaxis_title="Pas de temps",
        yaxis_title="Prix simulé (USD)",
        height=600,
        margin=dict(l=70, r=40, t=80, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)"
        )
    )
    
    fig_mc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    fig_mc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    st.plotly_chart(fig_mc, use_container_width=True)
    st.markdown("""
<div class="intro-box">
    <p style='text-align: justify; font-size: 1.08em; line-height: 1.8;'>
        La simulation Monte Carlo réalisée sur 500 trajectoires projette un prix futur de 
        <strong>Nintendo</strong> sur un horizon de 5 ans. Le scénario central indique une tendance 
        haussière progressive, cohérente avec le rendement annuel moyen observé historiquement.  
        <br><br>
        L'écart croissant entre les percentiles <strong>10</strong> et <strong>90</strong> illustre 
        une incertitude naturelle mais contenue : le modèle suggère que la probabilité d’un effondrement 
        significatif est très faible, tandis que les scénarios optimistes restent plausibles, surtout en cas 
        de lancement de nouvelles consoles ou d’expansion transversale de l’univers Nintendo (licensing, cinéma, mobile).  
        <br><br>
        Globalement, la distribution simulée soutient une thèse d’investissement de long terme avec 
        un profil rendement/risque équilibré.
    </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- GRAPHIQUE 4 : PROJECTION REVENUS (PLEINE LARGEUR) ----------
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
        hide_index=True,
        height=250
    )

    st.markdown("<br>", unsafe_allow_html=True)

    fig_fc = go.Figure()
    
    fig_fc.add_trace(
        go.Scatter(
            x=forecast["Année"],
            y=forecast["Prévision (JPY)"],
            mode="lines+markers",
            line=dict(width=5, color="#FF7F0E"),
            marker=dict(size=14, color="#FF7F0E", line=dict(width=3, color='white')),
            name="Revenus simulés",
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.2)'
        )
    )
    
    fig_fc.update_layout(
        title={
            'text': "Projection Total Revenue",
            'font': {'size': 20}
        },
        xaxis_title="Année",
        yaxis_title="Revenus (JPY)",
        height=600,
        margin=dict(l=70, r=40, t=80, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    fig_fc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    fig_fc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("""
<div class="intro-box">
    <p style='text-align: justify; font-size: 1.08em; line-height: 1.8;'>
        Les projections de revenus sur la période <strong>2025–2030</strong> s’appuient sur une 
        croissance progressive comprise entre 0 % et 25 %. Ce rythme est cohérent avec les cycles produits 
        observés chez Nintendo, caractérisés par des phases de montée en puissance lors du lancement d’une 
        nouvelle console suivies d’une stabilisation.  
        <br><br>
        Les résultats montrent une évolution prévisible et régulière, renforcée par la récurrence des ventes 
        logicielles et la force des franchises historiques. En milliards de JPY, la croissance projetée 
        traduit l’ancrage durable de Nintendo comme l’un des acteurs les plus rentables du secteur.  
        <br><br>
        Cette trajectoire suggère un risque faible de contraction durable du chiffre d’affaires, ce qui 
        constitue un signal positif pour les investisseurs à horizon moyen terme.
    </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------- GRAPHIQUE 5 : SCÉNARIOS (PLEINE LARGEUR) ----------
    st.markdown("### 🧪 Scénarios de résultat opérationnel")
    st.markdown("*Évaluation sous trois hypothèses de performance*")

    scenario_factors = {"Pessimiste": 0.85, "Central": 1.00, "Optimiste": 1.15}
    metric = "Operating Income"
    base_value = income_stmt.loc["Operating Income"].mean()

    df_scen = pd.DataFrame({
        "Scénario": list(scenario_factors.keys()),
        "Valeur (JPY)": [base_value * f for f in scenario_factors.values()]
    })
    
    df_scen["Valeur (Milliards JPY)"] = (df_scen["Valeur (JPY)"] / 1e9).round(2)

    st.dataframe(
        df_scen[["Scénario", "Valeur (Milliards JPY)"]], 
        use_container_width=True,
        hide_index=True,
        height=200
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig_scen = go.Figure()
    
    fig_scen.add_bar(
        x=df_scen["Scénario"],
        y=df_scen["Valeur (JPY)"],
        marker_color=["#E15759", "#4E79A7", "#59A14F"],
        text=df_scen["Valeur (Milliards JPY)"],
        texttemplate='%{text:.2f}B JPY',
        textposition='outside',
        textfont=dict(size=16, color='black', family='Arial')
    )
    
    fig_scen.update_layout(
        title={
            'text': "Operating Income par scénario",
            'font': {'size': 20}
        },
        yaxis_title="Revenus opérationnels (JPY)",
        height=600,
        margin=dict(l=70, r=40, t=80, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    fig_scen.update_xaxes(showgrid=False)
    fig_scen.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    
    st.plotly_chart(fig_scen, use_container_width=True)
    st.markdown("""
<div class="intro-box">
    <p style='text-align: justify; font-size: 1.08em; line-height: 1.8;'>
        L’analyse par scénarios permet de mesurer la sensibilité du <strong>résultat opérationnel</strong> 
        aux variations de performance. Le scénario pessimiste (-15 %) illustre une marge de sécurité 
        relativement élevée : même en cas de contexte défavorable, Nintendo maintient un niveau de rentabilité 
        important.  
        <br><br>
        Le scénario central correspond à la trajectoire historique, marquée par une efficacité opérationnelle 
        constante et une politique de coûts maîtrisée. Le scénario optimiste (+15 %) reflète l’impact potentiel 
        d’un nouveau cycle matériel ou d’un élargissement du revenu récurrent (licences, partenariats, contenus).  
        <br><br>
        Cette distribution par scénarios souligne une asymétrie favorable : le potentiel haussier est significatif, 
        tandis que la baisse potentielle reste limitée. Cela renforce la thèse d’un actif défensif avec un levier 
        de croissance crédible.
    </p>
</div>
""", unsafe_allow_html=True)
    
    st.info("**Hypothèses de scénarios :** Pessimiste (-15%), Central (baseline), Optimiste (+15%)")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Module Daisy : outil de support à la décision pour les investisseurs Nintendo.")

# ====================== PAGE PEACH FULL WIDTH ===========================================================================================================
if st.session_state["show_peach_page"]:

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center; margin-top:10px;'>🍑 Peach – Portfolio Optimization</h2>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="intro-box">
        <p style='text-align: justify; font-size: 1.05em; line-height: 1.7;'>
        Ce module permet d’optimiser un portefeuille centré sur <strong>Nintendo</strong> 
        en comparant l’approche <strong>M4 (Mean-Variance)</strong> et
        <strong>HRP (Hierarchical Risk Parity)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Retour au dashboard principal", key="close_peach"):
        st.session_state["show_peach_page"] = False
        st.rerun()

    # -------- CONFIG --------
    NINTENDO = "NTDOY"
    DEFAULT_PEERS = ["EA","TTWO","SONY","MSFT","7832.T","9697.T",
                     "9684.T","9766.T","UBI.PA","TCEHY"]
    START, END = "2015-09-30", "2025-09-30"

    TICKER_NAME = {
        "NTDOY": "Nintendo (ADR)",
        "7974.T": "Nintendo (Tokyo)",
        "EA": "Electronic Arts",
        "TTWO": "Take-Two Interactive",
        "SONY": "Sony Group",
        "MSFT": "Microsoft",
        "7832.T": "Bandai Namco",
        "9697.T": "Capcom",
        "9684.T": "Square Enix",
        "9766.T": "Konami",
        "UBI.PA": "Ubisoft",
        "TCEHY": "Tencent"
    }

    @dataclass
    class Constraints:
        min_center_weight: float = 0.10
        max_center_weight: float = 0.80
        max_weight_per_name: float = 0.25

    cons = Constraints()

    # ---------- UTILITAIRES ----------
    @st.cache_data(ttl=3600)
    def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        return data.ffill().dropna()

    def pct_returns(prices):
        return prices.pct_change().dropna()

    def ann_perf(r):
        ann_ret = (1+r).prod()**(252/len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-12)
        return ann_ret, ann_vol, sharpe

    def evaluate_portfolio(weights, returns):
        weights = weights / weights.sum()
        common = [t for t in weights.index if t in returns.columns]
        port_rets = (returns[common] * weights[common]).sum(axis=1)
        ann_ret, ann_vol, sharpe = ann_perf(port_rets)
        growth = (1 + port_rets).cumprod()
        return ann_ret, ann_vol, sharpe, port_rets, growth

    def herfindahl(w):
        w = w / w.sum()
        return float((w**2).sum())

    # ---------- OPTIMISATION MV ----------
    def optimize_mv_centered(mu, cov, tickers, center, cons, target_center_weight):
        if not HAS_CVXPY:
            weights = pd.Series(0.0, index=tickers)
            weights[center] = target_center_weight
            others = [t for t in tickers if t != center]
            rest = 1 - target_center_weight
            weights[others] = rest / len(others)
            return weights

        n = len(tickers)
        w = cp.Variable(n)
        idx_center = tickers.index(center)

        Sigma = cov.loc[tickers, tickers].values
        Sigma = 0.5*(Sigma+Sigma.T)
        eps = 1e-6*np.mean(np.diag(Sigma))
        np.fill_diagonal(Sigma, np.diag(Sigma)+eps)

        gamma = 10.0 / max(np.trace(Sigma), 1e-8)

        constraints = [cp.sum(w) == 1, w >= 0]
        for i in range(n):
            if i != idx_center:
                constraints.append(w[i] <= cons.max_weight_per_name)

        constraints.append(w[idx_center] == target_center_weight)

        objective = cp.Maximize(mu.loc[tickers].values @ w - 0.5 * gamma * cp.quad_form(w, Sigma))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except:
            prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            raise RuntimeError("Optimisation impossible")

        wv = np.array(w.value).ravel()
        return pd.Series(wv / wv.sum(), index=tickers)

    # ---------- HRP ----------
    def _correl_dist(corr):
        return np.sqrt(0.5 * (1 - corr))

    def _get_cluster_var(cov, items):
        sub = cov.loc[items, items]
        w = np.ones(len(sub)) / len(sub)
        return float(w @ sub.values @ w)

    @st.cache_data
    def build_hrp_weights(returns):
        corr = returns.corr()
        cov = returns.cov()
        dist = _correl_dist(corr)
        dist_cond = squareform(dist.values, checks=False)
        link = linkage(dist_cond, method="single")
        order = leaves_list(link)
        ordered = corr.index[order].tolist()

        weights = pd.Series(1.0, index=ordered)
        clusters = [ordered]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue
            split = len(cluster)//2
            c1, c2 = cluster[:split], cluster[split:]
            var1 = _get_cluster_var(cov, c1)
            var2 = _get_cluster_var(cov, c2)
            alloc2 = var1/(var1+var2)
            alloc1 = 1-alloc2
            weights[c1] *= alloc1
            weights[c2] *= alloc2
            clusters += [c1, c2]

        weights = weights.reindex(returns.columns)
        return weights / weights.sum()


    # ----------------- CHARGEMENT -----------------
    with st.spinner("📡 Téléchargement des données..."):
        UNIVERSE = [NINTENDO] + DEFAULT_PEERS
        PRICES = download_prices(UNIVERSE, START, END)
        RETURNS = pct_returns(PRICES)

        TICKERS = list(RETURNS.columns)
        CENTER = NINTENDO if NINTENDO in TICKERS else TICKERS[0]

        MU_ANN = RETURNS.mean() * 252
        COV_ANN = RETURNS.cov() * 252

        HRP_WEIGHTS = build_hrp_weights(RETURNS)

    st.success("Données prêtes ✔️")

    # ------------ SIDEBAR LOCALE ------------
    st.subheader("⚙️ Paramètres")

    target_return = st.slider("🎯 Rendement annuel cible (%)", 0.0, 30.0, 6.0) / 100
    horizon_years = st.slider("⏳ Horizon d'investissement (années)", 1, 20, 3)
    nintendo_weight = st.slider("🎮 Poids de Nintendo (%)", 
                                int(cons.min_center_weight*100),
                                int(cons.max_center_weight*100),
                                30) / 100

    if st.button("🚀 Lancer l’optimisation"):

        try:
            weights_m4 = optimize_mv_centered(
                MU_ANN, COV_ANN, TICKERS, CENTER, cons, target_center_weight=nintendo_weight
            )

            ann_ret, ann_vol, sharpe, _, growth_port = evaluate_portfolio(weights_m4, RETURNS)

            hrp_weights_full = HRP_WEIGHTS.reindex(TICKERS).fillna(0)
            hrp_ret, hrp_vol, hrp_sharpe, _, hrp_growth = evaluate_portfolio(
                hrp_weights_full, RETURNS
            )

            st.success("Optimisation terminée ✔️")
            st.write("### Résultats à analyser…")
            
            # === AFFICHAGE DES RÉSULTATS ===

            st.markdown("## 📊 Résultats du portefeuille optimisé (Méthode M4)")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Poids optimisés (M4)")
                st.dataframe(weights_m4.map(lambda x: round(x*100,2)))

            with colB:
                st.markdown("### Indicateurs de performance (M4)")
                st.write(f"**Rendement annuel :** {ann_ret:.2%}")
                st.write(f"**Volatilité annuelle :** {ann_vol:.2%}")
                st.write(f"**Sharpe ratio :** {sharpe:.2f}")
                st.write(f"**Indice Herfindahl :** {herfindahl(weights_m4):.4f}")

            # --- HRP ---
            st.markdown("---")
            st.markdown("## 🧩 Allocation HRP (benchmark)")

            colC, colD = st.columns(2)

            with colC:
                st.markdown("### Poids HRP")
                st.dataframe(hrp_weights_full.map(lambda x: round(x*100,2)))

            with colD:
                st.markdown("### Indicateurs HRP")
                st.write(f"**Rendement annuel :** {hrp_ret:.2%}")
                st.write(f"**Volatilité annuelle :** {hrp_vol:.2%}")
                st.write(f"**Sharpe ratio :** {hrp_sharpe:.2f}")
                st.write(f"**Indice Herfindahl :** {herfindahl(hrp_weights_full):.4f}")

            # --- Graphique comparatif ---
            st.markdown("---")
            st.markdown("## 📈 Comparaison : Portefeuille Optimisé vs HRP")

            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(growth_port, label="Portefeuille Optimisé (M4)")
            ax.plot(hrp_growth, label="HRP", linestyle="dashed")
            ax.set_title("Croissance cumulée du portefeuille")
            ax.set_xlabel("Date")
            ax.set_ylabel("Croissance")
            ax.legend()
            st.pyplot(fig)

            # --- Analyse textuelle (style intro-box) ---
            st.markdown("""
            <div class="intro-box">
                <p style='text-align: justify; font-size: 1.1em; line-height: 1.8;'>
                    L’optimisation centrée sur <strong>Nintendo</strong> montre une allocation 
                    construite autour d’un compromis rendement/risque supérieur au benchmark HRP. 
                    Le portefeuille optimisé affiche un <strong>Sharpe ratio plus élevé</strong>, 
                    indiquant une meilleure efficacité du risque. Bien que la pondération de 
                    Nintendo soit imposée par votre choix initial, l’optimiseur redistribue le 
                    reste du capital vers les titres ayant le meilleur couple rendement/variance.
                    <br><br>
                    Le benchmark <strong>HRP</strong>, basé sur la hiérarchie des corrélations, 
                    fournit une allocation plus équilibrée mais moins agressive. Cela se traduit par 
                    une volatilité plus faible mais un rendement inférieur. 
                    <br><br>
                    Au final, l’allocation optimisée présente un profil de croissance cumulée 
                    supérieur, ce qui en fait une approche adaptée pour un investisseur recherchant 
                    une <strong>allocation centrée sur Nintendo tout en maximisant la performance ajustée du risque</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            

        except Exception as e:
            st.error(f"Erreur : {e}")

# ====================== PAGE LUIGI FULL WIDTH ======================================================================================================
if st.session_state["show_luigi_page"]:

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center; margin-top:10px;'>👻 Luigi – Risk Management & Modeling </h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; opacity:0.85;'>Vue analyste complète : états financiers, performance boursière, simulations Monte Carlo et scénarios.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Retour au dashboard principal", key="close_luigi"):
        st.session_state["show_luigi_page"] = False
        st.rerun()

    with st.spinner("📊 Chargement des données Nintendo pour l'analyse de risque..."):
        try:
            # Télécharger les données
            nintendo_data = yf.download("NTDOY", start="2015-09-30", end="2025-09-30", progress=False)
            
            # ✅ CORRECTION : Gérer la structure MultiIndex ou simple
            if isinstance(nintendo_data.columns, pd.MultiIndex):
                # Si MultiIndex, extraire la colonne Close
                data = pd.DataFrame({'Close': nintendo_data['Close']['NTDOY']})
            else:
                # Si simple Index, renommer directement
                if 'Close' in nintendo_data.columns:
                    data = pd.DataFrame({'Close': nintendo_data['Close']})
                else:
                    # Si une seule colonne sans nom explicite
                    data = pd.DataFrame({'Close': nintendo_data.iloc[:, 0]})
            
            # Calculer les rendements logarithmiques
            data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data = data.dropna()
            
            # Vérifier que nous avons des données
            if len(data) == 0:
                st.error("❌ Aucune donnée disponible pour Nintendo")
                st.stop()
            
            st.success("✅ Données chargées avec succès")
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données : {str(e)}")
            st.exception(e)
            st.stop()
    
    # Paramètres de base
    last_price = data['Close'].iloc[-1]
    shares = 1000
    portfolio_value = last_price * shares
    mu = data['returns'].mean()
    sigma = data['returns'].std()
    alpha = 0.05  # Niveau de confiance 95%
    
    # Afficher les informations de base
    st.markdown("### 📊 Informations du portefeuille")
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Prix actuel", f"${last_price:.2f}")
    col_info2.metric("Nombre d'actions", f"{shares:,}")
    col_info3.metric("Valeur du portefeuille", f"${portfolio_value:,.2f}")
    
    st.markdown("---")
    
    # ==================== 1. Value-at-Risk (Approche Paramétrique) ====================
    st.markdown("### 1️⃣ Value-at-Risk (Approche Paramétrique)")
    
    z = stats.norm.ppf(1 - alpha)
    VaR = mu - z * sigma
    VaR_portfolio = portfolio_value * VaR
    
    col1, col2 = st.columns(2)
    col1.metric("VaR Paramétrique (5%)", f"{VaR*100:.2f}%")
    col2.metric("Perte potentielle", f"${abs(VaR_portfolio):,.0f}")
    
    # Simulation pour visualisation
    num_samples = 1000
    sim_returns = np.random.normal(mu, sigma, num_samples)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim_returns,
        nbinsx=50,
        opacity=0.7,
        name="Rendements simulés"
    ))
    fig.add_vline(
        x=VaR,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 5%: {VaR*100:.2f}%",
        annotation_position="top"
    )
    fig.update_layout(
        title="Distribution simulée - VaR Paramétrique",
        xaxis_title="Rendement",
        yaxis_title="Fréquence",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== 2. Value-at-Risk (Approche Historique) ====================
    st.markdown("### 2️⃣ Value-at-Risk (Approche Historique)")
    
    VaR_hist = data['returns'].quantile(alpha)
    VaR_hist_portfolio = VaR_hist * portfolio_value
    
    col1, col2 = st.columns(2)
    col1.metric("Historical VaR (5%)", f"{VaR_hist*100:.2f}%")
    col2.metric("Perte potentielle", f"${abs(VaR_hist_portfolio):,.0f}")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=data['returns'],
        nbinsx=40,
        opacity=0.7,
        name="Rendements historiques"
    ))
    fig2.add_vline(
        x=VaR_hist,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 5%: {VaR_hist*100:.2f}%",
        annotation_position="top"
    )
    fig2.update_layout(
        title="Distribution des rendements - VaR Historique",
        xaxis_title="Rendement",
        yaxis_title="Densité",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== 3. Backtesting du VaR ====================
    st.markdown("### 3️⃣ Backtesting du VaR (1%)")
    
    alpha_bt = 0.01
    z_bt = stats.norm.ppf(1 - alpha_bt)
    VaR_cutoff = mu - z_bt * sigma
    
    returns = data['returns']
    violations = returns[returns < VaR_cutoff]
    ratio = len(violations) / len(returns)
    
    col1, col2 = st.columns(2)
    col1.metric("Nombre de violations", len(violations))
    col2.metric("Taux de violation observé", f"{ratio*100:.2f}% (théorique: 1%)")
    
    if abs(ratio - 0.01) < 0.005:
        st.success("✅ Le modèle VaR est bien calibré")
    else:
        st.warning("⚠️ Le modèle VaR pourrait nécessiter un ajustement")
    
    st.markdown("---")
    
    # ==================== 4. Expected Shortfall (CVaR) ====================
    st.markdown("### 4️⃣ Expected Shortfall (CVaR)")
    
    # Parametric ES
    ES_param = mu - (stats.norm.pdf(z) / (1 - alpha)) * sigma
    ES_param_portfolio = ES_param * portfolio_value
    
    # Historical ES
    tail_losses = data['returns'][data['returns'] < VaR_hist]
    ES_hist = tail_losses.mean()
    ES_hist_portfolio = ES_hist * portfolio_value
    
    col1, col2 = st.columns(2)
    col1.metric("Expected Shortfall Paramétrique", f"{ES_param*100:.2f}%")
    col1.metric("Perte attendue", f"${abs(ES_param_portfolio):,.0f}")
    col2.metric("Expected Shortfall Historique", f"{ES_hist*100:.2f}%")
    col2.metric("Perte attendue", f"${abs(ES_hist_portfolio):,.0f}")
    
    st.info("""
    **💡 Expected Shortfall (ES)** : Mesure la perte moyenne au-delà du seuil VaR.
    C'est une mesure plus conservatrice que la VaR car elle prend en compte la queue de distribution.
    """)
    
    st.markdown("---")
    
    # ==================== 5. Credit Risk Modeling ====================
    st.markdown("### 5️⃣ Credit Risk Modeling (Simulation de défaut)")
    
    S0 = last_price
    T = 1
    I = 100000
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(I))
    
    L = 0.5  # Loss Given Default (50%)
    p = 0.01  # Probabilité de défaut (1%)
    D = np.random.poisson(p * T, I)
    D = np.where(D >= 1, 1, D)
    
    discount = np.exp(-mu * T)
    S0_CVA = discount * np.mean((1 - L * D) * ST)
    Credit_VaR = discount * np.mean(L * D * ST)
    S0_adj = S0 - Credit_VaR
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Prix ajusté au risque de crédit", f"${S0_adj:.2f}")
    col2.metric("Credit VaR estimé", f"${Credit_VaR:.4f}")
    col3.metric("Événements de défaut simulés", np.count_nonzero(L * D * ST))
    
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=L * D * ST, nbinsx=50, opacity=0.7))
    fig3.update_layout(
        title="Distribution des pertes liées au risque de crédit",
        xaxis_title="Perte",
        yaxis_title="Fréquence",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== Récapitulatif ====================
    st.markdown("### 📋 Récapitulatif des risques")
    
    summary_df = pd.DataFrame({
        "Mesure de risque": [
            "VaR Paramétrique (5%)",
            "VaR Historique (5%)",
            "Expected Shortfall Paramétrique",
            "Expected Shortfall Historique",
            "Credit VaR"
        ],
        "Perte potentielle": [
            f"${abs(VaR_portfolio):,.0f}",
            f"${abs(VaR_hist_portfolio):,.0f}",
            f"${abs(ES_param_portfolio):,.0f}",
            f"${abs(ES_hist_portfolio):,.0f}",
            f"${Credit_VaR:.2f}"
        ]
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.caption("🎮 Module Luigi - Analyse complète des risques financiers pour Nintendo")

# ====================== PAGE BOWSER FULL WIDTH ======================================================================================================
if st.session_state["show_bowser_page"]:

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center; margin-top:10px;'>👻 Bowser – Option Pricing </h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; opacity:0.85;'>Vue analyste complète : états financiers, performance boursière, simulations Monte Carlo et scénarios.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Retour au dashboard principal", key="close_bowser"):
        st.session_state["show_bowser_page"] = False
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

    st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TITRE PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.title("💰 Conseil en Pricing d'Options - NINTENDO (NTDOY)")
    st.markdown("---")
    st.markdown("""
    **Types d'options évalués:**
    - ✅ Options Européennes (Black-Scholes-Merton)
    - ✅ Options Américaines (Binomial Tree)
    - ✅ Options Bermudéennes (Binomial Tree modifié)
    - ✅ Options Exotiques - Asiatiques (Monte Carlo)
    - ✅ Greeks pour gestion du risque
    """)
 
    
    # Sélection du profil d'investisseur
    st.subheader("1️⃣ Profil d'Investisseur")
    profils_dict = {
        1: 'COUVERTURE (HEDGING)',
        2: 'SPÉCULATION HAUSSIÈRE',
        3: 'SPÉCULATION BAISSIÈRE',
        4: 'GÉNÉRATION DE REVENUS',
        5: 'VOLATILITÉ'
    }
    
    profil_key = st.radio(
        "Sélectionnez votre profil:",
        options=list(profils_dict.keys()),
        format_func=lambda x: profils_dict[x],
        index=1
    )
    
    # Paramètres de données
    st.subheader("2️⃣ Paramètres de Données")
    
    ticker = "NTDOY"
    start_date = "2015-09-01"
    end_date = "2025-09-30"
    
    r = st.slider("Taux sans risque (%)", 1.0, 10.0, 4.0, step=0.5,
    key="bowser_taux_sans_risque") / 100
    n_simulations = st.selectbox("Simulations Monte Carlo", [3000, 5000, 8000], index=1, key="bowser_simulations")
    
    # Paramètres de strikes et maturités
    st.subheader("3️⃣ Paramet Évaluation")
    
    strikes_min = st.slider("Strike minimum (% du prix)", 80, 100, 90, step=5, key="bowser_strikes_min")
    strikes_max = st.slider("Strike maximum (% du prix)", 100, 130, 110, step=5, key="bowser_strikes_max")
    
    maturity_min = st.slider("Maturité min (mois)", 1, 12, 3, step=1, key="bowser_maturity_min")
    maturity_max = st.slider("Maturité max (mois)", 1, 12, 12, step=1, key="bowser_maturity_max")
    
    if maturity_min > maturity_max:
        st.error("La maturité min doit être inférieure à max")
        maturity_min = maturity_max

        st.markdown("---")

    # Bouton pour lancer la simulation
    lancer_simulation = st.button("🚀 Lancer la simulation d'options")

    if lancer_simulation:

    
    # ═══════════════════════════════════════════════════════════════════════════
    # DÉFINITION DES PROFILS
    # ═══════════════════════════════════════════════════════════════════════════
    
        profils_investisseur = {
        1: {
            'nom': 'COUVERTURE (HEDGING)',
            'strategie_principale': 'Achat de Puts pour protection',
            'options_recommandees': ['Put Européen', 'Put Américain'],
            'horizon_typique': 'Court à Moyen terme (3-6 mois)',
            'delta_target': 'Négatif (protection)',
            'description': 'Minimiser les pertes en cas de baisse du sous-jacent'
        },
        2: {
            'nom': 'SPÉCULATION HAUSSIÈRE',
            'strategie_principale': 'Achat de Calls',
            'options_recommandees': ['Call Européen', 'Call Américain', 'Call Asiatique'],
            'horizon_typique': 'Moyen terme (6-12 mois)',
            'delta_target': 'Positif élevé (>0.5)',
            'description': 'Profiter d\'une hausse anticipée avec effet de levier'
        },
        3: {
            'nom': 'SPÉCULATION BAISSIÈRE',
            'strategie_principale': 'Achat de Puts',
            'options_recommandees': ['Put Européen', 'Put Américain'],
            'horizon_typique': 'Court à Moyen terme (3-9 mois)',
            'delta_target': 'Négatif (<-0.3)',
            'description': 'Profiter d\'une baisse anticipée'
        },
        4: {
            'nom': 'GÉNÉRATION DE REVENUS',
            'strategie_principale': 'Vente de Calls couverts (Covered Calls)',
            'options_recommandees': ['Call Européen OTM', 'Call Bermudéen'],
            'horizon_typique': 'Court terme répété (1-3 mois)',
            'delta_target': 'Légèrement positif (0.3-0.5)',
            'description': 'Collecter des primes en vendant des calls sur actions détenues'
        },
        5: {
            'nom': 'VOLATILITÉ',
            'strategie_principale': 'Straddle/Strangle',
            'options_recommandees': ['Call & Put Européens', 'Options Exotiques'],
            'horizon_typique': 'Court terme (1-3 mois)',
            'delta_target': 'Neutre (proche de 0)',
            'description': 'Profiter des mouvements de prix importants sans direction précise'
        }
        }
        
        profil = profils_investisseur[profil_key]
        
        # ═══════════════════════════════════════════════════════════════════════════
        # TÉLÉCHARGEMENT DES DONNÉES
        # ═══════════════════════════════════════════════════════════════════════════
        
        @st.cache_data
        def download_data(ticker, start, end):
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                return data['Close']
            except:
                st.error(f"Erreur lors du téléchargement de {ticker}")
                return None
        
        # Affichage du statut de chargement
        with st.spinner("📥 Téléchargement des données Nintendo..."):
            data = download_data(ticker, start_date, end_date)
        
        if data is None or len(data) == 0:
            st.error("❌ Aucune donnée de clôture disponible pour NTDOY sur la période sélectionnée.")
            st.stop()
        
        # Ici, on est sûr d'avoir des données
        S0 = float(data.iloc[-1])
        returns = np.log(data / data.shift(1)).dropna()
        volatility_hist = float(returns.std() * np.sqrt(252))
        
        # Affichage des métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💵 Prix actuel", f"${S0:.2f}")
        with col2:
            st.metric("📊 Volatilité historique", f"{volatility_hist*100:.2f}%")
        with col3:
            st.metric("📅 Jours de trading", len(data))
        with col4:
            st.metric("💹 Taux sans risque", f"{r*100:.2f}%")
            
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # MODÈLES DE PRICING
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Black-Scholes
        def black_scholes_call(S, K, T, r, sigma):
            if T <= 0:
                return max(S - K, 0)
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return call_price
        
        def black_scholes_put(S, K, T, r, sigma):
            if T <= 0:
                return max(K - S, 0)
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return put_price
        
        def bs_greeks(S, K, T, r, sigma, option_type='call'):
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                delta = -norm.cdf(-d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}
        
        # Binomial Tree
        def binomial_tree_american(S, K, T, r, sigma, N=100, option_type='call'):
            dt = T / N
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            
            stock_tree = np.zeros((N + 1, N + 1))
            option_tree = np.zeros((N + 1, N + 1))
            
            for i in range(N + 1):
                stock_tree[i, N] = S * (u ** (N - i)) * (d ** i)
            
            for i in range(N + 1):
                if option_type == 'call':
                    option_tree[i, N] = max(stock_tree[i, N] - K, 0)
                else:
                    option_tree[i, N] = max(K - stock_tree[i, N], 0)
            
            for j in range(N - 1, -1, -1):
                for i in range(j + 1):
                    stock_tree[i, j] = S * (u ** (j - i)) * (d ** i)
                    continuation = np.exp(-r * dt) * (p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1])
                    
                    if option_type == 'call':
                        exercise = max(stock_tree[i, j] - K, 0)
                    else:
                        exercise = max(K - stock_tree[i, j], 0)
                    
                    option_tree[i, j] = max(continuation, exercise)
            
            return option_tree[0, 0]
        
        # Monte Carlo
        def asian_option_monte_carlo(S, K, T, r, sigma, n_simulations=30000, n_steps=64, option_type='call'):
            dt = T / n_steps
            discount_factor = np.exp(-r * T)
            payoffs = []
            
            for _ in range(n_simulations):
                prices = [S]
                for _ in range(n_steps):
                    z = np.random.standard_normal()
                    S_next = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                    prices.append(S_next)
                
                avg_price = np.mean(prices)
                
                if option_type == 'call':
                    payoff = max(avg_price - K, 0)
                else:
                    payoff = max(K - avg_price, 0)
                
                payoffs.append(payoff)
            
            option_price = discount_factor * np.mean(payoffs)
            std_error = discount_factor * np.std(payoffs) / np.sqrt(n_simulations)
            
            return option_price, std_error
        
        # ═══════════════════════════════════════════════════════════════════════════
        # CALCUL DES RÉSULTATS
        # ═══════════════════════════════════════════════════════════════════════════
        
        st.subheader(f"📊 Profil: {profil['nom']}")
        st.write(f"**Stratégie:** {profil['strategie_principale']}")
        st.write(f"**Description:** {profil['description']}")
        
        # Paramètres selon le profil
        if profil_key == 1:  # COUVERTURE
            option_types_focus = ['put']
        elif profil_key == 2:  # HAUSSIER
            option_types_focus = ['call']
        elif profil_key == 3:  # BAISSIER
            option_types_focus = ['put']
        elif profil_key == 4:  # REVENUS
            option_types_focus = ['call']
        else:  # VOLATILITÉ
            option_types_focus = ['call', 'put']
        
        # Génération des strikes et maturités
        strikes_pct = np.linspace(strikes_min/100, strikes_max/100, 3)
        K_values = [S0 * mult for mult in strikes_pct]
        
        maturities = list(range(maturity_min, maturity_max + 1))
        
            
        T_values = [m/12 for m in maturities]
        
        st.markdown("---")
        st.subheader("⚙️ Configuration de l'évaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"✅ Strikes évalués: {len(K_values)}")
            st.info(f"✅ Maturités évaluées: {len(T_values)}")
        with col2:
            st.info(f"✅ Volatilité: {volatility_hist*100:.2f}%")
            st.info(f"✅ Taux sans risque: {r*100:.2f}%")
        
        # Calcul des options
        with st.spinner("⏳ Calcul des options en cours..."):
            results_all = []
            
            for K in K_values:
                for T in T_values:
                    T_months = int(T * 12)
                    moneyness = S0 / K
                    
                    if moneyness > 1.05:
                        status = 'ITM'
                    elif moneyness > 0.95:
                        status = 'ATM'
                    else:
                        status = 'OTM'
                    
                    result = {
                        'Strike': K,
                        'Maturité (mois)': T_months,
                        'Maturité (années)': T,
                        'Moneyness': moneyness,
                        'Status': status
                    }
                    
                    if 'call' in option_types_focus:
                        call_euro = black_scholes_call(S0, K, T, r, volatility_hist)
                        result['Call Européen'] = call_euro
                        
                        call_american = binomial_tree_american(S0, K, T, r, volatility_hist, N=100, option_type='call')
                        result['Call Américain'] = call_american
                        
                        call_asian, _ = asian_option_monte_carlo(S0, K, T, r, volatility_hist, n_simulations=n_simulations, option_type='call')
                        result['Call Asiatique'] = call_asian
                        
                        greeks_call = bs_greeks(S0, K, T, r, volatility_hist, 'call')
                        result['Call Delta'] = greeks_call['delta']
                        result['Call Gamma'] = greeks_call['gamma']
                        result['Call Vega'] = greeks_call['vega']
                        result['Call Theta'] = greeks_call['theta']
                    
                    if 'put' in option_types_focus:
                        put_euro = black_scholes_put(S0, K, T, r, volatility_hist)
                        result['Put Européen'] = put_euro
                        
                        put_american = binomial_tree_american(S0, K, T, r, volatility_hist, N=100, option_type='put')
                        result['Put Américain'] = put_american
                        
                        put_asian, _ = asian_option_monte_carlo(S0, K, T, r, volatility_hist, n_simulations=n_simulations, option_type='put')
                        result['Put Asiatique'] = put_asian
                        
                        greeks_put = bs_greeks(S0, K, T, r, volatility_hist, 'put')
                        result['Put Delta'] = greeks_put['delta']
                        result['Put Gamma'] = greeks_put['gamma']
                        result['Put Vega'] = greeks_put['vega']
                        result['Put Theta'] = greeks_put['theta']
                    
                    results_all.append(result)
            
            df_results = pd.DataFrame(results_all)
        
        st.success(f"✅ {len(df_results)} configurations d'options évaluées")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ONGLETS INTERACTIFS
        # ═══════════════════════════════════════════════════════════════════════════
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Résultats", "📈 Visualisations", "🎯 Recommandations", "📉 P&L", "📋 Tableau"]
        )
        
        # TAB 1: RÉSULTATS
        with tab1:
            st.subheader("Résultats des Évaluations")
            
            # Filtrage optionnel
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_status = st.multiselect("Filtrer par Status", ['ITM', 'ATM', 'OTM'], default=['ITM', 'ATM', 'OTM'])
            with col2:
                selected_maturity = st.multiselect("Filtrer par Maturité", sorted(df_results['Maturité (mois)'].unique()), 
                                                  default=sorted(df_results['Maturité (mois)'].unique()))
            with col3:
                precision = st.slider("Décimales", 2, 4, 2)
            
            # Filtrage
            df_filtered = df_results[
                (df_results['Status'].isin(selected_status)) &
                (df_results['Maturité (mois)'].isin(selected_maturity))
            ]
            
            # Arrondir les colonnes numériques
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            df_display = df_filtered.copy()
            for col in numeric_cols:
                df_display[col] = df_display[col].round(precision)
            
            st.dataframe(df_display, use_container_width=True)
            
            # Export
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name="nintendo_options.csv",
                mime="text/csv"
            )
        
        # TAB 2: VISUALISATIONS
        with tab2:
            st.subheader("Graphiques Interactifs")
            
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["3D Surface", "Comparaison", "Greeks"])
            
            # 3D Surface
            with sub_tab1:
                if 'call' in option_types_focus:
                    st.markdown("#### Surface 3D - Call Européen")
                    
                    strikes_unique = sorted(df_results['Strike'].unique())
                    maturities_unique = sorted(df_results['Maturité (années)'].unique())
                    
                    Z_call = []
                    for T in maturities_unique:
                        row = []
                        for K in strikes_unique:
                            val = df_results[(df_results['Strike'] == K) & 
                                            (df_results['Maturité (années)'] == T)]['Call Européen'].values
                            row.append(val[0] if len(val) > 0 else 0)
                        Z_call.append(row)
                    
                    fig_3d_call = go.Figure(data=[go.Surface(
                        x=strikes_unique,
                        y=maturities_unique,
                        z=Z_call,
                        colorscale='Viridis'
                    )])
                    
                    fig_3d_call.update_layout(
                        title='Prix Call Européen - Surface 3D',
                        scene=dict(
                            xaxis_title='Strike ($)',
                            yaxis_title='Maturité (années)',
                            zaxis_title='Prix ($)'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig_3d_call, use_container_width=True)
                
                if 'put' in option_types_focus:
                    st.markdown("#### Surface 3D - Put Européen")
                    
                    Z_put = []
                    for T in maturities_unique:
                        row = []
                        for K in strikes_unique:
                            val = df_results[(df_results['Strike'] == K) & 
                                            (df_results['Maturité (années)'] == T)]['Put Européen'].values
                            row.append(val[0] if len(val) > 0 else 0)
                        Z_put.append(row)
                    
                    fig_3d_put = go.Figure(data=[go.Surface(
                        x=strikes_unique,
                        y=maturities_unique,
                        z=Z_put,
                        colorscale='Reds'
                    )])
                    
                    fig_3d_put.update_layout(
                        title='Prix Put Européen - Surface 3D',
                        scene=dict(
                            xaxis_title='Strike ($)',
                            yaxis_title='Maturité (années)',
                            zaxis_title='Prix ($)'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig_3d_put, use_container_width=True)
            
            # Comparaison
            with sub_tab2:
                target_maturity = 6
                closest_maturity = min(maturities, key=lambda x: abs(x - target_maturity))
                df_comp = df_results[df_results['Maturité (mois)'] == closest_maturity].copy()
                
                if 'call' in option_types_focus and len(df_comp) > 0:
                    st.markdown(f"#### Comparaison des Calls (Maturité: {closest_maturity} mois)")
                    
                    fig_comp_call = go.Figure()
                    
                    fig_comp_call.add_trace(go.Scatter(
                        x=df_comp['Strike'],
                        y=df_comp['Call Européen'],
                        name='Call Européen',
                        mode='lines+markers',
                        line=dict(color='blue', width=3)
                    ))
                    
                    if 'Call Américain' in df_comp.columns:
                        fig_comp_call.add_trace(go.Scatter(
                            x=df_comp['Strike'],
                            y=df_comp['Call Américain'],
                            name='Call Américain',
                            mode='lines+markers',
                            line=dict(color='green', width=3)
                        ))
                    
                    fig_comp_call.add_vline(x=S0, line_dash="dash", line_color="red")
                    
                    fig_comp_call.update_layout(
                        title=f'Comparaison des Calls',
                        xaxis_title='Strike ($)',
                        yaxis_title='Prix ($)',
                        height=500
                    )
                    
                    st.plotly_chart(fig_comp_call, use_container_width=True)
                
                if 'put' in option_types_focus and len(df_comp) > 0:
                    st.markdown(f"#### Comparaison des Puts (Maturité: {closest_maturity} mois)")
                    
                    fig_comp_put = go.Figure()
                    
                    fig_comp_put.add_trace(go.Scatter(
                        x=df_comp['Strike'],
                        y=df_comp['Put Européen'],
                        name='Put Européen',
                        mode='lines+markers',
                        line=dict(color='blue', width=3)
                    ))
                    
                    if 'Put Américain' in df_comp.columns:
                        fig_comp_put.add_trace(go.Scatter(
                            x=df_comp['Strike'],
                            y=df_comp['Put Américain'],
                            name='Put Américain',
                            mode='lines+markers',
                            line=dict(color='green', width=3)
                        ))
                    
                    fig_comp_put.add_vline(x=S0, line_dash="dash", line_color="red")
                    
                    fig_comp_put.update_layout(
                        title=f'Comparaison des Puts',
                        xaxis_title='Strike ($)',
                        yaxis_title='Prix ($)',
                        height=500
                    )
                    
                    st.plotly_chart(fig_comp_put, use_container_width=True)
            
            # Greeks
            with sub_tab3:
                if len(df_comp) > 0:
                    if 'call' in option_types_focus and 'Call Delta' in df_comp.columns:
                        st.markdown("#### Greeks - Calls")

                        fig_greeks = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta')
                        )
                        
                        fig_greeks.add_trace(
                            go.Scatter(x=df_comp['Strike'], y=df_comp['Call Delta'],
                                      name='Delta', line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig_greeks.add_trace(
                            go.Scatter(x=df_comp['Strike'], y=df_comp['Call Gamma'],
                                      name='Gamma', line=dict(color='green')),
                            row=1, col=2
                        )
                        fig_greeks.add_trace(
                            go.Scatter(x=df_comp['Strike'], y=df_comp['Call Vega'],
                                      name='Vega', line=dict(color='orange')),
                            row=2, col=1
                        )
                        fig_greeks.add_trace(
                            go.Scatter(x=df_comp['Strike'], y=df_comp['Call Theta'],
                                      name='Theta', line=dict(color='red')),
                            row=2, col=2
                        )
                        
                        fig_greeks.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig_greeks, use_container_width=True)
        
        # TAB 3: RECOMMANDATIONS
        with tab3:
            st.subheader(f"🎯 Recommandations pour {profil['nom']}")
            
            if profil_key == 1:  # COUVERTURE
                st.markdown("""
                ### Stratégie de Couverture (Hedging)
                
                Vous détenez des actions Nintendo et voulez vous protéger contre une baisse.
                
                **Options recommandées:** Protective Puts
                """)
                
                best_hedge = df_results[df_results['Status'] == 'ATM'].nsmallest(3, 'Maturité (mois)')
                
                for i, (_, row) in enumerate(best_hedge.iterrows(), 1):
                    with st.expander(f"Option {i}: Put ${row['Strike']:.2f} ({int(row['Maturité (mois)'])} mois)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Prix", f"${row['Put Européen']:.2f}")
                        with col2:
                            st.metric("Delta", f"{row['Put Delta']:.3f}")
                        with col3:
                            st.metric("Gamma", f"{row['Put Gamma']:.6f}")
                        with col4:
                            st.metric("Theta", f"{row['Put Theta']:.6f}")
            
            elif profil_key == 2:  # HAUSSIER
                st.markdown("""
                ### Spéculation Haussière
                
                Vous anticipez une hausse - Achetez des Calls pour profiter de l'effet de levier.
                
                **Options recommandées:** Long Calls
                """)
                
                best_calls = df_results[df_results['Status'].isin(['ATM', 'OTM'])].nsmallest(3, 'Call Européen')
                
                for i, (_, row) in enumerate(best_calls.iterrows(), 1):
                    leverage = S0 / row['Call Européen'] if row['Call Européen'] > 0 else 0
                    breakeven = row['Strike'] + row['Call Européen']
                    required_move = ((breakeven / S0) - 1) * 100
                    
                    with st.expander(f"Option {i}: Call ${row['Strike']:.2f} ({int(row['Maturité (mois)'])} mois)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Prix", f"${row['Call Européen']:.2f}")
                        with col2:
                            st.metric("Levier", f"{leverage:.1f}x")
                        with col3:
                            st.metric("Point mort", f"${breakeven:.2f}")
                        with col4:
                            st.metric("Hausse requise", f"{required_move:.1f}%")
            
            elif profil_key == 3:  # BAISSIER
                st.markdown("""
                ### Spéculation Baissière
                
                Vous anticipez une baisse - Achetez des Puts pour profiter du mouvement baissier.
                
                **Options recommandées:** Long Puts
                """)
                
                best_puts = df_results[df_results['Status'].isin(['ATM', 'OTM'])].nsmallest(3, 'Put Européen')
                
                for i, (_, row) in enumerate(best_puts.iterrows(), 1):
                    leverage = S0 / row['Put Européen'] if row['Put Européen'] > 0 else 0
                    breakeven = row['Strike'] - row['Put Européen']
                    required_move = ((S0 / breakeven) - 1) * 100
                    
                    with st.expander(f"Option {i}: Put ${row['Strike']:.2f} ({int(row['Maturité (mois)'])} mois)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Prix", f"${row['Put Européen']:.2f}")
                        with col2:
                            st.metric("Levier", f"{leverage:.1f}x")
                        with col3:
                            st.metric("Point mort", f"${breakeven:.2f}")
                        with col4:
                            st.metric("Baisse requise", f"{required_move:.1f}%")
            
            elif profil_key == 4:  # REVENUS
                st.markdown("""
                ### Génération de Revenus (Covered Calls)
                
                Vous détenez des actions et voulez générer des revenus réguliers en vendant des Calls.
                
                **Options recommandées:** Covered Calls OTM
                """)
                
                covered_calls = df_results[
                    (df_results['Status'].isin(['ATM', 'OTM'])) &
                    (df_results['Maturité (mois)'] <= 3)
                ].sort_values('Maturité (mois)').head(3)
                
                for i, (_, row) in enumerate(covered_calls.iterrows(), 1):
                    annualized = (row['Call Européen'] / S0) * (12 / row['Maturité (mois)']) * 100
                    
                    with st.expander(f"Option {i}: Vendre Call ${row['Strike']:.2f} ({int(row['Maturité (mois)'])} mois)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Prime reçue", f"${row['Call Européen']:.2f}")
                        with col2:
                            st.metric("Annualisé", f"{annualized:.2f}%")
                        with col3:
                            st.metric("Strike au-dessus", f"{((row['Strike']/S0)-1)*100:.1f}%")
                        with col4:
                            st.metric("Delta", f"{row['Call Delta']:.3f}")
            
            else:  # VOLATILITÉ
                st.markdown("""
                ### Stratégie sur Volatilité (Straddle)
                
                Vous anticipez un mouvement important - Achetez un Call + Put ATM.
                
                **Options recommandées:** Long Straddle
                """)
                
                straddles = df_results[df_results['Status'] == 'ATM'].copy()
                straddles['Straddle Cost'] = straddles['Call Européen'] + straddles['Put Européen']
                straddles = straddles.nsmallest(3, 'Maturité (mois)')
                
                for i, (_, row) in enumerate(straddles.iterrows(), 1):
                    move_required = (row['Straddle Cost'] / S0) * 100
                    
                    with st.expander(f"Option {i}: Straddle ${row['Strike']:.2f} ({int(row['Maturité (mois)'])} mois)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Coût total", f"${row['Straddle Cost']:.2f}")
                        with col2:
                            st.metric("Mouvement requis", f"±{move_required:.1f}%")
                        with col3:
                            st.metric("Call Vega", f"{row['Call Vega']:.3f}")
                        with col4:
                            st.metric("Put Vega", f"{row['Put Vega']:.3f}")
        
        # TAB 4: P&L
        with tab4:
            st.subheader("📉 Analyse Profit & Loss")
            
            # Sélection d'une option
            option_selected = st.selectbox(
                "Sélectionnez une option pour analyser",
                options=range(len(df_results)),
                format_func=lambda x: f"{df_results.iloc[x]['Status']} - ${df_results.iloc[x]['Strike']:.2f} ({int(df_results.iloc[x]['Maturité (mois)'])} mois)"
            )
            
            selected_option = df_results.iloc[option_selected]
            
            if 'call' in option_types_focus:
                K_call = selected_option['Strike']
                premium_call = selected_option['Call Européen']
                
                price_range = np.linspace(S0 * 0.7, S0 * 1.3, 100)
                payoff_call = np.maximum(price_range - K_call, 0) - premium_call
                breakeven_call = K_call + premium_call
                
                fig_pl_call = go.Figure()
                
                fig_pl_call.add_trace(go.Scatter(
                    x=price_range,
                    y=payoff_call,
                    name='Long Call',
                    line=dict(color='green', width=3),
                    fill='tozeroy'
                ))
                
                fig_pl_call.add_hline(y=0, line_dash="solid", line_color="black")
                fig_pl_call.add_vline(x=S0, line_dash="dash", line_color="blue")
                fig_pl_call.add_vline(x=breakeven_call, line_dash="dash", line_color="red")
                
                fig_pl_call.update_layout(
                    title=f'Long Call: Strike ${K_call:.2f}, Prime ${premium_call:.2f}',
                    xaxis_title='Prix à maturité ($)',
                    yaxis_title='Profit / Perte ($)',
                    height=500
                )
                
                st.plotly_chart(fig_pl_call, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max loss", f"-${premium_call:.2f}")
                with col2:
                    st.metric("Max gain", "Illimité")
                with col3:
                    st.metric("Point mort", f"${breakeven_call:.2f}")
            
            if 'put' in option_types_focus:
                K_put = selected_option['Strike']
                premium_put = selected_option['Put Européen']
                
                price_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
                payoff_put = np.maximum(K_put - price_range, 0) - premium_put
                breakeven_put = K_put - premium_put
                
                fig_pl_put = go.Figure()
                
                fig_pl_put.add_trace(go.Scatter(
                    x=price_range,
                    y=payoff_put,
                    name='Long Put',
                    line=dict(color='red', width=3),
                    fill='tozeroy'
                ))
                
                fig_pl_put.add_hline(y=0, line_dash="solid", line_color="black")
                fig_pl_put.add_vline(x=S0, line_dash="dash", line_color="blue")
                fig_pl_put.add_vline(x=breakeven_put, line_dash="dash", line_color="red")
                
                fig_pl_put.update_layout(
                    title=f'Long Put: Strike ${K_put:.2f}, Prime ${premium_put:.2f}',
                    xaxis_title='Prix à maturité ($)',
                    yaxis_title='Profit / Perte ($)',
                    height=500
                )
                
                st.plotly_chart(fig_pl_put, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max loss", f"-${premium_put:.2f}")
                with col2:
                    st.metric("Max gain", f"${K_put - premium_put:.2f}")
                with col3:
                    st.metric("Point mort", f"${breakeven_put:.2f}")
        
        # TAB 5: TABLEAU COMPLET
        with tab5:
            st.subheader("📊 Tableau Complet")
            
            # Options d'affichage
            col1, col2 = st.columns(2)
            with col1:
                show_greeks = st.checkbox("Afficher les Greeks", value=True)
            with col2:
                decimals = st.slider("Décimales", 2, 6, 2)
            
            # Préparation du tableau
            df_display = df_results.copy()
            
            if not show_greeks:
                greek_cols = [col for col in df_display.columns if any(x in col for x in ['Delta', 'Gamma', 'Vega', 'Theta'])]
                df_display = df_display.drop(columns=greek_cols, errors='ignore')
            
            # Arrondir
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_display[col] = df_display[col].round(decimals)
            
            # Affichage
            st.dataframe(df_display, use_container_width=True)
            
            # Statistiques descriptives
            st.subheader("📈 Statistiques Descriptives")
            
            if 'call' in option_types_focus:
                st.markdown("#### Calls")
                call_stats = df_results[['Call Européen', 'Call Américain', 'Call Asiatique']].describe()
                st.dataframe(call_stats.round(decimals))
            
            if 'put' in option_types_focus:
                st.markdown("#### Puts")
                put_stats = df_results[['Put Européen', 'Put Américain', 'Put Asiatique']].describe()
                st.dataframe(put_stats.round(decimals))
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PIED DE PAGE
        # ═══════════════════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.markdown("""
        ### ⚠️ Avertissements Importants
        
        - **Modèles théoriques:** Les prix affichés sont des prix théoriques calculés avec des modèles mathématiques
        - **Hypothèses simplificatrices:** Volatilité constante, pas de dividendes, marchés parfaits, etc.
        - **Risque élevé:** Les options sont des produits complexes destinés à des investisseurs avertis
        - **Spread bid-ask:** Les prix réels incluent le spread du marché
        - **Volatilité implicite:** La volatilité du marché peut différer de l'historique
        - **Consultez un professionnel:** Avant toute décision d'investissement
        
        ---
        *Analyse réalisée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*
        """)

    else:
        st.info("👉 Configure les paramètres puis clique sur « 🚀 Lancer la simulation d'options » pour afficher les résultats.")


# ====================== PAGE BIRDO FULL WIDTH ===========================================================================================================================================================================
if st.session_state["show_birdo_page"]:

    # Titre principal
    st.markdown("""
    # 🎮 CONSEIL EN TRADING ALGORITHMIQUE - NINTENDO (NTDOY)
    **Période: Septembre 2015 - Septembre 2025**
    
    *Stratégies optimisées selon votre profil d'investisseur*
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Retour au dashboard principal", key="close_birdo"):
        st.session_state["show_birdo_page"] = False
        st.rerun()


    # =========================
    # PARAMÈTRES DANS L'ONGLET
    # =========================
    
    st.markdown("### ⚙️ Paramétrage de votre profil")
    
    # Sélecteur de profil (dans la page, plus dans la sidebar)
    profil_label = st.selectbox(
        "Choisissez votre profil d'investisseur :",
        options=[
            "CONSERVATEUR - Minimisation du risque, fréquence faible",
            "MODÉRÉ - Équilibre risque/rendement, fréquence moyenne",
            "DYNAMIQUE - Maximisation du rendement, fréquence élevée",
            "PERSONNALISÉ - Paramètres manuels"
        ],
        index=1
    )
    
    # Profils prédéfinis (inchangés)
    profils = {
        1: {
            'nom': 'CONSERVATEUR',
            'sma_short_range': range(50, 101, 10),
            'sma_long_range': range(200, 301, 20),
            'n_lags_regression': 10,
            'n_clusters_kmeans': 2
        },
        2: {
            'nom': 'MODÉRÉ',
            'sma_short_range': range(30, 71, 5),
            'sma_long_range': range(150, 281, 10),
            'n_lags_regression': 7,
            'n_clusters_kmeans': 3
        },
        3: {
            'nom': 'DYNAMIQUE',
            'sma_short_range': range(10, 51, 5),
            'sma_long_range': range(100, 201, 10),
            'n_lags_regression': 5,
            'n_clusters_kmeans': 4
        }
    }
    
    # Mapping du label vers l’ID de profil
    if profil_label.startswith("CONSERVATEUR"):
        profil_id = 1
    elif profil_label.startswith("MODÉRÉ"):
        profil_id = 2
    elif profil_label.startswith("DYNAMIQUE"):
        profil_id = 3
    else:
        profil_id = 4  # PERSONNALISÉ
    
    # Si profil personnalisé : afficher les inputs dans le corps
    if profil_id == 4:
        st.info("Mode **PERSONNALISÉ** activé : ajustez manuellement les paramètres ci-dessous.")
        col_p1, col_p2 = st.columns(2)
    
        with col_p1:
            sma_short_deb = st.number_input("Début de la plage du SMA court", min_value=5, max_value=100, value=10)
            sma_short_fin = st.number_input("Fin de la plage du SMA court", min_value=20, max_value=150, value=50)
            sma_short_pas = st.number_input("Pas du SMA court", min_value=1, max_value=20, value=5)
    
        with col_p2:
            sma_long_deb = st.number_input("Début de la plage du SMA long", min_value=50, max_value=200, value=100)
            sma_long_fin = st.number_input("Fin de la plage du SMA long", min_value=100, max_value=400, value=200)
            sma_long_pas = st.number_input("Pas du SMA long", min_value=1, max_value=50, value=10)
    
        n_lags_regression = st.number_input("Nombre de lags pour la régression OLS", min_value=1, max_value=20, value=5)
        n_clusters_kmeans = st.number_input("Nombre de clusters K-Means", min_value=2, max_value=8, value=4)
    
        profil = {
            'nom': 'PERSONNALISÉ',
            'sma_short_range': range(int(sma_short_deb), int(sma_short_fin) + 1, int(sma_short_pas)),
            'sma_long_range': range(int(sma_long_deb), int(sma_long_fin) + 1, int(sma_long_pas)),
            'n_lags_regression': int(n_lags_regression),
            'n_clusters_kmeans': int(n_clusters_kmeans)
        }
    else:
        profil = profils[profil_id]
    
    st.success(f"✅ Profil sélectionné : **{profil['nom']}**")
    
    # =========================
    # CHARGEMENT DES DONNÉES
    # =========================
    
    @st.cache_data
    def load_nintendo_data():
        ticker = "NTDOY"
        start_date = "2015-09-01"
        end_date = "2025-09-30"
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        data.name = 'Close'
        return data
    
    st.markdown("---")
    data_original = load_nintendo_data()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Prix initial", f"${data_original.iloc[0]:.2f}")
    col2.metric("💰 Prix actuel", f"${data_original.iloc[-1]:.2f}")
    col3.metric("⏱️ Période", f"{len(data_original)} jours")
    col4.metric("📊 Performance", f"{((data_original.iloc[-1]/data_original.iloc[0])-1)*100:.1f}%")
    
    # =========================
    # ONGLET(S) D’ANALYSE
    # =========================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 SMA Optimisée",
        "⚙️ Backtesting",
        "📈 Régression OLS",
        "🤖 K-Means ML"
    ])



    with tab1:
        st.header("🔍 Optimisation des Paramètres SMA")
        
        # Optimisation
        @st.cache_data
        def optimize_sma(profil):
            optimization_results = pd.DataFrame()
            for SMA1, SMA2 in product(profil['sma_short_range'], profil['sma_long_range']):
                if SMA1 >= SMA2: continue
                
                temp_data = data_original.copy()
                temp_data['SMA_Short'] = temp_data['Close'].rolling(SMA1).mean()
                temp_data['SMA_Long'] = temp_data['Close'].rolling(SMA2).mean()
                temp_data.dropna(inplace=True)
                
                temp_data['Position'] = np.where(temp_data['SMA_Short'] > temp_data['SMA_Long'], 1, -1)
                temp_data['Returns'] = np.log(temp_data['Close'] / temp_data['Close'].shift(1))
                temp_data['Strategy'] = temp_data['Position'].shift(1) * temp_data['Returns']
                temp_data.dropna(inplace=True)
                
                perf = np.exp(temp_data[['Returns', 'Strategy']].sum())
                volatility = temp_data['Strategy'].std() * np.sqrt(252)
                sharpe = (temp_data['Strategy'].mean() * 252) / volatility if volatility > 0 else 0
                
                result = pd.DataFrame({
                    'SMA_Short': [SMA1], 'SMA_Long': [SMA2],
                    'Strategy_Return': [perf['Strategy']], 'Sharpe_Ratio': [sharpe]
                })
                optimization_results = pd.concat([optimization_results, result], ignore_index=True)
            
            return optimization_results.sort_values('Sharpe_Ratio', ascending=False)
        
        optimization_results = optimize_sma(profil)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("⭐ Meilleurs paramètres", f"SMA {int(optimization_results.iloc[0]['SMA_Short'])}/{int(optimization_results.iloc[0]['SMA_Long'])}")
            st.metric("📈 Performance", f"{optimization_results.iloc[0]['Strategy_Return']:.2f}x")
            st.metric("⚖️ Sharpe Ratio", f"{optimization_results.iloc[0]['Sharpe_Ratio']:.3f}")
        
        with col2:
            st.dataframe(optimization_results.head(10)[['SMA_Short', 'SMA_Long', 'Strategy_Return', 'Sharpe_Ratio']].round(4))
        
        # Graphique 3D interactif
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=optimization_results['SMA_Short'], y=optimization_results['SMA_Long'], 
            z=optimization_results['Sharpe_Ratio'], mode='markers',
            marker=dict(size=6, color=optimization_results['Sharpe_Ratio'], colorscale='Viridis', showscale=True)
        )])
        fig_3d.update_layout(title=f"Optimisation SMA 3D - {profil['nom']}", height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab2:
        st.header("⚙️ Backtesting Vectorisé")
        
        # Application SMA optimale
        best_params = optimization_results.iloc[0]
        SMA_SHORT_OPT, SMA_LONG_OPT = int(best_params['SMA_Short']), int(best_params['SMA_Long'])
        
        data_sma = data_original.copy()
        data_sma['SMA_Short'] = data_sma['Close'].rolling(SMA_SHORT_OPT).mean()
        data_sma['SMA_Long'] = data_sma['Close'].rolling(SMA_LONG_OPT).mean()
        data_sma.dropna(inplace=True)
        data_sma['Position'] = np.where(data_sma['SMA_Short'] > data_sma['SMA_Long'], 1, -1)
        data_sma['Returns'] = np.log(data_sma['Close'] / data_sma['Close'].shift(1))
        data_sma['Strategy'] = data_sma['Position'].shift(1) * data_sma['Returns']
        data_sma.dropna(inplace=True)
        
        # Métriques
        cumulative_returns = np.exp(data_sma[['Returns', 'Strategy']].sum())
        volatility = data_sma[['Returns', 'Strategy']].std() * np.sqrt(252)
        sharpe = (data_sma[['Returns', 'Strategy']].mean() * 252) / volatility
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Performance SMA", f"{cumulative_returns['Strategy']:.2f}x")
        col2.metric("Sharpe Ratio", f"{sharpe['Strategy']:.3f}")
        col3.metric("Nb Trades", f"{int(data_sma['Position'].diff().ne(0).sum())}")
        
        # Graphiques
        fig_backtest = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   subplot_titles=('Prix & SMAs', 'Performance Cumulative'),
                                   row_heights=[0.6, 0.4])
        
        # Prix et SMAs
        fig_backtest.add_trace(go.Scatter(x=data_sma.index, y=data_sma['Close'], name='Prix', line=dict(color='blue')), row=1, col=1)
        fig_backtest.add_trace(go.Scatter(x=data_sma.index, y=data_sma['SMA_Short'], name=f'SMA {SMA_SHORT_OPT}', line=dict(color='orange')), row=1, col=1)
        fig_backtest.add_trace(go.Scatter(x=data_sma.index, y=data_sma['SMA_Long'], name=f'SMA {SMA_LONG_OPT}', line=dict(color='red')), row=1, col=1)
        
        # Performance
        cumulative_perf = data_sma[['Returns', 'Strategy']].cumsum().apply(np.exp)
        fig_backtest.add_trace(go.Scatter(x=cumulative_perf.index, y=cumulative_perf['Returns'], name='Buy & Hold', line=dict(color='blue')), row=2, col=1)
        fig_backtest.add_trace(go.Scatter(x=cumulative_perf.index, y=cumulative_perf['Strategy'], name='SMA Strategy', line=dict(color='green')), row=2, col=1)
        
        fig_backtest.update_layout(height=700, title=f"Backtesting SMA {SMA_SHORT_OPT}/{SMA_LONG_OPT} - {profil['nom']}")
        st.plotly_chart(fig_backtest, use_container_width=True)
    
    with tab3:
        st.header("📈 Stratégie Régression OLS")
        
        # Régression
        data_regression = data_original.copy()
        data_regression['returns'] = np.log(data_regression['Close'] / data_regression['Close'].shift(1))
        lags = profil['n_lags_regression']
        
        cols = [f'lag_{lag}' for lag in range(1, lags + 1)]
        for lag in range(1, lags + 1):
            data_regression[f'lag_{lag}'] = data_regression['returns'].shift(lag)
        data_regression.dropna(inplace=True)
        data_regression['direction'] = np.sign(data_regression['returns']).astype(int)
        
        # Modèles
        model_returns = LinearRegression().fit(data_regression[cols], data_regression['returns'])
        model_direction = LinearRegression().fit(data_regression[cols], data_regression['direction'])
        
        data_regression['pred_returns'] = model_returns.predict(data_regression[cols])
        data_regression['pred_direction'] = model_direction.predict(data_regression[cols])
        data_regression['pos_reg_returns'] = np.where(data_regression['pred_returns'] > 0, 1, -1)
        data_regression['pos_reg_direction'] = np.where(data_regression['pred_direction'] > 0, 1, -1)
        
        data_regression['strat_reg_returns'] = data_regression['pos_reg_returns'] * data_regression['returns']
        data_regression['strat_reg_direction'] = data_regression['pos_reg_direction'] * data_regression['returns']
        
        perf_regression = np.exp(data_regression[['returns', 'strat_reg_returns', 'strat_reg_direction']].sum())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Régr. Rendements", f"{perf_regression['strat_reg_returns']:.2f}x")
        col2.metric("Régr. Direction", f"{perf_regression['strat_reg_direction']:.2f}x")
        col3.metric("Précision", f"{((data_regression['direction'] == data_regression['pos_reg_returns']).mean()*100):.1f}%")
        
        # Graphique performances
        cumulative_reg = data_regression[['returns', 'strat_reg_returns', 'strat_reg_direction']].cumsum().apply(np.exp)
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=cumulative_reg.index, y=cumulative_reg['returns'], name='Buy & Hold'))
        fig_reg.add_trace(go.Scatter(x=cumulative_reg.index, y=cumulative_reg['strat_reg_returns'], name='Régr. Rendements'))
        fig_reg.add_trace(go.Scatter(x=cumulative_reg.index, y=cumulative_reg['strat_reg_direction'], name='Régr. Direction'))
        fig_reg.update_layout(title=f"Régression OLS - {lags} lags ({profil['nom']})", height=500)
        st.plotly_chart(fig_reg, use_container_width=True)
    
    with tab4:
        st.header("🤖 K-Means Clustering (Machine Learning)")
        
        # K-Means
        data_ml = data_regression[cols + ['returns', 'direction']].copy()
        data_ml['volatility_5'] = data_regression['returns'].rolling(5).std()
        data_ml['volatility_20'] = data_regression['returns'].rolling(20).std()
        data_ml['momentum_5'] = data_regression['returns'].rolling(5).mean()
        data_ml['momentum_20'] = data_regression['returns'].rolling(20).mean()
        data_ml.dropna(inplace=True)
        
        features = cols + ['volatility_5', 'volatility_20', 'momentum_5', 'momentum_20']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data_ml[features])
        
        n_clusters = profil['n_clusters_kmeans']
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data_ml['cluster'] = kmeans.fit_predict(scaled_features)
        
        cluster_returns = data_ml.groupby('cluster')['returns'].mean()
        cluster_to_position = {cluster: (1 if ret > 0 else -1) for cluster, ret in cluster_returns.items()}
        
        data_ml['pos_cluster'] = data_ml['cluster'].map(cluster_to_position)
        data_ml['strat_cluster'] = data_ml['pos_cluster'] * data_ml['returns']
        perf_cluster = np.exp(data_ml[['returns', 'strat_cluster']].sum())
        
        col1, col2 = st.columns(2)
        col1.metric("Performance K-Means", f"{perf_cluster['strat_cluster']:.2f}x")
        col2.metric("Précision", f"{((data_ml['direction'] == data_ml['pos_cluster']).mean()*100):.1f}%")
        
        # Graphiques clusters
        fig_clusters = make_subplots(rows=1, cols=2, subplot_titles=('Clusters (Lag1 vs Lag2)', 'Performance'))
        
        for cluster in range(n_clusters):
            cluster_data = data_ml[data_ml['cluster'] == cluster]
            fig_clusters.add_trace(go.Scatter(x=cluster_data['lag_1'], y=cluster_data['lag_2'], 
                                            mode='markers', name=f'Cluster {cluster}'), row=1, col=1)
        
        cumulative_cluster = data_ml[['returns', 'strat_cluster']].cumsum().apply(np.exp)
        fig_clusters.add_trace(go.Scatter(x=cumulative_cluster.index, y=cumulative_cluster['returns'], name='Buy & Hold'), row=1, col=2)
        fig_clusters.add_trace(go.Scatter(x=cumulative_cluster.index, y=cumulative_cluster['strat_cluster'], name='K-Means'), row=1, col=2)
        
        fig_clusters.update_layout(height=500, title=f"K-Means {n_clusters} clusters - {profil['nom']}")
        st.plotly_chart(fig_clusters, use_container_width=True)
    
    # Footer avec recommandations
    st.markdown("---")
    st.markdown("""
    ## 🎯 **Recommandations finales**
    
    **Pour votre profil {profil['nom']} :**
    - ✅ **Stratégie SMA optimale** : SMA {SMA_SHORT_OPT}/{SMA_LONG_OPT} (robuste et simple)
    - 🔄 **Fréquence de trading** : {data_sma['Position'].diff().ne(0).sum()} signaux
    - ⚖️ **Meilleur Sharpe Ratio** trouvé dans l'optimisation
    - 💡 **Diversification** : Combiner SMA + K-Means pour réduire le risque
    
    > *⚠️ Les performances passées ne préjugent pas des performances futures. Considérer les frais de transaction.*
    """.format(profil=profil['nom'], SMA_SHORT_OPT=SMA_SHORT_OPT, SMA_LONG_OPT=SMA_LONG_OPT))


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
