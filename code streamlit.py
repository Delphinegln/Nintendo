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

IMG=Path.cwd()/"images"

# HRP
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

warnings.filterwarnings("ignore")

#CARTE---------------
st.markdown("""
<style>
.card-container {
    background: rgba(255, 255, 255, 0.6); /* Fond blanc semi-transparent */
    backdrop-filter: blur(10px); /* Flou d'arrière-plan */
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin: 10px 8px 10px 8px; /* Espacement horizontal léger, vertical réduit */
}
.row-cards {
    display: flex;
    gap: 20px; /* Espacement entre les cartes dans la ligne */
}
</style>
""", unsafe_allow_html=True)

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

# ========== SESSION STATE GLOBAL (UNE SEULE FOIS) ==========
if "show_peach_page" not in st.session_state:
    st.session_state["show_peach_page"] = False

# ========== SESSION STATE GLOBAL (UNE SEULE FOIS) ==========
if "show_luigi_page" not in st.session_state:
    st.session_state["show_luigi_page"] = False

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

# ========== CSS : CARTES PLUS TRANSPARENTES ==========
st.markdown("""
<style>
    .main { background-color: transparent; }

    .custom-card {
        background-color: rgba(255, 255, 255, 0.25) !important; 
        backdrop-filter: blur(15px) !important; 
        -webkit-backdrop-filter: blur(15px) !important; 
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

# ========== GRID LAYOUT : CARTES ==========
if not (st.session_state["show_daisy_page"] or st.session_state["show_peach_page"] or st.session_state["show_luigi_page"]):
    col1, col2 = st.columns(2)
    
    col1, col2 = st.columns(2)

    col1, col2 = st.columns(2)
    
st.markdown('<div class="row-cards">', unsafe_allow_html=True)

st.markdown('<div class="card-container">', unsafe_allow_html=True)
st.image(str(IMG / "Daisy.png"), width=70)
st.markdown("### Financial Forecasting")
st.markdown("Daisy fait fleurir vos profits ! 🌼💰")
st.markdown("Module de prévision des tendances financières.")
if st.button("🔍 Ouvrir le module Daisy", key="open_daisy"):
    st.session_state["show_daisy_page"] = True
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card-container">', unsafe_allow_html=True)
st.image(str(IMG / "Peach.png"), width=70)
st.markdown("### Portfolio Optimization")
st.markdown("Peach your assets! 🍑💼")
st.markdown("Optimisation du portefeuille.")
if st.button("🔍 Ouvrir le module Peach", key="open_peach"):
    st.session_state["show_peach_page"] = True
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card-container">', unsafe_allow_html=True)
st.image(str(IMG / "Birdo.png"), width=70)
st.markdown("### Algorithmic Trading")
st.markdown("Vos trades, pondus et gérés par Birdo 🥚📈")
st.markdown("Stratégies automatisées et backtesting.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card-container">', unsafe_allow_html=True)
st.image(str(IMG / "Bowser.png"), width=70)
st.markdown("### Option Pricing")
st.markdown("Ne vous brûlez pas seul : Bowser hedge vos positions 🐢💼")
st.markdown("Modélisation et valorisation des options.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card-container">', unsafe_allow_html=True)
st.image(str(IMG / "Luigi.png"), width=70)
st.markdown("### Risk management")
st.markdown("Ne laissez pas vos risques vous hanter : Luigi est là 👻💸")
st.markdown("Analyse des risques financiers.")
if st.button("🔍 Ouvrir le module Luigi", key="open_luigi"):
    st.session_state["show_luigi_page"] = True
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


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
