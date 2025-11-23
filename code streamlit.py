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

        if st.button("🔍 Ouvrir le module Peach", key="open_peach"):
            st.session_state["show_daisy_page"] = True
            st.rerun()

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
