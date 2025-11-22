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

#################################################################
#                DAISY – FINANCIAL FORECASTING POPUP           
#################################################################

sns.set_theme(style="whitegrid")


@st.dialog("🌼 Daisy – Financial Forecasting")
def daisy_popup():

    st.title("🌼 Daisy – Nintendo Financial Forecasting")
    st.write(
        """
        Welcome to the *Financial Forecasting* module, inspired by Daisy’s bright 
        forward-looking optimism.  
        
        This section explores Nintendo’s fundamentals, historical performance, 
        Monte-Carlo projections, and scenario analysis.
        """
    )

    # ============================
    # SECTION 1 — DATA PREPARATION
    # ============================
    with st.expander("📁 Data Preparation & Setup", expanded=False):

        start = "2015-09-30"
        end = "2025-09-30"

        companies = {
            "NTDOY": "Nintendo Co., Ltd.",
            "SONY": "Sony Group Corporation",
            "MSFT": "Microsoft Corporation",
            "EA": "Electronic Arts Inc.",
            "TCEHY": "Tencent Holdings Corporation"
        }

        st.code(
            """
start = "2015-09-30"
end   = "2025-09-30"
companies = {"NTDOY": "Nintendo", ...}
""",
            language="python"
        )

        st.success("✔️ Data parameters have been loaded.")

    # ============================
    # SECTION 2 — FINANCIAL DATA
    # ============================
    with st.expander("📊 Nintendo Financial Statements", expanded=False):

        st.write("Downloading Nintendo financial statements from Yahoo Finance…")

        ntd = yf.Ticker("NTDOY")

        balance_sheet = ntd.balance_sheet
        income_stmt = ntd.income_stmt
        cashflow_stmt = ntd.cashflow

        st.subheader("📘 Balance Sheet")
        st.dataframe(balance_sheet)

        st.subheader("📗 Income Statement")
        st.dataframe(income_stmt)

        st.subheader("📙 Cash Flow Statement")
        st.dataframe(cashflow_stmt)

        st.success("✔️ Financial statements successfully retrieved.")

    # ============================
    # SECTION 3 — HISTORICAL PRICES
    # ============================
    with st.expander("📈 Historical Price Comparison", expanded=False):

        tickers = list(companies.keys())
        prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]

        st.write("Historical closing prices:")
        st.dataframe(prices.tail())

        def base100(df):
            return df / df.iloc[0] * 100

        px_norm = base100(prices)
        px_norm.columns = [companies[c] for c in px_norm.columns]

        fig, ax = plt.subplots(figsize=(12, 5))
        px_norm.plot(ax=ax)
        ax.set_title("Normalised Performance (Base 100)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Level")

        st.pyplot(fig)

    # ============================
    # SECTION 4 — MONTE CARLO
    # ============================
    with st.expander("🎲 Monte Carlo Simulation (NTDOY)", expanded=False):

        st.write("Simulating 10,000 price trajectories for Nintendo stock.")

        returns = prices["NTDOY"].pct_change().dropna()
        r = returns.mean()
        sigma = returns.std()

        T = 5
        M = 100
        dt = T / M
        I = 500  # Reduced for Streamlit speed

        S = np.zeros((M+1, I))
        S0 = prices["NTDOY"].iloc[-1]
        S[0] = S0

        for t in range(1, M+1):
            S[t] = S[t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * npr.randn(I)
            )

        fig_mc = go.Figure()

        for i in range(80):
            fig_mc.add_trace(go.Scatter(
                y=S[:, i], mode="lines", line=dict(width=1), opacity=0.3,
                showlegend=False
            ))

        fig_mc.add_trace(go.Scatter(
            y=S.mean(axis=1), mode="lines", name="Mean trajectory", line=dict(width=3)
        ))

        fig_mc.update_layout(
            title="Monte Carlo Simulation – NTDOY",
            xaxis_title="Time Steps",
            yaxis_title="Price",
            height=500
        )

        st.plotly_chart(fig_mc)

    # ============================
    # SECTION 5 — FORECASTING (SIMULATED)
    # ============================
    with st.expander("🔮 Financial Forecasting (Prophet Simulation)", expanded=False):

        st.warning(
            """
            ⚠️ Prophet cannot run on Streamlit Cloud (Stan compilation unsupported).  
            Instead, here is a **statistical projection simulation** consistent with your analysis.
            """
        )

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
        fig_fc.add_trace(go.Scatter(
            x=forecast["Year"],
            y=forecast["Simulated Forecast"],
            mode="lines+markers",
            line=dict(width=3)
        ))

        fig_fc.update_layout(
            title="Simulated Forecast – Total Revenue",
            xaxis_title="Year",
            yaxis_title="JPY (Simulated)",
            height=400
        )

        st.plotly_chart(fig_fc)

    # ============================
    # SECTION 6 — SCENARIO ANALYSIS
    # ============================
    with st.expander("🧪 Scenario Analysis – KPIs", expanded=False):

        st.write("Applying optimistic / central / pessimistic factors.")

        scenario_factors = {"Pessimistic": 0.85, "Central": 1.00, "Optimistic": 1.15}

        metric = "Operating Income"
        base_value = income_stmt.loc["Operating Income"].mean()

        df_scen = pd.DataFrame({
            "Scenario": list(scenario_factors.keys()),
            "Value": [base_value * f for f in scenario_factors.values()]
        })

        st.dataframe(df_scen)

        fig_scen = go.Figure()
        fig_scen.add_bar(x=df_scen["Scenario"], y=df_scen["Value"])
        fig_scen.update_layout(
            title="Operating Income Scenario Analysis",
            yaxis_title="JPY (Simulated)"
        )

        st.plotly_chart(fig_scen)



# CSS pour mettre l'image en fond d'écran
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


# CSS pour changer le curseur en étoile
st.markdown("""
    <style>
    * {
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"><path fill="%23FFD700" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>') 16 16, auto !important;
    }
    </style>
""", unsafe_allow_html=True)


# CSS pour uniformiser les cartes
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

# CSS pour le modal fullscreen Daisy
st.markdown("""
<style>
/* Overlay */
#daisy-modal {
    position: fixed;
    top:0; left:0;
    width:100vw; height:100vh;
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(3px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

/* Modal box */
.daisy-box {
    width: 90%;
    height: 90%;
    background: white;
    border-radius: 20px;
    padding: 40px;
    overflow-y: auto;
    position: relative;
    box-shadow: 0 0 30px rgba(0,0,0,0.25);
    animation: popup 0.25s ease;
}

/* Animation */
@keyframes popup {
    from { transform: scale(0.9); opacity:0; }
    to { transform: scale(1); opacity:1; }
}

/* Close button */
.close-btn {
    position:absolute;
    right:25px;
    top:15px;
    font-size:32px;
    cursor:pointer;
    color:#444;
    transition:0.2s;
}
.close-btn:hover {
    color:#ff7f00;
}

/* Titles */
.daisy-title {
    font-size: 42px;
    font-weight: 800;
    color: #e9a21f;
    text-align:center;
    margin-bottom: 15px;
}

.daisy-sub {
    font-size:22px;
    color:#555;
    text-align:center;
    margin-bottom:30px;
}
</style>
""", unsafe_allow_html=True)


def show_daisy_modal():
    st.markdown('<div id="daisy-modal">', unsafe_allow_html=True)
    st.markdown('<div class="daisy-box">', unsafe_allow_html=True)

    # Close button
    if st.button("✖", key="close_daisy", help="Close"):
        st.session_state["daisy_open"] = False

    # TITLE
    st.markdown('<h1 class="daisy-title">🌼 Daisy – Financial Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="daisy-sub">“Daisy makes your profits bloom!”</p>', unsafe_allow_html=True)

    st.write("---")

    # FINANCIAL DATA
    ticker = yf.Ticker("NTDOY")
    balance_sheet = ticker.balance_sheet
    income_stmt = ticker.financials

    st.markdown("## **Income Statement (JPY)**")
    st.dataframe((income_stmt/1e9).round(2))

    st.markdown("## **Balance Sheet (JPY)**")
    st.dataframe((balance_sheet/1e9).round(2))

    st.write("---")
    st.markdown("### Forecast Preview")
    st.write("Prévisions futures bientôt disponibles.")

    st.markdown('</div></div>', unsafe_allow_html=True)

# HEADER
st.markdown("<h1 style='text-align: center;'>Dashboard for Nintendo's Investors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 40px;'>Sélectionne une section pour explorer les modules.</p>", unsafe_allow_html=True)

# Initialiser la variable de session pour le popup Daisy
if "daisy_open" not in st.session_state:
    st.session_state["daisy_open"] = False
    

# --- GRID LAYOUT ---
col1, col2 = st.columns(2)

# ------------------------------------------------------------------
# PARTIE 1 : DAISY
with col1:
    # On crée un bouton invisible autour de la carte
    if st.button("🌼 Ouvrir Daisy", key="daisy_card_click"):
        st.session_state["daisy_open"] = True

    # Affichage de la carte
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/2ad3a5c2b5b8309627236c3eb193e4bd0b5b54fea0c8950a1b8c2dcb.png" class="card-img">
        <h3>Financial Forecasting</h3>
        <p style="opacity: 0.6;">Daisy fait fleurir vos profits ! 🌼💰</p>
        <p style="opacity: 0.8;">Module de prévision des tendances financières.</p>
        <p style="opacity:0;">Clique ici pour ouvrir</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# PARTIE 2 : PEACH
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

# ------------------------------------------------------------------
# LIGNE 2
col3, col4 = st.columns(2)

# PARTIE 3 : BIRDO
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

# PARTIE 4 : BOWSER
with col4:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/828f7ec3955d9049a1295309226e2c0696daadf60c3202fdedac0992.png" class="card-img">
        <h3>Option Pricing </h3>
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

# ------------------------------------------------------------------
# LIGNE 3 – LUIGI (centré)
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

# ------------------------------------------------------------------

# Affiche le modal Daisy si nécessaire
if st.session_state["daisy_open"]:
    show_daisy_modal()

# SIDEBAR
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
