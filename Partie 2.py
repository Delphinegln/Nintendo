
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# HRP : clustering hiérarchique
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# cvxpy optionnel (QP). Fallback prévu si absent.
try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

warnings.filterwarnings("ignore")

# Configuration Streamlit
st.set_page_config(
    page_title="Nintendo Portfolio Analyzer",
    page_icon="🎮",
    layout="wide"
)

# -------------------------------------
# Config de base
# -------------------------------------
NINTENDO = "NTDOY"  # Nintendo (ADR US)
DEFAULT_PEERS = ["EA", "TTWO", "SONY", "MSFT",
                 "7832.T", "9697.T", "9684.T", "9766.T",
                 "UBI.PA", "TCEHY"]
START, END = "2015-09-30", "2025-09-30"

TICKER_NAME = {
    "NTDOY": "Nintendo (ADR)",
    "7974.T": "Nintendo (Tokyo)",
    "EA": "Electronic Arts",
    "TTWO": "Take-Two Interactive",
    "SONY": "Sony Group (ADR)",
    "MSFT": "Microsoft",
    "7832.T": "Bandai Namco",
    "9697.T": "Capcom",
    "9684.T": "Square Enix",
    "9766.T": "Konami",
    "UBI.PA": "Ubisoft",
    "TCEHY": "Tencent (ADR)",
}

ORDER = "asc"  # tri d'affichage


# -------------------------------------
# Contraintes & optimisation
# -------------------------------------
@dataclass
class Constraints:
    min_center_weight: float = 0.10
    max_center_weight: float = 0.80
    max_weight_per_name: float = 0.25


cons = Constraints()


# -------------------------------------
# Fonctions utilitaires
# -------------------------------------
@st.cache_data(ttl=3600)
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Télécharge les prix depuis Yahoo Finance avec cache"""
    data = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.ffill().dropna()


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcule les rendements journaliers"""
    return prices.pct_change().dropna()


def ann_perf(r: pd.Series):
    """Retourne (rendement annuel, vol annuelle, Sharpe) sur une série de rendements quotidiens."""
    ann_ret = (1 + r).prod()**(252/len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    return ann_ret, ann_vol, sharpe


def evaluate_portfolio(weights: pd.Series, returns: pd.DataFrame):
    """Évalue les performances d'un portefeuille"""
    weights = weights / weights.sum()
    common = [t for t in weights.index if t in returns.columns]
    port_rets = (returns[common] * weights[common]).sum(axis=1)
    ann_ret, ann_vol, sharpe = ann_perf(port_rets)
    growth = (1 + port_rets).cumprod()
    return ann_ret, ann_vol, sharpe, port_rets, growth


def herfindahl(w: pd.Series) -> float:
    """Indice de concentration simple (HHI)."""
    w = w / w.sum()
    return float((w**2).sum())


# -------------------------------------
# Optimisation MV (M4)
# -------------------------------------
def optimize_mv_centered(mu: pd.Series, cov: pd.DataFrame,
                         tickers: List[str],
                         center: str,
                         cons: Constraints,
                         target_center_weight: float) -> pd.Series:
    """
    Optimisation moyenne-variance avec poids FIXE sur Nintendo.
    (Modèle M4)
    """
    n = len(tickers)
    if not HAS_CVXPY:
        # Fallback simple : poids Nintendo fixé, reste en équipondéré
        weights = pd.Series(0.0, index=tickers)
        weights[center] = target_center_weight
        others = [t for t in tickers if t != center]
        rest = 1.0 - weights.sum()
        if rest < 0:
            raise RuntimeError("Poids Nintendo trop élevé pour la heuristique.")
        if others:
            weights[others] = rest / len(others)
        return weights

    w = cp.Variable(n)
    idx_center = tickers.index(center)

    Sigma = cov.loc[tickers, tickers].values
    Sigma = 0.5 * (Sigma + Sigma.T)
    eps = 1e-6 * np.mean(np.diag(Sigma))
    np.fill_diagonal(Sigma, np.diag(Sigma) + eps)
    gamma = 10.0 / max(np.trace(Sigma), 1e-8)

    constraints = [cp.sum(w) == 1.0, w >= 0]
    for i in range(n):
        if i != idx_center:
            constraints.append(w[i] <= cons.max_weight_per_name)
    constraints.append(w[idx_center] == target_center_weight)

    objective = cp.Maximize(mu.loc[tickers].values @ w
                            - 0.5 * gamma * cp.quad_form(w, Sigma))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimisation MV centrée Nintendo impossible.")

    wv = np.array(w.value).ravel()
    wv = wv / wv.sum()
    return pd.Series(wv, index=tickers)


# -------------------------------------
# Modèle HRP
# -------------------------------------
def _correl_dist(corr: pd.DataFrame) -> np.ndarray:
    """Distance de corrélation pour HRP."""
    return np.sqrt(0.5 * (1 - corr))


def _get_cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
    """Variance d'un cluster"""
    sub = cov.loc[items, items]
    w = np.ones(len(sub)) / len(sub)
    return float(w @ sub.values @ w)


@st.cache_data
def build_hrp_weights(returns_df: pd.DataFrame) -> pd.Series:
    """
    Implémentation HRP (Hierarchical Risk Parity)
    """
    corr = returns_df.corr()
    cov = returns_df.cov()

    # Distance de corrélation -> matrice condensée pour linkage
    dist = _correl_dist(corr)
    dist_cond = squareform(dist.values, checks=False)

    # Clustering hiérarchique
    link = linkage(dist_cond, method="single")
    sort_idx = leaves_list(link)
    ordered_tickers = corr.index[sort_idx].tolist()

    # Bisection récursive
    weights = pd.Series(1.0, index=ordered_tickers)
    clusters = [ordered_tickers]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        c1 = cluster[:split]
        c2 = cluster[split:]

        var1 = _get_cluster_var(cov, c1)
        var2 = _get_cluster_var(cov, c2)
        # allocation inversement proportionnelle au risque
        alloc2 = var1 / (var1 + var2)
        alloc1 = 1.0 - alloc2

        weights[c1] *= alloc1
        weights[c2] *= alloc2

        clusters.append(c1)
        clusters.append(c2)

    # Remettre dans l'ordre original des colonnes
    weights = weights.reindex(returns_df.columns)
    weights = weights / weights.sum()
    return weights


# ==========================================
# APPLICATION STREAMLIT
# ==========================================

def main():
    st.title("🎮 Analyse de Portefeuille Nintendo")
    st.markdown("""
    Cette application permet d'optimiser un portefeuille centré sur Nintendo en utilisant :
    - **M4** : Optimisation Moyenne-Variance avec poids fixe sur Nintendo
    - **HRP** : Hierarchical Risk Parity (allocation alternative neutre en risque)
    """)

    # Sidebar - Paramètres
    st.sidebar.header("⚙️ Configuration du Portefeuille")
    
    # Chargement des données
    with st.spinner("📊 Chargement des données de marché..."):
        UNIVERSE = [NINTENDO] + DEFAULT_PEERS
        PRICES = download_prices(UNIVERSE, START, END)
        RETURNS = pct_returns(PRICES)
        
        TICKERS = list(RETURNS.columns)
        if NINTENDO in TICKERS:
            CENTER = NINTENDO
        elif "7974.T" in TICKERS:
            CENTER = "7974.T"
        else:
            CENTER = TICKERS[0]
        
        MU_ANN = RETURNS.mean() * 252
        COV_ANN = RETURNS.cov() * 252
        
        # Calcul HRP
        HRP_WEIGHTS = build_hrp_weights(RETURNS)
        HRP_WEIGHTS = HRP_WEIGHTS / HRP_WEIGHTS.sum()

    st.sidebar.success("✅ Données chargées avec succès")
    
    # Paramètres utilisateur
    st.sidebar.subheader("📈 Objectifs d'investissement")
    
    target_return = st.sidebar.slider(
        "Rendement annuel visé (%)",
        min_value=0.0,
        max_value=30.0,
        value=6.0,
        step=0.5,
        format="%.1f%%"
    ) / 100
    
    horizon_years = st.sidebar.slider(
        "Horizon d'investissement (années)",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )
    
    nintendo_weight = st.sidebar.slider(
        "Allocation à Nintendo (%)",
        min_value=int(cons.min_center_weight * 100),
        max_value=int(cons.max_center_weight * 100),
        value=30,
        step=1,
        format="%d%%"
    ) / 100

    # Bouton de calcul
    if st.sidebar.button("🚀 Calculer le portefeuille", type="primary"):
        
        with st.spinner("🔄 Optimisation en cours..."):
            # Optimisation M4
            try:
                weights_m4 = optimize_mv_centered(
                    MU_ANN,
                    COV_ANN,
                    TICKERS,
                    CENTER,
                    cons,
                    target_center_weight=nintendo_weight
                )
                
                ann_ret, ann_vol, sharpe, port_rets, growth_port = evaluate_portfolio(
                    weights_m4, RETURNS
                )
                
                # Résultats HRP
                hrp_weights_full = HRP_WEIGHTS.reindex(TICKERS).fillna(0)
                hrp_ret, hrp_vol, hrp_sharpe, _, hrp_growth = evaluate_portfolio(
                    hrp_weights_full, RETURNS
                )
                
                # Affichage des résultats
                st.success("✅ Optimisation terminée avec succès !")
                
                # Métriques principales
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Rendement annuel (M4)",
                        f"{ann_ret:.2%}",
                        delta=f"{(ann_ret - hrp_ret):.2%} vs HRP"
                    )
                
                with col2:
                    st.metric(
                        "Volatilité annuelle (M4)",
                        f"{ann_vol:.2%}",
                        delta=f"{(ann_vol - hrp_vol):.2%} vs HRP",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Ratio de Sharpe (M4)",
                        f"{sharpe:.2f}",
                        delta=f"{(sharpe - hrp_sharpe):.2f} vs HRP"
                    )
                
                # Onglets pour les différentes vues
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Allocation M4",
                    "🎯 Allocation HRP",
                    "📈 Comparaison",
                    "📉 Performance historique"
                ])
                
                # TAB 1 : Allocation M4
                with tab1:
                    st.subheader(f"Allocation du portefeuille M4 (Nintendo fixé à {nintendo_weight:.1%})")
                    
                    alloc_df = pd.DataFrame({"Poids": weights_m4})
                    alloc_df["Nom"] = alloc_df.index.map(lambda t: TICKER_NAME.get(t, t))
                    alloc_df = alloc_df.sort_values("Poids", ascending=False)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(
                            alloc_df[["Nom", "Poids"]].style.format({"Poids": "{:.2%}"}),
                            use_container_width=True
                        )
                        
                        hhi_m4 = herfindahl(weights_m4)
                        st.info(f"**Indice de concentration (HHI)** : {hhi_m4:.3f}")
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        alloc_plot = alloc_df.sort_values("Poids", ascending=True)
                        ax.barh(alloc_plot["Nom"], alloc_plot["Poids"], edgecolor="black")
                        ax.set_xlabel("Poids (fraction du portefeuille)")
                        ax.set_title("Allocation du portefeuille M4")
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # TAB 2 : Allocation HRP
                with tab2:
                    st.subheader("Allocation recommandée par HRP (Hierarchical Risk Parity)")
                    
                    hrp_df = pd.DataFrame({"Poids": hrp_weights_full})
                    hrp_df["Nom"] = hrp_df.index.map(lambda t: TICKER_NAME.get(t, t))
                    hrp_df = hrp_df.sort_values("Poids", ascending=False)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(
                            hrp_df[["Nom", "Poids"]].style.format({"Poids": "{:.2%}"}),
                            use_container_width=True
                        )
                        
                        hhi_hrp = herfindahl(hrp_weights_full)
                        st.info(f"**Indice de concentration (HHI)** : {hhi_hrp:.3f}")
                        
                        hrp_nintendo = hrp_weights_full.get(CENTER, 0.0)
                        st.warning(f"**Poids Nintendo (HRP)** : {hrp_nintendo:.1%}")
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        hrp_plot = hrp_df.sort_values("Poids", ascending=True)
                        ax.barh(hrp_plot["Nom"], hrp_plot["Poids"], 
                               edgecolor="black", color="orange", alpha=0.7)
                        ax.set_xlabel("Poids (fraction du portefeuille)")
                        ax.set_title("Allocation HRP (référence neutre en risque)")
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # TAB 3 : Comparaison
                with tab3:
                    st.subheader("📌 Comparaison M4 vs HRP")
                    
                    comparison_df = pd.DataFrame({
                        "Modèle": ["M4 (ton portefeuille)", "HRP (référence)"],
                        "Rendement annuel": [f"{ann_ret:.2%}", f"{hrp_ret:.2%}"],
                        "Volatilité annuelle": [f"{ann_vol:.2%}", f"{hrp_vol:.2%}"],
                        "Ratio de Sharpe": [f"{sharpe:.2f}", f"{hrp_sharpe:.2f}"],
                        "HHI (concentration)": [f"{herfindahl(weights_m4):.3f}", 
                                               f"{herfindahl(hrp_weights_full):.3f}"]
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Analyse comparative
                    st.markdown("### 💡 Analyse comparative")
                    
                    if hrp_sharpe > sharpe:
                        st.info("""
                        **HRP offre un meilleur couple rendement/risque** (Sharpe plus élevé).
                        
                        Il sacrifie éventuellement un peu de performance brute pour une meilleure 
                        efficacité ajustée du risque.
                        """)
                    elif (hrp_vol < ann_vol) and (hrp_ret < ann_ret):
                        st.info("""
                        **HRP propose un rendement plus faible, mais aussi une volatilité plus basse.**
                        
                        C'est une allocation plus prudente et plus diversifiée, qui accepte de renoncer 
                        à un peu de performance pour limiter les à-coups et la concentration du risque.
                        """)
                    else:
                        st.info("""
                        **HRP sert surtout de point de repère 'neutre' en risque.**
                        
                        Ton portefeuille M4 reflète davantage ta préférence pour Nintendo et le 
                        profil de rendement que tu vises.
                        """)
                    
                    if herfindahl(hrp_weights_full) < herfindahl(weights_m4):
                        st.success("""
                        ✅ **HRP est moins concentré** : le risque est mieux réparti entre les titres.
                        """)
                
                # TAB 4 : Performance historique
                with tab4:
                    st.subheader("📉 Évolution de la valeur du portefeuille")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(growth_port.index, growth_port.values, 
                           label="M4 (ton portefeuille)", linewidth=2)
                    ax.plot(hrp_growth.index, hrp_growth.values, 
                           label="HRP (référence)", linewidth=2, alpha=0.7)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Valeur du portefeuille (base 1)")
                    ax.set_title("Performance historique comparative")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info(f"""
                    **Horizon d'investissement déclaré** : {horizon_years} ans
                    
                    Ce graphique montre comment les deux stratégies auraient évolué historiquement 
                    sur la période {START} à {END}.
                    """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'optimisation : {str(e)}")
                st.exception(e)

    # Informations supplémentaires
    with st.expander("ℹ️ À propos des modèles"):
        st.markdown("""
        ### Modèle M4 (Mean-Variance)
        
        Optimisation classique de Markowitz avec une **contrainte de poids fixe** sur Nintendo.
        - Tu choisis le poids de Nintendo
        - L'algorithme optimise le reste du portefeuille pour maximiser le couple rendement/risque
        - Convient si tu as une forte conviction sur Nintendo
        
        ### Modèle HRP (Hierarchical Risk Parity)
        
        Approche alternative basée sur la **parité de risque hiérarchique** (Lopez de Prado).
        - Allocation purement neutre en risque
        - Utilise le clustering hiérarchique des corrélations
        - Meilleure diversification et stabilité
        - Ne reflète pas de convictions particulières
        
        ### Comment les interpréter ?
        
        - **M4** = ton portefeuille personnalisé avec ta conviction Nintendo
        - **HRP** = point de référence "neutre" pour comparer
        - Si HRP a un meilleur Sharpe → tu paies un coût pour ta conviction Nintendo
        - Si M4 a un meilleur Sharpe → ta conviction Nintendo est justifiée historiquement
        """)


if __name__ == "__main__":
    main()
