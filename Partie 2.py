def peach_module():

    import warnings
    from dataclasses import dataclass
    from typing import List

    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import streamlit as st

    # HRP
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    try:
        import cvxpy as cp
        HAS_CVXPY = True
    except Exception:
        HAS_CVXPY = False

    warnings.filterwarnings("ignore")

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

    # ======================================
    #         INTERFACE STREAMLIT
    # ======================================

    st.header("🍑 Peach — Portfolio Optimization Module")

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

        except Exception as e:
            st.error(f"Erreur : {e}")
