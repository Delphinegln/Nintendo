import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Nintendo Algorithmic Trading",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.markdown("""
# 🎮 CONSEIL EN TRADING ALGORITHMIQUE - NINTENDO (NTDOY)
**Période: Septembre 2015 - Septembre 2025**

*Stratégies optimisées selon votre profil d'investisseur*
""")

# Sidebar pour la configuration
st.sidebar.header("⚙️ Configuration")
profil_choisi = st.sidebar.selectbox(
    "Choisissez votre profil d'investisseur:",
    options={
        1: "CONSERVATEUR - Faible risque",
        2: "MODÉRÉ - Équilibre risque/rendement", 
        3: "DYNAMIQUE - Maximisation rendement",
        4: "PERSONNALISÉ"
    },
    index=1,
    format_func=lambda x: list(x.values())[0]
)

# Profils prédéfinis
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

if profil_choisi == 4:
    # Paramètres personnalisés
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sma_short_deb = st.number_input("SMA Court début", min_value=5, max_value=100, value=10)
        sma_short_fin = st.number_input("SMA Court fin", min_value=20, max_value=150, value=50)
        sma_short_pas = st.number_input("SMA Court pas", min_value=1, max_value=20, value=5)
    with col2:
        sma_long_deb = st.number_input("SMA Long début", min_value=50, max_value=200, value=100)
        sma_long_fin = st.number_input("SMA Long fin", min_value=100, max_value=400, value=200)
        sma_long_pas = st.number_input("SMA Long pas", min_value=1, max_value=50, value=10)
    
    profil = {
        'nom': 'PERSONNALISÉ',
        'sma_short_range': range(sma_short_deb, sma_short_fin+1, sma_short_pas),
        'sma_long_range': range(sma_long_deb, sma_long_fin+1, sma_long_pas),
        'n_lags_regression': st.sidebar.number_input("Lags régression", min_value=1, max_value=20, value=5),
        'n_clusters_kmeans': st.sidebar.number_input("Clusters K-Means", min_value=2, max_value=8, value=4)
    }
else:
    profil_key = [k for k, v in profils.items() if v['nom'] == profil_choisi.split(' - ')[0]][0]
    profil = profils[profil_key]

st.sidebar.success(f"✅ **Profil sélectionné**: {profil['nom']}")

# Téléchargement des données
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

# Onglet principal : Optimisation SMA
tab1, tab2, tab3, tab4 = st.tabs(["📊 SMA Optimisée", "⚙️ Backtesting", "📈 Régression OLS", "🤖 K-Means ML"])

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
