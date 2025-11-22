import streamlit as st
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

# HEADER
st.markdown("<h1 style='text-align: center;'>Dashboard for Nintendo's Investors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 40px;'>Sélectionne une section pour explorer les modules.</p>", unsafe_allow_html=True)

# --- GRID LAYOUT ---
col1, col2 = st.columns(2)

# ------------------------------------------------------------------
# PARTIE 1 : DAISY
with col1:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/2ad3a5c2b5b8309627236c3eb193e4bd0b5b54fea0c8950a1b8c2dcb.png" class="card-img">
        <h3>Financial Forecasting</h3>
        <p style="opacity: 0.6;">Daisy fait fleurir vos profits ! 🌼💰</p>
        <p style="opacity: 0.8;">Module de prévision des tendances financières.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Voir les détails et intégrer le code"):
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-text">Section à compléter par Daisy</div>
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
