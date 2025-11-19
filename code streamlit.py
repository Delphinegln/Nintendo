import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Data Science - Mario Project",
    page_icon="images/Daisy.png",  # Icône PNG
    layout="wide"
)

# CSS pour uniformiser les cartes
st.markdown("""
<style>
    .main { background-color: #FCF9F3; }

    .custom-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 3px solid rgba(0,0,0,0.1);
        text-align: center;
    }

    .card-img {
        width: 90px;
        margin-bottom: 10px;
    }

    .placeholder-box {
        background-color: rgba(94, 82, 64, 0.05);
        border: 2px dashed rgba(94, 82, 64, 0.3);
        border-radius: 8px;
        padding: 30px;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<h1 style='text-align: center;'>Dashboard Data Science – Projet Mario Group</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 40px;'>Sélectionne une section pour explorer les modules.</p>", unsafe_allow_html=True)

# --- GRID LAYOUT ---
col1, col2 = st.columns(2)

# ------------------------------------------------------------------
# PARTIE 1 : DAISY
with col1:
    st.markdown("""
    <div class="custom-card">
        <img src="https://nintendo-jx9pmih3bmjrbdhfzb8xd5.streamlit.app/~/+/media/2ad3a5c2b5b8309627236c3eb193e4bd0b5b54fea0c8950a1b8c2dcb.png" class="card-img">
        <h3>Partie 1 :  Daisy fait fleurir vos profits ! 🌼💰</h3>
        <p style="opacity: 0.6;">Daisy</p>
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
        <h3>Partie 2 : Peach your assets! 🍑💼</h3>
        <p style="opacity: 0.6;">Peach</p>
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
        <h3>Partie 3 : Trading algorithmique</h3>
        <p style="opacity: 0.6;">Birdo</p>
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
        <h3>Partie 4 : Pricing d'options</h3>
        <p style="opacity: 0.6;">Bowser</p>
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
        <h3>Partie 5 : Gestion des risques</h3>
        <p style="opacity: 0.6;">Luigi</p>
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
