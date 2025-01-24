# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import altair as alt

# Configuration de la page
st.set_page_config(
    page_title="Classification des Iris", 
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Barre latérale

# Initialiser page_selection dans l'état de session si pas déjà défini
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Page par défaut

# Fonction pour mettre à jour page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('🌼 Classification des Iris')
    
    # Navigation par boutons avec des icônes
    st.subheader("📂 Navigation")
    
    if st.button("🏷 À Propos", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("📊 Jeu de Données", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("🔍 Analyse Exploratoire", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("🧹 Nettoyage / Prétraitement", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("🤖 Apprentissage Automatique", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("🔮 Prédiction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("📜 Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Détails du projet
    st.subheader("Résumé")
    st.markdown("""
        Un tableau de bord interactif pour explorer et classifier les données des fleurs Iris.
        
        - 📊 [Jeu de Données](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
        - 📗 [Notebook Google Colab](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)
        - 🐙 [Dépôt GitHub](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)
        
        *Auteur :* [Zeraphim](https://jcdiamante.com)
    """)

# -------------------------

# Charger les données
try:
    df = pd.read_csv('iris.csv', delimiter=',')
except FileNotFoundError:
    st.error("Le fichier 'iris.csv' est introuvable. Veuillez vérifier son emplacement.")
    st.stop()

# Page principale
if st.session_state.page_selection == 'about':
    # Page À Propos
    st.title("🏷 À Propos")
    st.markdown("""
        Cette application explore le célèbre jeu de données *Iris* et propose :
        
        - Une exploration visuelle des données.
        - Un prétraitement et nettoyage des données.
        - La construction et l'évaluation de modèles d'apprentissage automatique.
        - Une interface interactive pour prédire l'espèce d'une fleur Iris.
        
        *Technologies utilisées :*
        - Python (Streamlit, Altair, Pandas)
        - Machine Learning (Scikit-learn)
        
        *Auteur : Stéphane C. K. Tékouabou*
        
        ✉ Contact : ctekouaboukoumetio@gmail.com
    """)

elif st.session_state.page_selection == 'dataset':
    # Page Jeu de Données
    st.title("📊 Jeu de Données")
    
    # Afficher les premières lignes du DataFrame
    if st.checkbox("Afficher le DataFrame"):
        nb_rows = st.slider("Nombre de lignes à afficher :", min_value=5, max_value=50, value=10)
        st.write(df.head(nb_rows))
    
    # Afficher les statistiques descriptives
    if st.checkbox("Afficher les statistiques descriptives"):
        st.write(df.describe())

elif st.session_state.page_selection == 'eda':
    # Page Analyse Exploratoire
    st.title("🔍 Analyse Exploratoire")
    
    # Graphique interactif : Longueur vs Largeur du pétale
    chart1 = alt.Chart(df).mark_circle(size=60).encode(
        x='petal_length',
        y='petal_width',
        color='species',
        tooltip=['petal_length', 'petal_width', 'species']
    ).interactive()
    
    # Graphique interactif : Longueur vs Largeur du sépale
    chart2 = alt.Chart(df).mark_circle(size=60).encode(
        x='sepal_length',
        y='sepal_width',
        color='species',
        tooltip=['sepal_length', 'sepal_width', 'species']
    ).interactive()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.altair_chart(chart1, use_container_width=True)
    
    with col2:
        st.altair_chart(chart2, use_container_width=True)

elif st.session_state.page_selection == 'prediction':
    # Page Prédiction
    from sklearn.neighbors import KNeighborsClassifier
       
    # Formulaire pour saisir les caractéristiques
    sepal_length = st.number_input("Longueur du sépale (cm)", min_value=0.0)
    sepal_width = st.number_input("Largeur du sépale (cm)", min_value=0.0)
    petal_length = st.number_input("Longueur du pétale (cm)", min_value=0.0)
    petal_width = st.number_input("Largeur du pétale (cm)", min_value=0.0)
    
    if st.button("Prédire"):
        try:
            knn_model = KNeighborsClassifier(n_neighbors=3)  # Exemple d'un modèle simple KNN
            X = df.drop('species', axis=1)
            y = df['species']
            knn_model.fit(X, y)  # Entraîner sur tout le jeu de données
            
            prediction = knn_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
            species_predicted = prediction[0]
            st.success(f"L'espèce prédite est : *{species_predicted}*")
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
