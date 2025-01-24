import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('Iris Classification')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Exploration des données des Iris')

st.header('Pré-analyse visuelles données des Iris TP1')  # On définit l'en-tête d'une section

# Afficher les premières lignes des données chargées
st.subheader('Description des données')  # Sets a subheader for a subsection
if st.checkbox("Boutons de prévisualisation du DataFrame"):
    if st.button("Head"):
        st.write(df.head(2))
    if st.button("Tail"):
        st.write(df.tail())
    if st.button("Infos"):
        st.write(df.info())
    if st.button("Shape"):
        st.write(df.shape)
else:
    st.write(df.head(2))

# -------------------------
# Préparation des données pour le modèle de prédiction

# Séparation des caractéristiques (features) et de la cible (target)
X = df.drop('species', axis=1)
y = df['species']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split des données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entraînement du modèle (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Filtres interactifs pour la prédiction

st.header("Prédiction d'espèce d'Iris")

# Utilisation de widgets Streamlit pour modifier les paramètres d'entrée
sepal_length = st.slider('Longueur du sépale (cm)', 4.0, 8.0, 5.0, 0.1)
sepal_width = st.slider('Largeur du sépale (cm)', 2.0, 5.0, 3.0, 0.1)
petal_length = st.slider('Longueur du pétale (cm)', 1.0, 7.0, 4.0, 0.1)
petal_width = st.slider('Largeur du pétale (cm)', 0.1, 3.0, 1.0, 0.1)

# Prédiction en fonction des valeurs saisies
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Afficher la prédiction
st.write(f"L'espèce prédite est : {prediction}")

# Affichage de graphiques
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color="species"
)

# Display chart
st.write(chart)

# Interactive chart
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive()

st.write(chart2)
