import streamlit as st
import pandas as pd
import altair as alt

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

st.header('Pré-analyse visuelles données données des Iris TP1')  # On définit l'en-tête d'une section

# Afficher les premières lignes des données chargées data
#st.write(df.head())
    
st.subheader('Description des données')  # Sets a subheader for a subsection

# Show Dataset
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

# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color="species"
)

# Display chart
st.write(chart)

# Interactive design representation 
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive()

st.write(chart2)

# -------------------------
# Add interactive filters for user input

st.sidebar.header("Paramètres d'Entrée pour la Prédiction")

# Sliders for Sepal and Petal dimensions
sepal_length = st.sidebar.slider('Longueur du Sépale', min_value=float(df['sepal_length'].min()), 
                                 max_value=float(df['sepal_length'].max()), 
                                 value=5.0, step=0.1)

sepal_width = st.sidebar.slider('Largeur du Sépale', min_value=float(df['sepal_width'].min()), 
                                max_value=float(df['sepal_width'].max()), 
                                value=3.0, step=0.1)

petal_length = st.sidebar.slider('Longueur du Pétale', min_value=float(df['petal_length'].min()), 
                                 max_value=float(df['petal_length'].max()), 
                                 value=3.5, step=0.1)

petal_width = st.sidebar.slider('Largeur du Pétale', min_value=float(df['petal_width'].min()), 
                                max_value=float(df['petal_width'].max()), 
                                value=1.0, step=0.1)

# Create a new dataframe with the inputted values
input_data = pd.DataFrame({
    'sepal_length': [sepal_length],
    'sepal_width': [sepal_width],
    'petal_length': [petal_length],
    'petal_width': [petal_width]
})

# Display the user's input data
st.write("Données d'entrée de l'utilisateur :")
st.write(input_data)

# -------------------------
# Machine Learning Prediction
# For the prediction part, we will train a simple classifier (Logistic Regression) and predict the species.

# Preprocess data and train model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = df.drop('species', axis=1)
y = df['species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the input data
input_data_scaled = scaler.transform(input_data)
predicted_species = model.predict(input_data_scaled)

# Display the prediction result
st.write(f"Prédiction de l'espèce : {predicted_species[0]}")

# -------------------------

# About
if st.button("About App"):
    st.subheader("App d'exploration des données des Iris")
    st.text("Construite avec Streamlit")
    st.text("Thanks to the Streamlit Team for the amazing work")

if st.checkbox("By"):
    st.text("Stéphane C. K. Tékouabou")
    st.text("ctekouaboukoumetio@gmail.com")
