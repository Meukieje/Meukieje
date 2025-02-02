import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("bank_model.pkl")

# Interface utilisateur
st.title("Prédiction d'abonnement bancaire")
st.write("Entrez les informations du client pour prédire s'il souscrira à un dépôt.")

# Entrée utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Statut marital", ["divorced", "married", "single", "unknown"])
education = st.selectbox("Éducation", ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"])
