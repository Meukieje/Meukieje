import streamlit as st
import pandas as pd
import pickle
# Charger le modèle
model = pickle.load(open("model.pkl", "rb"))
st.title("Prédiction de souscription à un dépôt à terme")
# Champs d'entrée utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
job = st.selectbox("Profession", ["admin.", "blue-collar", "technician", "services", "management", "entrepreneur", "retired", "self-employed", "student", "unemployed", "housemaid"])
marital = st.selectbox("Statut matrimonial", ["married", "single", "divorced"])
education = st.selectbox("Éducation", ["basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree", "illiterate"])
default = st.selectbox("Crédit en défaut", ["no", "yes"])
balance = st.number_input("Solde moyen", min_value=-10000, max_value=100000, value=500)
housing = st.selectbox("Crédit immobilier", ["no", "yes"])
loan = st.selectbox("Crédit personnel", ["no", "yes"])
# Encodage des entrées utilisateur
user_data = pd.DataFrame([[age, job, marital, education, default, balance, housing, loan]], columns=["age", "job", "marital", "education", "default", "balance", "housing", "loan"])
user_data = pd.get_dummies(user_data)
# Ajout des colonnes manquantes (même structure que les données d'entraînement)
model_columns = pickle.load(open("model_columns.pkl", "rb"))
for col in model_columns:
    if col not in user_data:
        user_data[col] = 0
# Prédiction
prediction = model.predict(user_data)
st.write("Résultat : *Oui" if prediction[0] == 1 else "Résultat : **Non*")
