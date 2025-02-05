import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler  # Important pour la cohérence

# Chargement du modèle
with open('bank_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Chargement du scaler (s'il a été utilisé lors de l'entraînement)
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    scaler = None  # Gérer le cas où le scaler n'a pas été sauvegardé

# Titre et description de l'application
st.title('Prédiction de souscription bancaire')
st.write('Entrez les informations du client pour prédire s\'il va souscrire à un dépôt à terme.')

# Formulaire de saisie des informations du client
# (Adapter ces champs aux features de votre modèle)
age = st.number_input('Age', min_value=18, value=30)
job = st.selectbox('Profession', ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'self-employed', 'housewife', 'entrepreneur', 'unemployed', 'student'])
marital = st.selectbox('Situation Maritale', ['married', 'single', 'divorced'])
education = st.selectbox('Niveau d\'éducation', ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox('Défaut de paiement', ['yes', 'no'])
housing = st.selectbox('Prêt immobilier', ['yes', 'no'])
loan = st.selectbox('Prêt personnel', ['yes', 'no'])
contact = st.selectbox('Type de contact', ['cellular', 'telephone'])
month = st.selectbox('Mois du dernier contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Jour de la semaine du dernier contact', ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input('Durée du dernier contact (secondes)', min_value=0, value=100)
campaign = st.number_input('Nombre de contacts pendant cette campagne', min_value=0, value=1)
pdays = st.number_input('Nombre de jours depuis le dernier contact de la campagne précédente', value=-1)  # -1 signifie inconnu
previous = st.number_input('Nombre de contacts avant cette campagne', min_value=0, value=0)
poutcome = st.selectbox('Résultat de la campagne précédente', ['unknown', 'other', 'failure', 'success'])
emp_var_rate = st.number_input('Taux de variation de l\'emploi', value=0.0)
cons_price_idx = st.number_input('Indice des prix à la consommation', value=90.0)
cons_conf_idx = st.number_input('Indice de confiance des consommateurs', value=-40.0)
euribor3m = st.number_input('Taux Euribor à 3 mois', value=0.0)
nr_employed = st.number_input('Nombre d\'employés', value=5000.0)

# Préparation des données pour la prédiction
data = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome,
    'emp_var_rate': emp_var_rate,
    'cons_price_idx': cons_price_idx,
    'cons_conf_idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr_employed': nr_employed
}

df = pd.DataFrame([data])

# Encodage des variables catégorielles (important pour la cohérence avec l'entraînement)
# ... (Utilisez la même méthode d'encodage que lors de l'entraînement de votre modèle)
# Exemple avec OneHotEncoder (si utilisé) :
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # adapter selon votre entraînement
# df_encoded = encoder.fit_transform(df[colonnes_cat])  # colonnes_cat = liste des colonnes catégorielles
# df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out())
# df = df.drop(colonnes_cat, axis=1)
# df = pd.concat([df, df_encoded], axis=1)


# Scaling des variables numériques (si utilisé lors de l'entraînement)
if scaler:
    # Identifier les colonnes numériques (à adapter à votre modèle)
    numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

# Prédiction
if st.button('Prédire'):
    prediction = model.predict(df)
    st.write(f'Prédiction : {prediction[0]}')  # Afficher le résultat
