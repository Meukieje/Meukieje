import streamlit as st
import numpy as np
import pandas as pd

# Fonction pour afficher le tableau du simplexe
def afficher_tableau(tableau, iteration):
    st.write(f"*Tableau Simplexe (Itération {iteration}) :*")
    st.dataframe(pd.DataFrame(tableau))

# Fonction de la méthode du simplexe
def methode_simplexe(c, A, b):
    num_variables = len(c)
    num_contraintes = len(b)
    
    # Création du tableau du simplexe
    tableau = np.hstack((A, np.eye(num_contraintes), b.reshape(-1, 1))).astype(float)

    # Ajout de la ligne de la fonction objectif
    ligne_c = np.hstack((c, np.zeros(num_contraintes + 1)))
    tableau = np.vstack([tableau, ligne_c])

    afficher_tableau(tableau, 0)

    iteration = 1
    steps = []

    while True:
        # Sélection de la colonne pivot (variable entrante)
        colonne_pivot = np.argmin(tableau[-1, :-1])  # Dernière ligne, sauf colonne des contraintes
        
        # Vérification de l'optimalité
        if tableau[-1, colonne_pivot] >= 0:
            break  # Plus de coefficients négatifs -> solution optimale trouvée

        # Sélection de la ligne pivot (variable sortante)
        ratios = tableau[:-1, -1] / tableau[:-1, colonne_pivot]
        ratios[ratios <= 0] = np.inf  # Ignorer les valeurs négatives ou nulles
        ligne_pivot = np.argmin(ratios)

        # Vérifications des dimensions
        st.write(f"*Itération {iteration}*")
        st.write(f"Taille du tableau : {tableau.shape}")
        st.write(f"Ligne pivot : {ligne_pivot}, Colonne pivot : {colonne_pivot}")
        
        # Normalisation de la ligne pivot
        pivot = tableau[ligne_pivot, colonne_pivot]
        tableau[ligne_pivot, :] /= pivot

        # Mise à zéro des autres lignes
        for i in range(len(tableau)):
            if i != ligne_pivot:
                facteur = tableau[i, colonne_pivot]
                tableau[i, :] -= facteur * tableau[ligne_pivot, :]

        # Enregistrement de l'étape
        steps.append(pd.DataFrame(tableau))
        afficher_tableau(tableau, iteration)
        iteration += 1

    # Extraction des résultats
    solution = np.zeros(num_variables)
    for i in range(num_contraintes):
        base_var = np.where(tableau[i, :num_variables] == 1)[0]
        if len(base_var) == 1:
            solution[base_var[0]] = tableau[i, -1]

    valeur_optimale = tableau[-1, -1]

    return solution, -valeur_optimale, steps

# Interface Streamlit
st.title("Méthode du Simplexe")

# Entrée des données
st.header("Données du problème")

# Fonction objectif
st.subheader("Fonction objectif :")
c_input = st.text_input("Coefficients (séparés par des virgules) :", "-4470, -2310, -2650")
c = np.array([float(x) for x in c_input.split(",")])

# Contraintes
st.subheader("Contraintes :")

# Contrainte 1
st.write("Contrainte 1 :")
A1_input = st.text_input("Coefficients :", "1, 1, 1")
b1_input = st.text_input("Second membre :", "1000")
A1 = np.array([float(x) for x in A1_input.split(",")])
b1 = float(b1_input)

# Contrainte 2
st.write("Contrainte 2 :")
A2_input = st.text_input("Coefficients :", "6400, 6400, 7200")
b2_input = st.text_input("Second membre :", "7000000")
A2 = np.array([float(x) for x in A2_input.split(",")])
b2 = float(b2_input)

# Contrainte 3
st.write("Contrainte 3 :")
A3_input = st.text_input("Coefficients :", "900, 600, 4500")
b3_input = st.text_input("Second membre :", "1600000")
A3 = np.array([float(x) for x in A3_input.split(",")])
b3 = float(b3_input)

# Bouton de résolution
if st.button("Résoudre"):
    # Création de la matrice A et du vecteur b
    A = np.array([A1, A2, A3])
    b = np.array([b1, b2, b3])

    # Vérifications des dimensions
    st.write(f"Dimensions de A : {A.shape}, Dimensions de b : {b.shape}, Dimensions de c : {c.shape}")

    # Résolution
    solution, valeur_optimale, steps = methode_simplexe(c, A, b)

    # Affichage des résultats
    st.header("Résultats :")
    for i in range(solution):
        st.write(f"X{i} : {round(solution[0], 2)} ")

    # Affichage des étapes
    st.header("Étapes du Simplexe :")
    for i, step in enumerate(steps):
        if i == 0:
            st.write('Forme Standard')
            st.dataframe(step)
        st.write(f"*Itération {i + 1}*")
        st.dataframe(step)
