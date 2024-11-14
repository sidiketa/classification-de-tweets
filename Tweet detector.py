import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("best_model.keras")

# Interface utilisateur
st.title("Détection de Tweets Suspects")
user_input = st.text_input("Entrez un tweet pour analyse :")

if user_input:
    # Traitement et prédiction
    processed_input = preprocess(user_input)  # Fonction de prétraitement
    prediction = model.predict([processed_input])
    prob = model.predict_proba([processed_input])[0]

    # Affichage des résultats
    st.write(f"Prédiction : {'Suspect' if prediction == 1 else 'Non Suspect'}")
    st.write(f"Probabilité : {max(prob):.2f}")
