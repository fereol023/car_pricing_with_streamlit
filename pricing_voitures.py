import streamlit as st
import numpy as np
import joblib

st.title("Prédiction du prix d'une voiture en fonction de ses caractéristiques")
st.subheader("Application réalisée par Féréol")
st.markdown("*Cette application se base sur un modèle de machine learning pour estimer le prix d'une voiture selon les données du marché (au sens vague).*")


# definition d'une fonction d'inférence
def inference(symboling, normalized_losses, wheel_base, length, width, height, curb_weight, engine_size, compression_ratio, horse_power, peak_rpm, city_mpg, highway_mpg) :
    """
    Prends les caracteristiques de la voiture et renvoie le prix selon le modele de ML sauvegardé.
    """
    new_data = np.array([
        symboling, normalized_losses, wheel_base, length, width, height, curb_weight, engine_size, 
        compression_ratio, horse_power, peak_rpm, city_mpg, highway_mpg
    ])

    # chargement du modèle
    model = joblib.load("final_model.joblib")

    # prediction
    return model.predict(new_data.reshape(1, -1))

# features saisies par l'utilisateur
# données par défaut (valeurs médianes)
symboling = st.number_input(label='symboling:', min_value=0, value=3)
normalized_losses = st.number_input('normalized_losses', value=100)
wheel_base = st.number_input('wheel_base', value=90)
length = st.number_input('length', value=150)
width = st.number_input('width', value=65)
height = st.number_input('height', value=50)
curb_weight = st.number_input('curb_weight', value=200)
engine_size = st.number_input('engine_size', value=120)
compression_ratio = st.number_input('compression_ratio', value=9)
horse_power = st.number_input('horse_power', value=110)
peak_rpm = st.number_input('peak_rpm', value=5200)
city_mpg = st.number_input('city_mpg', value=24)
highway_mpg = st.number_input('highway_mpg', value=30)

# creation du bouton "predict" qui retourne la prediction
if st.button("Estimer le prix") :
    prediction = inference(
        symboling, normalized_losses, wheel_base, length, width, height, curb_weight,
         engine_size, compression_ratio, horse_power, peak_rpm, city_mpg, highway_mpg
    )
    res = "Le prix de cette voiture est : " + str(prediction[0]) + " $."
    st.success(res)









