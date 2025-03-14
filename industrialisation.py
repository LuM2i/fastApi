from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# Charger le modèle, le préprocesseur et les noms des features
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
feature_names = joblib.load("feature_names.joblib")  # Liste des features utilisées après transformation

# Créer une instance FastAPI
app = FastAPI()

# Configurer CORS (garde ton paramétrage initial)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Remplacez "*" par ["http://localhost:8000"] pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route principale
@app.get("/")
def home():
    return {"message": "API de prédiction d'inondation sur votre département"}

# Route pour récupérer les features attendues
@app.get("/features")
def get_features():
    return {"features": feature_names}

# Définition du schéma des données attendues
class PredictionInput(BaseModel):
    data: dict

# Route pour effectuer une prédiction
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convertir les données en DataFrame
        df = pd.DataFrame([input_data.data])
        # Tenter de convertir chaque colonne en numérique
        df = df.apply(pd.to_numeric, errors='raise')

        # Vérifier que toutes les features sont bien présentes
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            return {"error": f"Features manquantes: {missing_features}"}

        # Réordonner les colonnes pour correspondre au modèle
        df = df[feature_names]

        # Identifier les variables numériques nécessitant RobustScaler
        numeric_features = ["resultat_obs_elab", "vent_moyen", "humidite", 
                            "pluie_24h", "vent_direction", "nb_rafales_10min", 
                            "nb_c_insee_meteo"]

        # Appliquer RobustScaler uniquement aux variables numériques
        df[numeric_features] = preprocessor.transform(df)[:, -len(numeric_features):]

        # # Transformer les données avec le préprocesseur
        # df_transformed = preprocessor.transform(df)

        # Faire la prédiction
        prediction = model.predict(df)  #df_transformed

        return {"prediction_inondations_dans_votre_departement": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
