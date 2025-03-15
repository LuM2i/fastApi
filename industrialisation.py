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
        numeric_to_scale = ["resultat_obs_elab", "vent_moyen", "pluie_24h", "nb_rafales_10min"]
        # Réordonner selon feature_names
        numeric_to_scale_ordered = [col for col in feature_names if col in numeric_to_scale]

        # Appliquer RobustScaler uniquement sur ces colonnes
        
        df[numeric_to_scale_ordered] = preprocessor.named_transformers_["extreme"].transform(df[numeric_to_scale_ordered])


        


        # # Transformer les données avec le préprocesseur
        # df_transformed = preprocessor.transform(df)

        # Faire la prédiction
        prediction = model.predict(df)  #df_transformed

        pred_value = int(prediction[0])
        
        # Appliquer la logique conditionnelle pour créer un message personnalisé
        if pred_value == 0:
            annonce = "Risque faible d'inondation"
        elif pred_value == 1:
            annonce = "Attention, le risque d'inondation dans votre département est important"
        else:
            annonce = "{}"
        
        # Retourner à la fois la prédiction et le message personnalisé
        return {
            "prediction_inondations_dans_votre_departement": pred_value,
            "annonce": annonce
        }

    except Exception as e:
        return {"error": str(e)}
    
    

