from fastapi import FastAPI
from fastapi.testclient import TestClient
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json

app = FastAPI()

# Définir l'URI de suivi pour utiliser le système de fichiers local
# mlflow.set_tracking_uri("file:///C:/Users/PC/Desktop/OC/Projet7_OBAME_ONIANE_Landry/mlruns")

# Définir l'URI de suivi pour utiliser le système de fichiers local
mlflow.set_tracking_uri("http://localhost:5000")

# Charger le modèle depuis MLflow
model_uri = "runs:/017e8de687ae4339816dd8a75eac8645/model"
model = mlflow.sklearn.load_model(model_uri)

# Préparation data
app_test = pd.read_csv("app.csv")
expected_features = model.feature_names_in_
app_test_aligned = app_test[expected_features]

# Convertir une ligne en dictionnaire pour prédiction
sample_features = app_test_aligned.iloc[0].to_dict()

# Convertir le dictionnaire en une chaîne JSON correctement formatée
sample_features = json.dumps(sample_features, ensure_ascii=False)

# Seuil pour la classification
threshold = 0.5

@app.post("/predict")
def predict(features: dict):
    """
    Prend un dictionnaire de caractéristiques, retourne la probabilité de défaut
    et la classe correspondante (accepté/refusé).
    """
    try:
        
        data = np.array([list(features.values())])
        
        # Prédire la probabilité de défaut
        prob = model.predict_proba(data)[:, 1][0]  # [0] pour récupérer la probabilité unique
        
        # Décision basée sur le seuil
        decision = "Refusé" if prob >= threshold else "Accepté"
        
        return {"probabilité": float(prob), "décision": decision}
    except Exception as e:
        return {"erreur": str(e)}

# Tester l'API localement
if __name__ == "__main__":
    # Extraire un échantillon
    #sample_features = app_test_aligned.iloc[0].to_dict()
    
    # Tester localement avec FastAPI
    client = TestClient(app)
    response = client.post("/predict", json=sample_features)
    print(response.json())