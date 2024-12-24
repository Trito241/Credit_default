from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json

app = FastAPI()

# Définir l'URI de suivi pour utiliser le système de fichiers local ou serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Utiliser l'URI de votre serveur MLflow si déployé

# Charger le modèle depuis MLflow
model_uri = "runs:/017e8de687ae4339816dd8a75eac8645/model"
model = mlflow.sklearn.load_model(model_uri)

# Obtenir les noms des caractéristiques attendues par le modèle
expected_features = model.feature_names_in_

# Seuil pour la classification
threshold = 0.5

@app.post("/predict_single")
async def predict_single(file: UploadFile = File(...), line_index: int = 0):
    """
    Charge un fichier CSV, sélectionne une seule ligne (par son index) pour la prédiction.
    """
    try:
        # Lire le fichier CSV
        contents = await file.read()
        app_test = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Vérifier si les colonnes nécessaires sont présentes
        missing_features = [feature for feature in expected_features if feature not in app_test.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Les colonnes suivantes sont manquantes dans le fichier : {missing_features}"
            )

        # Aligner les colonnes avec celles attendues par le modèle
        app_test_aligned = app_test[expected_features]

        # Vérifier que l'index demandé est valide
        if line_index >= len(app_test_aligned):
            raise HTTPException(
                status_code=400,
                detail=f"Ligne {line_index} introuvable dans le fichier. Le fichier contient {len(app_test_aligned)} lignes."
            )

        # Sélectionner une seule ligne pour la prédiction
        row = app_test_aligned.iloc[line_index]

        # Convertir la ligne en dictionnaire
        sample_features = row.to_dict()

        # Convertir le dictionnaire en JSON compatible
        sample_features_json = json.dumps(sample_features, ensure_ascii=False)

        # Reconversion du JSON en dictionnaire
        sample_features_dict = json.loads(sample_features_json)

        # Préparer les données pour la prédiction
        data = np.array([list(sample_features_dict.values())])  # Convertir en tableau numpy

        # Prédire la probabilité
        prob = model.predict_proba(data)[:, 1][0]

        # Décision basée sur le seuil
        decision = "Refusé" if prob >= threshold else "Accepté"

        return {"client_data": sample_features_dict, "probabilité": float(prob), "décision": decision}

    except Exception as e:
        return {"erreur": str(e)}
