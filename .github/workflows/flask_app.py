from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd

# Créer une instance de l'application Flask
flask_app = Flask(__name__)

# Spécifiez le chemin vers le fichier model.pkl et le fichier input_example.json
model_path = 'model.pkl'
json_path = 'input_example.json'

# Charger le modèle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Charger le fichier JSON pour obtenir les colonnes
with open(json_path, 'r') as f:
    input_example = json.load(f)
columns = input_example.get("columns", [])

# Route pour la page d'accueil
@flask_app.route('/', methods=['GET'])
def accueil():
    return jsonify({
        "Accueil": "Bienvenue sur l'API du crédit"
    })

# Route pour prédire avec de nouvelles données envoyées en JSON
@flask_app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données JSON envoyées dans la requête
    data = request.get_json()

    # Vérifier que toutes les colonnes nécessaires sont présentes dans les données JSON
    missing_columns = [col for col in columns if col not in data]
    if missing_columns:
        return jsonify({"error": f"Les colonnes suivantes sont manquantes : {', '.join(missing_columns)}"}), 400

    # Créer un DataFrame avec les données reçues
    input_data = {col: data.get(col) for col in columns}
    input_df = pd.DataFrame([input_data])

    # Effectuer la prédiction avec le modèle
    prediction_proba = model.predict_proba(input_df)
    probability = prediction_proba[0][1]  # La probabilité de la classe positive (prêt accordé)

    # Seuil pour la prédiction
    threshold = 0.5  # Vous pouvez ajuster ce seuil selon vos besoins

    # Décision basée sur la probabilité
    result = "prêt accordé" if probability <= threshold else "prêt refusé"

    # Retourner la probabilité et la décision sous forme de réponse JSON
    return jsonify({
        "probabilité": probability,
        "décision": result
    })

# Lancer l'application Flask
if __name__ == '__main__':
    flask_app.run(debug=True)
