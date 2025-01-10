from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd
import numpy as np


# Créer une instance de l'application Flask
flask_app = Flask(__name__)

# Spécifiez le chemin vers le fichier model.pkl et le fichier input_example.json
model_path = './data/model.pkl'
json_path = './data/input_example.json'
data_path = './data/data_test.csv'

# Charger le fichier JSON des features 
with open(json_path, 'r') as f:
    input_example = json.load(f)
columns = input_example.get("columns", [])

def load():
    """fonction qui charge le modèle entrainé, le dataset sur lequel va porter l'api et les features 
    utilisés par le modèle"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    data_test = pd.read_csv(data_path)
    data_test.set_index("SK_ID_CURR", inplace=True)
    # Charger le fichier JSON des features 
    with open(json_path, 'r') as f:
        input_example = json.load(f)
    columns = input_example.get("columns", [])
    data_test = data_test[columns]
    return model, data_test

def create_df_proba(df, seuil:float):
    """fonction qui calcule les probabilités d'un client de faire défault à partir du modèle récupéré via la fonction
    load() et le seuil optimisé"""
    proba = model.predict_proba(df)
    df_proba = pd.DataFrame({'client_num':df.index, "proba_no_default":proba.transpose()[0], "proba_default":proba.transpose()[1]})
    df_proba["prediction"] = np.where(df_proba["proba_default"] > seuil, 1, 0)
    return df_proba


model, data_test = load()
seuil_predict = 0.50  #cf. travaux de modélisation 
pred_data = create_df_proba(data_test, seuil_predict)


# Route pour la page d'accueil
@flask_app.route('/', methods=['GET'])
def accueil():
    return jsonify({
        "Accueil": "Bienvenue sur l'API du crédit"
    })

@flask_app.route('/id_client', methods=['POST'])
def get_list_id():
    """fonction qui renvoie les liste des id client (nécessaire pour identifier les clients pour lesquels on souhaite avoir la proba de défault) 
    et la liste des variables du modèle (nécessaire pour explication des résultats)"""
    list_id = data_test.index.to_list()
    return {"list_id":list_id,
            "list_feat":columns}

@flask_app.route('/get_data', methods=['POST'])
def get_data():
    df_data = data_test.to_dict()
    return {"data": df_data}
    
# Route pour prédire avec de nouvelles données 
@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer l'entrée utilisateur
        item = request.get_json()
        if not item or "client_num" not in item:
            return jsonify({"status": "error", "message": "Données invalides. Fournissez un client_num."}), 400
        
        client_num = item["client_num"]

        # Filtrer les résultats
        results = pred_data[pred_data["client_num"] == client_num]
        if results.empty:
            return jsonify({"status": "error", "message": "Client non trouvé."}), 404

        # Préparer la réponse
        verdict = "Demande de crédit acceptée ✅" if results["prediction"].values[0] == 0 else "Demande de crédit refusée ⛔"
        proba = f"Nous estimons la probabilité de défaut du client à {results['proba_default'].values[0]*100:.2f}%"
        
        return jsonify({
            "status": "success",
            "verdict": verdict,
            "proba": proba
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Lancer l'application Flask
if __name__ == '__main__':
    flask_app.run(debug=True)
