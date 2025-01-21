from flask import Flask, request, jsonify
import shap
import pickle
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Créer une instance de l'application Flask
flask_app = Flask(__name__)

# Spécifiez le chemin vers le fichier model.pkl et le fichier input_example.json
model_path = './data/model.pkl'
json_path = './data/input_example.json'
data_path = './data/data_test.csv.zip'

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
explainer = shap.LinearExplainer(model, data_test)

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

# Route pour afficher la jauge de probabilité
@flask_app.route('/gauge', methods=['POST'])
def gauge():
    """visualisation de la probabilité de défaut d'un client donné sous forme de jauge"""
    item = request.get_json()
    client_num = item.get("client_num")
    
    # Récupérer la probabilité et la décision
    value = pred_data[pred_data["client_num"] == client_num]["proba_no_default"].values[0]
    prediction = pred_data[pred_data["client_num"] == client_num]["prediction"].values[0]
    
    # Décision basée sur la prédiction
    if prediction == 0:
        verdict = "Demande de crédit acceptée ✅"
    else:
        verdict = "Demande de crédit refusée ⛔"
    
    # Définir la couleur en fonction du seuil
    if value > 1 - seuil_predict:
        color = "green"
    else:
        color = "orange"
    
    # Créer la jauge Plotly
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=value,
        mode="gauge+number+delta",
        title={'text': "Score", 'font': {'size': 15}},
        delta={'reference': 1 - seuil_predict, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={'axis': {'range': [None, 1]},
               'bar': {'color': color},
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1 - seuil_predict}}))
    
    fig.update_layout(autosize=False, width=400, height=350)
    fig_html = fig.to_html(full_html=False)  # Convertir en HTML

    # Retourner la jauge, la décision et le score
    return {
        "fig": fig_html,
        "verdict": verdict,
        "score": f"Probabilité de défaut : {(1- value) * 100:.2f}%"
    }

# Route d'explication des décisions du modèle
@flask_app.route('/explanation', methods=['POST'])
def get_explanation():
    """Renvoie l'explication des décisions du modèle pour un client donné"""
    try:
        # Récupérer les données de l'utilisateur
        item = request.get_json()
        if not item or "client_num" not in item:
            return jsonify({"status": "error", "message": "Données invalides. Fournissez un client_num."}), 400
        
        client_num = item.get("client_num")

        # Récupérer les données du client
        client_data = data_test.loc[client_num].values.reshape(1, -1)
        
        # Calculer les valeurs SHAP pour le client
        shap_values = explainer.shap_values(client_data).astype(float)
        shap.initjs()
        # Créer un graphique force plot SHAP pour visualiser la contribution des features pour ce client
        shap_force_plot = shap.force_plot(explainer.expected_value, shap_values[0:], client_data, feature_names=data_test.columns)
        shap_html = f"<head>{shap.getjs()}</head><body>{shap_force_plot.html()}</body>"
        
        # Créer un graphique summary plot SHAP pour visualiser l'importance globale des features
        shap_summary_plot = shap.summary_plot(shap_values, client_data, feature_names=data_test.columns, show=False)
        shap_summary_html = shap.get_html()  # Convertir en HTML pour l'affichage dans Streamlit
        
        return jsonify({
            "status": "success",
            "shap_force_plot": shap_html,
            "shap_summary_plot": shap_summary_html
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    flask_app.run(debug=True)
