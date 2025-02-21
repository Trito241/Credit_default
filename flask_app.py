import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flask import Flask, request, jsonify
import shap
import pickle
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import os

# Désactiver l'interface Tkinter pour éviter les erreurs liées aux threads
matplotlib.use('Agg')

# Créer une instance de l'application Flask
flask_app = Flask(__name__, static_url_path='/static', static_folder='static')

# Spécifiez le chemin vers le fichier model.pkl et le fichier input_example.json
model_path = './data/model.pkl'
json_path = './data/input_example.json'
data_path = './data/data_test.csv.zip'  # pour le dashboard et avoir des distributions plus robustes
    
# Charger le fichier JSON des features 
with open(json_path, 'r') as f:
    input_example = json.load(f)
columns = input_example.get("columns", [])

def load():
    """fonction qui charge le modèle entrainé, le dataset sur lequel va porter l'api et les features 
    utilisés par le modèle"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    data_test =  pd.read_csv(data_path, compression='zip')
    # data_test_ori = pd.read_csv(data_path_ori)
    data_test.set_index("SK_ID_CURR", inplace=True)
    # data_test_ori.set_index("SK_ID_CURR", inplace=True)
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
seuil_predict = 0.30  #cf. travaux de modélisation 
pred_data = create_df_proba(data_test, seuil_predict)
explainer = shap.LinearExplainer(model, data_test)

# Générer une fois l'explication globale des features
shap_values_global = explainer.shap_values(data_test)
shap_values_global = np.array(shap_values_global, dtype=float)  # Assurer le bon type
plt.figure()
shap.summary_plot(shap_values_global, data_test, max_display=10, show=False)
global_image_path = "./static/shap_summary.png"
plt.title("Feature importance global")
plt.savefig(global_image_path, bbox_inches='tight')
plt.close()


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
    """fonction qui renvoie, le score de probabilité et la décision"""
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


@flask_app.route('/explanation', methods=['POST'])
def get_explanation():
    """
    Renvoie les explications locales et globales des décisions du modèle pour un client donné.
    """
    try:
        # Récupérer les données envoyées par le client
        item = request.get_json()
        client_num = item.get("client_num")
        if client_num not in data_test.index:
            return jsonify({"status": "error", "message": "Client non trouvé."}), 404
            
        # Vérification que SHAP est bien configuré
        if not explainer:
            return jsonify({"status": "error", "message": "Explainer SHAP non configuré."}), 500

        num_id = data_test.index.get_loc(client_num)
        shap_values_local = explainer.shap_values(data_test)
        shap_values_local = np.array(shap_values_local, dtype=float)  # Assurer le bon type
        client_data = data_test.loc[client_num]
        
        shap.initjs()  # Initialisation des scripts JS nécessaires pour SHAP

        # Créer le graphique Waterfall pour les explications locales
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_local[num_id, :],  # SHAP values pour le client
                base_values=explainer.expected_value,
                data=client_data,
                feature_names=data_test.columns,
            ),
            show=False
        )
        
        local_image_path = f"./static/shap_waterfall_{client_num}.png"
        plt.title(f"Feature importance du client {client_num}")
        plt.savefig(local_image_path,bbox_inches='tight')
        plt.close()

        return jsonify({
            "status": "success",
            "local_explanation": f"/static/shap_waterfall_{client_num}.png",
            "global_explanation": f"/static/shap_summary.png"
        })
        
    except Exception as e:
        # Gestion des erreurs et retour d'une réponse appropriée
        print(f"Erreur dans /explanation : {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@flask_app.route('/info_client', methods=['POST'])
def get_perso():
    """Pour un client donné, renvoie un ensemble d'informations (âge, sexe, métier...)"""
    try:
        # Récupérer les données envoyées par le client
        item = request.get_json()
        client_num = item.get("client_num")

        # Vérifier si le client est dans les données
        if client_num not in data_test.index.values:
            return jsonify({"status": "error", "message": f"Client {client_num} non trouvé."}), 404

        # Convertir les index pour un accès plus rapide
        df = data_test.copy()

        # Récupérer les informations du client
        client_data = df.loc[client_num]

        gender = "Male" if client_data["CODE_GENDER_M"] == 1 else "Female"
        nb_child = int(client_data["CNT_CHILDREN"])
        income_amount = float(client_data["AMT_INCOME_TOTAL"])
        credit = float(client_data["AMT_CREDIT"])

        # Sélectionner les colonnes qui commencent par NAME_INCOME_TYPE ou NAME_FAMILY_STATUS
        list_col = [col for col in df.columns if col.startswith("NAME_INCOME_TYPE") or col.startswith("NAME_FAMILY_STATUS")]

        # Filtrer pour récupérer les informations du client
        input_data = client_data[list_col]

        # Trouver les colonnes où la valeur est 1
        list_comp = [c for c in list_col if input_data[c] == 1]

        # Extraire les informations
        income_type = str(list_comp[0].rsplit('_')[-1]) if len(list_comp) > 0 else "Inconnu"
        family = str(list_comp[1].rsplit('_')[-1]) if len(list_comp) > 1 else "Inconnu"

        return jsonify({
            "status": "success",
            "gender": gender,
            "nb_child": nb_child,
            "income_amount": income_amount,
            "credit": credit,
            "income_type": income_type,
            "family": family
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@flask_app.route('/data_comparaison', methods=['POST'])
def get_comparison_data():
    """Renvoie les données nécessaires pour comparer un client aux autres."""
    try:
        # Sélectionner les variables pertinentes
        variables = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3","AMT_CREDIT", 
                     "DAYS_EMPLOYED", "CREDIT_TERM", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"]
        
        data_test_com = data_test.copy()
        data_test_com = data_test_com[variables]
        data_test_com = data_test_com.reset_index()

        # Vérifier si les colonnes existent dans data_test
        available_vars = [var for var in variables if var in data_test.columns]

        if not available_vars:
            return jsonify({"status": "error", "message": "Aucune variable valide trouvée."}), 400

        # Extraire uniquement les colonnes nécessaires
        data_subset = data_test_com.dropna().to_dict(orient="records")

        return jsonify({"status": "success", "data": data_subset})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    flask_app.run(debug=True)