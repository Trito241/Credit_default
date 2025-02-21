import pytest
import matplotlib
from flask_app import flask_app

@pytest.fixture
def client():
    """Crée un client de test Flask."""
    flask_app.testing = True
    with flask_app.test_client() as client:
        yield client

def test_home(client):
    """Test de la route d'accueil."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {"Accueil": "Bienvenue sur l'API du crédit"}

def test_id_client(client):
    """Test de la route /id_client pour récupérer les ID clients et les colonnes."""
    response = client.post('/id_client')
    assert response.status_code == 200
    assert "list_id" in response.json
    assert "list_feat" in response.json

def test_get_data(client):
    """Test de la route /get_data pour vérifier les données."""
    response = client.post('/get_data')
    assert response.status_code == 200
    assert "data" in response.json

def test_predict_missing_fields(client):
    """Test de la route /predict avec des champs manquants."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert response.json["status"] == "error"
    assert "Données invalides" in response.json["message"]

def test_predict_client_not_found(client):
    """Test de la route /predict avec un client non existant."""
    payload = {"client_num": 999999}  # ID client inexistant
    response = client.post('/predict', json=payload)
    assert response.status_code == 404
    assert response.json["status"] == "error"
    assert "Client non trouvé" in response.json["message"]

def test_predict_valid_accept(client):
    """Test de la route /predict avec un client valide et prédiction acceptée."""
    payload = {"client_num": 100028}  # Remplacez avec un ID existant
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json["status"] == "success"

def test_predict_valid_refuse(client):
    """Test de la route /predict avec un client valide et prédiction refusée."""
    payload = {"client_num": 100013}  # Remplacez avec un ID existant
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json["status"] == "success"

def test_explanation_valid_client(client):
    """Test de la route /explanation avec un client valide."""
    payload = {"client_num": 100028}  # ID client existant
    response = client.post('/explanation', json=payload)
    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert "local_explanation" in response.json
    assert "global_explanation" in response.json

def test_get_perso_valid_client(client):
    """Test de la route /info_client avec un client valide."""
    payload = {"client_num": 100028}  # ID client existant
    response = client.post('/info_client', json=payload)
    assert response.status_code == 200
    assert response.json["status"] == "success"
    expected_keys = {"gender", "nb_child", "income_amount", "credit", "income_type", "family"}
    assert expected_keys.issubset(response.json.keys())  

def test_get_comparison_data(client):
    """Test de la route /data_comparaison pour comparer un client aux autres."""
    response = client.post('/data_comparaison')
    assert response.status_code == 200
    assert "data" in response.json
    assert isinstance(response.json["data"], list)

if __name__ == '__main__':
    pytest.main()
