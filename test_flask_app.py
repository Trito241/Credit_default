import pytest
from flask_app import flask_app

@pytest.fixture
def client():
    # Configuration de Flask pour les tests
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
    payload = {"client_num": 999999}  # Un ID client qui n'existe pas
    response = client.post('/predict', json=payload)
    assert response.status_code == 404
    assert response.json["status"] == "error"
    assert "Client non trouvé" in response.json["message"]

def test_predict_valid_accept(client):
    """Test de la route /predict avec un client valide et prédiction acceptée."""
    payload = {"client_num": 144092}  # Remplacez par un ID client existant avec prédiction acceptée
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert response.json["verdict"] == "Demande de crédit acceptée ✅"

def test_predict_valid_refuse(client):
    """Test de la route /predict avec un client valide et prédiction refusée."""
    payload = {"client_num": 100038}  # Remplacez par un ID client existant avec prédiction refusée
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert response.json["verdict"] == "Demande de crédit refusée ⛔"

