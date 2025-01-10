import pytest
from flask_app import flask_app

@pytest.fixture
def client():
    flask_app.testing = True
    return flask_app.test_client()

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {"Accueil": "Bienvenue sur l'API du crédit"}

def test_predict_missing_fields(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "colonnes suivantes sont manquantes" in response.json["error"]
