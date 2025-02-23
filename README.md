Ce Projet a été développé dans le cadre du parcours Data Scientist d'OpenClassrooms en partenariat avec CentralSupélec (RNCP niveau 7)

Ce dépôt comprend les fichiers nécessaires au déploiement de l'API.

- [flask_app.py](flask_app.py) : fichier python API
- [test_flask_app.py](test_flask_app.py) : fichier de test de l'API
- [requirements.txt](requirements.txt), [Procfile](Procfile), [runtime.txt](runtime.txt) : fichiers indispensables pour le déploiement via Heroku

Dans le dossier data se trouvent les fichiers necessaires au backend de l'API:

- Un subset des données clients soumis au modèle pour tester l'API [data_test.csv](data/data_test.csv.zip)
- Le modèle entrainé sérialisé [model.pkl](data/model.pkl)
- Les features retenus lors de l'entrainement des modèles [input_example.json](data/input_example.json)

Dans le workflows se trouve le fichier [deploy.yml](.github/workflows/deploy.yml) : fichier qui contient le pipeline de déploiement de l'application sur Héroku
