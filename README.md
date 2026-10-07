# Cardiovascular Risks Project

Prédiction du risque de maladie cardio-vasculaire par régression logistique, à partir du jeu de données Kaggle « Cardiovascular Disease dataset » (70 000 patients).

Code écrit en 2023 ; revue et corrections de 2026 faites avec Claude.

## Contenu

- `main.ipynb` : le notebook principal (exploration, nettoyage, modèles, prédiction pour un patient fictif « Arthur »).
- `data/cardio_train.csv` : les données, à télécharger (voir ci-dessous) ; non versionnées.
- `custom_class.py`, `custom_functions.py` : début d'un modèle de régression logistique écrit à la main (inachevé, non utilisé dans le notebook).
- `requirements.txt` : les bibliothèques nécessaires.

## Données

Jeu de données « [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) » de Svetlana Ulianova sur Kaggle. Sa licence est indiquée « Unknown » : il n'est donc pas redistribué dans ce dépôt.

Pour l'obtenir, le télécharger depuis la page Kaggle et placer `cardio_train.csv` dans le dossier `data/`, ou bien :

```
pip install kagglehub
python -c "import kagglehub, shutil, os; p = kagglehub.dataset_download('sulianova/cardiovascular-disease-dataset'); os.makedirs('data', exist_ok=True); shutil.copy(os.path.join(p, 'cardio_train.csv'), 'data/')"
```

Format : séparateur `;`, une ligne par patient (70 000), âge en jours, colonne cible `cardio`.

## Installation et exécution

Testé avec Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install jupyterlab
jupyter lab
```

Ouvrir `main.ipynb` puis « Run All » (ou l'ouvrir dans VS Code avec l'extension Jupyter). Le notebook doit être lancé depuis le dossier du projet (chemin relatif `data/cardio_train.csv`).

Les sorties (tableaux et graphiques) sont enregistrées dans le notebook : on peut lire les résultats sur GitHub sans rien exécuter.

## Ce que fait le notebook

1. Exploration des données : répartition de la cible, de l'âge, de la taille, du poids, du genre, des facteurs de mode de vie et cliniques.
2. Nettoyage : suppression des tailles supérieures à 2,20 m ou inférieures à 1,20 m et des tensions artérielles nulles, négatives ou supérieures à 200 mm Hg (68 880 patients restants).
3. Régression logistique avec statsmodels (`Logit`) puis scikit-learn (`LogisticRegression`, données normalisées avec `StandardScaler`), sur 80 % des données ; validation sur les 20 % restants.
4. Évaluation : précision d'environ 72 % et aire sous la courbe ROC d'environ 0,79 sur le jeu de validation.
5. Prédiction pour Arthur : le modèle le classe à risque.

## Limites connues

- La section 5 (modèle personnalisé) n'est pas terminée.
- Le modèle statsmodels est ajusté sans constante (pas d'ordonnée à l'origine).

## Pour aller plus loin

D'autres valeurs invraisemblables restent dans les données après le nettoyage :

- 6 patients pèsent moins de 30 kg ;
- 272 patients ont une tension diastolique (`ap_lo`) supérieure à la systolique (`ap_hi`), ce qui est physiologiquement impossible (probablement des valeurs inversées à la saisie).

On pourrait les supprimer de la même façon que les petites tailles et observer l'effet sur les résultats.
