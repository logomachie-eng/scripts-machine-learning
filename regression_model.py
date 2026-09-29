# Script d'entraînement expérimental : Régression Linéaire
# Basé sur les concepts d'autoformation en Machine Learning (Régression et Classification)

import numpy as np
from sklearn.linear_model import LinearRegression

# Jeu de données factice pour l'analyse
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Initialisation et entraînement du modèle de régression
modele = LinearRegression()
modele.fit(X, y)

# Affichage de la prédiction
prediction = modele.predict(np.array([[6]]))
print(f"Test de prédiction pour X=6 : {prediction[0]}")
