# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:59:37 2024

@author: IVAN
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Cargar el dataset de cáncer de mama
data = load_breast_cancer()
X, y = data.data, data.target

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el scaler y ajustarlo solo en el conjunto de entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo de regresión logística
model = LogisticRegression()

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    # 'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    # 'C': [0.1, 1.0, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Ejecutar GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Obtener los mejores parámetros
best_params = grid_search.best_params_
print("\nMejores parámetros encontrados:", best_params)

# Evaluar el mejor modelo en el conjunto de prueba
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test_scaled, y_test)
print(f"\nPrecisión en el conjunto de prueba: {accuracy:.4f}")
