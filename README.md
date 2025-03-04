# IA-Guia-6

### 1. Considere la figura 6.1, tome una ecuación determinada, por ejemplo una raíz cúbica, o un seno, genere un data set con muchos valores. Con base en ese data set y utilizando una herramienta de ML, encuentre un modelo para el cálculo de la raíz cuadrada. Úselo con 10 ejemplos y compare los resultados con los que da la función del lenguaje.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generar dataset
X = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
y = np.sin(X).ravel()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo (red neuronal)
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# Comparar resultados
plt.scatter(X_test, y_test, label='Valor real', color='blue')
plt.scatter(X_test, y_pred, label='Predicción', color='red', alpha=0.5)
plt.legend()
plt.title("Comparación: Función seno vs Modelo ML")
plt.show()
```
### 2. Estudie el programa SVM con todo detalle, mejore su documentación y con base en el haga cambios para una aplicación.

```python
# Código en Python
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Cargar datos
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar SVM
svm_model = SVC(kernel='rbf', C=10, gamma=0.001)
svm_model.fit(X_train, y_train)

# Evaluar
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))
```
### 3. Busque un ejemplo se utilice un algoritmo de K- Nearest, o árboles de decisión.
```python
# Código en Python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluar
print("Precisión:", knn.score(X_test, y_test))
```
### 4. Desarrolle un problema de su escogencia.

```python
# Código en Python
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor

# Cargar datos
boston = fetch_openml(name='boston', version=1)
X, y = boston.data, boston.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
print("R²:", model.score(X_test, y_test))
```
