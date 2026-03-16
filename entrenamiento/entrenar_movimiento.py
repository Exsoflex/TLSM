import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# cargar dataset
dataset = joblib.load("dataset_movimiento.pkl")

X = []
y = []

for secuencia, etiqueta in dataset:

    # convertir secuencia (20x42) en vector plano (840)
    vector = np.array(secuencia).flatten()

    X.append(vector)
    y.append(etiqueta)

X = np.array(X)
y = np.array(y)

# dividir datos entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# crear modelo
modelo = RandomForestClassifier(n_estimators=200)

# entrenar
modelo.fit(X_train, y_train)

# probar
pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Precisión del modelo:", accuracy)

# guardar modelo
joblib.dump(modelo, "modelo_movimiento.pkl")

print("Modelo guardado")