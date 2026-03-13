import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# cargar dataset
data = pd.read_csv("dataset_señas.csv")

# separar datos y etiquetas
X = data.drop("letra", axis=1)
y = data["letra"]

# dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# crear modelo
modelo = RandomForestClassifier()

# entrenar modelo
modelo.fit(X_train, y_train)

# probar modelo
predicciones = modelo.predict(X_test)

precision = accuracy_score(y_test, predicciones)

print("Precisión del modelo:", precision)

# guardar modelo
joblib.dump(modelo, "modelo_señas.pkl")

print("Modelo guardado correctamente.")