import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Carregue o dataset FILTRADO (o .csv)
print("Carregando dataset...")
df = pd.read_csv("earthquakes_filtred.csv")

# Carregue o MODELO
# (Use o modelo treinado com as 26 features, sem 'risco_terra', etc.)
print("Carregando modelo...")
model = joblib.load("../model/best_model.joblib")

# Pegue as colunas (features) do dataset, exceto o alvo
features = df.drop(columns=["properties.tsunami"]).columns
importances = model.feature_importances_

# Crie um DataFrame para visualização
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Importância das Features:")
print(importance_df.head(10))

# Plotar
plt.figure(figsize=(12, 10))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df.head(15) # Mostrar as top 15
)
plt.title('Importância das Features - O Modelo é Teimoso?')
plt.xlabel('Importância (Peso da Feature)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("feature_importance.png")

print("\nGráfico 'feature_importance.png' salvo.")
print("Verifique esse gráfico. Você vai ver que 'properties.mag' é 90% de tudo.")