import sys
import os
import numpy as np
import pandas as pd
import joblib
import logging
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib import pyplot as plt

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuração de semente para reprodutibilidade
SEED = 18498

# Caminho para importar os dados locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados

# 1. Validação dos dados
required_columns = ['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO', 'CHURN']
if not all(col in dados.columns for col in required_columns):
    raise ValueError(f"O arquivo de dados deve conter as colunas: {required_columns}")

# 2. Dados e divisão
X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']]
y = dados['CHURN']
cat_cols = X.columns.tolist()

# Tratamento de valores ausentes
if X.isnull().any().any():
    X.fillna('Desconhecido', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

if X_train.isnull().any().any():
    logging.warning("Valores ausentes encontrados no conjunto de treino. Substituindo por 'Desconhecido'.")
    X_train.fillna('Desconhecido', inplace=True)
    X_test.fillna('Desconhecido', inplace=True)

# Filtrar apenas colunas numéricas
numeric_cols = X_train.select_dtypes(include=[np.number]).columns

# Calcular variância apenas nas colunas numéricas
numeric_cols = X_train.select_dtypes(include=[np.number]).columns

if len(numeric_cols) > 0:
    low_variance_cols = X_train[numeric_cols].var()[X_train[numeric_cols].var() < 1e-5].index.tolist()

    if low_variance_cols:
        logging.warning(f"Colunas com baixa variância removidas: {low_variance_cols}")
        X_train.drop(columns=low_variance_cols, inplace=True)
        X_test.drop(columns=low_variance_cols, inplace=True)

class_counts = y_train.value_counts()
logging.info(f"Distribuição das classes no treino: {class_counts}")
if class_counts.min() < 5:
    logging.warning("Uma das classes tem menos de 5 exemplos. Verifique os dados.")

# 3. Pré-processamento
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

# 4. Pipeline LightGBM + SMOTE
pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=SEED)),
    ('clf', LGBMClassifier(random_state=SEED, n_jobs=-1))
])

# 5. Hiperparâmetros para busca
param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [5, 10],
    'clf__learning_rate': [0.05, 0.1],
    'clf__class_weight': ['balanced'],
    'clf__min_child_samples': [20, 50],  # Ajuste aqui
    'clf__min_split_gain': [0.01, 0.1],  # Ajuste aqui
    'smote__sampling_strategy': [0.5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

grid = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_iter=10,  # Testa 10 combinações aleatórias
    n_jobs=-1,
    verbose=1,
    random_state=SEED
)

# 6. Treinamento
logging.info("Iniciando o treinamento...")
grid.fit(X_train, y_train)
logging.info("Treinamento concluído.")

# 7. Probabilidades e curva F1
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 200)
f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# Função para previsões com threshold customizado
def prever_com_threshold(modelo, dados, threshold):
    proba = modelo.predict_proba(dados)[:, 1]
    return (proba >= threshold).astype(int)

# 8. Previsão com threshold otimizado
y_pred_otimo = prever_com_threshold(grid.best_estimator_, X_test, best_threshold)

# 9. Relatórios
logging.info(f"Melhores hiperparâmetros: {grid.best_params_}")
logging.info(f"Melhor threshold pelo F1-score: {best_threshold:.2f} (F1 = {best_f1:.3f})")
logging.info(f"AUC-ROC no teste: {roc_auc_score(y_test, y_proba):.3f}")

print("\n📊 Relatório final:")
print(classification_report(y_test, y_pred_otimo, target_names=['Não Evadido', 'Evadido']))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_otimo))

# 10. Curva F1
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label='F1-score', color='green')
plt.axvline(best_threshold, linestyle='--', color='red', label=f'Threshold ótimo = {best_threshold:.2f}')
plt.title('F1-score por Threshold (LightGBM)')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 11. Exportar modelo + threshold
output_path = "modelo_churn_lgbm.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump({
    "modelo": grid.best_estimator_,
    "threshold": best_threshold
}, output_path)

logging.info(f"Modelo LightGBM salvo como '{output_path}'")

lgb.set_config(verbosity=1)  # Define o nível de log para detalhado
