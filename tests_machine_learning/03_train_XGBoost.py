import sys
import os
import numpy as np
import joblib
import logging
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
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

# Validação dos dados
required_columns = ['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO', 'CHURN']
if not all(col in dados.columns for col in required_columns):
    raise ValueError(f"O arquivo de dados deve conter as colunas: {required_columns}")

# Dados e divisão
X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']]
y = dados['CHURN']
cat_cols = X.columns.tolist()

X.fillna('Desconhecido', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# Remover colunas com baixa variância se existirem
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    low_variance_cols = X_train[numeric_cols].var()[X_train[numeric_cols].var() < 1e-5].index.tolist()
    if low_variance_cols:
        logging.warning(f"Colunas com baixa variância removidas: {low_variance_cols}")
        X_train.drop(columns=low_variance_cols, inplace=True)
        X_test.drop(columns=low_variance_cols, inplace=True)

logging.info(f"Distribuição das classes no treino:\n{y_train.value_counts()}")

# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=SEED)),
    ('clf', XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss'))
])

# Hiperparâmetros
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10],
    'clf__learning_rate': [0.05, 0.1],
    'clf__scale_pos_weight': [1, 3, 5],
    'clf__subsample': [0.6, 0.8, 1.0],  # Ajuste de subsample
    'clf__colsample_bytree': [0.6, 0.8, 1.0],  # Ajuste de colsample_bytree
    'clf__gamma': [0, 0.1, 0.5],  # Ajuste de gamma
    'smote__sampling_strategy': [0.3, 0.5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

grid = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_iter=15,
    n_jobs=-1,
    verbose=1,
    random_state=SEED
)

# Treinamento
grid.fit(X_train, y_train)

# Avaliação
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 200)
f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

def prever_com_threshold(modelo, dados, threshold):
    proba = modelo.predict_proba(dados)[:, 1]
    return (proba >= threshold).astype(int)

y_pred_otimo = prever_com_threshold(grid.best_estimator_, X_test, best_threshold)

# Relatórios
logging.info(f"Melhores hiperparâmetros: {grid.best_params_}")
logging.info(f"Melhor threshold pelo F1-score: {best_threshold:.2f} (F1 = {best_f1:.3f})")
logging.info(f"AUC-ROC no teste: {roc_auc_score(y_test, y_proba):.3f}")

print("\n📊 Relatório final:")
print(classification_report(y_test, y_pred_otimo, target_names=['Não Evadido', 'Evadido']))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_otimo))

# Curva F1
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label='F1-score', color='green')
plt.axvline(best_threshold, linestyle='--', color='red', label=f'Threshold ótimo = {best_threshold:.2f}')
plt.title('F1-score por Threshold (XGBoost)')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Exportar modelo + threshold
output_path = "modelo_churn_xgboost.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump({
    "modelo": grid.best_estimator_,
    "threshold": best_threshold
}, output_path)

logging.info(f"Modelo XGBoost salvo como '{output_path}'")
