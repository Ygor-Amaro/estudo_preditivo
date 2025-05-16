import sys
import os
# Adiciona o diretório 'src' ao sys.path para importar módulos locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline especial para SMOTE

SEED = 18498  # Semente para reprodutibilidade

# Seleção de features e target
X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']]
Y = dados['CHURN']

cat_cols = ['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']
num_cols = []

# Split dos dados em treino e teste, mantendo a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    stratify=Y)

# Pré-processamento: OneHotEncoder para variáveis categóricas
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

smote = SMOTE(
    sampling_strategy = 0.5,
    k_neighbors = 3,
    random_state = SEED
)

# Pipeline com pré-processamento, SMOTE e RandomForest
pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', smote),  # Etapa de oversampling
    ('clf', RandomForestClassifier(random_state=SEED))  # Classificador
])

# Parâmetros para busca em grade (GridSearch)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__class_weight': [None, 'balanced'],
    'smote__sampling_strategy': [0.3, 0.5],
}

# Validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Configuração do GridSearchCV
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1  # Paraleliza o processamento
)

# Treinamento do modelo com busca de hiperparâmetros
grid.fit(X_train, y_train)  # Deve vir antes das métricas!

# Avaliação no conjunto de teste
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
y_pred = grid.best_estimator_.predict(X_test)

print(f'\nAUC-ROC no teste: {roc_auc_score(y_test, y_proba):.3f}')
print('\nRelatório Completo:')
print(classification_report(y_test, y_pred, target_names=['Não Evadido', 'Evadido']))

# --- Otimização do threshold para maximizar recall ---
# Calcula a curva precisão-recall
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Encontra o threshold que maximiza o recall
optimal_threshold = thresholds[recall[:-1] == max(recall[:-1])][0]

# Reclassifica as previsões usando o novo threshold
y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
print('\nRelatório com threshold otimizado para máximo recall:')
print(classification_report(y_test, y_pred_optimized, target_names=['Não Evadido', 'Evadido']))

# --- Otimização do threshold para maximizar F1-score ---
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[f1_scores[:-1].argmax()]

y_pred_f1 = (y_proba >= best_threshold).astype(int)
print('\nRelatório com threshold otimizado para máximo F1-score:')
print(classification_report(y_test, y_pred_f1, target_names=['Não Evadido', 'Evadido']))

"""
Script para treinamento e avaliação de um modelo Random Forest com SMOTE para detecção de evasão (churn).
Inclui pipeline de pré-processamento, oversampling, busca de hiperparâmetros e avaliação de métricas.
"""