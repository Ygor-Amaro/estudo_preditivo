import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline especial para SMOTE

SEED = 18498

X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']]
Y = dados['CHURN']

cat_cols = ['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    stratify=Y)

# Pré-processamento
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline com SMOTE
pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=SEED)),  # Etapa de oversampling
    ('clf', RandomForestClassifier(random_state=SEED))  # Removido class_weight
])

# Parâmetros para GridSearch
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [3, 5, 7],
    'smote__sampling_strategy': [0.3, 0.5],
    'smote__k_neighbors': [3, 5]
}

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Configuração do GridSearch
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1  # Paraleliza o processamento
)

# Treinamento
grid.fit(X_train, y_train)  # Deve vir antes das métricas!

# Avaliação
print(f'\nMelhores parâmetros: {grid.best_params_}')
print(f'Melhor AUC (CV): {grid.best_score_:.3f}')

# Métricas no teste
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
y_pred = grid.best_estimator_.predict(X_test)

print(f'\nAUC-ROC no teste: {roc_auc_score(y_test, y_proba):.3f}')
print('\nRelatório Completo:')
print(classification_report(y_test, y_pred, target_names=['Não Evadido', 'Evadido']))