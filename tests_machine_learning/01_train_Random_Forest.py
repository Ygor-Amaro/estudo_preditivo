import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


X = dados['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'ATIVO/EVADIDO', 'PERIODO', 'SEXO']
Y = dados['CHURN']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pré-processamento: imputação e one-hot encoding
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline com modelo
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state = 8412))
])

# Validação cruzada
scores = cross_val_score(pipeline, X, Y, cv=5, scoring='roc_auc')
print(f'AUC média (5-fold CV): {scores.mean():.3f}')

# Ajuste de hiperparâmetros (exemplo para RandomForest)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
grid.fit(X, Y)
print(f'Melhores parâmetros: {grid.best_params_}')
print(f'Melhor AUC: {grid.best_score_:.3f}')