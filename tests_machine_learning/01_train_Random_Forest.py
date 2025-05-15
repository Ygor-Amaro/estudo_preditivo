import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

SEED = 68943

# Corrigido: seleção das colunas
X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'ATIVO/EVADIDO', 'PERIODO', 'SEXO']]
Y = dados['CHURN']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    stratify=Y)

# Pré-processamento: imputação e one-hot encoding
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline com modelo
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=SEED))
])

# Validação cruzada (apenas no treino)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f'AUC média (5-fold CV): {scores.mean():.3f}')

# Ajuste de hiperparâmetros (exemplo para RandomForest)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
print(f'Melhores parâmetros: {grid.best_params_}')
print(f'Melhor AUC (CV): {grid.best_score_:.3f}')

# Avaliação no conjunto de teste
test_score = grid.score(X_test, y_test)
print(f'Acurácia no teste: {test_score:.3f}')