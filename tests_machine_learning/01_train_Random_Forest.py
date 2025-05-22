import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Caminho para importar os dados locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 18498

# 1. Dados e variáveis
X = dados[['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']]
y = dados['CHURN']
cat_cols = ['CURSOS', 'FILIAL', 'SEGMENTO', 'TURNO', 'PERIODO', 'SEXO']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 2. Pré-processamento + Pipeline com SMOTE + RF
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=SEED)),
    ('clf', RandomForestClassifier(random_state=SEED))
])

param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [5, 10, None],
    'clf__class_weight': [None, 'balanced'],
    'smote__sampling_strategy': [0.3, 0.5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)

# 3. Treinamento
grid.fit(X_train, y_train)

# 4. Obter probabilidades no teste
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]

# 5. Gerar curva F1 por threshold
thresholds = np.linspace(0, 1, 200)
f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# 6. Função reutilizável para previsões com threshold customizado
def prever_com_threshold(modelo, dados, threshold):
    probas = modelo.predict_proba(dados)[:, 1]
    return (probas >= threshold).astype(int)

# 7. Avaliação com threshold ótimo
y_pred_otimo = prever_com_threshold(grid.best_estimator_, X_test, best_threshold)

# 8. Resultados
print(f'\n✅ Melhor combinação de hiperparâmetros: {grid.best_params_}')
print(f'\n🎯 Melhor threshold pelo F1: {best_threshold:.2f} (F1 = {best_f1:.3f})')
print(f'\n📈 AUC-ROC no teste: {roc_auc_score(y_test, y_proba):.3f}')
print('\n📊 Relatório com threshold otimizado:')
print(classification_report(y_test, y_pred_otimo, target_names=['Não Evadido', 'Evadido']))

# 9. Plot F1 por threshold
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label='F1-score', color='blue')
plt.axvline(best_threshold, linestyle='--', color='red', label=f'Melhor threshold = {best_threshold:.2f}')
plt.title('F1-score por Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Avaliação com threshold otimizado
y_pred_opt = (y_proba >= best_threshold).astype(int)
print(f"\nMelhor threshold pelo F1-score: {best_threshold:.2f} (F1 = {best_f1:.3f})\n")
print(classification_report(y_test, y_pred_opt, target_names=['Não Evadido', 'Evadido']))

# Agrupar o modelo e o threshold ótimo em um dicionário
objeto_exportacao = {
    "modelo": grid.best_estimator_,
    "threshold": best_threshold
}

# Exportar para arquivo .pkl
joblib.dump(objeto_exportacao, "modelo_churn_rf.pkl")

print("✅ Modelo salvo como 'modelo_churn_rf.pkl'")