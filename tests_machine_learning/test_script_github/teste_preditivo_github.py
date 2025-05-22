# 1 - Importando as bibliotecas

# Análise de dados
# Visualização de dados
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configurações para visualização
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 100  # Define a resolução padrão das figuras
sns.set_theme(style='darkgrid')  # Define o tema padrão do Seaborn

# Tratamento de warnings
from warnings import simplefilter

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
# Classificadores - Algoritmos de Machine Learning
from sklearn.linear_model import LogisticRegression
# Métricas e Visualização
from sklearn.metrics import classification_report, confusion_matrix
# Otimização dos hiperparâmetros
# Treino e teste
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
# Pipeline
from sklearn.pipeline import Pipeline
# Padronização, balanceamento e tratamentos
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)

simplefilter(action='ignore', category=FutureWarning)

# 2 - Acessando a base de dados
arquivo = 'https://raw.githubusercontent.com/wallacecarlis/arquivos_ml/refs/heads/main/Churn_Prediction_Telecom.csv'
df = pd.read_csv(arquivo)

# Calculando a utilização de memória do DataFrame
memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
print(f'Memória usada pelo DataFrame: {memory_usage.round(2)} MB')

# 3 - Verificando a quantidade de linhas e colunas
df.shape

# 4 - Verificando os tipos dos dados das colunas
df.info()

# 5 - Visualizando as primeiras linhas
df.head()

# 6 - Visualizando as últimas linhas
df.tail()

# 7 - Verificando o resumo estatístico
df.describe()

# 8 - Confirmando a quantidade de cada valor na coluna Senior Citizen
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(5, 4))
barras = sns.countplot(
    x='SeniorCitizen',
    data=df,
    ax=ax,
    palette='cividis',
    hue='SeniorCitizen',
    legend=False,
)

ax.set_title(
    'Quantidade por classe\nSenior Citizen',
    color='grey',
    fontsize=18,
    fontweight='bold',
    pad=10,
)

for i in barras.patches:
    ax.annotate(
        int(i.get_height()),
        xy=(
            i.get_x() + (i.get_width() / 2),
            i.get_height() - (i.get_width() * 200),
        ),
        color='white',
        fontweight='bold',
        ha='center',
        va='top',
        fontsize=14,
    )

for i, (nome, valor) in enumerate(
    zip(
        df.SeniorCitizen.value_counts().index,
        df.SeniorCitizen.value_counts().values,
    )
):
    ax.text(
        i,
        valor - valor + 200,
        f'Classe {nome}',
        ha='center',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )

ax.axis('off')
plt.tight_layout()

# 9 - Confirmando a quantidade de cada valor na coluna gender
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(5, 4))
barras = sns.countplot(
    x='gender', data=df, ax=ax, palette='cividis', hue='gender', legend=False
)

ax.set_title(
    'Quantidade por gender',
    color='grey',
    fontsize=18,
    fontweight='bold',
    pad=10,
)

for i in barras.patches:
    ax.annotate(
        int(i.get_height()),
        xy=(
            i.get_x() + (i.get_width() / 2),
            i.get_height() - (i.get_width() * 170),
        ),
        color='white',
        fontweight='bold',
        ha='center',
        va='top',
        fontsize=14,
    )

for i, (nome, valor) in enumerate(
    zip(df.gender.value_counts().index, df.gender.value_counts().values)
):
    ax.text(
        i,
        valor - valor + 200,
        nome,
        ha='center',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )

ax.axis('off')
plt.tight_layout()

# 10 - Confirmando a quantidade de cada valor na coluna Payment Method
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(10, 4))
barras = sns.countplot(
    y='PaymentMethod',
    data=df,
    ax=ax,
    order=df.PaymentMethod.value_counts().index,
    palette='cividis',
    hue='PaymentMethod',
    legend=False,
)

ax.set_title(
    'Quantidade por PaymentMethod',
    color='grey',
    fontsize=20,
    fontweight='bold',
    pad=10,
)

for i, (nome, valor) in enumerate(
    zip(
        df.PaymentMethod.value_counts().index,
        df.PaymentMethod.value_counts().values,
    )
):
    ax.text(
        valor - valor + 20,
        i,
        nome,
        ha='left',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )
    ax.text(
        valor - 20,
        i,
        valor,
        ha='right',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )

ax.axis('off')
plt.tight_layout()

# 11 - Confirmando a quantidade de cada valor na coluna Contract
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(10, 4))
barras = sns.countplot(
    y='Contract',
    data=df,
    ax=ax,
    order=df.Contract.value_counts().index,
    palette='cividis',
    hue='Contract',
    legend=False,
)

ax.set_title(
    'Quantidade por Contract',
    color='grey',
    fontsize=20,
    fontweight='bold',
    pad=10,
)

for i, (nome, valor) in enumerate(
    zip(df.Contract.value_counts().index, df.Contract.value_counts().values)
):
    ax.text(
        valor - valor + 20,
        i,
        nome,
        ha='left',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )
    ax.text(
        valor - 20,
        i,
        valor,
        ha='right',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )

ax.axis('off')
plt.tight_layout()

# 12 - Gerando uma cópia da base de dados
df_clean = df.copy()

# 13 - Criando uma nova coluna para receber valores do tipo "float"
df_clean['TotalCharges'] = pd.to_numeric(df_clean.iloc[:, 19], errors='coerce')
df_clean['TotalCharges'].isnull().sum()

# 14 - Inserindo valores medianos aos valores ausentes na coluna Total Charges, e transformando-os em float
valores_medianos = SimpleImputer(missing_values=np.nan, strategy='median')
df_clean['TotalCharges'] = valores_medianos.fit_transform(
    df_clean.iloc[:, 19].values.reshape(-1, 1)
).astype(float)

# 15 - Verificando se existem valores nulos
df_clean['TotalCharges'].isnull().sum()

# 16 - Verificando novamente o resumo estatístico após os tratamentos
df_clean.describe().round(1)

# 17 - Visualizando o boxplot da coluna Total Charges
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 4))
ax = sns.boxplot(
    x=df_clean['TotalCharges'], orient='h', linewidth=2.5, color='#A69D75'
)
ax.set_title(
    'Boxplot - Total Charges',
    fontsize=22,
    color='grey',
    fontweight='semibold',
    pad=20,
)
ax.tick_params(axis='x', labelsize=14, colors='darkgrey')
ax.set_xlabel('')
plt.tight_layout()

# 18 - Visualizando o boxplot da coluna Total Charges
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 4))
ax = sns.boxplot(
    x=df_clean['MonthlyCharges'], orient='h', linewidth=2.5, color='#A69D75'
)
ax.set_title(
    'Boxplot - Monthly Charges',
    fontsize=22,
    color='grey',
    fontweight='semibold',
    pad=20,
)
ax.tick_params(axis='x', labelsize=14, colors='darkgrey')
ax.set_xlabel('')
plt.tight_layout()

# 19 - Visualizando a quantidade de Churn
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(5, 4))
barras = sns.countplot(
    x='Churn',
    data=df_clean,
    ax=ax,
    palette='cividis',
    hue='Churn',
    legend=False,
)

ax.set_title(
    'Quantidade por Churn',
    color='grey',
    fontsize=18,
    fontweight='bold',
    pad=10,
)

for i in barras.patches:
    ax.annotate(
        int(i.get_height()),
        xy=(
            i.get_x() + (i.get_width() / 2),
            i.get_height() - (i.get_width() * 170),
        ),
        color='white',
        fontweight='bold',
        ha='center',
        va='top',
        fontsize=14,
    )

for i, (nome, valor) in enumerate(
    zip(
        df_clean.Churn.value_counts().index,
        df_clean.Churn.value_counts().values,
    )
):
    ax.text(
        i,
        valor - valor + 200,
        nome,
        ha='center',
        color='white',
        fontsize=14,
        fontweight='bold',
        va='center',
    )

ax.axis('off')
plt.tight_layout()

# 20 - Visualizando a quantidade de clientes por contrato e cancelamento

# Configurando a base de dados
df_melted = (
    df.groupby(['Churn', 'Contract']).size().reset_index(name='Clientes')
)

# Definindo a estrutura e paleta
sns.set_style('white')
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
colors = sns.color_palette(
    'cividis', n_colors=len(df_melted['Contract'].unique())
)

# Plotando a linha de inclinação
for i, contract in enumerate(df_melted['Contract'].unique()):
    valores = df_melted[df_melted['Contract'] == contract]['Clientes'].values
    ax.plot(
        [0, 1],
        valores,
        marker='o',
        color=colors[i],
        label=contract,
        linewidth=4.5,
    )

# Plotando título e anotações
ax.set_title(
    'Quantidade de Clientes por Contrato e Cancelamento - Churn',
    color='grey',
    fontsize=26,
    fontweight='bold',
    pad=50,
    ha='center',
)

ax.text(
    -0.1,
    df_melted['Clientes'].max() * 1.15,
    'Clientes que permaneceram',
    color='darkgrey',
    fontsize=13,
    fontweight='semibold',
)

ax.text(
    0.75,
    df_melted['Clientes'].max() * 1.15,
    'Clientes que cancelaram',
    color='darkgrey',
    fontsize=13,
    fontweight='semibold',
)

# Plotando as informações da legenda
df_filtered = df_melted[df_melted['Churn'] == 'No']
primeiros = (
    df_filtered.groupby('Contract')['Clientes'].nlargest(1).reset_index()
)

ax.text(
    -0.25,
    primeiros.Clientes[0] * 0.985,
    primeiros.Contract[0],
    color='#4c556c',
    fontsize=15,
    fontweight='semibold',
    ha='center',
)

ax.text(
    -0.25,
    primeiros.Clientes[2] * 0.985,
    primeiros.Contract[2],
    color='#b1a570',
    fontsize=15,
    fontweight='semibold',
    ha='left',
)

ax.text(
    -0.25,
    primeiros.Clientes[1] * 0.985,
    primeiros.Contract[1],
    color='#6c6e72',
    fontsize=15,
    fontweight='semibold',
    ha='left',
)

# Plotando os valores
for i, contrato in enumerate(df_melted['Contract'].unique()):
    valores = df_melted[df_melted['Contract'] == contrato]['Clientes'].values
    ax.text(
        -0.02,
        valores[0],
        f'{valores[0]}',
        va='center',
        ha='right',
        fontsize=14,
        color=colors[i],
        fontweight='bold',
    )
    ax.text(
        1.02,
        valores[1],
        f'{valores[1]}',
        va='center',
        ha='left',
        fontsize=14,
        color=colors[i],
        fontweight='bold',
    )

ax.axis('off')
plt.tight_layout()
# plt.savefig("QTD_CHURN.jpeg", bbox_inches = "tight")

# 21 - Visualizando a quantidade de clientes por churn, contrato e gênero
ccg = df.groupby(['Churn', 'Contract', 'gender']).size().unstack()

sns.set_style('dark')
fig, ax = plt.subplots(figsize=(12, 6))
colors = {'Male': '#4c556c', 'Female': '#b1a570'}
barras = ccg.plot(
    kind='barh',
    ax=ax,
    color=[colors[col] for col in df['gender'].unique()],
    width=0.9,
    linewidth=2,
)

for i in barras.containers:
    ax.bar_label(
        i,
        fmt='%.0f',
        padding=5,
        fontsize=13,
        color='#6c6e72',
        fontweight='bold',
    )

ax.set_title(
    'Quantidade de clientes por tipo de churn, contrato e gênero',
    fontsize=24,
    fontweight='bold',
    color='grey',
    pad=20,
)

ax.set_yticklabels(
    [i.get_text() for i in ax.get_yticklabels()], fontweight='bold'
)
ax.tick_params(axis='y', labelsize=15, colors='darkgrey')
ax.set_ylabel('')

ax.text(
    ccg.Male[0] * 0.92,
    5.2,
    'Gender',
    color='#6c6e72',
    fontsize=16,
    fontweight='semibold',
)
ax.text(
    ccg.Male[0] * 0.92,
    4.95,
    'Male',
    color='#4c556c',
    fontsize=14,
    fontweight='semibold',
)
ax.text(
    ccg.Male[0] * 0.92,
    4.7,
    'Female',
    color='#b1a570',
    fontsize=14,
    fontweight='semibold',
)

ax.legend().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.tight_layout()

# 22 - Visualização da distribuição da coluna tenure
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(12, 5))
bins = 25
ax = sns.histplot(df_clean.tenure, lw=5, bins=bins, color='#4c556c')
ax.set_title(
    'Tenure - qtd por período mensal como cliente',
    fontsize=24,
    color='grey',
    fontweight='semibold',
    pad=20,
)
ax.tick_params(axis='x', labelsize=14, colors='darkgrey')
ax.tick_params(axis='y', labelsize=14, colors='darkgrey')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(True, axis='y', linestyle='--', color='darkgrey', alpha=0.3)
plt.tight_layout()

# 23 - Visualização da distribuição da coluna Monthly Charges
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(12, 5))
bins = 25
ax = sns.histplot(df_clean.MonthlyCharges, lw=5, bins=bins, color='#4c556c')
ax.set_title(
    'MonthlyCharges - qtd por pagamentos mensais',
    fontsize=24,
    color='grey',
    fontweight='semibold',
    pad=20,
)
ax.tick_params(axis='x', labelsize=14, colors='darkgrey')
ax.tick_params(axis='y', labelsize=14, colors='darkgrey')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(True, axis='y', linestyle='--', color='darkgrey', alpha=0.3)
plt.tight_layout()

# 24 - Visualização da distribuição da coluna Total Charges
sns.set_style('dark')
fig, ax = plt.subplots(figsize=(12, 5))
bins = 25
ax = sns.histplot(df_clean.TotalCharges, lw=5, bins=bins, color='#4c556c')
ax.set_title(
    'TotalCharges - qtd por pagamentos totais',
    fontsize=24,
    color='grey',
    fontweight='semibold',
    pad=20,
)
ax.tick_params(axis='x', labelsize=14, colors='darkgrey')
ax.tick_params(axis='y', labelsize=14, colors='darkgrey')
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(True, axis='y', linestyle='--', color='darkgrey', alpha=0.3)
plt.tight_layout()

# 25 - Excluindo a coluna customerID
df_clean.drop('customerID', axis=1, inplace=True)

# 26 - Visualizando a quantidade de valores únicos das colunas
np.unique(df_clean.select_dtypes('object').values)
print(f'Quantidade de valores únicos por coluna:\n{(df_clean.nunique())}\n')

# 27 - Gerando uma nova cópia da base de dados atualizada
df_ml = df_clean.copy()

# 28 - Separando as variáveis em binário, numérico e categórico

# para binário: apenas colunas que contenham 2 valores
binario = df_ml.nunique()[df_ml.nunique() == 2].keys().tolist()
binario.remove('Churn')

# para numérico: apenas colunas que contenham valores numéricos ("int" e "float")
numerico = [
    col
    for col in df_ml.select_dtypes(['int', 'float']).columns.tolist()
    if col not in binario
]

# para categórico: as demais colunas que não se enquadrem nas duas condições acima
categorico = [
    col for col in df_ml.columns.tolist() if col not in binario + numerico
]
categorico.remove('Churn')

# 29 - Criando um objeto para o Label Encoding da variável alvo
le = LabelEncoder()
df_ml['Churn'] = le.fit_transform(df_ml.Churn)

# 30 - Separando a base total em treino e teste
train_ratio = 0.8
train_size = int(len(df_ml) * train_ratio)
df_shuffled = df_ml.sample(frac=1, random_state=42)
df_train = df_shuffled[:train_size]
df_test = df_shuffled[train_size:]

# 31 - Definindo X e y nas bases de treino e teste
X = df_train.drop(columns=['Churn'])
y = df_train.Churn

X_test = df_test.drop(columns=['Churn'])
y_test = df_test.Churn

# 32 - Separando a base de treino em treino e validação
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 33 - Pipelines para cada tipo de dado
binario_pipeline = Pipeline([('onehot', OneHotEncoder(drop='if_binary'))])

numerico_pipeline_standard = Pipeline([('scaler', StandardScaler())])

numerico_pipeline_minmax = Pipeline([('minmax', MinMaxScaler())])

categorico_pipeline = Pipeline(
    [('onehot', OneHotEncoder(handle_unknown='ignore'))]
)

# 34 - Column Transformer para preprocessamento para cada tipo de dado
preprocessor_standard = ColumnTransformer(
    [
        ('binario', binario_pipeline, binario),
        ('numerico', numerico_pipeline_standard, numerico),
        ('categorico', categorico_pipeline, categorico),
    ]
)

preprocessor_minmax = ColumnTransformer(
    [
        ('binario', binario_pipeline, binario),
        ('numerico', numerico_pipeline_minmax, numerico),
        ('categorico', categorico_pipeline, categorico),
    ]
)

# 35 - Dicionários de Pipelines com algoritmos, balanceamento e pré-processadores
model_pipelines = {
    'MLPClassifier': ImbPipeline(
        [
            ('preprocessor', preprocessor_minmax),
            ('rus', RandomUnderSampler(random_state=42)),
            (
                'classifier',
                MLPClassifier(
                    hidden_layer_sizes=(50,), max_iter=1000, random_state=42
                ),
            ),
        ]
    ),
    'LogisticRegression': ImbPipeline(
        [
            ('preprocessor', preprocessor_standard),
            ('rus', RandomUnderSampler(random_state=42)),
            ('classifier', LogisticRegression(max_iter=500, random_state=42)),
        ]
    ),
    'RandomForest': ImbPipeline(
        [
            ('preprocessor', preprocessor_standard),
            ('rus', RandomUnderSampler(random_state=42)),
            (
                'classifier',
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    ),
}

# 36 - Verificação do resultado da validação cruzada na base de treinamento
for name, pipeline in model_pipelines.items():
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=10, scoring='recall'
    )
    print(
        f'{name} - Recall Médio: {scores.mean():.4f} (+/- {scores.std():.4f})\n'
    )

# 37 - Visualização da matriz de confusão de cada modelo
for name, pipeline in model_pipelines.items():

    # Após a validação cruzada, ajustamos o modelo com o conjunto de treino completo
    pipeline.fit(X_train, y_train)

    # Fazendo previsões no conjunto de validação
    y_pred = pipeline.predict(X_valid)

    # Gerando a matriz de confusão
    cm = confusion_matrix(y_valid, y_pred, normalize='true')

    # Plotando a matriz de confusão
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=['Negativo', 'Positivo'],
        yticklabels=['Negativo', 'Positivo'],
    )
    ax.set_title(
        f'{name} - Matriz de Confusão',
        color='grey',
        fontsize=24,
        fontweight='bold',
        pad=18,
    )
    ax.set_xlabel(
        'Predições', color='darkgrey', fontsize=14, fontweight='semibold'
    )
    ax.tick_params(axis='x', labelsize=12, colors='grey')
    ax.set_ylabel('Real', color='darkgrey', fontsize=14, fontweight='semibold')
    ax.tick_params(axis='y', labelsize=12, colors='grey')
    plt.tight_layout()
    print()
    print()

    # 38 - Configurando o parâmetro n_estimators
param_grid_1 = {'classifier__n_estimators': [50, 100, 150, 200]}

grid_search_1 = GridSearchCV(
    model_pipelines['RandomForest'],
    param_grid_1,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=2,
)

grid_search_1.fit(X_train, y_train)

best_n_estimators = grid_search_1.best_params_['classifier__n_estimators']
print(f'Melhor n_estimators: {best_n_estimators}')

# 39 - Configurando o parâmetro max_depth
param_grid_2 = {'classifier__max_depth': [5, 10, 20, 30]}

pipeline_rf = model_pipelines['RandomForest']
pipeline_rf.set_params(classifier__n_estimators=best_n_estimators)

grid_search_2 = GridSearchCV(
    pipeline_rf, param_grid_2, cv=5, scoring='recall', n_jobs=-1, verbose=2
)

grid_search_2.fit(X_train, y_train)

best_max_depth = grid_search_2.best_params_['classifier__max_depth']
print(f'Melhor max_depth: {best_max_depth}')

# 40 - Configurando o parâmetro min_samples_split
param_grid_3 = {'classifier__min_samples_split': [2, 5, 10]}

pipeline_rf.set_params(
    classifier__n_estimators=best_n_estimators,
    classifier__max_depth=best_max_depth,
)

grid_search_3 = GridSearchCV(
    pipeline_rf, param_grid_3, cv=5, scoring='recall', n_jobs=-1, verbose=2
)

grid_search_3.fit(X_train, y_train)

best_min_samples_split = grid_search_3.best_params_[
    'classifier__min_samples_split'
]
print(f'Melhor min_samples_split: {best_min_samples_split}')

# 41 - Configurando o min_samples_leaf
param_grid_4 = {'classifier__min_samples_leaf': [1, 2, 4]}

pipeline_rf.set_params(
    classifier__n_estimators=best_n_estimators,
    classifier__max_depth=best_max_depth,
    classifier__min_samples_split=best_min_samples_split,
)

grid_search_4 = GridSearchCV(
    pipeline_rf, param_grid_4, cv=5, scoring='recall', n_jobs=-1, verbose=2
)

grid_search_4.fit(X_train, y_train)

best_min_samples_leaf = grid_search_4.best_params_[
    'classifier__min_samples_leaf'
]
print(f'Melhor min_samples_leaf: {best_min_samples_leaf}')

# 42 - Configurando o max_features
param_grid_5 = {'classifier__max_features': ['sqrt', 'log2']}

pipeline_rf.set_params(
    classifier__n_estimators=best_n_estimators,
    classifier__max_depth=best_max_depth,
    classifier__min_samples_split=best_min_samples_split,
    classifier__min_samples_leaf=best_min_samples_leaf,
)

grid_search_5 = GridSearchCV(
    pipeline_rf, param_grid_5, cv=5, scoring='recall', n_jobs=-1, verbose=2
)

grid_search_5.fit(X_train, y_train)

best_max_features = grid_search_5.best_params_['classifier__max_features']
print(f'Melhor max_features: {best_max_features}')

# 43 - Configurando o class_weight
param_grid_6 = {'classifier__class_weight': [None, 'balanced']}

# Atualizando o pipeline com os melhores parâmetros encontrados até agora
pipeline_rf.set_params(
    classifier__n_estimators=best_n_estimators,
    classifier__max_depth=best_max_depth,
    classifier__min_samples_split=best_min_samples_split,
    classifier__min_samples_leaf=best_min_samples_leaf,
    classifier__max_features=best_max_features,
)

# Realizando a busca em grade para o parâmetro class_weight
grid_search_6 = GridSearchCV(
    pipeline_rf,
    param_grid_6,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1,  # Reduzido para evitar excesso de logs
)

# Ajustando o modelo e obtendo o melhor parâmetro
grid_search_6.fit(X_train, y_train)
best_class_weight = grid_search_6.best_params_['classifier__class_weight']
print(f'Melhor class_weight: {best_class_weight}')

# 44 - Configurando o random forest com os melhores parâmetros
best_rf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    max_features=best_max_features,
    class_weight=best_class_weight,
    random_state=42,
)

# Criando o pipeline final
final_pipeline = ImbPipeline(
    [
        ('preprocessor', preprocessor_standard),
        ('rus', RandomUnderSampler(random_state=42)),
        ('classifier', best_rf),
    ]
)

# Treinando o modelo final
final_pipeline.fit(X_train, y_train)

# Fazendo previsões na base de validação
y_pred_valid = final_pipeline.predict(X_valid)

# Exibindo métricas
print('Avaliação na base de treino:')
print(classification_report(y_valid, y_pred_valid))

# 45 - Treinamento Final e Avaliação na Base de Teste
y_pred_test = final_pipeline.predict(X_test)

print('Avaliação na Base de Teste')
print(classification_report(y_test, y_pred_test))

# 46 - Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred_test, normalize='true')

# Exibindo a matriz de confusão
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
ax = sns.heatmap(
    cm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=['Negativo', 'Positivo'],
    yticklabels=['Negativo', 'Positivo'],
)
ax.set_title(
    f'{name} Base de Teste - Matriz de Confusão',
    color='grey',
    fontsize=22,
    fontweight='bold',
    pad=18,
)
ax.set_xlabel(
    'Predições', color='darkgrey', fontsize=14, fontweight='semibold'
)
ax.tick_params(axis='x', labelsize=12, colors='grey')
ax.set_ylabel('Real', color='darkgrey', fontsize=14, fontweight='semibold')
ax.tick_params(axis='y', labelsize=12, colors='grey')
plt.tight_layout()
