import os
import pandas as pd
from src.estudo_preditivo.transform_csv import dados

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


print(type(dados))

# y = dados['ATIVO/EVADIDO']
# x = dados[['RM', 'CURSOS', 'FILIAL', 'PERIODO', 'SEXO']]

# SEED = 6518

# treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
#                                                         random_state = SEED,
#                                                         stratify = y)

# print(f'treinamento realizado com {len(treino_x)} elementos')
# print(f'testaremos realizado com {len(teste_x)} elementos')

# modelo = LinearSVC()
# modelo.fit(treino_x, treino_y)
# previsoes = modelo.predict(teste_x)

# acuracia = accuracy_score(teste_y, previsoes) * 100
# print(f'Acuracia do modelo foi de {round(acuracia,1)}%')