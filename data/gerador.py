import pandas as pd
import random
import numpy as np
import json
import os


# Configurações para geração de dados
num_rows = 60000  # Número de linhas de dados fictícios
random.seed(65432)  # Para reprodutibilidade

# Carregando os pesos de cursos a partir do JSON
with open("data/input/dados.json", "r", encoding="utf-8") as file:
    dados = json.load(file)

cursos = list(dados["CURSOS"].keys())
pesos = list(dados["CURSOS"].values())

# Geração de dados fictícios com porcentagens ajustadas
data = {
    "RM": [f"CUST-{i:05d}" for i in range(1, num_rows + 1)],
    "CURSOS": random.choices(cursos, weights=pesos, k=num_rows),
    "ATIVO/EVADIDO": random.choices(
        ["Ativo", "Evadido", "Formado"], 
        weights=[0.78606, 0.14413, 0.06981], k=num_rows),
    "TURNO": random.choices(
        ["Integral", "Integral - MS", "Matutino", "Noturno", "Turno EAD", "Turno EAD - VOT", "Vespertino"], 
        weights=[0.00044, 0.00449, 0.42913, 0.28739, 0.17588, 0.00556, 0.09711], k=num_rows),
    "SEGMENTO": random.choices(
        ["EAD", "INTEGRAL", "PHYGITAL", "PHYGITAL II", "PRESENCIAL"], 
        weights=[0.16450, 0.00449, 0.01694, 0.01626, 0.79781], k=num_rows),
    "FILIAL": random.choices(
        ["Cidade Jardim", "Paraíso", "Vila Mariana", "Votorantim"], 
        weights=[0.03814, 0.13970, 0.79934, 0.02282], k=num_rows),
    "PERIODO": random.choices(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        weights=[0.00008, 0.22421, 0.18466, 0.15668, 0.13278, 0.08074, 0.06665, 0.06046, 0.05988, 0.01458, 0.01928], k=num_rows),
    "SEXO": random.choices(
        ["F", "M", "Sem identificação"], 
        weights=[0.69950, 0.29213, 0.00837], k=num_rows)
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Salvando os dados em um arquivo CSV na pasta data/output
output_file = "data/output/dados_ficticios.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Garante que o diretório exista
df.to_csv(output_file, index=False)

print(f"Arquivo '{output_file}' gerado com sucesso!")