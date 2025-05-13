import pandas as pd
import random
import numpy as np
import json
import os

"""
Script para geração de dados fictícios baseados em pesos extraídos de um arquivo JSON.

Este script realiza as seguintes etapas:
1. Carrega os dados de um arquivo JSON contendo informações sobre cursos, filiais, segmentos e turnos.
2. Calcula os pesos para cada combinação de curso, filial, segmento e turno.
3. Gera dados fictícios com base nos pesos calculados.
4. Salva os dados gerados em um arquivo CSV.

Configurações:
- Número de linhas de dados fictícios gerados: 60.000
- Arquivo de entrada: data/input/dados.json
- Arquivo de saída: data/output/dados_ficticios.csv
"""

# Configurações para geração de dados
num_rows = 60000  # Número de linhas de dados fictícios
random.seed(65432)  # Para reprodutibilidade

# Carregando os pesos de cursos a partir do JSON
with open("data/input/dados.json", "r", encoding="utf-8") as file:
    dados = json.load(file)

def calcular_pesos(dados):
    """
    Calcula os pesos para cada combinação de curso, filial, segmento e turno.

    Args:
        dados (dict): Dados carregados do arquivo JSON.

    Returns:
        tuple: Uma tupla contendo:
            - cursos (list): Lista de combinações no formato "Curso - Filial - Segmento - Turno".
            - pesos (list): Lista de pesos correspondentes a cada combinação.
    """
    cursos = []
    pesos = []

    for curso, filiais in dados["CURSOS"].items():
        peso_curso = 0  # Soma dos pesos do curso
        for filial, segmentos in filiais.items():
            peso_filial = 0  # Soma dos pesos da filial
            for segmento, turnos in segmentos.items():
                peso_segmento = 0  # Soma dos pesos do segmento
                if isinstance(turnos, dict):  # Verifica se turnos é um dicionário
                    for turno, peso in turnos.items():
                        if isinstance(peso, (int, float)):  # Verifica se o peso é numérico
                            # Adiciona o curso completo à lista
                            cursos.append(f"{curso} - {filial} - {segmento} - {turno}")
                            pesos.append(peso)
                            peso_segmento += peso  # Soma o peso do turno
                # Adiciona o peso do segmento à filial
                peso_filial += peso_segmento
            # Adiciona o peso da filial ao curso
            peso_curso += peso_filial
    return cursos, pesos

# Calculando os cursos e pesos
cursos, pesos = calcular_pesos(dados)

# Geração de dados fictícios com porcentagens ajustadas
cursos_selecionados = random.choices(cursos, weights=pesos, k=num_rows)  # Seleciona cursos com base nos pesos

data = {
    "RM": [f"CUST-{i:05d}" for i in range(1, num_rows + 1)],
    "CURSOS": [curso.split(" - ")[0] for curso in cursos_selecionados],  # Extrai apenas o nome do curso
    "FILIAL": [curso.split(" - ")[1] for curso in cursos_selecionados],  # Extrai a filial
    "SEGMENTO": [curso.split(" - ")[2] for curso in cursos_selecionados],  # Extrai o segmento
    "TURNO": [curso.split(" - ")[3] for curso in cursos_selecionados],  # Extrai o turno
    "ATIVO/EVADIDO": random.choices(
        ["Ativo", "Evadido", "Formado"], 
        weights=[0.78606, 0.14413, 0.06981], k=num_rows),
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