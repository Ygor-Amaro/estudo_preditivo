import json
import os
import random
import pandas as pd

"""
Script para geração de dados fictícios baseados em pesos extraídos de um arquivo JSON.

Este script realiza as seguintes etapas:
1. Carrega os dados de um arquivo JSON contendo informações sobre cursos, filiais, segmentos, turnos e períodos.
2. Calcula os pesos para cada combinação de curso, filial, segmento, turno e período.
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

# Função para extrair todas as combinações possíveis e seus pesos de ATIVOS/EVADIDOS, incluindo PERÍODO
def extrair_combinacoes(dados):
    combinacoes = []
    for curso, filiais in dados["CURSOS"].items():
        for filial, modalidades in filiais.items():
            for modalidade, turnos in modalidades.items():
                for turno, periodos in turnos.items():
                    for periodo, churns in periodos.items():
                        if isinstance(churns, dict):
                            ativos = churns.get("ATIVOS", 0)
                            evadidos = churns.get("EVADIDOS", 0)
                            if ativos > 0:
                                combinacoes.append({
                                    "CURSOS": curso,
                                    "FILIAL": filial,
                                    "SEGMENTO": modalidade,
                                    "TURNO": turno,
                                    "PERIODO": int(periodo) if periodo.isdigit() else periodo,
                                    "ATIVO/EVADIDO": "Ativo",
                                    "peso": ativos
                                })
                            if evadidos > 0:
                                combinacoes.append({
                                    "CURSOS": curso,
                                    "FILIAL": filial,
                                    "SEGMENTO": modalidade,
                                    "TURNO": turno,
                                    "PERIODO": int(periodo) if periodo.isdigit() else periodo,
                                    "ATIVO/EVADIDO": "Evadido",
                                    "peso": evadidos
                                })
    return combinacoes

# Extrai todas as combinações possíveis
combinacoes = extrair_combinacoes(dados)

# Prepara listas para random.choices
labels = [
    (
        c["CURSOS"],
        c["FILIAL"],
        c["SEGMENTO"],
        c["TURNO"],
        c["PERIODO"],
        c["ATIVO/EVADIDO"]
    )
    for c in combinacoes
]
pesos = [c["peso"] for c in combinacoes]

# Gera as linhas de acordo com os pesos de cada combinação
selecionados = random.choices(labels, weights=pesos, k=num_rows)

# Gera o campo SEXO aleatório
sexos = random.choices(
    ["F", "M", "Sem identificação"],
    weights=[0.69640, 0.29512, 0.00847],
    k=num_rows
)

# Monta o DataFrame
data = {
    "RM": [f"CUST-{i:05d}" for i in range(1, num_rows + 1)],
    "CURSOS": [s[0] for s in selecionados],
    "FILIAL": [s[1] for s in selecionados],
    "SEGMENTO": [s[2] for s in selecionados],
    "TURNO": [s[3] for s in selecionados],
    "PERIODO": [s[4] for s in selecionados],
    "ATIVO/EVADIDO": [s[5] for s in selecionados],
    "SEXO": sexos
}

df = pd.DataFrame(data)

# Salvando os dados em um arquivo CSV na pasta data/output
output_file = "data/output/dados_ficticios.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Garante que o diretório exista
df.to_csv(output_file, index=False)

print(f"Arquivo '{output_file}' gerado com sucesso!")