import os
import pandas as pd
import numpy as np


# Construindo o caminho absoluto para o arquivo:
base_dir = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(base_dir, '..', '..', 'data', 'output', 'dados_ficticios.csv')

# Verificando se o arquivo existe:
if not os.path.exists(caminho):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

# Carregando o arquivo CSV
dados = pd.read_csv(caminho, encoding='utf-8')

# Função para criar a coluna 'Churn'
def create_coluna_churn(df: pd.DataFrame) -> pd.DataFrame:
    if 'ATIVO/EVADIDO' not in df.columns:
        raise KeyError("A coluna 'ativo/evadido' não foi encontrada no DataFrame.")
    status_map = {
        'ATIVO': 1,
        'EVADIDO': 0,
    }
    df['CHURN'] = df['ATIVO/EVADIDO'].map(status_map)
    return df

# Aplicando as transformações
dados = create_coluna_churn(dados)

# Exibindo as primeiras linhas do DataFrame transformado
# Contando quantas vezes "ARQUITETURA E URBANISMO" aparece na coluna "CURSOS"
arquitetura = dados.query('CURSOS = "ARQUITETURA E URBANISMO"').head()
arquitetura
