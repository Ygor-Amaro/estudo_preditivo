import glob
import os
import pandas as pd
from typing import List
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def extract_csv_files(path: str) -> List[pd.DataFrame]:
    # Verifica se o caminho existe
    if not os.path.exists(path):
        raise FileNotFoundError(f"O caminho especificado não existe: {path}")
    
    # Busca todos os arquivos CSV no diretório
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"Não foram encontrados arquivos CSV no caminho: {path}")
    
    dataframe_list = []
    for file in all_files:
        try:
            # Detecta a codificação do arquivo
            encoding = detect_encoding(file)
            # Carrega o arquivo CSV com a codificação detectada
            df = pd.read_csv(file, sep=",", encoding=encoding, on_bad_lines='skip')
            dataframe_list.append(df)
        except Exception as e:
            print(f"Erro ao carregar o arquivo {file}: {e}")
    
    # Verifica se algum arquivo foi carregado com sucesso
    if not dataframe_list:
        raise ValueError("Nenhum arquivo CSV foi carregado com sucesso.")
    
    return dataframe_list

# Caminho para os arquivos CSV
path_1 = os.path.abspath("TOTVS/data/input/Alunos")

# Extração dos arquivos CSV
Lista_csv_Alunos = extract_csv_files(path_1)
