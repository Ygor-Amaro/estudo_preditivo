import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estudo_preditivo.transform_csv import dados
from sklearn.ensemble import RandomForestClassifier


print(dados.head())

