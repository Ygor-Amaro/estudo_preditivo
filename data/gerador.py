import pandas as pd
import random
import numpy as np

# Configurações para geração de dados
num_rows = 60000  # Número de linhas de dados fictícios
random.seed(65432)  # Para reprodutibilidade

# Geração de dados fictícios com porcentagens ajustadas
data = {
    "RM": [f"CUST-{i:05d}" for i in range(1, num_rows + 1)],
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
        weights=[0.69950, 0.29213, 0.00837], k=num_rows),
    "MultipleLines": random.choices(["Yes", "No", "No phone service"], weights=[40, 50, 10], k=num_rows),
    "InternetService": random.choices(["DSL", "Fiber optic", "No"], weights=[35, 50, 15], k=num_rows),
    "OnlineSecurity": random.choices(["Yes", "No", "No internet service"], weights=[25, 60, 15], k=num_rows),
    "OnlineBackup": random.choices(["Yes", "No", "No internet service"], weights=[30, 55, 15], k=num_rows),
    "DeviceProtection": random.choices(["Yes", "No", "No internet service"], weights=[28, 57, 15], k=num_rows),
    "TechSupport": random.choices(["Yes", "No", "No internet service"], weights=[20, 65, 15], k=num_rows),
    "StreamingTV": random.choices(["Yes", "No", "No internet service"], weights=[40, 45, 15], k=num_rows),
    "StreamingMovies": random.choices(["Yes", "No", "No internet service"], weights=[42, 43, 15], k=num_rows),
    "Contract": random.choices(["Month-to-month", "One year", "Two year"], weights=[55, 25, 20], k=num_rows),
    "PaperlessBilling": random.choices(["Yes", "No"], weights=[60, 40], k=num_rows),
    "PaymentMethod": random.choices(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        weights=[35, 20, 25, 20],
        k=num_rows,
    ),
    "MonthlyCharges": np.round(np.random.uniform(18.25, 118.75, num_rows), 2),
    "TotalCharges": lambda: np.round(
        np.random.uniform(18.25, 8684.8, num_rows), 2
    ),
    "Churn": random.choices(["Yes", "No"], weights=[27, 73], k=num_rows),  # 27% de churn
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Corrigindo TotalCharges para clientes com tenure = 0
df.loc[df["tenure"] == 0, "TotalCharges"] = 0

# Salvando os dados em um arquivo CSV
output_file = "dados_ficticios.csv"
df.to_csv(output_file, index=False)

print(f"Arquivo '{output_file}' gerado com sucesso!")