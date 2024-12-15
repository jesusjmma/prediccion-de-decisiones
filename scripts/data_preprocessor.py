import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re

ventanas = [100,250,500,750,1000,1500,2000,2500,3000,5000,7500,10000]

script_path = os.path.dirname(os.path.abspath(__file__))
origen_data_path = f"{script_path}/../data/Muse EEG Subconscious Decisions Dataset"
local_path = f"{origen_data_path}/Local"
muse_path = f"{origen_data_path}/Muse"
processed_data_path = f"{script_path}/../data/processed"
splits_file = f"{script_path}/../splits.csv"
sujetos_file = f"{script_path}/../sujetos.csv"
responses_count = {}

def count_responses(subject, window):
    if subject not in responses_count:
        responses_count[subject] = [subject, 0] + [0]*len(ventanas)
    responses_count[subject][1] += 1
    responses_count[subject][ventanas.index(window)+2] += 1

def write_sujetos_file():
    with open(sujetos_file, 'w') as file:
        file.write("Sujeto,NumRespuestasTotal")
        for ventana in ventanas:
            file.write(f",NumRespuestas{ventana}ms")
        file.write("\n")
        for key in responses_count:
            file.write(",".join(map(str, responses_count[key])) + "\n")

def init_control_file():
    with open(splits_file, 'w') as file:
        file.write("Sujeto,Sesion,Respuesta,Ventana (ms),NumTrozos,Entrenamiento,Fold\n")
        

def add_splits_row(sujeto, sesion, respuesta, ventana, num_trozos):
    with open(splits_file, 'a') as file:
        file.write(f"{sujeto},{sesion},{respuesta},{ventana},{num_trozos},,\n")

def load_museData(file_path):
    data_slice = pd.read_csv(file_path, low_memory=False)
    
    # Columnas relevantes
    selected_columns = ['TimeStamp','Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10']
    data_slice = data_slice[selected_columns]
    
    data_slice['TimeStamp'] = pd.to_datetime(data_slice['TimeStamp'])
    data_slice.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    data_slice_clean = data_slice.dropna()
    timestamp_column = data_slice_clean['TimeStamp']
    features = data_slice_clean.drop(columns=['TimeStamp'])

    # Normalizar los datos
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    data_slice_normalized = pd.DataFrame(features_normalized, columns=features.columns)
    data_slice_normalized.insert(0, 'TimeStamp', timestamp_column.reset_index(drop=True))

    print(data_slice_normalized.describe())

    return data_slice_normalized

if __name__ == "__main__":
    os.makedirs(processed_data_path, exist_ok=True)

    # Crear estructura de directorios
    print("Creando estructura de directorios y archivos...")

    results_files = sorted([file for file in os.listdir(local_path) if re.match(r"^results\d+\.csv$", file)], key=lambda x: int(re.search(r"\d+", x).group()))

    for file in results_files:
        results = pd.read_csv(f"{local_path}/{file}")
        
        num_sujeto = results["ID del participante"].unique()[0]
        print(f"Creando estructura para sujeto {num_sujeto}...")
        sujeto_path = f"{processed_data_path}/sujeto{num_sujeto}"
        os.makedirs(sujeto_path, exist_ok=True)

        museData = load_museData(f"{muse_path}/museData{num_sujeto}.csv")
        
        init_control_file()

        num_sesiones = results["Trial"].max()+1
        for num_sesion in range(num_sesiones):
            print(f"Creando estructura para sujeto {num_sujeto}, sesión {num_sesion}...")
            sesion_path = f"{sujeto_path}/sesion{num_sesion}"
            os.makedirs(sesion_path, exist_ok=True)

            num_respuestas = results[results["Trial"] == num_sesion]["Respuesta"].max()+1
            for num_respuesta in range(num_respuestas):
                print(f"Creando estructura para sujeto {num_sujeto}, sesión {num_sesion}, respuesta {num_respuesta}...")
                respuesta_path = f"{sesion_path}/respuesta{num_respuesta}"
                os.makedirs(respuesta_path, exist_ok=True)
                
                # Aquí también se podría coger el "Tiempo de aparición de letras", pero hay:
                # 3 casos en que la ventana de tiempo es negativa,
                # 6 casos en que es menor que 100ms,
                # 11 casos en que es menor que 250ms y
                # 40 casos en que es menor que 500ms
                inicio = pd.to_datetime(results[(results["Trial"] == num_sesion) & (results["Respuesta"] == num_respuesta)]["Tiempo de inicio"].iloc[0])
                fin = pd.to_datetime(results[(results["Trial"] == num_sesion) & (results["Respuesta"] == num_respuesta)]["Tiempo de la pulsación"].iloc[0])
                
                duracion = round((fin - inicio).total_seconds()*1000)

                # Si la duración es negativa, no se tiene en cuenta
                if duracion < 0:
                    continue

                for ventana in ventanas:
                    if ventana > duracion:
                        continue
                    print(f"Creando estructura para sujeto {num_sujeto}, sesión {num_sesion}, respuesta {num_respuesta}, ventana {ventana}ms...")
                    ventana_path = f"{respuesta_path}/{ventana}ms"
                    os.makedirs(ventana_path, exist_ok=True)

                    trozos = int(duracion/ventana)
                    for i in range(trozos):
                        inicio_trozo = fin-pd.Timedelta(ventana*(trozos-i), unit="ms")
                        fin_trozo = fin-pd.Timedelta(ventana*(trozos-i-1), unit="ms")

                        museData_trozo = museData[(museData["TimeStamp"] >= inicio_trozo) & (museData["TimeStamp"] < fin_trozo)]
                        museData_trozo.to_csv(f"{ventana_path}/trozo{i}.csv", index=False)
                    
                    add_splits_row(num_sujeto, num_sesion, num_respuesta, ventana, trozos)
                    count_responses(num_sujeto, ventana)

    write_sujetos_file()
