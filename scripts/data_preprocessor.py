"""
Author: Jesús Maldonado
Description: Script para preprocesar datos de EEG y crear conjuntos de entrenamiento y test.

data_preprocessor.py llama a sets_creator.py al final del proceso, por lo que bastará con ejecutar el primero para que se ejecute todo el flujo de preprocesado y creación de splits.

Flujo de ejecución:
1. Creación y preparación de directorios y archivos.
2. Búsqueda de archivos de resultados en la carpeta Local.
3. Procesamiento por cada sujeto.
4. Lectura y normalización de datos EEG (museDataX.csv).
5. Organización en sesiones y respuestas.
6. Cálculo de duraciones e identificación de ventanas.
7. Generación de *chunks* en cada ventana.
8. Registro de información en splits.csv y contadores.
9. Creación de subjects.csv.
10. Ejecución automática de sets_creator.py.
11. Llamada a sets_creator.py para crear los splits de entrenamiento y test:
   1. Descubrir los sujetos procesados.
   2. Separar sujetos en entrenamiento y test.
   3. Cargar y analizar el contenido de subjects.csv.
   4. Generar folds.
   5. Reescribir splits.csv.

Al final tendrás todos los datos EEG normalizados, troceados y clasificados por sujeto, sesión, respuesta y ventana, además de los archivos .csv que resumen el contenido y permiten trabajar directamente con los conjuntos de entrenamiento, validación y test.
"""

from configparser import ConfigParser
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
from sets_creator import SetsCreator

ROOT_BASE_PATH = ".."   # Relative to this file folder
CONFIG_INI_FILE_RELATIVE_PATH = "config.ini"   # Relative to ROOT_BASE_PATH

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_BASE_PATH)
CONFIG_INI_FILE = os.path.join(ROOT_PATH, os.path.normpath(CONFIG_INI_FILE_RELATIVE_PATH))

config = ConfigParser()
config.read(CONFIG_INI_FILE)

WINDOWS = list(map(int, config['Data']['window_sizes'].split(',')))

LOCAL_PATH = os.path.join(ROOT_PATH, os.path.normpath(config['Paths']['local_raw_data_path']))
MUSE_PATH = os.path.join(ROOT_PATH, os.path.normpath(config['Paths']['muse_raw_data_path']))
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, os.path.normpath(config['Paths']['processed_data_path']))
SPLITS_FILE = os.path.join(ROOT_PATH, os.path.normpath(config['Paths']['splits_file']))
SUBJECTS_FILE = os.path.join(ROOT_PATH, os.path.normpath(config['Paths']['subjects_file']))

responses_count = {}

def count_responses(subject, window):
    if subject not in responses_count:
        responses_count[subject] = [subject, 0] + [0]*len(WINDOWS)
    responses_count[subject][1] += 1
    responses_count[subject][WINDOWS.index(window)+2] += 1

def write_subjects_file():
    with open(SUBJECTS_FILE, 'w') as file:
        file.write("Subject,TotalResponsesNum")
        for window in WINDOWS:
            file.write(f",{window}msResponsesCount")
        file.write("\n")
        for key in responses_count:
            file.write(",".join(map(str, responses_count[key])) + "\n")

def init_splits_file():
    with open(SPLITS_FILE, 'w') as file:
        file.write("Subject,Session,Response,Window (ms),ChunksCount,Training,Fold\n")
        

def add_splits_row(subject_num, session_num, response_num, window, chunks_num):
    with open(SPLITS_FILE, 'a') as file:
        file.write(f"{subject_num},{session_num},{response_num},{window},{chunks_num},,\n")

def load_museData(file_path):

    data_slice = pd.read_csv(file_path, low_memory=False)
    
    # Relevant columns
    selected_columns = ['TimeStamp','Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10']
    data_slice = data_slice[selected_columns]
    
    data_slice['TimeStamp'] = pd.to_datetime(data_slice['TimeStamp'])
    data_slice.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    data_slice_cleaned = data_slice.dropna()
    timestamp_column = data_slice_cleaned['TimeStamp']
    features = data_slice_cleaned.drop(columns=['TimeStamp'])

    # Data normalization
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)

    data_slice_normalized = pd.DataFrame(features_normalized, columns=features.columns)
    data_slice_normalized.insert(0, 'TimeStamp', timestamp_column.reset_index(drop=True))

    print(data_slice_normalized.describe())

    return data_slice_normalized

def extract_number_from_filename(filename: str) -> int:
    match = re.search(r"\d+", filename)
    assert match is not None, f"No se encontró número en el nombre del archivo: {filename}"
    return int(match.group())


if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Create directories and files structure
    print("Creating directories and files structure...")

    RESULTS_FILES = sorted([file for file in os.listdir(LOCAL_PATH) if re.match(r"^results\d+\.csv$", file)], key=extract_number_from_filename)

    init_splits_file()

    for file in RESULTS_FILES:
        results = pd.read_csv(f"{LOCAL_PATH}/{file}")
        
        subject_num = results["ID del participante"].unique()[0]
        print(f"Creating structure for subject {subject_num}...")
        subject_path = f"{PROCESSED_DATA_PATH}/subject{subject_num}"
        os.makedirs(subject_path, exist_ok=True)

        museData = load_museData(f"{MUSE_PATH}/museData{subject_num}.csv")

        sessions_count = results["Trial"].max()+1
        for session_num in range(sessions_count):
            print(f"Creating structure for subject {subject_num}, session {session_num}...")
            session_path = f"{subject_path}/session{session_num}"
            os.makedirs(session_path, exist_ok=True)

            num_respuestas = results[results["Trial"] == session_num]["Respuesta"].max()+1
            for response_num in range(num_respuestas):
                print(f"Creating structure for subject {subject_num}, session {session_num}, response {response_num}...")
                response_path = f"{session_path}/response{response_num}"
                os.makedirs(response_path, exist_ok=True)
                
                # Here we could also take the "Tiempo de aparición de letras", but there are:
                # 3 cases where the time window is negative,
                # 6 cases where it is less than 100ms,
                # 11 cases where it is less than 250ms and
                # 40 cases where it is less than 500ms
                start = pd.to_datetime(results[(results["Trial"] == session_num) & (results["Respuesta"] == response_num)]["Tiempo de inicio"].iloc[0])
                end = pd.to_datetime(results[(results["Trial"] == session_num) & (results["Respuesta"] == response_num)]["Tiempo de la pulsación"].iloc[0])
                
                duration = round((end - start).total_seconds()*1000)

                # If the duration is negative, it will be ignored
                if duration < 0:
                    continue

                for window in WINDOWS:
                    if window > duration:
                        continue
                    print(f"Creating structure for subject {subject_num}, session {session_num}, response {response_num}, window {window}ms...")
                    window_path = f"{response_path}/{window}ms"
                    os.makedirs(window_path, exist_ok=True)

                    chunks_num = int(duration/window)
                    for i in range(chunks_num):
                        chunk_start = end-pd.Timedelta(window*(chunks_num-i), unit="ms")
                        chunk_end = end-pd.Timedelta(window*(chunks_num-i-1), unit="ms")

                        museData_chunk = museData[(museData["TimeStamp"] >= chunk_start) & (museData["TimeStamp"] < chunk_end)]
                        museData_chunk.to_csv(f"{window_path}/chunk{i}.csv", index=False)
                    
                    add_splits_row(subject_num, session_num, response_num, window, chunks_num)
                    count_responses(subject_num, window)

    write_subjects_file()

    sets_creator = SetsCreator(ROOT_BASE_PATH, CONFIG_INI_FILE_RELATIVE_PATH)
    sets_creator.create_sets()
