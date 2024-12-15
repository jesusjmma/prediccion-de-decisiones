import os
import random
from datetime import datetime

# Variables a elegir por el usuario
TRAINING_DATA_RATIO = 0.8
TOTAL_FOLDS = 3
WINDOWS = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000]
SPLITS_FILE_NAME = "splits"
SEED = 83504

random.seed(SEED) # Semilla fijada para poder reproducir los resultados
script_path = os.path.dirname(os.path.abspath(__file__))
splits_file = os.path.join(script_path, "..", f"{SPLITS_FILE_NAME}.csv")

def split_training_test_data():
    processed_data_path = os.path.join(script_path, "..", "data", "processed")
    subjects = [int(name.replace('sujeto','')) for name in os.listdir(processed_data_path) if os.path.isdir(os.path.join(processed_data_path, name))]
    total_subjects = len(subjects)

    training_sujetos_size = int(total_subjects * TRAINING_DATA_RATIO)
    test_sujetos_size = total_subjects - training_sujetos_size

    training_test_set = training_sujetos_size*[1] + test_sujetos_size*[0]
    random.shuffle(training_test_set)

    training_test_dict = {sujeto: training_test for sujeto, training_test in zip(subjects, training_test_set)}   
    return total_subjects, training_test_dict

def generate_folds_set(training_data_size_per_window):

    folds_set = {ventana: [] for ventana in WINDOWS}
    
    for ventana in WINDOWS:
        folds_size = training_data_size_per_window[ventana] // TOTAL_FOLDS
        remaining_folds_size = training_data_size_per_window[ventana] % TOTAL_FOLDS
        folds_sizes = [folds_size]*TOTAL_FOLDS
        for i in range(remaining_folds_size):
            folds_sizes[i] += 1

        folds_set[ventana] = []
        for i in range(TOTAL_FOLDS):
            folds_set[ventana].extend([i]*folds_sizes[i])
        random.shuffle(folds_set[ventana])
    
    return folds_set

def split_data_into_folds(total_subjects, training_test_dict):
    
    subjects_file = os.path.join(script_path, "..", "sujetos.csv")
    with open(subjects_file, 'r') as file:
        text = file.readlines()

    subject_responses_count = []
    for i in range(1, len(text)-1):
        subject_responses_count.append(text[i].strip().split(','))

    training_data_size_per_window = {ventana: 0 for ventana in WINDOWS}

    for i in range(len(WINDOWS)):
        for j in range(total_subjects):
            if training_test_dict[int(subject_responses_count[j][0])] == 1:
                training_data_size_per_window[WINDOWS[i]] += int(subject_responses_count[j][i+2])

    folds_set = generate_folds_set(training_data_size_per_window)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_splits_file = os.path.join(script_path, "..", f"{SPLITS_FILE_NAME}_{timestamp}.csv")


    with open(splits_file, 'r') as file, open(temp_splits_file, 'w') as temp_file:
        first_line = file.readline()
        temp_file.write(first_line)
        
        for line in file:
            subject, session, response, window, chunk_count, _, _ = line.strip().split(',')
            subject = int(subject)
            is_training = training_test_dict[subject]
            fold = -1 if is_training == 0 else folds_set[int(window)].pop()
            temp_file.write(f"{subject},{session},{response},{window},{chunk_count},{is_training},{fold}\n")

    os.replace(temp_splits_file, splits_file)

if __name__ == "__main__":
    total_subjects, training_test_dict = split_training_test_data()
    split_data_into_folds(total_subjects, training_test_dict)

    print("Datos categorizados correctamente.\n")
    print(f"Se ha creado un archivo {SPLITS_FILE_NAME}.csv' con los datos categorizados en:")
    print(f"- {int(TRAINING_DATA_RATIO*100)}% entrenamiento")
    print(f"- {100-int(TRAINING_DATA_RATIO*100)}% test\n")
    print(f"Y los datos categorizados como de 'entrenamiento' se han dividido en {TOTAL_FOLDS} folds con aproximadamente un {100//TOTAL_FOLDS}% de los datos cada uno.")
