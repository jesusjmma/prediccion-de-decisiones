"""
Author: Jesús Maldonado
Description: Script complementario para crear conjuntos de entrenamiento y test a partir de datos EEG procesados.
"""

from configparser import ConfigParser
import os
import random
from datetime import datetime

class SetsCreator:
    """Creates training and test sets for EEG data processing.
    This class is responsible for splitting the data into training and test sets based on the configuration provided in a .ini file. It also generates folds for cross-validation.

    Args:
        ROOT_BASE_PATH (str): Base path for the project. Default is "..".
        CONFIG_INI_FILE_RELATIVE_PATH (str): Relative path to the configuration file. Default is "config.ini".

    Attributes:
        ROOT_PATH (str): Root path of the project.
        CONFIG_INI_FILE (str): Path to the configuration file.
        config (ConfigParser): Configuration parser object.
        WINDOWS (list): List of window sizes.
        TRAINING_DATA_RATIO (float): Ratio of training data.
        TOTAL_FOLDS (int): Number of folds for cross-validation.
        SPLITS_FILE (str): Path to the splits file.
        SEED (int): Random seed for reproducibility.
    
    Methods:
        __init__(self, ROOT_BASE_PATH = "..", CONFIG_INI_FILE_RELATIVE_PATH = "config.ini"): Initializes the SetsCreator object.
        __split_training_test_data(self): Splits the data into training and test sets.
        __generate_folds_set(self, training_data_size_per_window): Generates folds for cross-validation.
        __split_data_into_folds(self, TOTAL_SUBJECTS, TRAINING_TEST_DICT): Splits the data into folds and writes to the splits file.
        create_sets(self): Creates the training and test sets and writes to the splits file.
    
    Usage:
        sets_creator = SetsCreator()
        sets_creator.create_sets()
    
    Example:
        from sets_creator import SetsCreator
        sets_creator = SetsCreator()
        sets_creator.create_sets()
    """

    def __init__(self, ROOT_BASE_PATH = "..", CONFIG_INI_FILE_RELATIVE_PATH = "config.ini"):
        self.ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_BASE_PATH)
        self.CONFIG_INI_FILE = os.path.join(self.ROOT_PATH, os.path.normpath(CONFIG_INI_FILE_RELATIVE_PATH))

        self.config = ConfigParser()
        self.config.read(self.CONFIG_INI_FILE)

        self.WINDOWS = list(map(int, self.config['Data']['window_sizes'].split(',')))

        self.TRAINING_DATA_RATIO = float(self.config['Data']['training_data_ratio'])
        self.TOTAL_FOLDS = int(self.config['Data']['folds_number'])
        self.SPLITS_FILE = os.path.join(self.ROOT_PATH, os.path.normpath(self.config['Paths']['splits_file']))
        self.SEED = int(self.config['Random']['seed'])

        random.seed(self.SEED)

    def __split_training_test_data(self):
        """Splits the data into training and test sets.
        This method reads the processed data path from the configuration file, discovers the subjects, and randomly assigns them to training and test sets.
        The ratio of training data is defined in the configuration file.

        Args:
            None
        
        Returns:
            TOTAL_SUBJECTS (int): Total number of subjects.
            TRAINING_TEST_DICT (dict): Dictionary mapping subjects to training/test sets.

        Raises:
            FileNotFoundError: If the processed data path does not exist.
        
        """
        PROCESSED_DATA_PATH = os.path.join(self.ROOT_PATH, os.path.normpath(self.config['Paths']['processed_data_path']))
        SUBJECTS = [int(name.replace('subject','')) for name in os.listdir(PROCESSED_DATA_PATH) if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, name))]
        TOTAL_SUBJECTS = len(SUBJECTS)

        TRAINING_SUBJECTS_SIZE = int(TOTAL_SUBJECTS * self.TRAINING_DATA_RATIO)
        TEST_SUBJECTS_SIZE = TOTAL_SUBJECTS - TRAINING_SUBJECTS_SIZE

        training_test_set = [1] * TRAINING_SUBJECTS_SIZE + [0] * TEST_SUBJECTS_SIZE
        random.shuffle(training_test_set)

        TRAINING_TEST_DICT = {subject: training_test for subject, training_test in zip(SUBJECTS, training_test_set)}
        return TOTAL_SUBJECTS, TRAINING_TEST_DICT

    def __generate_folds_set(self, training_data_size_per_window):
        """Generates folds for cross-validation.
        This method takes the training data size per window and generates folds for cross-validation.

        Args:
            training_data_size_per_window (dict): Dictionary mapping window sizes to training data sizes.
        
        Returns:
            folds_set (dict): Dictionary mapping window sizes to folds.
        """

        folds_set = {window: [] for window in self.WINDOWS}
        
        for window in self.WINDOWS:
            folds_size = training_data_size_per_window[window] // self.TOTAL_FOLDS
            remaining_folds_size = training_data_size_per_window[window] % self.TOTAL_FOLDS
            folds_sizes = [folds_size]*self.TOTAL_FOLDS
            for i in range(remaining_folds_size):
                folds_sizes[i] += 1

            folds_set[window] = []
            for i in range(self.TOTAL_FOLDS):
                folds_set[window].extend([i]*folds_sizes[i])
            random.shuffle(folds_set[window])
        
        return folds_set

    def __split_data_into_folds(self, TOTAL_SUBJECTS, TRAINING_TEST_DICT):
        """Splits the data into folds and writes to the splits file.
        This method takes the total number of subjects and the training/test dictionary, generates folds, and writes the data to the splits file.

        Args:
            TOTAL_SUBJECTS (_type_): _description_
            TRAINING_TEST_DICT (_type_): _description_

        Returns:
            None

        Raises:
            FileNotFoundError: If the splits file does not exist.
        """
        
        SUBJECTS_FILE = os.path.join(self.ROOT_PATH, os.path.normpath(self.config['Paths']['subjects_file']))
        with open(SUBJECTS_FILE, 'r') as file:
            text = file.readlines()

        subject_responses_count = []
        for i in range(1, len(text)):
            subject_responses_count.append(text[i].strip().split(','))
        
        training_data_size_per_window = {window: 0 for window in self.WINDOWS}

        for i in range(len(self.WINDOWS)):
            for j in range(TOTAL_SUBJECTS):
                if TRAINING_TEST_DICT[int(subject_responses_count[j][0])] == 1:
                    training_data_size_per_window[self.WINDOWS[i]] += int(subject_responses_count[j][i+2])

        folds_set = self.__generate_folds_set(training_data_size_per_window)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_splits_file = f"{self.SPLITS_FILE}_{timestamp}.csv"


        with open(self.SPLITS_FILE, 'r') as file, open(temp_splits_file, 'w') as temp_file:
            first_line = file.readline()
            temp_file.write(first_line)
            
            for line in file:
                subject, session, response, window, chunk_count, _, _ = line.strip().split(',')
                subject = int(subject)
                is_training = TRAINING_TEST_DICT[subject]
                fold = -1 if is_training == 0 else folds_set[int(window)].pop()
                temp_file.write(f"{subject},{session},{response},{window},{chunk_count},{is_training},{fold}\n")

        os.replace(temp_splits_file, self.SPLITS_FILE)

    def create_sets(self):
        """Creates the training and test sets and writes to the splits file.
        This method initializes the splits file, splits the data into training and test sets, generates folds, and writes the data to the splits file.
        
        Args:
            None

        Returns:
            None

        Raises:
            FileNotFoundError: If the splits file does not exist.
        """
        TOTAL_SUBJECTS, TRAINING_TEST_DICT = self.__split_training_test_data()
        self.__split_data_into_folds(TOTAL_SUBJECTS, TRAINING_TEST_DICT)

        print("Data successfully categorized.\n")
        print(f"A file named '{os.path.basename(self.SPLITS_FILE)}' has been created with the data categorized into:")
        print(f"- {int(self.TRAINING_DATA_RATIO*100)}% for training")
        print(f"- {100-int(self.TRAINING_DATA_RATIO*100)}% for testing\n")
        print(f"And the data categorized as 'training' has been divided into {self.TOTAL_FOLDS} folds, with approximately {100//self.TOTAL_FOLDS}% of the data in each.")


"""
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
