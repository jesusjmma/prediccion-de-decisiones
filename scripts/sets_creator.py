from configparser import ConfigParser
import os
import random
from datetime import datetime

class SetsCreator:
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
        TOTAL_SUBJECTS, TRAINING_TEST_DICT = self.__split_training_test_data()
        self.__split_data_into_folds(TOTAL_SUBJECTS, TRAINING_TEST_DICT)

        print("Data successfully categorized.\n")
        print(f"A file named '{os.path.basename(self.SPLITS_FILE)}' has been created with the data categorized into:")
        print(f"- {int(self.TRAINING_DATA_RATIO*100)}% for training")
        print(f"- {100-int(self.TRAINING_DATA_RATIO*100)}% for testing\n")
        print(f"And the data categorized as 'training' has been divided into {self.TOTAL_FOLDS} folds, with approximately {100//self.TOTAL_FOLDS}% of the data in each.")
