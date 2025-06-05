from pathlib import Path
import pandas as pd
import numpy as np
import code

from scripts.config import Config

def main():
    data_dir: Path = Config().PROCESSED_MUSEDATA_PATH
    file_name: Path = Config().MUSEDATA_FILES_PREFIX
    file: Path = data_dir / (str(file_name) + '1.csv')
    print (f"Reading file: {file}")
    with open(file, 'r') as f:
        header = f.readline().strip().split(',')
    
    columns = Config().MUSEDATA_COLUMNS
    dtype_map = {'Set': bool, 'Fold': np.uint8, 'Trial': np.uint8, 'Respuesta': np.uint8}
    selected_columns = [col for col in columns if col in header]
    dtypes = {col: dtype_map[col] for col in selected_columns if col in dtype_map}
    df: pd.DataFrame = pd.read_csv(file, low_memory=False, date_format="%Y-%m-%d %H:%M:%S.%f", parse_dates=[0], index_col=0, dtype=dtypes)

    # cut the dataframe erasing -1 values
    i=23
    w=100
    dataframe = df[df.columns[i]]
    dataframe = dataframe[dataframe != -1]
    count = dataframe.value_counts()
    windows = count[count >= 0] / int(w / Config().SAMPLING_OFFSET)

    index = dataframe.drop_duplicates().index

    # Open interactive python shell
    code.interact(local=locals())

    df[df.columns[i]].value_counts()

    '''
    for i in range(23, df.shape[1]-1):
        counts[df.columns[i]] = df.iloc[:,[20,21,i]].loc[df.iloc[:,i] != -1].groupby(['Trial', 'Respuesta', df.columns[i]]).count().iloc[:,0].values
    #'''

    count = df.iloc[:,i].loc[df.iloc[:,i] != -1].value_counts()



if __name__ == "__main__":
    main()
