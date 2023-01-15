import os
import json
import pandas as pd
import warnings

def read_config(main_config_dir):
    schema_files = [file for file in os.listdir(main_config_dir)
     if file.endswith('json') and 'schema' in file]

    if len(schema_files) != 1:
        write_error('backend-utils',
         f"{main_config_dir} must have exactly one schema file")
        raise FileExistsError if len(schema_files) > 1 else FileNotFoundError

    schema_path = os.path.join(main_config_dir, schema_files[0])
    with open(schema_path, 'r') as f:
        data = json.load(f)
    return data['inputDatasets']['textClassificationBaseMainInput']

def write_error(source_script:str, msg: str, error_file_path, mode=None):
    if mode is None:
        if os.path.exists(os.path.join(error_file_path, f"{source_script}.txt")):
            mode = 'a'
        else:
            mode = 'w'

    with open(os.path.join(error_file_path, f"{source_script}.txt"), mode) as f:
        f.write(msg)


def read_data(main_dir:str, has_key=False):
    if 'test' in main_dir.lower():
        s = 'testing'
        t = 'testing'
    elif 'train' in main_dir.lower():
        s = 'preprocess'
        t = 'train'

    if has_key:
        files = [file for file in os.listdir(main_dir)
            if file.endswith('.csv') and 'key' in file]
        if len(files) == 0:
            warnings.warn("There is no testing_key file to evaluate the model performance on testing set")
            return None
    else:
        files = [file for file in os.listdir(main_dir)
            if file.endswith('.csv') and 'key' not in file]

        if len(files) == 0:
            write_error(f'backend-{s}',
            f"There are no {t} files in {main_dir}")
            raise e

    if len(files) == 1:
        df = pd.read_csv(os.path.join(main_dir, files[0]))
    else:
        df = None
        try:
            for file in files:
                df_temp = pd.read_csv(os.path.join(main_dir, file))
                if df is None:
                    df = df_temp
                else:
                    df = pd.concat([df, df_temp], axis=0)
        except Exception as e:
            write_error(f'backend-{s}', 
            f'There was an error while appending all {t} csv files,'
            f'make sure that all are in the correct format, error: {e}')
            raise e
    
    return df