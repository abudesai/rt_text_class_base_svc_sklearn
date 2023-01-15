import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/backend')
sys.path.append('./src/frontend')
sys.path.append('/usr/src/frontend')
sys.path.append('/usr/src/')
sys.path.append('/usr/src/backend')

import utils
import os

MAIN_DIR = "/opt/ml_vol"

MAIN_DATA_CONFIG = "inputs/data_config"
MAIN_DATA_TRAINING= "inputs/data/training/textClassificationBaseMainInput"
MAIN_DATA_TESTING= "inputs/data/testing/textClassificationBaseMainInput"
MAIN_MODELS = "model/artifacts"
MAIN_OUTPUT = "outputs"


# DATA_CONFIG = "inputs/data_config/clickbait_schema.json"
# DATA_TRAINING = "inputs/data/training/textClassificationBaseMainInput/clickbait_train.csv"
# DATA_TESTING = "inputs/data/testing/textClassificationBaseMainInput/clickbait_test.csv"
# DATA_TESTING_KEY = "inputs/data/testing/textClassificationBaseMainInput/clickbait_test_key.csv"

MODELS_ARTIFACT_PREPROCESS = "model/artifacts/preprocess_pipeline.p"
MODELS_ARTIFACT_LABEL_PREPROCESS = "model/artifacts/preprocess_label_pipeline.p"
MODELS_ARTIFACT_MODELS = "model/artifacts/model_pipeline.p"

OUTPUT_ERRORS = "outputs/errors"
OUTPUT_HPT = "outputs/hpt_errors"
OUTPUT_TESTING = "outputs/testing_outputs/test_predictions.csv"



DATA = utils.read_config(os.path.join(MAIN_DIR, MAIN_DATA_CONFIG))
id_field = DATA['idField']
target_class = DATA['targetField']
document_field = DATA['documentField']