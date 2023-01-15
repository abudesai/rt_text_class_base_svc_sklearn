import joblib
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import json

from utils import write_error, read_data
import constants


def infer(df):

    print("making predictions")
    df_pred, X = get_predict_proba(df, model, pmodel, plmodel)

    df_pred["__label"] = pd.DataFrame(
        df_pred[plmodel.classes_], columns=plmodel.classes_
    ).idxmax(axis=1)

    
    id_field_name = constants.id_field
    df_pred.index.name=id_field_name
    df_pred.reset_index(drop=False, inplace=True)
    print(df_pred)

    # convert to the json response specification
    predictions_response = []
    for rec in df_pred.to_dict(orient="records"):
        pred_obj = {}
        pred_obj[id_field_name] = rec[id_field_name]
        pred_obj["label"] = rec["__label"]
        pred_obj["probabilities"] = {
            str(k): np.round(v, 5)
            for k, v in rec.items()
            if k not in [id_field_name, "__label"]
        }
        predictions_response.append(pred_obj)
    predictions_response = json.dumps({"predictions": predictions_response})
    return predictions_response




def get_model(): 
    print("reading model")
    if not os.path.exists(
        os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_MODELS)
    ):
        print("Model can not be found. Run train script first")
        constants.write_error("testing", "Model can't be found. Run train script first")
        return

    with open(
        os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_MODELS), "rb"
    ) as f:
        model = joblib.load(f)

    if not os.path.exists(
        os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_PREPROCESS)
    ):
        print("There is no a preprocess model")
        warnings.warn("There is no preprocessing model in models/artifact")
        pmodel = None
    else:
        with open(
            os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_PREPROCESS), "rb"
        ) as f:
            pmodel = joblib.load(f)

    print("processing data")
    if not os.path.exists(
        os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_LABEL_PREPROCESS)
    ):
        print("There is no a preprocess model for labels")
        warnings.warn("There is no preprocessing model in models/artifact for labels")
        plmodel = None
    else:
        with open(
            os.path.join(
                constants.MAIN_DIR, constants.MODELS_ARTIFACT_LABEL_PREPROCESS
            ),
            "rb",
        ) as f:
            plmodel = joblib.load(f)
    
    return model, pmodel, plmodel


def get_predict_proba(df, model, pmodel, plmodel):
    if pmodel:
        X = pmodel.transform(df[constants.document_field])
    else:
        X = df[constants.document_field].values
    ypred_p = model.predict_proba(X)
    df_pred = pd.DataFrame(data=ypred_p, index=df.index, columns=plmodel.classes_)
    return df_pred, X


def batch_predict():

    print("reading data")
    df = read_data(
        os.path.join(constants.MAIN_DIR, constants.MAIN_DATA_TESTING), has_key=False
    )
    df.set_index(constants.id_field, inplace=True)

    df_true = read_data(
        os.path.join(constants.MAIN_DIR, constants.MAIN_DATA_TESTING), has_key=True
    )

    # load model, proprocessor, and label_encoder
    print("Loading model")
    model, pmodel, plmodel = get_model()

    # get predictions and preprocessed inputs
    print("making predictions")
    df_pred, X = get_predict_proba(df, model, pmodel, plmodel)
    
    
    df_pred.to_csv(os.path.join(constants.MAIN_DIR, constants.OUTPUT_TESTING))

    if df_true is not None:
        df_true.set_index(constants.id_field, inplace=True)
        ytrue = df_true[constants.target_class]
        ytrue = plmodel.transform(ytrue)
        ypred = model.predict(X)
        print("confusion matrix: ", confusion_matrix(ytrue, ypred))
        print("balanced accuracy score: ", balanced_accuracy_score(ytrue, ypred))
        print("f1-score: ", f1_score(ytrue, ypred))
        print("precision: ", precision_score(ytrue, ypred))
        print("recall: ", recall_score(ytrue, ypred))

    print("Predictions completed")
    return



print("Loading model")
model, pmodel, plmodel = get_model()    


if __name__ == "__main__":
    batch_predict()
