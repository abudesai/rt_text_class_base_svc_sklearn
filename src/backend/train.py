from sklearn.model_selection import train_test_split
import datetime as dt
from preprocess import preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import constants
import joblib
import os

now = dt.datetime.today().strftime("%y-%m-%d-%H-%M-%S")

if __name__ == "__main__":
    Xtrain, Xval = preprocess()

    X_train, y_train = Xtrain[:, :-1], Xtrain[:, -1]
    X_val, y_val = Xval[:, :-1], Xval[:, -1]

    # clc = MultinomialNB()
    clc = SVC(C = 1, degree = 3, kernel = 'rbf', probability=True)
    clc.fit(X_train, y_train)
    with open(
        os.path.join(constants.MAIN_DIR, constants.MODELS_ARTIFACT_MODELS), "wb"
    ) as f:
        joblib.dump(clc, f)

    print(f"Training score: ", clc.score(X_train, y_train))
    print(f"Validation score: ", clc.score(X_val, y_val))
    print("Training complete.")
