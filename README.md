# Support Vector Classifier for Text Classification with tf-idf and SVD preprocessing

## Description (TL/DR)

This project implements the specifications of [ReadyTensor](https://readytensor.com/).

All python scripts can be found within `./src/` directory.

```
src
├──backend
│   ├── preprocess.py # Preprocess text column using scikit-learn pipeline
│   ├── constants.py # Containing all directory and utility functions
│   ├── train.py # Main script for training. Training Entry point
│   └── predict # Main script for prediction. Testing entry point
├──frontend
│   └── preprocess.py # Preprocess text column using scikit-learn pipeline

```

## How to run

1. Build the docker image.<br>
   `sudo docker buildx build -t <imagename> .`
2. Create a docker volume to mount your data<br>
   `sudo docker volume create --name <vname> --opt --type=none --opt device=<absolute-path-to-ml_vol> --opt o=bind`
3. Run the training script.<br>
   `sudo docker run --rm --name <container-name> --mount source=<vname>,target=/opt/ml_vol <imagename> train`
4. Check the testing results of the trained model. <br>
   `sudo docker run --rm --name <container-name> --mount source=<vname>,target=/opt/ml_vol <imagename> test`
5. To start up a server.<br>
   `sudo docker run -p 5000:5000 -d --rm --name <container-name> --mount source=<vname>,target=/opt/ml_vol <imagename> test` <br>
   Step 5 will start a service on [localhost](localhost:5000).
