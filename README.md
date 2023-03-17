# NeuralWorks Challenge Machine Learning Engineer: Chilean flight delay predictor

This repository contains the proposed solution for the challenge of choosing and optimising a machine learning model for the prediction of delays on Chilean flights.
Down below the answers to the questions asked will be provided:


1.- The model chosen for the problem was the XGBoost, mainly due to the imbalanced nature of the data. This machine learning model displays resilience when facing data with these properties. It is also worth pointing out that in the literature, this model has been used to solve problems of similar parameters, where a XGBoost model with  the Over-Sampling technique, randomised SMOTE obtained a validation accuracy of 85.73% which is. [https://ieeexplore.ieee.org/document/8876970]

2.- The following changes were tested on the file to-expose.ipynb:
Optimization of hyperparameters using CV Matrix.
Downsampling on majority class to half, and to the same size of minority class.
Upsampling of minority class.
Downsampling of the majority class using SMOTE.
Of these methods, the last one obtained better results, with an AUC SCORE of 0.663. Now, this result still has room for improvement, as the amount of false positives is still significant.

3.- The model with Downsampling SMOTE was serialised on the file xgboost_smote.joblib, and a Flask Rest API was implemented. This API receives requests on the /PredictFlightDelay path, where it expects a POST request with content-type: application/json. The Json expected on the body consists of an array of Json objects, having each object the format and data of a row of the file dataset_SCL.csv, The array can be of any size, and more than one object can be sent at a time. The api answers with code 200 if the request was correct, returning a Json object with an array property, which contains either a 0 or a 1 depending on the prediction of the model


An example of its use is presented down below:
Example request:
```json
[
  {
    "Fecha-I": "2017-01-01 23:30:00",
    "Vlo-I": 226,
    "Ori-I": "SCEL",
    "Des-I": "KMIA",
    "Emp-I": "AAL",
    "Fecha-O": "2017-01-01 23:33:00",
    "Vlo-O": 226,
    "Ori-O": "SCEL",
    "Des-O": "KMIA",
    "Emp-O": "AAL",
    "DIA": 1,
    "MES": 1,
    "AÑO": 2017,
    "DIANOM": "Domingo",
    "TIPOVUELO": "I",
    "OPERA": "American Airlines",
    "SIGLAORI": "Santiago",
    "SIGLADES": "Miami"
  }
]
```
Example answer:
```json
{
  "delay": [0]
}
```

4.- The API is already configured and contained in a docker container, and it can be easily deployed to either a Google Cloud computing instance or an Amazon Web Services instance. For deploying it to GCP a google cloud project must be initialised, and in Amazon, an Amazon ECS instance must be configured.

5.- wrk benchmark was run with the following parameters:
```bash
wrk -t12 -c400 -d30s --latency http://127.0.0.1:8080
```
The output is displayed on the file output.txt
Now, the application was tested out in local, in debug mode, as deploying the application to a server for the testing was beyond the scope of the challenge. For this, the performance can be greatly improved by deploying the application to any of the two cloud servers mentioned on the previous section, and also configuring gunicorn for better management of the server resources.



