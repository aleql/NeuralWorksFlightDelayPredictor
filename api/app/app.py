import os
import pandas as pd
from flask import Flask, request
from flask_caching import Cache
from joblib import load

# Configure flask cache and app
config = {
    "DEBUG": os.environ.get("DEBUG", True),
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)


@app.route('/PredictFlightDelay', methods=['POST'])
@cache.cached(timeout=50)
def predict_flight_delay():

    # Check if request is correct
    if ~request.data:
        return "Json array data not found or incorrect on body", 400
    
    # Read body from request, and transform to usable df by the ml classifier
    resquest_df=pd.DataFrame(request.json)
    
    prediction_df = pd.DataFrame(
        columns=['OPERA_Aerolineas Argentinas', 'OPERA_Aeromexico', 'OPERA_Air Canada',
       'OPERA_Air France', 'OPERA_Alitalia', 'OPERA_American Airlines',
       'OPERA_Austral', 'OPERA_Avianca', 'OPERA_British Airways',
       'OPERA_Copa Air', 'OPERA_Delta Air', 'OPERA_Gol Trans',
       'OPERA_Grupo LATAM', 'OPERA_Iberia', 'OPERA_JetSmart SPA',
       'OPERA_K.L.M.', 'OPERA_Lacsa', 'OPERA_Latin American Wings',
       'OPERA_Oceanair Linhas Aereas', 'OPERA_Plus Ultra Lineas Aereas',
       'OPERA_Qantas Airways', 'OPERA_Sky Airline', 'OPERA_United Airlines',
       'TIPOVUELO_I', 'TIPOVUELO_N', 'MES_1', 'MES_2', 'MES_3', 'MES_4',
       'MES_5', 'MES_6', 'MES_7', 'MES_8', 'MES_9', 'MES_10', 'MES_11',
       'MES_12']
    )
    data = resquest_df[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']]
    features = pd.concat(
        [pd.get_dummies(data['OPERA'], prefix = 'OPERA'),pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
    )
    
    prediction_df = prediction_df.append(features).fillna(0)
    y = modelxgb.predict(prediction_df)

    # convert numpy array to dictionary, and return json
    return { "delay":y.tolist() }

@app.route("/", methods=['GET', 'POST'])
def root_instructions():
    return "This api allows for the prediction of delays on chilean flights, for this perform an HTTP POST with flights data as json on the body to request to the url:{0}/PredictFlightDelay".format(os.environ.get("HOST", "0.0.0.0"))

if __name__ == "__main__":
    # Preemptively load model
    global modelxgb
    modelxgb = load('../../xgboost_smote.joblib')

    app.run(debug=os.environ.get("DEBUG", True), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8080)))