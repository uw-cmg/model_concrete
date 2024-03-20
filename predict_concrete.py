import os
import pandas as pd
import numpy as np
import joblib
import dill
from mastml.feature_generators import ElementalFeatureGenerator, OneHotGroupGenerator

def get_preds_ebars_domains(df_test):
    d = 'model_concrete'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))
    recal_params = pd.read_csv(os.path.join(d, 'recal_dict.csv'))

    features = df_features.columns.tolist()
    df_test = df_test[features]

    X = scaler.transform(df_test)

    # Make predictions
    preds = model.predict(X)

    # Get ebars and recalibrate them
    errs_list = list()
    a = recal_params['a'][0]
    b = recal_params['b'][0]
    for i, x in X.iterrows():
        preds_list = list()
        for pred in model.model.estimators_:
            preds_list.append(pred.predict(np.array(x).reshape(1, -1))[0])
        errs_list.append(np.std(preds_list))
    ebars = a * np.array(errs_list) + b 

    # Get domains
    with open(os.path.join(d, 'model.dill'), 'rb') as f:
        model_domain = dill.load(f)

    domains = model_domain.predict(X)

    return preds, ebars, domains


def make_predictions(df_test):

    # Process data
    X_train = pd.read_csv('model_concrete/X_train.csv')
    feature_names = X_train.columns.tolist()
    
    # Check the data
    cols_in = df_test.columns.tolist()
    for c_in in cols_in:
        if c_in not in feature_names:
            print('Error with input feature', c_in)
            print('Input features should be', feature_names)
            break

    # Get the ML predicted values
    preds, ebars, domains = get_preds_ebars_domains(df_test)

    pred_dict = {'Predicted concrete compressive strength (MPa)': preds,
                 'Ebar concrete compressive strength (MPa)': ebars}

    for d in domains.columns.tolist():
        pred_dict[d] = domains[d]

    del pred_dict['y_pred']
    #del pred_dict['d_pred']
    del pred_dict['y_stdu_pred']
    del pred_dict['y_stdc_pred']

    for f in feature_names:
        pred_dict[f] = np.array(df_test[f]).ravel()

    return pd.DataFrame(pred_dict)