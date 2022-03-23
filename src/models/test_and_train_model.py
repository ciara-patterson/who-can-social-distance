import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn import metrics

# function for creating a random forest model from sci-kit learn's standard RandomForestRegression
def scale_data(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return (X_train, X_test)

def build_rand_forest_reg(features, results, scale_test_data = True, 
                          estimators = 100, max_depth = 80, output = True):
    '''Builds random forest regressor that predicts the results col from the other information included in 
    features dataset. Returns either 2 dataframes, one with the test values and predicted values for each county 
    or the model itself is returned.'''

    reg = RandomForestRegressor(n_estimators=estimators, max_depth = max_depth, oob_score = True, random_state = 42)
    # X = features_dataset.drop(columns = [results_col])
    # y = features_dataset[results_col]  # Labels (Target)
    X = features 
    y = results
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% test
    print('Test size', y_test.shape)

    if scale_test_data == True:
        X_train, X_test = scale_data(X_train, X_test)

    #Train the model using the training sets 
    reg.fit(X_train,y_train)

    y_pred_reg = reg.predict(X_test)
    
    # Get numerical feature importances
    importances = list(reg.feature_importances_)

    # List of tuples with variable and importance
    feature_list = X.columns
    feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    if output:
        # Print out the feature and importances 
        print('\n---\nFEATURE IMPORTANCE \n---\n')
        [print('{:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    feat_importance_df = pd.DataFrame(feature_importances, columns = ['feature', 'importance'])
    
    if output:
        # print model scores
        print('\n---\nMODEL SCORES \n---\n')
        
        print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(reg.score(X_train, y_train), 
                                                                                                reg.oob_score_,
                                                                                                reg.score(X_test, y_test)))
        # print model error
        print('\n---\nMODEL ERROR \n---\n')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_reg))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_reg))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))

    # create a dataframe with predicted values
    preds_df = pd.DataFrame(zip(features.index, y_test, y_pred_reg), columns = ['fips', 'y_test', 'RRF_pred'])

    # # return either model or predictions with
    # if return_model:
    #     return reg
    # else:
    #     return preds_df, feat_importance_df
    return reg, preds_df, feat_importance_df