from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_data(X_train):
    '''
	scales data using standardization or minmax 

    Parameters
    ==========
    X_train: pandas df. Feature data already encoded in Real space and clean
    at this stage. 

    Returns
    ==========
    the scaled data and the object for future queries or test set. 
    '''

    scaler_std = StandardScaler()
    scaler_std = scaler_std.fit(X_train)
    X_train_std = scaler_std.transform(X_train)
    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    scaler_minmax = scaler_minmax.fit(X_train)
    X_train_minmax = scaler_minmax.transform(X_train)

    return X_train_std, X_train_minmax, scaler_std, scaler_minmax
