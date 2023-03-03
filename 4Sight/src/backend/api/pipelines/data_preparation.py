import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import IsolationForest


def nan(df):
    """
    Function which process missing data. Will depend on the number of Nan
    Parameters
    ----------
    df : Pandas DataFrame
      DataFrame to clean

    Returns
    -------
    df : Pandas DataFrame 
    """
    print("Process Nan...")
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    for col in numeric_cols:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
        if pct_missing < 4:                                                       #* if NaN < 4% : replace by median
            med = df[col].median()
            df[col] = df[col].fillna(med)
        if pct_missing >= 20:                                                     #* if NaN > 20% : drop features
            df = df.drop(columns=[col])
        if (pct_missing < 20) & (pct_missing >= 4) :                              #* if NaN < 20% & > 4% : drop lines
            df = df.dropna(subset=[col])

    df_non_numeric = df.select_dtypes(exclude=[np.number])                        #* Repeat process with non numerics variables
    non_numeric_cols = df_non_numeric.columns.values
    for col in non_numeric_cols:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
        if pct_missing >= 15:
            df = df.drop(columns=[col])
        if pct_missing < 15 :
            df = df.dropna(subset=[col])
    print(df.shape)
    return df

def fix_typos(df):
    """
    Function to correct the typos of columns to switch them to object into another more appropriate 
    Parameters
    ----------
    df : Pandas DataFrame 
      DataFrame to clean

    Returns
    -------
    df :Pandas DataFrame 
    """
    print("Fixing Typos...")
    obj = [col  for col, dt in df.dtypes.items() if dt == object]
    for col in obj:
        df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].str.upper()
        df[col] = df[col].str.strip()
    print(df.shape)
    return df

def dummification(df, cat=None):                                                    
    """
    Function to encode objects variable 
    Parameters
    ----------
    df : Pandas DataFrame
      Pandas DataFrame to encode
    cat : list
      list of columns to encode

    Returns
    -------
    df : Pandas DataFrame 
    """
    print("Encoding categorical varible(s)...")
    if cat is not None:
        pd.get_dummies(data=df, columns=cat)
    print(df.shape)
    return df

def drop_bad_periods(df, mult_var=None, l_periods:dict=None):                      
    """
    Function to remove periods selected by user
    Parameters
    ----------
    df : Pandas DataFrame
      Pandas DataFrame to clean
    mult_var : str
      Name of the columns containning the differents machines. Only if we have long format Dataframe 
    l_periods : dict
      dictionnary with in key the machine if long format Dataframe (if not in long format and only one machine, leave empty or set the id of the machine). In value set date range.

    Returns
    -------
    df : Pandas DataFrame 
    """
    print("Removing Time periods...")
    if l_periods is not None:
        for key, value in l_periods.items():
            if (mult_var is not None) & (len(mult_var) != 0):
                df = df[~(df.index.strftime('%Y-%m-%d').isin(value) & df[mult_var == key])]
            else:
                df = df[~(df.index.strftime('%Y-%m-%d').isin(value))]
    print(df.shape)                                                     
    return df

def drop_date(df, dow=None, month=None):                                            #* dow and month are list
    """
    Function to remove regular moment of date as month or day of week
    Parameters
    ----------
    df : Pandas DataFrame
      Pandas DataFrame to clean
    dow : list
      List of day to remove 
    month
      List of month to remove

    Returns
    -------
    df : Pandas DataFrame 
    """
    print("Removing date")
    if dow is not None:
        print("Removing day of week...")
        mask = df.index.dayofweek.isin(dow)
        df = df[~mask]                                         
    if month is not None:
        print("Removing useless month...")
        mask = df.index.month.isin(month)
        df = df[~mask]  
    print(df.shape)
    return df

def custom_rules(df, column:str, value):                                            #! make custom filter (depend on UI)
    """
    Function to remove regular moment of date as month or day of week
    Parameters
    ----------
    df : Pandas DataFrame
      Pandas DataFrame to clean
    column : str
      column of the dataset to filter based on value
    value : any
      Filter value

    Returns
    -------
    df : Pandas DataFrame 
    """
    if type(value) == list:
        if value[0] == str:
            return df[df[column].isin(value)]
        else:
            return df[(df[column] >= value[0]) & (df[column] >= value[1])]
    else:
        return df[df[column]==value]


def data_prep(df, dow=None, month=None, cat=None, mult_var=None, l_periods=None):
    """
    Function who call all the function above
    Parameters
    ----------
    df : Pandas DataFrame
      Pandas DataFrame to clean
    ...

    Returns
    -------
    df : Pandas DataFrame 
    """
    df = df.drop_duplicates(keep='last')                                            #* Keep only most recent duplicatas
    df = drop_bad_periods(df, mult_var=mult_var, l_periods=l_periods)               #* Drop periods selected
    df = drop_date(df, dow=dow, month=month)                                        #* Drop useless date
    df = fix_typos(df)                                                              #* Set a good typos for categorical features
    df = dummification(df, cat)                                                     #* Encode categorical variables 
    dfn = df.apply(pd.to_numeric, errors='ignore')                                                      #* Assign good type for the modelling phase
    print(dfn.dtypes)
    #df = df.select_dtypes(exclude=['object'])                                       #* Remove Object and String columns who are irrelevant
    dfn = nan(dfn)                                                                    #* Process empty values based on several conditions
    dfn = dfn.convert_dtypes()                                                       #* Assign good type for the modelling phase
    
    print("End of preprocessing\n")                                                
    return dfn



###################################
#* Isolation Forest
###################################

def custom_silhouette(estimator, X):
    """
    Custom scoring function for clustering (higher is better)
    Parameters
    ----------
    estimator : sk-learn model
      model to evaluater
    X :
      data used for eval

    Returns
    -------
    sil_score : float
    """
    sil_score = silhouette_score(X, estimator.predict(X))
    print("{}   -     ".format(round(sil_score, 4)), end = '')
    return sil_score

def custom_DBScrore(estimator, X):
    """
    Custom scoring function for clustering (higher is better)
    Parameters
    ----------
    estimator : sk-learn model
      model to evaluate
    X :
      data used for eval

    Returns
    -------
    sil_score : float
    """
    davboul_score = davies_bouldin_score(X, estimator.predict(X))
    print("{}   -     ".format(round(davboul_score, 4)), end = '')
    return davboul_score

def fit_IF(train):
    """
    Function who fit Isolation Forest
    Parameters
    ----------
    train : Pandas Dataframe
      training data for the model
    
    Returns
    -------
    best_model : sklearn RandomizedSearchCV model
    """
    clf = IsolationForest(random_state=42)
    # Define the parameter grid for the hyperparameters that will be randomly sampled
    param_grid = {
          'n_estimators': list(range(100, 1000, 10)),     
          'bootstrap': [True, False]
          }   
    # Define the hyperparameter that will be tried for every possible valu
    fixed_param = {
          'contamination': np.arange(0.0, 0.1, 0.01)
          }

    # Combine the parameter grid and the fixed hyperparameter into a single dictionary
    param_grid.update(fixed_param)     

    grid_isol = RandomizedSearchCV(clf, 
                                    param_grid,
                                    scoring=custom_silhouette,              #? Davies Bouldin Score     or      Silhouette Score  
                                    refit=True,
                                    cv=3, 
                                    return_train_score=True)

    best_model = grid_isol.fit(train.values)

    print('\nOptimum parameters', best_model.best_params_)
    custom_silhouette(best_model, train.values)
    return best_model

def predict_IF(best_model, train):
    """
    Function who predict Isolation Forest
    Parameters
    ----------
    train : Pandas Dataframe
      data to cluster
    best_model : sk-learn RandomizedSearchCV model
    
    Returns
    -------
    best_model : sklearn RandomizedSearchCV model
    """
    y_pred = best_model.predict(train.values)
    train_clustered = train.assign(Cluster=y_pred)
    train_clustered['Cluster'] = train_clustered['Cluster'].replace({-1: "Anomaly", 1: "Regular"})
    train_clustered = train_clustered[train_clustered['Cluster'] == "Regular"]
    train_clustered = train_clustered.drop(columns=['Cluster'])
    return train_clustered

def save_final_dataframe(data_clustered):
    """
    Save data previously clusterized
    Parameters
    ----------
    data_clustered : Pandas Dataframe
      final dataset to save 
  
    """
    train_Bearing_Nasa = data_clustered[data_clustered['Cluster'] == "Regular"]
    train_Bearing_Nasa = data_clustered.drop(columns=['Cluster'])
    train_Bearing_Nasa.to_csv('../data/processed/train_Bearing_Nasa.csv')
