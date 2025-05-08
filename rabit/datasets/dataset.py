import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def process_dataset(
    df, 
    target_column, 
    target_name,
    immutable_columns=[],
    unincreasable_columns=[],
    irreducible_columns=[],
    sensitive_columns=[],
    prefix_sep=':'
):  
    y = (df[target_column].values == target_name).astype(np.int64)
    df_processed = pd.get_dummies(df.drop(target_column, axis=1), prefix_sep=prefix_sep)
    X = df_processed.values.astype(np.float64)
    constrains = _make_constraints(immutable=immutable_columns, unincreasable=unincreasable_columns, irreducible=irreducible_columns)
    params = _get_feature_params(df_processed, constrains, sensitive_columns, prefix_sep)
    params['target_column'] = target_column
    params['target_name'] = target_name
    return X, y, params


def _make_constraints(immutable, unincreasable, irreducible):
    constraints = {}
    if len(immutable) > 0:
        constraints['immutable'] = immutable
    if len(unincreasable) > 0:
        constraints['unincreasable'] = unincreasable
    if len(irreducible) > 0:
        constraints['irreducible'] = irreducible
    return constraints


def _get_feature_params(df_processed, constrains, sensitive_columns, prefix_sep):

    feature_names = df_processed.columns.values
    n_features = len(feature_names)

    is_binary = np.zeros(n_features, dtype=bool)
    is_integer = np.zeros(n_features, dtype=bool)
    for d, feature in enumerate(feature_names):
        if df_processed[feature].dtype == float:
            continue
        elif np.array_equal(np.array([0, 1]), np.sort(df_processed[feature].unique())):
            is_binary[d] = True
            is_integer[d] = True
        else:
            is_integer[d] = True

    is_immutable = np.zeros(n_features, dtype=bool)
    is_unincreasable = np.zeros(n_features, dtype=bool)
    is_irreducible = np.zeros(n_features, dtype=bool)
    is_sensitive = np.zeros(n_features, dtype=bool)
    for d, feature in enumerate(feature_names):
        if prefix_sep in feature:
            feature, _ = feature.split(prefix_sep)
        if 'immutable' in constrains and feature in constrains['immutable']:
            is_immutable[d] = True
        elif 'unincreasable' in constrains and feature in constrains['unincreasable']:
            is_unincreasable[d] = True
        elif 'irreducible' in constrains and feature in constrains['irreducible']:
            is_irreducible[d] = True
        if feature in sensitive_columns:
            is_sensitive[d] = True
    
    categories = []
    prefix = ''
    _categories = []
    for d, feature in enumerate(feature_names):
        if prefix_sep not in feature:
            continue
        prefix_d, _ = feature.split(prefix_sep)
        if prefix == prefix_d:
            _categories.append(d)
        else:
            if len(_categories) > 0:
                categories.append(_categories)
            prefix = prefix_d
            _categories = [d]
    if len(_categories) > 0:
        categories.append(_categories)
        
    params = {
        'n_features': n_features,
        'feature_names': feature_names, 
        'is_binary': is_binary,
        'is_integer': is_integer,
        'is_immutable': is_immutable,
        'is_unincreasable': is_unincreasable,
        'is_irreducible': is_irreducible,
        'is_sensitive': is_sensitive,
        'categories': categories
    }
    return params
    


class Dataset():
    
    def __init__(
        self,
        df, 
        target_column, 
        target_name,        
        immutable_columns=[],
        unincreasable_columns=[],
        irreducible_columns=[],
        sensitive_columns=[],
    ):
        
        self.df = df
        self.target_column = target_column
        self.target_name = target_name
        self.immutable_columns = immutable_columns
        self.unincreasable_columns = unincreasable_columns
        self.irreducible_columns = irreducible_columns
        self.sensitive_columns = sensitive_columns
        
        X, y, params = process_dataset(df, target_column, target_name, immutable_columns, unincreasable_columns, irreducible_columns, sensitive_columns)
        self.X = X
        self.y = y
        self.params = params
        
        self.n_features = params['n_features']
        self.feature_names = params['feature_names']
        self.is_binary = params['is_binary']
        self.is_integer = params['is_integer']
        self.is_immutable = params['is_immutable']
        self.is_unincreasable = params['is_unincreasable']
        self.is_irreducible = params['is_irreducible']
        self.is_sensitive = params['is_sensitive']
        self.categories = params['categories']
        
        
    def get_dataset(self, split=False, test_size=0.25, validation_size=0.0):

        if split:
            X_tr, X_ts, y_tr, y_ts = train_test_split(self.X, self.y, test_size=test_size+validation_size, stratify=self.y)
            if validation_size > 0:
                X_vl, X_ts, y_vl, y_ts = train_test_split(X_ts, y_ts, test_size=test_size/(test_size+validation_size), stratify=y_ts)
                return X_tr, X_vl, X_ts, y_tr, y_vl, y_ts
            else:
                return X_tr, X_ts, y_tr, y_ts
        else:
            return self.X, self.y        
        
    
    def get_details(self):
        
        features = self.feature_names
        types = ['Binary' if self.is_binary[d] else ('Integer' if self.is_integer[d] else 'Real') for d in range(self.n_features)]
        mins, maxs = self.X.min(axis=0), self.X.max(axis=0)
        immutables = ['Yes' if self.is_immutable[d] else 'No' for d in range(self.n_features)]
        constraints = ['Fix' if self.is_immutable[d] else ('Unincreasable' if self.is_unincreasable[d] else ('Irreducible' if self.is_irreducible[d] else 'Nothing')) for d in range(self.n_features)]
        sensitive = ['Yes' if self.is_sensitive[d] else 'No' for d in range(self.n_features)]
        details = {
            'Feature': features,
            'Type': types, 
            'Min': mins,
            'Max': maxs, 
            'Immutable': immutables, 
            'Constraint': constraints,
            'Sensitive': sensitive,
        }
        return pd.DataFrame(details)        
    
    
    def get_sensitive_indices(self):
        return np.where(self.is_sensitive)[0]
    
    

class FicoDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/fico.csv')
        immutable_columns = [
            'ExternalRiskEstimate',
            'MSinceOldestTradeOpen',
            'MSinceMostRecentTradeOpen',
            'AverageMInFile',
            'NumTrades60Ever2DerogPubRec',
            'NumTrades90Ever2DerogPubRec',
            'NumTotalTrades',
            'NumTradesOpeninLast12M'
        ]
        unincreasable_columns = []
        irreducible_columns = []
        sensitive_columns = []
        super().__init__(df, 'RiskPerformance', 'Good', immutable_columns, unincreasable_columns, irreducible_columns, sensitive_columns)
        self.name = 'FICO'
        


class CompasDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/compas.csv')
        immutable_columns = ['race', 'sex']
        unincreasable_columns = []
        irreducible_columns = ['age']
        sensitive_columns = ['race']
        super().__init__(df, 'two_year_recid', 'No', immutable_columns, unincreasable_columns, irreducible_columns, sensitive_columns)
        self.name = 'COMPAS'
        


class AdultDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/adult.csv')
        immutable_columns = ['marital-status', 'race', 'gender', 'native-country']
        unincreasable_columns = []
        irreducible_columns = ['age']
        sensitive_columns = ['race']
        super().__init__(df, 'income', '>50K', immutable_columns, unincreasable_columns, irreducible_columns, sensitive_columns)
        self.name = 'Adult'
    


class BailDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/bail.csv')
        immutable_columns = ['White', 'Super', 'Married', 'Felon', 'Propty', 'Person', 'Male', 'Tservd', 'Follow']
        unincreasable_columns = []
        irreducible_columns = ['Age']
        sensitive_columns = []
        super().__init__(df, 'Recidivate', 'No', immutable_columns, unincreasable_columns, irreducible_columns, sensitive_columns)
        self.name = 'Bail'
        


class CreditDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/credit.csv')
        immutable_columns = ['Married', 'Single', 'TotalOverdueCounts', 'TotalMonthsOverdue']
        unincreasable_columns = []
        irreducible_columns = ['Age']
        sensitive_columns = []
        super().__init__(df, 'DefaultNextMonth', 'No', immutable_columns, unincreasable_columns, irreducible_columns, sensitive_columns)
        self.name = 'Credit'
        
            

