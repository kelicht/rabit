import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.parallel import Parallel, delayed

from rabit._utils import map_to_bins


def _get_thresholds(X_d, n_bins):
    
    X_sorted = np.sort(X_d)
    unique_values = np.unique(X_sorted)
    
    if len(unique_values) < n_bins:
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    
    else:
        percentiles = np.linspace(0, 100, n_bins+1)[1:-1]
        thresholds = np.percentile(X_sorted, percentiles, method='midpoint')
        thresholds = np.unique(thresholds)
    
    return thresholds



class BinningTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        n_bins=256,
        n_jobs=None
    ):
        
        self.n_bins = n_bins
        self.n_jobs = n_jobs
    
    
    def fit(self, X, y=None):
        self.n_samples_, self.n_features_ = X.shape       
        self.thresholds_ = Parallel(
            n_jobs=self.n_jobs,
            backend='threading',
            verbose=0
        )(delayed(_get_thresholds)(X[:, d], self.n_bins) for d in range(self.n_features_))
        self.n_thresholds_ = np.array([len(thresholds) for thresholds in self.thresholds_])
        return self
    
    
    def transform(self, X):
        X_bin = Parallel(
            n_jobs=self.n_jobs,
            backend='threading',
            verbose=0
        )(delayed(map_to_bins)(X[:, d], self.thresholds_[d]) for d in range(self.n_features_))
        return np.array(X_bin, dtype=np.int64).T
    
    
