import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.utils.parallel import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from rabit._booster import RecourseBooster
from rabit._utils import Y_TARGET, compute_optimal_k



class RecourseBoostingClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        action,
        n_estimators=100,
        max_depth=8,
        min_value_leaf=0.001,
        min_split_gain=0.001,
        max_features='sqrt',
        n_bins=128,
        alpha=0.001,
        gamma=0.001,
        learning_rate=0.1,
        only_actionable_features=False,
        random_state=None,   
        n_jobs=None,     
    ):
        
        self.action = action
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_value_leaf = min_value_leaf
        self.min_split_gain = min_split_gain
        self.max_features = max_features
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.only_actionable_features = only_actionable_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        

    def fit(self, X, y):
        
        self.n_samles_, self.n_features_ = X.shape
        
        z = np.zeros_like(y)     
        beta = np.zeros_like(y)
        estimators = []
        for t in range(self.n_estimators):
            
            estimator = RecourseBooster(
                action=self.action,
                max_depth=self.max_depth,
                min_value_leaf=self.min_value_leaf,
                min_split_gain=self.min_split_gain,
                max_features=self.max_features,
                n_bins=self.n_bins,
                alpha=self.alpha,
                gamma=self.gamma,
                only_actionable_features=self.only_actionable_features,
                n_jobs=self.n_jobs,
            )
            
            X_sample, y_sample, z_sample, beta_sample = resample(X, y, z, beta)
            estimator = estimator.fit(X_sample, y_sample, z_sample, beta_sample)
            estimator.tree_.value = estimator.tree_.value * self.learning_rate
            z = z + estimator.predict(X)
            beta = beta + estimator.get_min_value()
            estimators.append(estimator)
            
        self.estimators_ = estimators    
        self.intercept_ = 0.0
        return self
            
    
    def decision_function(self, X):
        
        y = np.zeros(X.shape[0])
        for estimator in self.estimators_:
            y = y + estimator.predict(X)
        y = y + self.intercept_
        return y
    

    def predict(self, X):
        
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)
    

    def predict_proba(self, X):
        
        z = self.decision_function(X)
        y_proba = 1 / (1 + np.exp(-z))
        return np.array([1 - y_proba, y_proba]).T
    
    
    def transform(self, X):
        
        X_transformed = Parallel(
            n_jobs=self.n_jobs,
            prefer="threads",
            verbose=0,
        )(delayed(estimator.transform)(X) for estimator in self.estimators_)
        return hstack(X_transformed)
    
    
    @property
    def feature_importances_(self):
        all_importances = np.zeros(self.n_features_)
        for estimator in self.estimators_:
            all_importances += estimator.feature_importances_
        return all_importances / all_importances.sum()


    def optimize_intercept(self, X_cf, epsilon=0.1, delta=0.05):
        
        n_samples = X_cf.shape[0]
        k_opt = compute_optimal_k(n_samples, epsilon, delta)
        probs = np.sort(self.predict_proba(X_cf)[:, Y_TARGET])
        p = probs[k_opt]        
        theta = np.log((1 - p) / p)
        theta = max(theta, 0.0)
        self.intercept_ = theta
        return self
    
    
    def optimize_weights(self, X, y, X_cf, eta=0.1, C=0.1):
        
        n_samples = X.shape[0]
        X_all = np.vstack([X, X_cf])
        X_all = self.transform(X_all)
        X_cf = X_all[n_samples:]
        y_all = np.concatenate([y, Y_TARGET * np.ones(n_samples)])
        estimator = LogisticRegression(solver='saga', C=C, max_iter=1000, warm_start=True)
        
        w_all = np.ones(2 * n_samples)
        w_all[n_samples:] = eta
        estimator = estimator.fit(X_all, y_all, sample_weight=w_all)
            
        coefs = estimator.coef_[0]
        ptr = 0
        for t in range(self.n_estimators):
            is_leaf = self.estimators_[t].tree_.is_leaf
            n_leaves = is_leaf.sum()
            self.estimators_[t].tree_.value[is_leaf] = coefs[ptr:ptr+n_leaves]
            ptr = ptr + n_leaves
        self.intercept_ = estimator.intercept_[0]
        
        return self
    
    
