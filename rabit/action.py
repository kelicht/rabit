import numpy as np
from scipy.stats import median_abs_deviation
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest

from rabit._utils import MIN_VAL, MAX_VAL, compute_candidate_actions, compute_all_actions, is_feasible



class Action():
    
    def __init__(
        self,
        params,
        cost_type='MPS',
        cost_budget=0.5,
        cost_ord=-1,
    ):
        
        self.n_features = params['n_features']
        self.feature_names = params['feature_names']
        self.is_binary = params['is_binary']
        self.is_integer = params['is_integer']
        self.is_immutable = params['is_immutable']
        self.is_unincreasable = params['is_unincreasable']
        self.is_irreducible = params['is_irreducible']
        self.is_sensitive = params['is_sensitive']
        self.categories = params['categories']
        self.target_column = params['target_column']
        self.target_name = params['target_name']

        self.cost_type = cost_type
        self.cost_budget = cost_budget
        self.cost_ord = cost_ord
                                
    
    def fit(self, X, y):
        if self.cost_type in ['MPS', 'TLPS']:
            self._weight = None
            self._percentile = self._get_percentile(X)
        else:
            self._weight = self._get_weight(X)
            self._percentile = None     
        self._plausibility = IsolationForest(n_estimators=100).fit(X)
        return self
            
        
    def _get_weight(self, X):
        weight = np.ones(self.n_features)
        if self.cost_type == 'MAD':
            weight = (median_abs_deviation(X) + 1e-4) ** -1
            weight[self.is_binary] = (X[:, self.is_binary] * 1.4826).std(axis=0)
        elif self.cost_type == 'STD':
            weight = np.std(X, axis=0) ** -1
            weight[self.is_binary] = (X[:, self.is_binary] * 1.4826).std(axis=0)
        elif self.cost_type == 'NRM':
            weight = (X.max(axis=0) - X.min(axis=0)) ** -1
        return weight
    
    
    def _get_percentile(self, X, l_buff=1e-6, r_buff=1e-6, l_quantile=0.001, r_quantile=0.999, grid_size=100):
        percentile = []
        for d in range(self.n_features):
            if self.is_immutable[d]:
                percentile.append(None)
                continue
            kde_estimator = kde(X[:, d])
            grid = np.linspace(np.quantile(X[:, d], l_quantile), np.quantile(X[:, d], r_quantile), grid_size)
            pdf = kde_estimator(grid)
            cdf_raw = np.cumsum(pdf)
            total = cdf_raw[-1] + l_buff + r_buff
            cdf = (l_buff + cdf_raw) / total
            p_d = interp1d(x=grid, y=cdf, copy=False, fill_value=(l_buff, 1.0 - r_buff), bounds_error=False, assume_sorted=False)
            percentile.append(p_d)
        return percentile
    

    def _get_action(self, X, thresholds):
        return compute_candidate_actions(X, thresholds, self.is_binary, self.is_integer)          
    
    
    def _get_cost(self, X, A, ptr):
        n_samples, n_thresholds = A.shape[0], A.shape[1]
        C = np.zeros((n_samples, n_thresholds), dtype=np.float64)
        for d in range(X.shape[1]):
            if self.is_immutable[d]:
                C[:, ptr[d]:ptr[d+1]] = np.inf
            else:
                A_d = A[:, ptr[d]:ptr[d+1], 1]
                if self.cost_type in ['MPS', 'TLPS']:
                    q_d = self._percentile[d]
                    q_0 = q_d(X[:, d])
                    X_cf = np.tile(X[:, d].reshape(-1, 1), A_d.shape[1]) + A_d
                    if self.cost_type == 'MPS':
                        C[:, ptr[d]:ptr[d+1]] = abs(np.tile(q_0.reshape(-1, 1), A_d.shape[1]) - q_d(X_cf))
                    else:
                        C[:, ptr[d]:ptr[d+1]] = abs(np.log2((1 - np.tile(q_0.reshape(-1, 1), A_d.shape[1])) / (1 - q_d(X_cf))))
                else:
                    C[:, ptr[d]:ptr[d+1]] = self._weight[d] * abs(A_d)
                if self.is_unincreasable[d]:
                    C[:, ptr[d]:ptr[d+1]][A_d>0] = np.inf
                if self.is_irreducible[d]:
                    C[:, ptr[d]:ptr[d+1]][A_d<0] = np.inf
        return C
    
    
    def get_flip_cost(self, X, thresholds, ptr):
        A = self._get_action(X, thresholds)
        C = self._get_cost(X, A, ptr)
        return C
    
    
    def _feasify(self, x, A, regions, cost_category):
        for cats in self.categories:
            i = A[:, cats].sum(axis=1)
            if (i == 1).sum() > 0:
                d = cats[np.where(x[cats] == 1)[0][0]]
                A[i == 1, d] = -1.0
            if (i == -1).sum() > 0:
                for l in np.arange(regions.shape[0])[i == -1]:
                    r = regions[l]
                    cats_feasible = [cats[j] for j, r_d in enumerate(r[cats]) if r_d[1] != 0.5]
                    d_feasible = cats_feasible[np.argmin([c_d for r_d, c_d in zip(r[cats], cost_category[cats]) if r_d[1] != 0.5])]
                    A[l, d_feasible] = 1.0
        return A  
    
    
    def _cost(self, x, A):
        if self.cost_type in ['MPS', 'TLPS']:
            C = np.zeros(A.shape[0])
            for d in range(A.shape[1]):
                q_d = self._percentile[d]
                if q_d is None:
                    continue
                if self.cost_type == 'MPS':
                    C = np.maximum(C, abs(q_d(x[d]) - q_d(x[d] + A[:, d])))
                else:
                    C += abs( np.log2( (1 - q_d(x[d] + A[:, d])) / (1 - q_d(x[d])) ) )
        else:
            if self.cost_ord == -1:
                C = np.max(abs(A) * self._weight, axis=1)
            else:
                C = (abs(A) ** self.cost_ord).dot(self._weight) 
        return C


    def enumerate_actions(self, X, regions, max_features):
        
        n_samples, n_features = X.shape
                
        cost_category = np.zeros(n_features, dtype=np.float64)
        for cats in self.categories:
            if self.cost_type == 'MPS':
                cost_category[cats] += np.array([MAX_VAL if self._percentile[d] is None 
                                                 else abs(self._percentile[d](1) - self._percentile[d](0)) for d in cats])                
            elif self.cost_type == 'TLPS':
                cost_category[cats] += np.array([MAX_VAL if self._percentile[d] is None 
                                                 else abs(np.log2((1-self._percentile[d](1)) / (1-self._percentile[d](0)))) for d in cats])                
            else:
                cost_category[cats] += self._weight[cats]
                
        null_region = np.zeros((1, n_features, 2), dtype=np.float64)
        null_region[:, :, 0], null_region[:, :, 1] = MIN_VAL, MAX_VAL
        regions = np.concatenate([regions, null_region])
        
        A_all = compute_all_actions(X, regions, self.is_binary, self.is_integer)
        A = np.array([self._feasify(X[i], A_all[i], regions, cost_category) for i in range(n_samples)], dtype=np.float64)
        F = is_feasible(A, self.is_immutable, self.is_unincreasable, self.is_irreducible, max_features)
        C = np.array([self._cost(X[i], A[i]) for i in range(n_samples)])
        
        return A, C, F