import numpy as np

from rabit._utils import Y_TARGET, find_best_actions



class RecourseExplainer():
    
    def __init__(
        self,
        estimator,
        action,
        max_features=-1,
        confidence=0.5, 
        plausibility=1.0,
        max_search=1000,         
    ):
        
        self.estimator = estimator
        self.action = action
        self.max_features = max_features
        self.confidence = confidence
        self.plausibility = plausibility
        self.max_search = max_search
            
    
    def _get_regions(self):
        regions = []
        for estimator in self.estimator.estimators_:
            leaves = np.array([ j for j in range(estimator.tree_.node_count) if estimator.tree_.feature[j] < 0 and estimator.tree_.value[j] > 0])
            if len(leaves) == 0: continue
            regions.append(estimator.regions_[leaves])
        if len(regions) == 0: 
            return []
        regions = np.concatenate(regions, axis=0)
        if regions.shape[0] > self.max_search:
            regions = regions[np.random.choice(regions.shape[0], size=self.max_search, replace=False)]
        return regions


    def _get_validity(self, X, A, F):
        V = F * (self.estimator.predict_proba(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0))[:, Y_TARGET] >= self.confidence).reshape(F.shape)
        if self.plausibility < 1.0:
            P = -1 * self.action._plausibility.score_samples(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0))
            V = V * (P <= self.plausibility).reshape(F.shape)
        return V

    
    def _get_counterfactuals(self, X, A_opt):
        X_cf = X + A_opt
        return X_cf
    
    
    def explain_recourse(self, X):
       
        is_target = (self.estimator.predict(X) == Y_TARGET)
        regions = self._get_regions()

        if len(regions) == 0:
            results = {
                'budget': self.action.cost_budget,
                'is_target': is_target,
                'X': X, 
                'action': np.zeros_like(X), 
                'X_cf': X, 
                'cost': np.zeros(X.shape[0]), 
                'valid': np.zeros(X.shape[0], dtype=np.bool_), 
            }
            return Recourse(**results)
        
        A, C, F = self.action.enumerate_actions(X, regions, self.max_features)
        V = self._get_validity(X, A, F)
            
        CA = find_best_actions(X, A, V, C)
        A_opt, C_opt = CA[:, 1:], CA[:, 0]

        X_cf = self._get_counterfactuals(X, A_opt)
        V_opt = (self.estimator.predict(X_cf) == Y_TARGET)
        P_opt = -1 * self.action._plausibility.score_samples(X_cf)
        
        results = {
            'budget': self.action.cost_budget,
            'is_target': is_target,
            'X': X, 
            'action': A_opt, 
            'X_cf': X_cf, 
            'cost': C_opt, 
            'valid': V_opt, 
            'plausibility': P_opt,
        }
        return Recourse(**results)    
    

    def generate_recourse_calibration_samples(self, X):
        
        regions = self._get_regions()
        A, C, F = self.action.enumerate_actions(X, regions, self.max_features)
        F = F * (C <= self.action.cost_budget)
        probabilities = (self.estimator.predict_proba(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0))[:, Y_TARGET]).reshape(F.shape)
        PA = find_best_actions(X, A, F, 1-probabilities)
        X_cf = X + PA[:, 1:]
        return X_cf

    

class CausalRecourseExplainer(RecourseExplainer):
    
    def __init__(
        self, 
        estimator, 
        action, 
        adjacency_matrix,
        max_features=-1, 
        confidence=0.5, 
        algorithm='ft', 
        max_search=1000,
    ):
        
        super().__init__(estimator, action, max_features, confidence, algorithm, max_search)
        self.adjacency_matrix = adjacency_matrix
        self._causal_effects = self._get_causal_effects(adjacency_matrix)


    def _get_causal_effects(self, adjacency_matrix):
        n_features = adjacency_matrix.shape[0]
        causal_effects = np.linalg.inv(np.eye(n_features) - adjacency_matrix)
        np.fill_diagonal(causal_effects, 0.0)
        return causal_effects    


    def _do_intervention(self, A):
        return self._causal_effects.dot(A.T).T
    
    
    def _get_validity(self, X, A, F):
        A_causal = A + self.do_intervention(np.concatenate(A, axis=0)).reshape(A.shape)
        V = F * (self.estimator.predict_proba(np.repeat(X, A.shape[1], axis=0) + np.concatenate(A_causal, axis=0))[:, Y_TARGET] >= self.confidence).reshape(F.shape)
        return V

    
    def _get_counterfactuals(self, X, A_opt):
        X_cf = X + A_opt + self.action.do_intervention(A_opt)            
        return X_cf



class Recourse():
    
    def __init__(
        self, 
        budget,
        is_target,
        X,
        action,
        X_cf,
        cost,
        valid,
        plausibility,
    ):
        
        self.budget = budget
        self.is_target = is_target
        self.X = X
        self.action = action
        self.X_cf = X_cf
        self.cost = cost
        self.valid = valid
        self.plausibility = plausibility
              
        
    def get_recourse(self, budget=True):
        if budget:
            return (self.valid & (self.cost <= self.budget)).mean()
        else:
            return self.valid.mean()

    def get_validity(self, budget=True):
        if budget:
            return (self.valid[~self.is_target] & (self.cost[~self.is_target] <= self.budget)).mean()
        else:
            return self.valid[~self.is_target].mean()

    def get_cost(self, valid=True):
        if valid:
            return self.cost[(~self.is_target) & self.valid].mean()
        else:
            return self.cost[~self.is_target].mean()
    
    def get_plausibility(self, valid=True):
        if valid:
            return self.plausibility[(~self.is_target) & self.valid].mean()
        else:
            return self.plausibility[~self.is_target].mean()

    def get_sparsity(self, valid=True):
        if valid:
            non_zeros = np.count_nonzero(self.action[(~self.is_target) & self.valid], axis=1)
        else:
            non_zeros = np.count_nonzero(self.action[~self.is_target], axis=1)
        return non_zeros.mean() 
    
    def get_recourse_for_each_group(self, sensitive_indices):
        Z = self.X[:, sensitive_indices]
        recourse = (self.valid & (self.cost <= self.budget))
        recourse_group = np.array([recourse[Z[:, i] == 1].mean() if (Z[:, i] == 1).sum() > 0 else 0.0 for i in range(Z.shape[1])])
        return recourse_group
    
    def get_unfairness(self, sensitive_indices):
        recourse_group = self.get_recourse_for_each_group(sensitive_indices)
        return recourse_group.max() - recourse_group.min()