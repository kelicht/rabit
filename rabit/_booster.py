import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from rabit._tree import Node, Tree
from rabit._binning import BinningTransformer
from rabit._utils import split_sort_idx, compute_gain



class RecourseBooster(BaseEstimator, RegressorMixin):
    
    def __init__(
        self,
        action,
        max_depth=8,
        min_value_leaf=0.0,
        min_split_gain=0.0,
        max_features=None,
        n_bins=128,
        alpha=0.001,
        gamma=0.001,
        nu=1.0,
        only_actionable_features=False,
        n_jobs=None,
    ):

        self.action = action
        self.max_depth = max_depth
        self.min_value_leaf = min_value_leaf
        self.min_split_gain = min_split_gain
        self.max_features = max_features
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.only_actionable_features = only_actionable_features
        self.n_jobs = n_jobs
        

    def fit(self, X, y, z, beta):

        self.n_samples_, self.n_features_ = X.shape
        self.feature_count_ = np.zeros(self.n_features_)
        
        if self.max_features == 'sqrt':
            self.max_features_ = max(1, int(np.sqrt(self.n_features_)))
        elif type(self.max_features) == int:
            self.max_features_ = max(1, min(self.n_features_, self.max_features))
        else:
            self.max_features_ = self.n_features_
                    
        root = self._grow_tree(X, y, z, beta)
        self.tree_ = Tree().compile(root)
        self.regions_ = self.tree_.region(self.n_features_)
        
        return self
    
    
    def _get_grad_and_hess_of_loss(self, y, z):
        y_prob = 1 / (1 + np.exp(-z))
        grad = y_prob - y
        hess = y_prob * (1 - y_prob)
        return grad, hess
    
    
    def _get_cross_grad_and_hess_of_recs(self, beta):
        loss_grad = np.exp(-beta) / (1 + np.exp(-beta))
        loss_hess = loss_grad / (1 + np.exp(-beta))
        grad_x = 0.5 * loss_grad
        hess_x = 0.5 * loss_hess / self.nu
        hess_xy = (loss_grad ** 2) / (4 * self.nu)
        return grad_x, hess_x, hess_xy
    
    
    def _grow_tree(self, X, y, z, beta):
        
        transformer = BinningTransformer(n_bins=self.n_bins, n_jobs=self.n_jobs)
        transformer = transformer.fit(X)

        thresholds = []
        ptr = [0]
        for d in range(self.n_features_):
            thresholds.append(np.stack([np.array([d]*transformer.n_thresholds_[d]), transformer.thresholds_[d]], axis=1))
            ptr.append(transformer.n_thresholds_[d] + ptr[-1])
        thresholds = np.concatenate(thresholds, axis=0)
        flip_cost = self.action.get_flip_cost(X, thresholds, ptr)
        is_flip_threshold_root = (flip_cost <= self.action.cost_budget)
        
        if self.only_actionable_features:
            actionable_features = np.where(~self.action.is_immutable)[0]
            n_actionable_features = len(actionable_features)

        g_loss, h_loss = self._get_grad_and_hess_of_loss(y, z)
        g_hat, h_hat = self._get_grad_and_hess_of_loss(np.ones_like(y), beta)
        g_til, h_til, q_til = self._get_cross_grad_and_hess_of_recs(beta)
        
        weight_loss = 1 / self.n_samples_
        weight_recs = self.gamma / self.n_samples_

        value_root = - (weight_loss * g_loss.sum() + weight_recs * g_hat.sum()) / (weight_loss * h_loss.sum() + weight_recs * h_hat.sum() + self.alpha)
        idx_in_root = np.stack([np.argsort(X[:, d]) for d in range(X.shape[1])], axis=1)
        idx_reach_root = idx_in_root.copy()

        root = Node(
            value=value_root,
            depth=0,
            n_node_samples=self.n_samples_,
            is_in_node=np.ones(self.n_samples_, dtype=np.bool_),
            is_reach_node=np.ones(self.n_samples_, dtype=np.bool_),
            idx_in=idx_in_root,
            idx_reach=idx_reach_root           
        )

        stack = [root]
        node_list = []

        while stack:
            
            node = stack.pop()
            if node.depth >= self.max_depth:
                node_list.append(node)
                continue
            
            feature_mask = np.zeros(self.n_features_, dtype=int)
            if self.only_actionable_features:
                features_selected = actionable_features[np.random.choice(n_actionable_features, self.max_features_, replace=False)]
            else:
                features_selected = np.random.choice(self.n_features_, self.max_features_, replace=False)
            feature_mask[features_selected] = 1
            
            if self.action.cost_type == 'MPS' or self.action.cost_ord == -1:
                is_flip_threshold = is_flip_threshold_root
            else:
                scheduling_rate = ((node.depth + 1) / self.max_depth) ** 0.3
                cost_budget = self.action.cost_budget * scheduling_rate
                is_flip_threshold = (flip_cost <= cost_budget)
            
            results = compute_gain(
                X, 
                g_loss, 
                h_loss, 
                g_hat, 
                h_hat, 
                g_til,
                h_til,
                q_til,
                thresholds, 
                is_flip_threshold, 
                node.idx_in, 
                node.idx_reach, 
                feature_mask,
                weight_loss,
                weight_recs,
                self.alpha, 
            )
            
            j = np.argmax(results[:, 2])
            feature, threshold = int(results[j, 0]), results[j, 1]
            gain = results[j, 2]
            value_left, value_right = results[j, 3], results[j, 4]
            is_branch = (X[:, feature] <= threshold)

            if min(np.sum(is_branch[node.is_in_node]), np.sum(~is_branch[node.is_in_node])) == 0 or gain < self.min_split_gain or min(abs(value_left), abs(value_right)) <= self.min_value_leaf:
                node_list.append(node)
                continue

            is_in_node_left = node.is_in_node * is_branch
            is_in_node_right = node.is_in_node * (~is_branch)
            is_reach_left = node.is_reach_node * (is_branch | is_flip_threshold[:, j])
            is_reach_right = node.is_reach_node * ((~is_branch) | is_flip_threshold[:, j])
            
            idx_in_left = np.zeros((np.sum(is_in_node_left), self.n_features_), dtype=int)
            idx_in_right = np.zeros((np.sum(is_in_node_right), self.n_features_), dtype=int)
            split_sort_idx(is_in_node_left, is_in_node_right, node.idx_in, idx_in_left, idx_in_right)
            
            idx_reach_left = np.zeros((np.sum(is_reach_left), self.n_features_), dtype=int)
            idx_reach_right = np.zeros((np.sum(is_reach_right), self.n_features_), dtype=int)
            split_sort_idx(is_reach_left, is_reach_right, node.idx_reach, idx_reach_left, idx_reach_right)
            
            left_node = Node(
                value=value_left,
                depth=node.depth+1,
                n_node_samples=np.sum(is_in_node_left),
                is_in_node=is_in_node_left,
                is_reach_node=is_reach_left,
                idx_in=idx_in_left,
                idx_reach=idx_reach_left
            )
            right_node = Node(
                value=value_right,
                depth=node.depth+1,
                n_node_samples=np.sum(is_in_node_right),
                is_in_node=is_in_node_right,
                is_reach_node=is_reach_right,
                idx_in=idx_in_right,
                idx_reach=idx_reach_right
            )            
            node.feature = feature
            node.threshold = threshold
            node.children_left = left_node
            node.children_right = right_node

            node_list.append(node)
            stack.append(right_node)
            stack.append(left_node)
            self.feature_count_[feature] += 1
                        
        return root

        
    def predict(self, X):
        J = self.tree_.apply(X)
        return self.tree_.value[J]
    
    
    def transform(self, X):
        X_transformed = self.tree_.transform(X)
        return X_transformed
    

    def get_min_value(self):
        min_values = self.tree_.get_min_value()
        return min_values
    
    @property
    def feature_importances_(self):
        sum_feature_count = np.sum(self.feature_count_)
        if sum_feature_count == 0:
            return np.zeros(self.n_features_)
        else:
            return self.feature_count_ / sum_feature_count

