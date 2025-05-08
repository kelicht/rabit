import numpy as np
from sklearn.preprocessing import OneHotEncoder

from rabit._utils import MIN_VAL, MAX_VAL, apply, region, find_min_value



class Node:
    
    def __init__(
        self, 
        feature=-2, 
        threshold=-2, 
        value=None, 
        depth=None,
        n_node_samples=None, 
        children_left=None, 
        children_right=None, 
        is_in_node=None,
        is_reach_node=None,
        idx_in=None,
        idx_reach=None,
    ):
        
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.depth = depth
        self.n_node_samples = n_node_samples
        self.children_left = children_left
        self.children_right = children_right
        self.is_in_node = is_in_node
        self.is_reach_node = is_reach_node
        self.idx_in = idx_in
        self.idx_reach = idx_reach

    @property
    def is_leaf(self):
        return self.feature < 0

    def set_index(self, index):
        self.index = index
        return self



class Tree:

    def compile(self, root):
        
        node_list = []
        stack = [root]
        max_depth = 0
        i = 0
        while stack:
            node = stack.pop()
            node_list.append(node)
            node = node.set_index(i)
            i = i + 1
            if not node.is_leaf:
                stack.append(node.children_right)
                stack.append(node.children_left)
            else:
                max_depth = max(max_depth, node.depth)
        self.node_list = node_list
        
        self.feature = np.array([node.feature for node in node_list])
        self.threshold = np.array([node.threshold for node in node_list])
        self.value = np.array([node.value for node in node_list])
        self.n_node_samples = np.array([node.n_node_samples for node in node_list])
        self.children_left = np.array([-1 if node.is_leaf else node.children_left.index for node in node_list])
        self.children_right = np.array([-1 if node.is_leaf else node.children_right.index for node in node_list])
        self.is_reach = np.array([node.is_reach_node for node in node_list])
        self.is_leaf = np.array([node.is_leaf for node in node_list])
        self.leaf_labels = np.where(self.is_leaf)[0]
        self.node_count = len(node_list)
        self.max_depth = max_depth
        
        self.encoder = OneHotEncoder()
        self.encoder = self.encoder.fit(self.leaf_labels.reshape(-1, 1))

        return self
    
    def apply(self, X):
        if self.node_count > 1:
            J = apply(X, self.feature, self.threshold, self.children_left, self.children_right)
        else:
            J = np.zeros(X.shape[0], dtype=np.int64)
        return J

    def region(self, n_features):
        if self.node_count > 1:
            R = region(self.feature, self.threshold, self.children_left, self.children_right, n_features, self.node_count)
        else:
            R = np.zeros((1, n_features, 2), dtype=np.float64)
            R[:, :, 0] = MIN_VAL
            R[:, :, 1] = MAX_VAL
        return R 
    
    def transform(self, X):
        X_leaf = self.apply(X)
        X_transform = self.encoder.transform(X_leaf.reshape(-1, 1))
        return X_transform
    
    def get_min_value(self):
        leaves = (self.feature < 0)
        is_reach = self.is_reach[leaves]
        value = self.value[leaves]
        V = find_min_value(is_reach, value)
        return V