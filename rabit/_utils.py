import numpy as np
import numba

Y_TARGET = 1
MIN_VAL = - 1e+8
MAX_VAL =   1e+8


@numba.njit("int64(int64, float64, float64)", cache=True)
def compute_optimal_k(n_samples, epsilon, delta):
    r = 0.0
    s = 0.0
    for n in range(n_samples + 1):
        if n == 0:
            r = n_samples * np.log(1.0 - epsilon)
        else:
            r += np.log(n_samples - n + 1) - np.log(n) + np.log(epsilon) - np.log(1.0 - epsilon)
        s += np.exp(r)
        if s > delta:
            if n == 0:
                raise Exception('No valid threshold')
            else:
                return n - 1
    return n_samples


@numba.njit("int64[:](float64[:, :], int64[:], float64[:], int64[:], int64[:])", parallel=True, cache=True)
def apply(X, feature, threshold, children_left, children_right):
    J = np.zeros(X.shape[0], dtype=np.int64)
    for i in numba.prange(X.shape[0]):
        while feature[J[i]] >= 0:
            if X[i, feature[J[i]]] <= threshold[J[i]]:
                J[i] = children_left[J[i]]
            else:
                J[i] = children_right[J[i]]
    return J


@numba.njit("float64[:, :, :](int64[:], float64[:], int64[:], int64[:], int64, int64)", cache=True)
def region(feature, threshold, children_left, children_right, n_features_in, node_count):
    R = np.zeros((node_count, n_features_in, 2), dtype=np.float64)
    R[:, :, 0] = MIN_VAL
    R[:, :, 1] = MAX_VAL
    for j in range(node_count):
        if feature[j] < 0:
            continue
        else:
            R[children_left[j]] = R[j]
            R[children_right[j]] = R[j]
            R[children_left[j], feature[j], 1] = threshold[j]
            R[children_right[j], feature[j], 0] = threshold[j]
    return R


@numba.njit("float64[:](boolean[:, :], float64[:])", parallel=True, cache=True)
def find_min_value(is_reach, value):
    n_samples = is_reach.shape[1]
    V = np.zeros(n_samples, dtype=np.float64)
    for i in numba.prange(n_samples):
        for l in range(value.shape[0]):
            if is_reach[l, i]:
                V[i] = min(value[l], V[i])
    return V


@numba.njit("void(boolean[:], boolean[:], int64[:, :], int64[:, :], int64[:, :])", parallel=True, cache=True)
def split_sort_idx(is_in_node_left, is_in_node_right, sort_idx, sort_idx_left, sort_idx_right):
    for d in numba.prange(sort_idx.shape[1]):
        l, r = 0, 0
        for i in sort_idx[:, d]:
            if is_in_node_left[i]:
                sort_idx_left[l, d] = int(i)
                l = l + 1
            if is_in_node_right[i]:
                sort_idx_right[r, d] = int(i)
                r = r + 1


@numba.njit("float64[:, :](float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :], boolean[:, :], int64[:, :], int64[:, :], int64[:], float64, float64, float64)", cache=True)
def compute_gain(
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
    idx_in, 
    idx_reach, 
    feature_mask,
    weight_loss,
    weight_recs,
    alpha, 
):
    
    G_loss, H_loss = g_loss[idx_in[:, 0]].sum(), h_loss[idx_in[:, 0]].sum()
    G_hat, H_hat = g_hat[idx_reach[:, 0]].sum(), h_hat[idx_reach[:, 0]].sum()
    loss_current = - (1/2) * (weight_loss * G_loss + weight_recs * G_hat)**2 / (H_loss + weight_recs * H_hat + alpha)
    
    n_thresholds = thresholds.shape[0]
    results = np.zeros((n_thresholds, 5), dtype=np.float64)
    results[:, 2] = MIN_VAL
    
    d_prev = -1
    for k, (d, t) in enumerate(thresholds):

        results[k, 0] = d
        results[k, 1] = t       
        d = int(d)

        if not feature_mask[d]:
            continue

        if d != d_prev:
            d_prev = d
            G_loss_left, H_loss_left = 0.0, 0.0
            G_loss_right, H_loss_right = G_loss, H_loss
            i = 0
            G_hat_left, H_hat_left = 0.0, 0.0
            G_til, H_til, Q_til = 0.0, 0.0, 0.0
            G_hat_right, H_hat_right = G_hat, H_hat
            j_left, j_right = 0, 0

        while i < idx_in.shape[0] and X[idx_in[i, d], d] <= t:
            G_loss_left = G_loss_left + g_loss[idx_in[i, d]]
            H_loss_left = H_loss_left + h_loss[idx_in[i, d]]
            G_loss_right = G_loss_right - g_loss[idx_in[i, d]]
            H_loss_right = H_loss_right - h_loss[idx_in[i, d]]
            i = i + 1

        if weight_recs > 0.0:

            while j_left < idx_reach.shape[0] and (X[idx_reach[j_left, d], d] <= t and (not is_flip_threshold[idx_reach[j_left, d], k])):            
                G_hat_left = G_hat_left + g_hat[idx_reach[j_left, d]]
                H_hat_left = H_hat_left + h_hat[idx_reach[j_left, d]]
                G_til = G_til - g_til[idx_reach[j_left, d]]
                H_til = H_til - h_til[idx_reach[j_left, d]]
                Q_til = Q_til - q_til[idx_reach[j_left, d]]
                j_left = j_left + 1
                
            while j_right < idx_reach.shape[0] and (X[idx_reach[j_right, d], d] > t and is_flip_threshold[idx_reach[j_right, d], k]):
                G_til = G_til + g_til[idx_reach[j_right, d]]
                H_til = H_til + h_til[idx_reach[j_right, d]]
                Q_til = Q_til + q_til[idx_reach[j_right, d]]
                G_hat_right = G_hat_right - g_hat[idx_reach[j_right, d]]
                H_hat_right = H_hat_right - h_hat[idx_reach[j_right, d]]
                j_right = j_right + 1

        G_left = weight_loss * G_loss_left + weight_recs * (G_hat_left + G_til)
        G_right = weight_loss * G_loss_right + weight_recs * (G_hat_right + G_til)
        H_left = weight_loss * H_loss_left + weight_recs * (H_hat_left + H_til) + alpha
        H_right = weight_loss * H_loss_right + weight_recs * (H_hat_right + H_til) + alpha

        denominator = H_left * H_right - (weight_recs * Q_til)**2
        value_left = (weight_recs * Q_til * G_right - H_right * G_left) / denominator
        value_right = (weight_recs * Q_til * G_left - H_left * G_right) / denominator
        loss = G_left * value_left + 0.5 * H_left * value_left**2 + G_right * value_right + 0.5 * H_right * value_right**2 + weight_recs * Q_til * value_left * value_right
        
        results[k, 2] = loss_current - loss
        results[k, 3] = value_left
        results[k, 4] = value_right
                    
    return results


@numba.njit("float64[:, :, :](float64[:, :], float64[:, :], boolean[:], boolean[:])", parallel=True, cache=True)
def compute_candidate_actions(X, thresholds, is_binary, is_integer):
    A = np.zeros((X.shape[0], thresholds.shape[0], 2), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for j in numba.prange(thresholds.shape[0]):
            d, b = thresholds[j]
            A[i, j, 0] = d
            d = int(d)
            if X[i, d] <= b:
                if is_binary[d]:
                    A[i, j, 1] = 1.0
                elif is_integer[d]:
                    A[i, j, 1] = int(b) - X[i, d] + 1.0
                else:
                    A[i, j, 1] = b - X[i, d] + 1e-8
            else:
                if is_binary[d]:
                    A[i, j, 1] = - 1.0
                elif is_integer[d]:
                    A[i, j, 1] = int(b) - X[i, d]
                else:
                    A[i, j, 1] = b - X[i, d]                    
    return A


@numba.njit("float64[:, :, :](float64[:, :], float64[:, :, :], boolean[:], boolean[:])", parallel=True, cache=True)
def compute_all_actions(X, regions, is_binary, is_integer):
    A = np.zeros((X.shape[0], regions.shape[0], X.shape[1]), dtype=np.float64)
    for i in numba.prange(X.shape[0]):
        for l in numba.prange(regions.shape[0]):
            for d in numba.prange(X.shape[1]):
                if X[i, d] <= regions[l][d][0]:
                    if is_binary[d]:
                        A[i, l, d] = 1.0
                    elif is_integer[d]:
                        A[i, l, d] = int(regions[l][d][0]) - X[i, d] + 1.0
                    else:
                        A[i, l, d] = regions[l][d][0] - X[i, d] + 1e-8
                elif X[i, d] <= regions[l][d][1]:
                    A[i, l, d] = 0.0
                else:
                    if is_binary[d]:
                        A[i, l, d] = - 1.0
                    elif is_integer[d]:
                        A[i, l, d] = int(regions[l][d][1]) - X[i, d]
                    else:
                        A[i, l, d] = regions[l][d][1] - X[i, d]
    return A


@numba.njit("boolean[:, :](float64[:, :, :], boolean[:], boolean[:], boolean[:], int64)", parallel=True, cache=True)
def is_feasible(A, is_immutable, is_unincreasable, is_irreducible, max_features):
    F = np.ones((A.shape[0], A.shape[1]), dtype=np.bool_)
    is_fix = (is_immutable.sum() > 0)
    is_inc = (is_unincreasable.sum() > 0)
    is_red = (is_irreducible.sum() > 0)
    for i in numba.prange(A.shape[0]):
        if is_fix:
            F[i] = F[i] * (np.count_nonzero(A[i][:, is_immutable], axis=1) == 0)
        if is_inc:
            F[i] = F[i] * (np.count_nonzero(np.clip(A[i][:, is_unincreasable], 0.0, None), axis=1) == 0)
        if is_red:
            F[i] = F[i] * (np.count_nonzero(np.clip(A[i][:, is_irreducible], None, 0.0), axis=1) == 0)
        if max_features > 0: 
            F[i] = F[i] * (np.count_nonzero(A[i], axis=1) <= max_features)
    return F


@numba.njit("float64[:, :](float64[:, :], float64[:, :, :], boolean[:, :], float64[:, :])", parallel=True, cache=True)
def find_best_actions(X, A, V, C):
    CA = np.zeros((X.shape[0], 1 + X.shape[1]), dtype=np.float64)
    C_opt = np.ones(X.shape[0], dtype=np.float64) * MAX_VAL
    for i in numba.prange(X.shape[0]):
        if V[i].sum() == 0: continue
        for l in range(V.shape[1]):
            if V[i, l] and (C[i, l] < C_opt[i]):
                CA[i, 0] = C[i, l]
                CA[i, 1:] = A[i, l]
                C_opt[i] = C[i, l]
    return CA


@numba.njit("int64(float64, float64[:])", cache=True)
def binary_search(x_d, thresholds_d):
    left, right = 0, thresholds_d.shape[0]
    while left < right:
        mid = left + (right - left - 1) // 2
        if x_d <= thresholds_d[mid]:
            right = mid
        else:
            left = mid + 1
    return left


@numba.njit("int64[:](float64[:], float64[:])", parallel=True, cache=True)
def map_to_bins(X_d, thresholds_d):
    X_bin = np.zeros_like(X_d, dtype=np.int64)
    for i in numba.prange(X_d.shape[0]):
        X_bin[i] = binary_search(X_d[i], thresholds_d)
    return X_bin


