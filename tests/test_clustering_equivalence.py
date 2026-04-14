"""Verify the batch Gram clustering computation is numerically equivalent to the old roll-based method."""

import numpy as np
import pytest


def _old_variance_covariance(var_i, var_m, N, cluster_ids):
    """Old roll-based implementation for reference."""
    K = var_i.shape[1]
    variance_covariance = 1 / N * np.array([
        var_i.T @ var_i, var_m.T @ var_m, var_i.T @ var_m
    ])
    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        for cid in unique_clusters:
            index = np.where(cluster_ids == cid)[0]
            v1_l = var_i[index, :]
            v2_l = var_m[index, :]
            v1_c = v1_l.copy()
            v2_c = v2_l.copy()
            for k in range(len(index) - 1):
                v1_c = np.roll(v1_c, 1, axis=0)
                v2_c = np.roll(v2_c, 1, axis=0)
                variance_covariance = variance_covariance + 1 / N * np.array([
                    v1_l.T @ v1_c, v2_l.T @ v2_c, v1_l.T @ v2_c
                ])
    return variance_covariance


def _new_block_gram(var, N, cluster_ids):
    """New batch Gram implementation."""
    M, _, K = var.shape
    if cluster_ids is not None:
        cluster_ids_flat = cluster_ids.flatten()
        unique_clusters = np.unique(cluster_ids_flat)
        C = len(unique_clusters)
        cluster_map = {c: idx for idx, c in enumerate(unique_clusters)}
        cidx = np.array([cluster_map[c] for c in cluster_ids_flat])
        cluster_sums = np.zeros((M, C, K), dtype=var.dtype)
        for m in range(M):
            np.add.at(cluster_sums[m], cidx, var[m])
        cs_flat = cluster_sums.transpose(1, 0, 2).reshape(C, M * K)
        return (1 / N) * cs_flat.T @ cs_flat
    else:
        var_flat = var.transpose(1, 0, 2).reshape(N, M * K)
        return (1 / N) * var_flat.T @ var_flat


@pytest.mark.parametrize("N,M,K,C,cluster_size", [
    (100, 2, 3, 10, 10),
    (200, 5, 5, 20, 10),
    (50, 3, 4, 5, 10),
])
class TestClusteringEquivalence:
    def test_with_clustering(self, N, M, K, C, cluster_size):
        rng = np.random.default_rng(42)
        cluster_ids = np.repeat(np.arange(C), cluster_size).reshape(-1, 1)
        var = rng.standard_normal((M, N, K))
        gram = _new_block_gram(var, N, cluster_ids)
        for i_idx in range(M):
            for m_idx in range(i_idx + 1, M):
                old = _old_variance_covariance(var[i_idx], var[m_idx], N, cluster_ids)
                new_ii = gram[i_idx * K:(i_idx + 1) * K, i_idx * K:(i_idx + 1) * K]
                new_mm = gram[m_idx * K:(m_idx + 1) * K, m_idx * K:(m_idx + 1) * K]
                new_im = gram[i_idx * K:(i_idx + 1) * K, m_idx * K:(m_idx + 1) * K]
                np.testing.assert_allclose(new_ii, old[0], atol=1e-12)
                np.testing.assert_allclose(new_mm, old[1], atol=1e-12)
                np.testing.assert_allclose(new_im, old[2], atol=1e-12)

    def test_without_clustering(self, N, M, K, C, cluster_size):
        rng = np.random.default_rng(42)
        var = rng.standard_normal((M, N, K))
        gram = _new_block_gram(var, N, None)
        for i_idx in range(M):
            for m_idx in range(i_idx + 1, M):
                old = _old_variance_covariance(var[i_idx], var[m_idx], N, None)
                new_ii = gram[i_idx * K:(i_idx + 1) * K, i_idx * K:(i_idx + 1) * K]
                new_mm = gram[m_idx * K:(m_idx + 1) * K, m_idx * K:(m_idx + 1) * K]
                new_im = gram[i_idx * K:(i_idx + 1) * K, m_idx * K:(m_idx + 1) * K]
                np.testing.assert_allclose(new_ii, old[0], atol=1e-12)
                np.testing.assert_allclose(new_mm, old[1], atol=1e-12)
                np.testing.assert_allclose(new_im, old[2], atol=1e-12)
