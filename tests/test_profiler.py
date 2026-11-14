"""Disclaimer:
   All of the test cases are AI generated to independently check the logic of the code as well as the edge cases."""

import numpy as np
import pytest
from profiler import classify_opponent, kmeans, profiler


# ==========================================
# 1. K-Means Algorithm & Edge Cases
# ==========================================


def test_kmeans_output_shapes_and_bounds():
  np.random.seed(42)
  X = np.random.randn(60, 3)
  k = 3
  centroids, labels = kmeans(X, k=k, iterations=50)

  assert centroids.shape == (k, 3)
  assert labels.shape == (60,)
  assert set(labels).issubset({0, 1, 2})


def test_kmeans_separates_distinct_clusters():
  np.random.seed(42)
  c0 = np.random.normal(loc=-10.0, scale=0.1, size=(20, 3))
  c1 = np.random.normal(loc=0.0, scale=0.1, size=(20, 3))
  c2 = np.random.normal(loc=10.0, scale=0.1, size=(20, 3))
  X = np.vstack([c0, c1, c2])

  centroids, labels = kmeans(X, k=3, iterations=100)

  # Each contiguous block of 20 points must share a single unique cluster label
  assert len(set(labels[:20])) == 1
  assert len(set(labels[20:40])) == 1
  assert len(set(labels[40:60])) == 1
  # All three blocks must be assigned to distinct clusters
  assert len({labels[0], labels[20], labels[40]}) == 3


def test_kmeans_single_iteration_convergence():
  np.random.seed(42)
  # Identical points in each cluster should converge immediately without NaN
  X = np.array(
      [[0.0, 0.0, 0.0]] * 10
      + [[5.0, 5.0, 5.0]] * 10
      + [[10.0, 10.0, 10.0]] * 10
  )
  centroids, labels = kmeans(X, k=3, iterations=50)

  assert not np.isnan(centroids).any()
  assert len(set(labels[:10])) == 1
  assert len(set(labels[10:20])) == 1
  assert len(set(labels[20:30])) == 1


def test_kmeans_k_equals_one_edge_case():
  np.random.seed(42)
  X = np.random.normal(loc=[2.0, 4.0, 6.0], scale=0.5, size=(30, 3))
  centroids, labels = kmeans(X, k=1, iterations=50)

  assert centroids.shape == (1, 3)
  assert np.all(labels == 0)
  assert np.allclose(centroids[0], np.mean(X, axis=0))


def test_kmeans_empty_cluster_safety():
  np.random.seed(42)
  # Only 2 unique points with k=3 forces at least one empty cluster
  X = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
  centroids, labels = kmeans(X, k=3, iterations=20)

  assert centroids.shape == (3, 3)
  assert not np.isnan(centroids).any()


# ==========================================
# 2. Profiler & Label-Switching Invariance
# ==========================================


def test_profiler_return_structures():
  np.random.seed(42)
  means, stds, centroids, cluster_to_alpha = profiler(visual=False)

  assert means.shape == (3,)
  assert stds.shape == (3,)
  assert np.all(stds > 0)
  assert centroids.shape == (3, 3)
  assert isinstance(cluster_to_alpha, dict)
  assert set(cluster_to_alpha.keys()) == {0, 1, 2}
  assert set(cluster_to_alpha.values()) == {0.75, 0.50, 0.25}


def test_profiler_label_switching_invariance():
  """
    Verifies that regardless of random K-Means initialization seed,
    the lowest-VPIP cluster always maps to 0.75 (Nit), middle to 0.50 (TAG),
    and highest to 0.25 (Maniac).
    """
  for seed in [0, 7, 21, 99, 123]:
    np.random.seed(seed)
    means, stds, centroids, cluster_to_alpha = profiler(visual=False)
    unscaled_centroids = centroids * stds + means

    # Sort cluster IDs by ascending VPIP (column 0)
    sorted_ids = np.argsort(unscaled_centroids[:, 0])
    assert cluster_to_alpha[sorted_ids[0]] == 0.75
    assert cluster_to_alpha[sorted_ids[1]] == 0.50
    assert cluster_to_alpha[sorted_ids[2]] == 0.25


# ==========================================
# 3. Out-of-Sample Opponent Classification
# ==========================================


def test_classify_canonical_archetypes():
  np.random.seed(42)
  means, stds, centroids, cluster_to_alpha = profiler(visual=False)

  # Canonical Nit: low VPIP, low PFR, low AF
  assert (
      classify_opponent([0.11, 0.07, 0.75], means, stds, centroids, cluster_to_alpha)
      == 0.75
  )
  # Canonical TAG: moderate VPIP/PFR, high AF
  assert (
      classify_opponent([0.22, 0.19, 2.80], means, stds, centroids, cluster_to_alpha)
      == 0.50
  )
  # Canonical Maniac: very high VPIP, PFR, and AF
  assert (
      classify_opponent([0.72, 0.62, 4.60], means, stds, centroids, cluster_to_alpha)
      == 0.25
  )


def test_classify_extreme_and_boundary_inputs():
  np.random.seed(42)
  means, stds, centroids, cluster_to_alpha = profiler(visual=False)

  # Extreme ultra-tight zero-stat rock -> must classify as Nit (0.75)
  assert (
      classify_opponent([0.0, 0.0, 0.0], means, stds, centroids, cluster_to_alpha)
      == 0.75
  )
  # Extreme hyper-loose 100% VPIP player -> must classify as Maniac (0.25)
  assert (
      classify_opponent([1.0, 1.0, 10.0], means, stds, centroids, cluster_to_alpha)
      == 0.25
  )
  # Accepts tuple or NumPy array inputs cleanly, not just Python lists
  alpha_tuple = classify_opponent(
      (0.21, 0.18, 2.7), means, stds, centroids, cluster_to_alpha
  )
  alpha_array = classify_opponent(
      np.array([0.21, 0.18, 2.7]), means, stds, centroids, cluster_to_alpha
  )
  assert alpha_tuple == 0.50
  assert alpha_array == 0.50


