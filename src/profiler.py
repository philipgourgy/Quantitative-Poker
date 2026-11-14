import matplotlib.pyplot as plt
import numpy as np

def kmeans(player_Z, k, iterations=100):
  idx = np.random.choice(len(player_Z), size=k, replace=False)
  centroids = player_Z[idx]

  for _ in range(iterations):
    old_centroids = centroids.copy()

    labels = []
    for player in player_Z:
      v = player - centroids
      distances = np.linalg.norm(v, axis=1)
      closest_pin = np.argmin(distances)
      labels.append(closest_pin)
    labels = np.array(labels)

    for c in range(k):
      team = player_Z[labels == c]
      if len(team) > 0:
        centroids[c] = np.mean(team, axis=0)

    if np.allclose(old_centroids, centroids):
      break

  return centroids, labels


def profiler(visual=False):
  nit_vpip = np.random.normal(0.12, 0.02, 30)
  nit_pfr = np.random.normal(0.08, 0.015, 30)
  nit_af = np.random.normal(0.8, 0.25, 30)

  tag_vpip = np.random.normal(0.22, 0.03, 30)
  tag_pfr = np.random.normal(0.19, 0.025, 30)
  tag_af = np.random.normal(2.8, 0.4, 30)

  maniac_vpip = np.random.normal(0.7, 0.07, 30)
  maniac_pfr = np.random.normal(0.6, 0.07, 30)
  maniac_af = np.random.normal(4.5, 0.6, 30)

  nits = np.column_stack([nit_vpip, nit_pfr, nit_af])
  tags = np.column_stack([tag_vpip, tag_pfr, tag_af])
  maniacs = np.column_stack([maniac_vpip, maniac_pfr, maniac_af])
  players = np.vstack([nits, tags, maniacs])

  means = np.mean(players, axis=0)
  stds = np.std(players, axis=0)
  players_Z = (players - means) / stds
  centroids, labels = kmeans(players_Z, 3, 100)

  player_stats = centroids * stds + means
  team = np.argsort(player_stats[:, 0])
  nit_idx, tag_idx, maniac_idx = team
  cluster_to_alpha = {
      nit_idx: 0.75,
      tag_idx: 0.50,
      maniac_idx: 0.25,
  }

  if visual:
    plt.scatter(players[:, 0], players[:, 1])
    plt.xlabel("VPIP")
    plt.ylabel("PFR")
    plt.title("Opponent Archetype Clusters")
    plt.show()

  return means, stds, centroids, cluster_to_alpha


def classify_opponent(new_stats, means, stds, centroids, cluster_to_alpha):
  z = (np.array(new_stats) - means) / stds
  distances = np.linalg.norm(centroids - z, axis=1)
  closest_cluster = np.argmin(distances)
  return cluster_to_alpha[closest_cluster]
