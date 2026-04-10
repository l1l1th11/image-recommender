# Pilot Experiment Report

## Objective

The pilot-scale experiment evaluates clustering behavior on a small dataset (1,000 embeddings) to inspect semantic groupings.

---

## Setup

- **Dataset:** 1,000 images (small enough for fast inspection while providing sufficient diversity to observe semantic groupings)
- **Features:** image embeddings

### Dimensionality Reduction (UMAP)

- **n_neighbors = 15**
  Chosen to preserve local neighborhood structure in a small dataset while enabling clear cluster separation.

- **min_dist = 0.1**
  Low value to maintain tight clusters and highlight local variations.

- **n_clusters = 10**
  Chosen to quickly assess whether clear semantic structure emerges at small scale.

- **Projections:** 2D and 3D

### Visualization

- **Point size = 10.0**
  Slightly larger to emphasize individual points in a small dataset

- **Alpha = 0.5**
  Balanced to show density without overwhelming sparse clusters

### Reproducibility

- **Seed:** DR_SEED = 42

---

## Results

The embedding space shows distinct clusters reflecting semantic coherence, with clear local structure and some sparse or ambiguous regions.

### Cluster Structure

- **Cluster 1:** Faces and small tight subgroup of sunglasses
- **Cluster 2:** Plants, animals and ambiguous items
- **Cluster 3:** Nature and landscapes
- **Cluster 4:** Urban scenes (buildings, streets, vehicles)
- **Cluster 5:** Sports
- **Cluster 6:** Indoor spaces (kitchen, bedroom) with a small subgroup of cats
- **Cluster 7:** Food (lunch, desserts) with birthday scenes and teddy bears
- **Cluster 8:** Undefinable images
- **Cluster 9:** Wild animals (elephants, lions, giraffes)
- **Cluster 10:** Airplanes, boats and other transport

---

### Spatial Relationships

- Clusters generally reflect semantic proximity:
  - indoor environments <-> food
  - nature <-> wild animals
  - urban scenes <-> transportation

- Some clusters show overlap due to mixed content (e.g. Cluster 2 spreads across multiple semantic categories).

---

### Notable Patterns

- Sunglasses form a distinct subgroup within Cluster 1
- Cluster 2 overlaps with 6, 7 and 8 due to semantic diversity
- Cats appear consistently within indoor spaces (Cluster 6)
- Food-related subgroups reveal semantic coherence and human interaction (Cluster 7)
- Sparse clusters (Cluster 8) highlight ambiguous or low-semantic images
- Overlaps validate pipeline sensitivity to semantic similarity and dataset heterogeneity

---

### 2D vs. 3D

- **2D:** provides clear visualization for exploratory inspection  
- **3D:** reveals internal cluster shapes, subgroup distributions and separation of overlapping clusters  

---

<div style="display: flex; gap: 10px;">

  <div style="flex: 1; text-align: center;">
    <img src="../../../data/experiments/viz/pilot/2d_clusters.png" style="width: 500px;" />
    <div>2D Clusters</div>
  </div>

  <div style="flex: 1; text-align: center;">
    <img src="../../../data/experiments/viz/pilot/3d_clusters.png" style="width: 500px;" />
    <div>3D Clusters</div>
  </div>

</div>

---

## Limitations

- Small dataset size limits generalizability  
- Some clusters reflect visual similarity rather than purely semantic content  

---

## Conclusion

The pilot experiment confirms that the embedding pipeline produces coherent semantic clusters.
Subclusters and overlaps illustrate semantic structure, while 2D and 3D visualizations support exploratory analysis of the embedding space.