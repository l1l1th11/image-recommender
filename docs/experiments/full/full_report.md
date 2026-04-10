# Full Experiment Report

## Objective

The full-scale experiment evaluates clustering behavior on the complete dataset to analyze global semantic structure, stability and scaling effects beyond the medium setting.

---

## Setup

- **Dataset:** 504,703 images (maximal scale to reveal global structure and long-range semantic relationships)
- **Features:** image embeddings

### Dimensionality Reduction (UMAP)

- **n_neighbors = 30**
  Increased to capture broader neighborhood structure in a large dataset while preserving local relationships.

- **min_dist = 0.3**
  Higher value to reduce tight clustering and allow smoother transitions in the 2D/3D layout.

- **n_clusters = 20**
  Slightly increased compared to medium to reflect additional fine-grained structure while preserving stable macro-clusters.

- **Projections:** 2D and 3D

### Visualization

- **Point size = 0.3**
  Optimized for dense global layout

- **Alpha = 0.01**
  Highlights density structure and avoids overplotting

### Reproducibility

- **Seed:** DR_SEED = 42

---

## Results

The full embedding space reveals a highly structured semantic layout with both stable macro-clusters and fine-grained local substructures.

---

### Cluster Structure

- **Cluster 1:** Low-semantic, mostly single-color images
- **Cluster 2:** Snow landscapes and skiing
- **Cluster 3:** Office environments (laptops, keyboards, workspaces)
- **Cluster 4:** Faces
- **Cluster 5:** Humans, partially overlapping with sports-related scenes
- **Cluster 6:** Animals
  - left: giraffes
  - right: zebras
  - subregion at bottom: tiger
- **Cluster 7:** Landscapes (mountains, nature scenes)
- **Cluster 8:** Humans (non-face images, full-body and contextual scenes)
- **Cluster 9:** Sports (football, baseball, tennis)
- **Cluster 10:** Drinks and beverages
- **Cluster 11:** Traffic and transportation (roads, cars, trains, signals)
- **Cluster 12:** Animals
  - central: horses
  - upper region: elephants
- **Cluster 13:** Buildings (houses, churches, architectural structures)
- **Cluster 14:** Animals (birds, chickens, general wildlife, bottom subcluster of peacocks)
- **Cluster 15:** Humans with low color intensity / grayscale-like images
- **Cluster 16:** Plants
  - top: flowers
  - bottom: mushrooms
- **Cluster 17:** Food
  - top: desserts
  - bottom: pizza
- **Cluster 18:** Airplanes
- **Cluster 19:** Indoor environments (rooms)
  - left: kitchens (transition toward food)
  - lower region: bathrooms and utility spaces
  - mixed region: cats with toilets, laundry baskets, laptops
- **Cluster 20:** Water-related scenes (ocean, beach, surfing, boats)

---

### Spatial Relationships

- Clusters generally reflect semantic proximity:
  - landscapes <-> buildings (built structures in natural environments)
  - drinks <-> food (consumption context)
  - water scenes <-> landscapes (via waterfalls and natural water environments)
  - snow/ski scenes <-> airplanes (sky and brightness characteristics)
  - humans <-> indoor environments (contextual co-occurrence)

- Transitional regions indicate semantic blending rather than strict separation.

---

### Notable Patterns

- Low-semantic images form a strong baseline cluster and are spatially connected to office/work scenes due to low color diversity
- The animal cluster shows apparent spatial ordering (giraffes -> zebras -> tigers with interruptions from neighboring animal subgroups)
- A small intrusion of landscape-with-animal images appears within the animal cluster
- A localized peacock subcluster emerges within the bird cluster
- Space-related, sky-like images appear between landscapes and water

---

### 2D vs. 3D

- **2D:** highlights the dominance of the face cluster (Cluster 4), which appears as the largest and most dense region
- **3D:** confirms the spatial isolation of several animal-related clusters (Clusters 6, 12 and 14)

---

<div style="display: flex; gap: 10px;">

  <div style="flex: 1; text-align: center;">
    <img src="../../../data/experiments/viz/full/2d_clusters.png" style="width: 500px;" />
    <div>2D Clusters</div>
  </div>

  <div style="flex: 1; text-align: center;">
    <img src="../../../data/experiments/viz/full/3d_clusters.png" style="width: 500px;" />
    <div>3D Clusters</div>
  </div>

</div>

---

## Limitations

- KMeans enforces discrete boundaries on continuous semantic space
- Some clusters still reflect visual similarity rather than strict semantic categories
- Overlapping and densely packed regions indicate limits of separability at scale

---

## Conclusion

At full scale, the embedding space forms a stable and highly structured semantic layout with clear macro-clusters and meaningful transitions between related concepts.
The results show consistent semantic organization across scales, with robust cluster structure and suggest increasing stability at larger dataset size.
The observed spatial relationships indicate that the model captures meaningful high-level semantic continuity rather than purely visual similarity.