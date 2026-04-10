# Medium Experiment Report

## Objective

The medium-scale experiment evaluates clustering behavior on a substantially larger sample (100,000 embeddings) to analyze stability, structure and parameter suitability beyond the pilot setting.

---

## Setup

- **Dataset:** 100,000 images (large enough to reveal stable global structure while remaining computationally tractable)
- **Features:** image embeddings

### Dimensionality Reduction (UMAP)

- **n_neighbors = 20**
  Increased compared to the pilot to better capture global structure in a larger dataset while still preserving local relationships.

- **min_dist = 0.2**
  Slightly increased to reduce over-clustering and allow smoother transitions between dense regions at higher sample size.

- **n_clusters = 18**
  Increased to capture finer semantic structure emerging at larger scale, where more distinct clusters begin to separate.

- **Projections:** 2D and 3D

### Visualization

- **Point size = 1.0**
  Reduced to avoid overplotting in dense regions.

- **Alpha = 0.1**
  Lowered to emphasize density patterns and prevent saturation in large clusters.

### Reproducibility

- **Seed:** DR_SEED = 42

---

## Results

The embedding space exhibits a well-defined global structure with multiple coherent semantic regions and meaningful transitions between them.

### Cluster Structure

- **Cluster 1:** Human mobility (walking and cycling)
- **Cluster 2:** Faces and small tight subgroup of sunglasses
- **Cluster 3:** Sports (football, baseball, tennis)
- **Cluster 4:** Landscapes (mountains, rivers)
- **Cluster 5:** Snow landscapes and skiing (adjacent to Cluster 4)
- **Cluster 6:** Drinks and table objects
- **Cluster 7:** Diffuse, low-semantic or visually homogeneous images, spread across central regions between clusters 9, 10, 12, 14 and 15
- **Cluster 8:** Animals
  - upper region: amphibians
  - lower-left region: birds
  - lower-right region: cats and dogs
- **Cluster 9:** Office/work environments (laptops, documents, books)
- **Cluster 10:** Posters, comics, advertisements
- **Cluster 11:** Water scenes (ocean, beach, surfing), located between clusters 1 and 4
- **Cluster 12:** Indoor scenes (rooms), with a subgroup of bathrooms/toilets on the far left
- **Cluster 13:** Food
  - left region: food with people
  - right region: isolated food items
- **Cluster 14:** Buildings (houses, castles), partially overlapping with landscapes
- **Cluster 15:** Transportation
  - right region: cars and buses
  - left region: trains and trams
- **Cluster 16:** Land animals
  - center: horses
  - upper region: elephants
  - lower region: giraffes
- **Cluster 17:** General human images, located below Cluster 2
- **Cluster 18:** Plants and flowers
  - upper region: transition toward vegetables (Cluster 13)
  - lower region: transition toward insects (Cluster 8)
  - far left: small subgroup of mushrooms

---

### Spatial Relationships

- Clusters generally reflect semantic proximity:
  - buildings <-> landscapes (buildings in natural environments)
  - food <-> plants (vegetables)
  - drinks & table objects <-> food (consumption context)
  - office/work environments <-> indoor scenes (laptops and books)
  - plants <-> animals (insects)
  - sports <-> outdoor environments

- Transitional regions connect major clusters rather than strictly separating them.

---

### Notable Patterns

- The diffuse cluster (Cluster 7) spans multiple regions, indicating low semantic specificity
- The animal cluster (Cluster 8) shows internal structure, while Cluster 16 forms a spatially distinct subregion within the broader animal space.
- Subgroups emerge based on visual features (e.g. colorful birds vs. other animals)
- Object-heavy clusters (e.g. office scenes) are clearly separated from natural content
- Small outlier groups appear:
  - mushrooms within the plant region
  - empty or unrecognized images forming isolated points

---

### 2D vs. 3D

- **2D:** highlights outliers and density structure more clearly, isolated groups (e.g. mushrooms on the far left) and the bridge-like structure of Cluster 16 between parts of Cluster 8 are more apparent.
- **3D:** improves separation of overlapping regions and confirms that Clusters 8 and 16 are spatially more isolated.

---

<div style="display: flex; gap: 10px;">

  <div style="flex: 1; text-align: center;">
    <img src="../../../data/experiments/viz/medium/2d_clusters.png" style="width: 500px;" />
    <div>2D Clusters</div>
  </div>

  <div style="flex: 1; text-align: center;">
    <img src="../../../data/experiments/viz/medium/3d_clusters.png" style="width: 500px;" />
    <div>3D Clusters</div>
  </div>

</div>

---

## Limitations

- KMeans enforces hard boundaries on inherently continuous data
- Some clusters reflect visual similarity rather than semantic meaning
- Diffuse regions indicate limits of embedding separability

---

## Conclusion

At medium scale, the embedding space forms stable and interpretable semantic structures.
Clusters align well with real-world categories while preserving continuous transitions between related concepts.