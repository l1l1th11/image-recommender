# Image Recommender System

A modular image recommender for large scale datasets using multiple feature types and linear as well as annoy based similarity search.

---

## Features

- Feature types:  
    HSV (color histograms)  
    Embeddings (ResNet-based)  
    pHash (perceptual hashing)

- Search backends:  
    Linear (exact search)  
    Annoy (approximate search)

- Query modes:  
    Single image query  
    Multi image query (score aggregation)

- Performance optimizations:  
    Id to vector mappings for fast candidate lookup  
    Subset based re ranking for annoy  
    Persistent query loop for repeated queries

---

## Setup

conda env create -f environment.yml  
conda activate image_recommender

---

## Feature Extraction

Example:

python -m image_recommender.cli.main extract --feature-type hsv --input-mode db --run-dir data/features/db

---

## Query (Single Run)

python -m image_recommender.cli.main query --image-path "path/to/image.jpg" --run-dir data/features/db --backend annoy --display


## Query Loop (Recommended)

python -m image_recommender.cli.main query-loop --run-dir data/features/db --backend annoy --display

Inside the loop:

path/to/image1.jpg  
path/to/image1.jpg;path/to/image2.jpg

---

## Profiling

python -m image_recommender.cli.main profile-query --mode single --image-path "path/to/image.jpg" --run-dir data/features/db

Outputs:

- profile.stats (cProfile output)
- bottleneck visualization (PNG)

---

## Testing

pytest

---

## Summary

Image similarity search with feature based retrieval and optimized query performance on large datasets.
