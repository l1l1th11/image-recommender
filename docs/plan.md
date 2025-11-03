A:
* **Block 2** — DB & Pilot
* **Block 5** — HSV
* **Block 6** — Embeddings
* **Block 8** — Search backends (Annoy + linear)
* **Block 11** — Evaluation & Profiling
* **Block 12** — Docs & slides *(secondary)*

B:
* **Block 1** — Repo & CI
* **Block 3** — I/O & Reliability
* **Block 4** — Feature Storage
* **Block 7** — Perceptual Hash
- **Block 9** — Recommender & CLI
* **Block 10** — Visualization
* **Block 12** — Docs & slides *(primary)*

---


**Week 1**

- A: **Block 2 — DB & Pilot** (schema, ingest, pilot list)    
- B: **Block 1 — Repo & CI**, CLI base
    

**Week 2**

- A: **Block 5 — HSV (samples-only)**: extractor + distances; self-contained read helper  
- B: **Block 3 — I/O & Reliability (complete)**: loader (RGB uint8, skip-bad, log) + atomic writes + sharding math & CLI flags + resume markers & pending-shards helper
- B: **Block 4 — Feature Storage (minimal)**: `.npy` + `ids.txt` + `meta.json` (atomic)
    

**Week 3**

- A: **Block 5 — HSV (pilot on HDD → finish full)** and **Block 8 — Linear top-k** (E2E search on HSV)
- B: **Block 7 — Perceptual Hash (complete)**: extractor + distances (Hamming) + **samples** + **pilot** + **finish (full set)**
    

**Week 4**

- A: **Block 6 — Embeddings (samples)**: extractor + distances   
- B: **Block 9 — Recommender & CLI**: default backend = linear; flag to switch to Annoy
    

**Week 5**

- A: **Block 8 — Annoy (wiring & build, pilot)**; index persistence  
- B: **Block 10 — Visualization**: pilot map; subsampled full; exports
    

**Week 6**

- A: **Block 8 — Annoy (full build)**
- B: **Block 9 — polish** (params/UX) + **Docs/README baseline**
    

**Week 7**

- A: **Block 11** — Evaluation & Profiling: timing hooks, pilot+full evals, summaries
- B: **Block 12 — Docs & Slides (main work)**: outline, core sections, key visuals, draft slides
    

**Week 8**

- A+B: **Block 12 — Docs & Slides (finalize)**: edits, manifests, polish; release/tag; final CI check
