A:
* **Block 2** — DB & Pilot
* **Block 6** — Embeddings
* **Block 8** — Search backends (Annoy + linear)
* **Block 9** — Recommender & CLI *(solo)*
* **Block 11** — Evaluation & Profiling
* **Block 12** — Docs & slides *(secondary)*

B:
* **Block 1** — Repo & CI
* **Block 3** — I/O & Reliability
* **Block 4** — Feature Storage
* **Block 5** — HSV
* **Block 7** — Perceptual Hash
* **Block 10** — Visualization
* **Block 12** — Docs & slides *(primary)*

---

**Week 1**
A: **Block 2** — DB & Pilot (schema, full metadata ingest, pilot list)
B: **Blocks 1, 3, 4** — Repo & CI; I/O & Reliability; Feature Storage

**Week 2**
A: **Block 6** — Embeddings (extractor, distances, samples)
B: **Block 5** — HSV (extractor, distances, pilot on HDD → start full)

**Week 3**
A: **Block 6** — Embeddings (pilot on HDD → start full)
B: **Block 7** — Perceptual Hash (hash extractor, Hamming, samples)

**Week 4**
A: **Block 8** — Search backends (linear top-k, Annoy)
B: **Blocks 7, 10** — Perceptual Hash (pilot → start full); Viz wrappers

**Week 5**
A: **Blocks 8, 9** — Annoy build (pilot/full); Recommender & CLIs
B: **Blocks 1, 12** — CI tighten (e2e on samples); docs skeleton/README

**Week 6**
A: **Block 11** — Eval & profiling (hooks, latency/size, quality proxy; pilot evals)
B: **Block 10** — Mapping & visuals (pilot map; optional subsampled full; exports)

**Week 7**
A: **Block 11** — Full eval & demo (full runs, query demo rehearsal)
B: **Block 12** — Docs & slides (metrics/profiling/analysis; CI green)

**Week 8**
A: **Block 12** — Docs finalization, release/tag, CI check
B: **Blocks 10, 12** — Visual polish & slides; verify manifests
