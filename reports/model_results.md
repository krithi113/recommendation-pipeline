# Model Evaluation Results

K = 10; a test item is relevant if its rating >= 4. Evaluated on 95 users with training history (2863 test ratings of known users and items; 194 cold-start test users skipped).

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity baseline | 0.1105 (±0.0177) | 0.0867 (±0.0179) | 0.1381 (±0.0213) | 0.0464 | 0.7491 | 1.0373 | 0.8336 |
| SVD matrix factorisation | 0.1200 (±0.0177) | 0.1285 (±0.0235) | 0.1781 (±0.0250) | 0.1560 | 0.7254 | 1.1503 | 0.9251 |

Values in brackets are standard errors across users.
Coverage = share of the item catalogue appearing in any user's top-K list. Diversity = mean pairwise genre dissimilarity (1 - cosine) inside a top-K list.