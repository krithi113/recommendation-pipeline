# Model Comparison

Two models were trained on the same training data (79,897 interactions before the time-split cutoff of 1998-03-07) and evaluated on the same test data, at K = 10.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity baseline | 0.1105 (±0.0177) | 0.0867 (±0.0179) | 0.1381 (±0.0213) | 0.0464 | 0.7491 | 1.0373 | 0.8336 |
| SVD matrix factorisation | 0.1200 (±0.0177) | 0.1285 (±0.0235) | 0.1781 (±0.0250) | 0.1560 | 0.7254 | 1.1503 | 0.9251 |

Values in brackets are standard errors across users.

## The models

- **Popularity baseline:** recommends the items with the most training ratings (`item_popularity_count` from the feature store). Its rating prediction is the item's average rating.
- **SVD (PureSVD):** TruncatedSVD with 20 factors on the mean-centred user-item matrix (explained variance 0.239). Its rating prediction is the user's average rating plus the latent score, clipped to 1-5.

## Evaluation protocol

- Test set: the latest 20% of interactions by time.
- Only users and items seen in training are evaluated (warm users and items). 2,863 test ratings from 95 users qualified.
- An item is relevant if its test rating is 4 or higher. Metrics are averaged over users with at least one relevant test item.
- Coverage is the share of warm items that appear in any user's top-10 list. Diversity is the mean pairwise genre dissimilarity (1 - cosine) inside a top-10 list.
- RMSE and MAE are computed for predicted ratings on the same warm test rows.

## Reading the results

- SVD is better on recall (0.1285 vs 0.0867), NDCG (0.1781 vs 0.1381) and coverage (0.1560 vs 0.0464, about 3.4 times).
- The precision gap (0.1200 vs 0.1105) is smaller than the standard error, so it should not be called a real difference.
- The popularity baseline predicts ratings better (RMSE 1.037 vs 1.150, MAE 0.834 vs 0.925) and its lists are slightly more diverse (0.749 vs 0.725).
- Recommending popular items is a strong baseline on MovieLens, and it is the right comparison for a personalised model.

## Limitations

- Only 95 users could be evaluated, so differences between the models are noisy.
- 192 test users and 67 test items do not appear in training (cold start). They are counted and excluded, and neither model serves them.
- The focus of the assignment is the data pipeline, not maximising accuracy, and the models were not tuned.
