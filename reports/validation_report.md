# Data Validation Report

Generated: 2026-11-14 17:05:11

## Summary

| Table | Rows in | Valid | Rejected |
|---|---:|---:|---:|
| users | 956 | 943 | 13 |
| items | 1689 | 1682 | 7 |
| interactions | 100050 | 99870 | 180 |

## Rejected rows by reason (detected vs injected)

| Table | Reason | Detected | Injected | Match |
|---|---|---:|---:|:--:|
| users | invalid_age | 4 | 4 | yes |
| users | missing_age | 3 | 3 | yes |
| users | invalid_gender | 3 | 3 | yes |
| users | duplicate | 3 | 3 | yes |
| items | missing_title | 4 | 4 | yes |
| items | duplicate | 3 | 3 | yes |
| interactions | duplicate | 50 | 50 | yes |
| interactions | invalid_rating | 30 | 30 | yes |
| interactions | invalid_timestamp | 30 | 30 | yes |
| interactions | orphan_item | 20 | 20 | yes |
| interactions | missing_rating | 20 | 20 | yes |
| interactions | orphan_user | 20 | 20 | yes |
| interactions | bad_dtype | 10 | 10 | yes |

## Checks implemented

- Missing values in required columns
- Incorrect data types (text where a number is expected)
- Invalid ratings (outside 1-5)
- Invalid timestamps (<= 1990 or in the future)
- Invalid user age and gender values
- Duplicate users, items and (user, item) interactions
- Referential integrity (interactions must reference a valid user and item)

Rejected rows are stored in `data/errors/`, valid rows in `data/processed/`.