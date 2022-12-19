| Feature | Description |
| --- | --- |
| **Name** | `en_entity_categorization` |
| **Version** | `0.0.0` |
| **spaCy** | `>=3.4.3,<3.5.0` |
| **Default Pipeline** | `transformer`, `spancat` |
| **Components** | `transformer`, `spancat` |
| **Vectors** | 0 keys, 0 unique vectors (0 dimensions) |
| **Sources** | n/a |
| **License** | n/a |
| **Author** | [n/a]() |

### Label Scheme

<details>

<summary>View label scheme (14 labels for 1 components)</summary>

| Component | Labels |
| --- | --- |
| **`spancat`** | `DATE negative`, `GPE negative`, `GPE positive`, `QUANTITY negative`, `DATE positive`, `PRODUCT negative`, `PRODUCT positive`, `MONEY positive`, `FAC negative`, `QUANTITY positive`, `MONEY negative`, `PERCENT positive`, `PERCENT negative`, `FAC positive` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `SPANS_SC_F` | 70.00 |
| `SPANS_SC_P` | 75.38 |
| `SPANS_SC_R` | 65.33 |
| `TRANSFORMER_LOSS` | 237.04 |
| `SPANCAT_LOSS` | 2543.82 |