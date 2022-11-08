| Feature | Description |
| --- | --- |
| **Name** | `en_spacy_causal_carbon` |
| **Version** | `0.0.0` |
| **spaCy** | `>=3.4.2,<3.5.0` |
| **Default Pipeline** | `tok2vec`, `spancat` |
| **Components** | `tok2vec`, `spancat` |
| **Vectors** | 0 keys, 0 unique vectors (0 dimensions) |
| **Sources** | n/a |
| **License** | n/a |
| **Author** | [n/a]() |

### Label Scheme

<details>

<summary>View label scheme (10 labels for 1 components)</summary>

| Component | Labels |
| --- | --- |
| **`spancat`** | `cause`, `effect`, `status`, `technology`, `core reference`, `emissions`, `timeline`, `location`, `carbon investment`, `financial information` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `SPANS_SC_F` | 0.02 |
| `SPANS_SC_P` | 0.01 |
| `SPANS_SC_R` | 32.12 |
| `TOK2VEC_LOSS` | 702164.31 |
| `SPANCAT_LOSS` | 3153215.62 |