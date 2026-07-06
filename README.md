# RAQG-QPP: Query Performance Prediction with Retrieved Query Variants and Retrieval Augmented Query Generation

This repository contains the code (implemented with **PyTerrier**) for our paper:

**RAQG-QPP: Query Performance Prediction with Retrieved Query Variants and Retrieval Augmented Query Generation**

![RAQG-Overview-full](https://github.com/user-attachments/assets/01b13c55-4ccf-4fdb-a5cd-b0da49c28c35)

## Overview

Query Performance Prediction (QPP) aims to estimate the effectiveness of a query before relevance judgments are available. In this work, we propose **RAQG-QPP**, a QPP framework that leverages query variants (QVs) to improve the prediction accuracy. Here, query variants refer to the queries that contain similar information needs as the target query.

We obtain query variants using two methods:

1. **Retrieved query variants** — QVs retrieved from a training set, e.g., MS MARCO training set, by lexical or semantical match. We also extend the first-hop QVs (the queries that are directly similar to the target query) by a second-hop QV retrieval (see our paper for more details).
2. **Retrieval-augmented generated query variants** — **Contextually generated** QVs taking retrieved QVs as context, which informs the query generation process how real user queries resemble.

The query variants are further re-ranked by RBO, a measurement of the similarity between the information needs underlying two queries. The top-ranked QVs after the re-ranking are leveraged into the QPP estimation process.

Our repo includes the code for indexing queries, retrieving queries (by either 1-hop or 2-hop QV retrieval), re-ranking queries and QV-based QPP with a range of QPP methods. The experiment results will be a dataframe containing the QPP accuracies of the given setting with combination coefficient lambda at {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.

## Repository Structure

```text
RAQG_QPP/
├── doc_indices/
├── query_indices/
├── qpp_methods/
├── qv_res/
├── res/
├── experiment_qv_qpp.py
├── hopper.py
├── query_retrievers.py
├── retrieve_qvs.py
├── generate_qvs.py
└── rerank_gen_qv.py
```

## Example scripts:

### Prepare QVs:
#### Retrieving QVs:
python retrieve_qvs.py --dataset_name dl_19 --q_retriever sbert --hop_num 1

#### Generating QVs:
python generate_qvs.py --dataset_name dl_20 --q_retriever sbert --hop_num 2 --p 3

#### Reranking Generated QVs
python generate_qvs.py --dataset_name dl_20 --q_retriever sbert --hop_num 2 --p 3

### QPP Experiment:
python experiment_qv_qpp.py --dataset_name dl_22 --retrieval bm25_monoT5 --base_predictor nqc --q_retriever bm25 --hop_num 2 --k 3 --p 1
