# Eval run: baseline-generation

_2026-11-02 22:53 UTC_ · 34 questions

## Config

```json
{
  "chunker": "structure",
  "retrieval_mode": "hybrid",
  "reranker": "BAAI/bge-reranker-base",
  "rerank_max_length": 512,
  "final_k": 5,
  "fused_k": 20,
  "embed_model": "Qwen/Qwen3-Embedding-0.6B",
  "llm_model": "gemma4:12b",
  "chunk_max_tokens": 512,
  "golden_set": "golden_set.yaml",
  "db": "pka"
}
```

## Overall

| metric | value |
|---|---|
| recall@5 | 0.982 |
| recall@20 | 1.000 |
| mrr | 0.905 |
| ndcg@5 | 0.914 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.000 |
| answers_with_citation | 1.000 |
| invalid_citation_rate | 0.000 |
| faithfulness | 1.000 |
| ttft_p50_ms | 11183 |
| ttft_p95_ms | 18503 |

## By category

| category | n | recall@5 | recall@20 | mrr | ndcg@5 | refusal_accuracy | false_refusal_rate | answers_with_citation | invalid_citation_rate | faithfulness |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 2 | 1.000 | 1.000 | 0.750 | 0.815 | – | 0.000 | 1.000 | 0.000 | 1.000 |
| identifier | 7 | 1.000 | 1.000 | 0.929 | 0.947 | – | 0.000 | 1.000 | 0.000 | 1.000 |
| lookup | 15 | 1.000 | 1.000 | 0.933 | 0.951 | – | 0.000 | 1.000 | 0.000 | 1.000 |
| negative | 6 | – | – | – | – | 1.000 | – | – | – | – |
| synthesis | 4 | 0.875 | 1.000 | 0.833 | 0.769 | – | 0.000 | 1.000 | 0.000 | 1.000 |

## Per question

| id | R@5 | MRR | refused | cites | notes |
|---|---|---|---|---|---|
| L01 | 1.000 | 0.500 | False | 1 |  |
| L02 | 1.000 | 1.000 | False | 1 |  |
| L03 | 1.000 | 1.000 | False | 1 |  |
| L04 | 1.000 | 1.000 | False | 1 |  |
| L05 | 1.000 | 1.000 | False | 1 |  |
| L06 | 1.000 | 1.000 | False | 1 |  |
| L07 | 1.000 | 1.000 | False | 1 |  |
| L08 | 1.000 | 1.000 | False | 1 |  |
| L09 | 1.000 | 1.000 | False | 1 |  |
| L10 | 1.000 | 1.000 | False | 1 |  |
| L11 | 1.000 | 1.000 | False | 1 |  |
| L12 | 1.000 | 0.500 | False | 1 |  |
| L13 | 1.000 | 1.000 | False | 1 |  |
| L14 | 1.000 | 1.000 | False | 1 |  |
| L15 | 1.000 | 1.000 | False | 1 |  |
| I01 | 1.000 | 1.000 | False | 1 |  |
| I02 | 1.000 | 1.000 | False | 2 |  |
| I03 | 1.000 | 0.500 | False | 3 |  |
| I04 | 1.000 | 1.000 | False | 1 |  |
| I05 | 1.000 | 1.000 | False | 1 |  |
| I06 | 1.000 | 1.000 | False | 1 |  |
| I07 | 1.000 | 1.000 | False | 1 |  |
| S01 | 1.000 | 1.000 | False | 3 |  |
| S02 | 1.000 | 1.000 | False | 3 |  |
| S03 | 0.500 | 1.000 | False | 2 |  |
| S04 | 1.000 | 0.333 | False | 3 |  |
| C01 | 1.000 | 1.000 | False | 1 |  |
| C02 | 1.000 | 0.500 | False | 2 |  |
| N01 | – | – | True | 0 |  |
| N02 | – | – | True | 0 |  |
| N03 | – | – | True | 0 |  |
| N04 | – | – | True | 0 |  |
| N05 | – | – | True | 0 |  |
| N06 | – | – | True | 0 |  |
