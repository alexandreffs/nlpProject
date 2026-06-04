# BioGen NL Agent — Phases 1, 2 and 3

This repository contains the implementation and outputs for the BioGen NL Agent project for **Natural Language Processing and Search 2025/2026**.

The project is divided into three phases:

1. **Phase 1 — Data Parsing, Indexing and Search**  
   Builds a PubMed retrieval system using OpenSearch and compares BM25, Jelinek-Mercer language modelling, Dirichlet language modelling, and KNN dense retrieval.

2. **Phase 2 — Factually Grounded RAG**  
   Extends the retrieval system into a biomedical retrieval-augmented generation pipeline. Retrieved PubMed articles are split into sentences, ranked with a biomedical cross-encoder, and used to generate short citation-bearing answers.

3. **Phase 3 — Deep Research Agent**  
   Extends the RAG pipeline into an ADaPT-inspired biomedical Deep Research Agent. The system decomposes each question into subtopics, retrieves and selects evidence for each subtopic, aggregates the evidence, generates a structured report, validates citations, and evaluates the reports with an LLM-as-a-judge.

---

## Authors

- Alexandre Santos — 72970
- Miguel Mestre — 73018

Course: **Natural Language Processing and Search**

---

## Environment Variables

The notebooks use API keys through environment variables. Before running the LLM judge cells, define:

```bash
export IAEDU_API_KEY="your_iaedu_api_key"
export IAEDU_API_URL="your_iaedu_api_url"
```

For NovaSearch model calls, configure the endpoint and API key in the notebook if required.

---

## Phase 1: Retrieval

Phase 1 parses PubMed records into structured fields:

- `pmid`
- `title`
- `abstract`

The final retrieval query is formed by concatenating:

- `topic`
- `question`
- `narrative`

The OpenSearch index contains lexical fields for BM25, Jelinek-Mercer, and Dirichlet retrieval, as well as a dense `embedding` field for KNN retrieval.

The evaluated retrieval methods are:

- BM25
- Jelinek-Mercer language model
- Dirichlet language model
- KNN dense retrieval

The main retrieval metrics are:

- P@10
- R@100
- NDCG@10
- NDCG@100
- MAP
- MRR

KNN was the strongest method and is therefore used as the retrieval backbone for Phases 2 and 3.

---

## Phase 2: Factually Grounded RAG

Phase 2 uses the KNN retrieval output from Phase 1.

Pipeline:

1. Retrieve the top PubMed documents for each test query.
2. Split each abstract into sentences.
3. Score query-sentence pairs with `ncbi/MedCPT-Cross-Encoder`.
4. Select up to 3 reference sentences per retrieved article.
5. Generate a short grounded answer with `google/medgemma-4b-it`.
6. Evaluate reference sentence alignment and answer entailment with GPT-4o through the IAedu API.

Outputs:

```text
phase2_reference_sentences.json
phase2_generated_answers.json
phase2_judge_results.json
```

The Phase 2 judge labels are:

Sentence alignment:

- Required
- Borderline
- Unnecessary
- Inappropriate

Answer entailment:

- Supported
- Partially Supported
- Unsupported

---

## Phase 3: Deep Research Agent

Phase 3 implements an **ADaPT-inspired** biomedical Deep Research Agent.

Pipeline:

1. **Plan**  
   Generate focused biomedical subtopics for each patient question.

2. **Browse / Explore**  
   For each subtopic, run KNN retrieval and MedCPT sentence selection.

3. **Aggregate Evidence**  
   Group evidence by subtopic and remove duplicate PMID-sentence pairs.

4. **Synthesize Report**  
   Generate a structured patient-facing biomedical report using only the aggregated evidence.

5. **Validate Citations**  
   Extract cited PMIDs from the report and verify that they appear in the aggregated evidence.

6. **Judge Evaluation**  
   Use GPT-4o-mini through the IAedu API to evaluate factual support, citation correctness, completeness, and safety.

Outputs:

```text
phase3_plans.json
phase3_subtopic_evidence.json
phase3_reports.json
phase3_judge_results.json
```

The Phase 3 judge dimensions are:

- Factual support
- Citation correctness
- Completeness
- Safety

---

## Running the Project

### 1. Build or load the Phase 1 index

Make sure the OpenSearch index exists and contains the PubMed documents with lexical fields and the dense `embedding` field.

### 2. Run Phase 1 retrieval

Run the Phase 1 notebook to generate retrieval results. The KNN run should be saved as:

```text
outputs/phase1_outputs/retrieved_docs_knn.json
```

### 3. Run Phase 2

Run the Phase 2 notebook to generate:

```text
phase2_reference_sentences.json
phase2_generated_answers.json
phase2_judge_results.json
```

### 4. Run Phase 3

Run the Phase 3 notebook to generate:

```text
phase3_plans.json
phase3_subtopic_evidence.json
phase3_reports.json
phase3_judge_results.json
```

If the judge API fails, rerun the judge cells. The notebook should resume from the last saved successful judgment.
