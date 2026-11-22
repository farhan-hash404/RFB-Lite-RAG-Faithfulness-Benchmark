# RFB-Lite: RAG Faithfulness Benchmark (Lite)

A small, rigorous, reproducible benchmark that measures how faithfully **small open-weight LLMs** ground their
answers in retrieved passages, **claim by claim** and on a **graded** scale.

- **Dataset**: 150 (question, 5 BM25-retrieved passages, reference answer, claim annotations) triples, 75
  Wikipedia QA (Natural Questions) + 75 biomedical (PubMedQA), plus 6 held-out few-shot exemplars.
- **Metric**: Citation-Faithfulness Score (CFS): supported 1.0, partially supported 0.5, unsupported 0.0,
  contradicted -0.5, averaged over an answer's atomic claims. Reported with Hallucination Rate, Contradiction
  Rate, Claim Coverage, Abstention, Citation Precision and Misattribution.
- **Models**: Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B-v0.3 and Phi-3.5-mini (instruct), greedy decoding, 3-shot
  (primary) and 0-shot, plus a 4-bit NF4 vs FP16 ablation. All of it runs on one free Kaggle notebook.
- **Rigor**: 95% bootstrap CIs, paired bootstrap tests with Holm correction, a scorer validated against human
  labels (Spearman rho), and Cohen's kappa on a 40-triple double-annotated overlap set.

This is complementary to RAGAS, HHEM, FActScore, ALCE, RAGChecker, MiniCheck and FaithEval (see `paper/`).
Its niche is a *graded, claim-level, citation-aware* metric applied to the small-open-model ecosystem, packaged
as a dataset small enough to audit.

## Repository layout

```
rfb/                  harness (python -m rfb.cli <command>)
  build.py            question sampling + frozen BM25 retrieval
  annotate.py         reference answers, claim decomposition, LLM pre-labels
  annotator.py/.html  local web UI for human verification
  finalize.py         merge human labels, % corrected, Cohen's kappa
  generate.py         run evaluated models (transformers; FP16 / NF4; resumable)
  decompose.py        split model answers into atomic claims (LLM splitter, sentence fallback)
  scorers.py          per-passage NLI + MiniCheck probabilities -> 4-level labels, taxonomy
  metrics.py stats.py CFS/HR/CR/Coverage, bootstrap CIs, paired tests, Holm
  validation.py       scorer vs human labels (Spearman rho, weighted kappa, F1)
  analysis.py         tables, figures, failure examples
  report.py           fills paper/template.tex -> paper/main.tex
  kaggle.py           builds the self-contained Kaggle notebook
  hub.py              HF dataset card + upload
configs/models.json   evaluated models
data/rfb_v0.1/        pool.jsonl, fewshot.jsonl, triples.jsonl (LLM pre-labelled), rfb_v0.1.jsonl (final)
annotations/<name>/   human labels written by the UI
runs/<model>__<prec>__<k>shot/  generations, claims, scores (from the Kaggle run)
results/              results.json, tables/*.md|tex, figures/*.pdf|png
paper/                template.tex, main.tex (generated), refs.bib
notebooks/            rfb_kaggle_sweep.ipynb (generated, self-contained)
docs/annotation_guidelines.md
tests/                unit + end-to-end tests (pytest)
```

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY (free tier works with batching)

# 1. dataset (CPU + free LLM API)
python -m rfb.cli build         # sample 75+75 questions, BM25 top-5 (recall@5: wiki 0.87, biomed 0.96)
python -m rfb.cli annotate      # reference answers + atomic claims + LLM pre-labels (8 items per API call)
python -m rfb.cli annotate-ui --annotator <you>                    # verify every label by hand
python -m rfb.cli annotate-ui --annotator <friend> --overlap-only  # second annotator, 40-triple overlap
python -m rfb.cli finalize      # -> data/rfb_v0.1/rfb_v0.1.jsonl, % corrected, Cohen's kappa

# 2. model sweep (GPU): build the notebook, run it on Kaggle (T4 x2), unzip rfb_outputs.zip here
python -m rfb.cli kaggle        # -> notebooks/rfb_kaggle_sweep.ipynb
#    (or locally on a GPU: generate -> decompose -> score)

# 3. scorer validation (~80 model claims, labelled blind)
python -m rfb.cli validate-prep
python -m rfb.cli annotate-ui --annotator <you> --task scorer_validation
python -m rfb.cli validate

# 4. results + paper
python -m rfb.cli analyze       # results/tables, results/figures
python -m rfb.cli report        # paper/main.tex (upload paper/ to Overleaf or arXiv)
python -m rfb.cli push-hf --repo <user>/rag-faith-bench   # needs HF_TOKEN
pytest -q
```

### Kaggle run
1. Kaggle → Create → New Notebook → File → Import Notebook → `notebooks/rfb_kaggle_sweep.ipynb`.
2. Settings: Accelerator **GPU T4 x2**, Internet **On**. No API keys or HF token needed.
3. Run All. Stage 1 (FP16, 3-shot + 0-shot, ~2–2.5 h) writes `rfb_outputs.zip`; stage 2 (NF4 ablation, ~1 h)
   rewrites it. Download it from the Output panel and unzip it into this folder, then run `analyze` and `report`.

Edit `MODELS` / `CORE` / `ABLATION` / `LIMIT` in the notebook's first code cell (e.g. `LIMIT = 3` for a quick test).
Model answers are split into claims on the GPU by `Qwen/Qwen3-4B-Instruct-2507` (not one of the evaluated models),
using the same decomposition rules as the dataset, so the evaluation itself needs no API.

## Metric details
| Label | Score | Definition |
|---|---|---|
| Supported | 1.0 | Fully entailed by at least one retrieved passage |
| Partially supported | 0.5 | Some evidence, but the claim extends beyond the passages |
| Unsupported | 0.0 | No evidence in any passage (extrinsic) |
| Contradicted | -0.5 | Directly conflicts with passage content |

- **CFS** = mean claim score per answer, macro-averaged over non-abstaining answers (range -0.5 to 1).
- **HR** = (unsupported + contradicted) / claims, **CR** = contradicted / claims (pooled).
- **Coverage** = fraction of the grounded reference claims entailed by the answer (abstention = 0).
- **CitePrec** = cited claims supported by one of their cited passages; **MisAttr** = supported claims whose cited passages don't support them.
- Automated labels: MiniCheck-RoBERTa-Large (support) + DeBERTa-v3-large NLI (contradiction), per passage, with
  thresholds fixed a priori (`rfb/scorers.py:TH`). `--method nli|minicheck|hybrid` switches the scorer.

## Limitations
- 150 questions, two English domains: aggregate comparisons are the primary result; per-domain results are directional.
- Reference answers, reference claims and pre-labels come from one LLM (`gemini-3.5-flash-lite`, free tier, batched);
  human verification is required before the labels count as gold. The dataset card states the verification
  status and correction rate. Two few-shot exemplar answers were hand-edited (flagged in `fewshot.jsonl`).
- Model claims are labelled automatically; the scorer's agreement with human labels bounds which differences are meaningful.
- BM25 over a few thousand passages is easier than web-scale retrieval.
- Llama-3.1 and Gemma-2 load from the ungated `unsloth` mirrors (same weights as the gated official repos).

## License
Code: MIT. Annotations: CC-BY-4.0. Passages keep their source licences (Wikipedia CC-BY-SA, PubMedQA MIT).
