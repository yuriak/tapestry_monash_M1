# Centralized reranker recipes

Run the shell scripts in `scripts/reranker/<task>/` from a source checkout. Each script invokes its Python module directly. Recipe defaults live in `configs/reranker/recipes/`. CaseHOLD and NFCorpus use their existing JSON configs. Training scripts write the loss, adapters, predictions and evaluation metrics under the output directory you supply. Use a distinct output directory per dataset, recipe and model. Several evaluation routines resume from saved scores or metrics by path.

## Environment and inputs

Use Python 3.11 and a BF16-capable CUDA device for model runs. For CaseHOLD and NFCorpus use the M1 environment and model asset layout documented in the root README. For the research recipes use the requirements in the corresponding config directory, e.g. `uv pip install --python "$PYTHON" -r configs/reranker/recipes/bioasq/requirements.txt` in a dedicated environment. Evidence Inference also installs the scientific English sentence model. The shell scripts default to the active `python`. Set `PYTHON` to override it. `GPUS` sets the number of processes launched on one machine, with one GPU per process for distributed stages. CaseHOLD's epoch config specifies four. Encoding and merging must use the same shard count. For a multi-host allocation invoke the same Python module via your distributed launcher with `PYTHONPATH=src`.

Use local snapshots of Qwen/Qwen3-Reranker-0.6B (`e61197ed45024b0ed8a2d74b80b4d909f1255473`) and Qwen/Qwen3-Embedding-4B (`5cf2132abc99cad020ac570b19d031efec650f2b`). The examples use `MODEL` and `EMBED` for their directories, `D` for prepared data, `R` for a frozen retrieval directory, and `OUT` for run outputs. Choose these paths before running a recipe. W&B logging defaults to disabled. Set `WANDB_MODE=online` and your own W&B environment credentials to enable it.

## Mini-BioASQ and binary relevance datasets

Use the published parquet files from `enelpol/rag-mini-bioasq`, revision `8a845907dc1cff31d42fa6f7bb9c6eef5f3ae6f6`. Preserve `text-corpus/test-*.parquet` and `question-answer-passages/{train,test}-*.parquet` under `$D/raw/`. Preparation retains the published test split and builds the dev split from deduplicated training questions.

```bash
S=scripts/reranker/bioasq
"$S/prepare.sh" --model "$MODEL" --data "$D"
"$S/encode.sh" --model "$EMBED" --data "$D" --output "$R"
"$S/freeze.sh" --data "$D" --output "$R"
for recipe in bce basic shifter; do
    "$S/train_$recipe.sh" --data "$D" --retrieval "$R" --model "$MODEL" --output "$OUT/$recipe"
done
"$S/evaluate.sh" --model "$MODEL" --data "$D" --retrieval "$R" \
    --adapter "$OUT/bce/checkpoint-step-00254/adapter" --output "$OUT/rescore-bce"
```

Each epoch visits all training positive relations. Four dense negatives exclude every known gold for the query. BCE scores individual pairs. Basic and Shifter optimize pairwise logistic loss, with Shifter sampling from a progressively shorter prefix. Evaluation reports MAP@10, nDCG@10, MRR@10 and recall on the original Top100/500/1000 candidate prefixes. Omit `--adapter` for zero-shot scoring.

**Add your own binary-relevance dataset.** Supply two UTF-8 JSONL files. A document row has the form `{"id":"d1","text":"A passage"}`. A query row has the form `{"id":"q1","text":"A question","split":"train","positive_ids":["d1"]}`. Provide all three splits `train/dev/test` and globally unique IDs within each collection. Each training query needs at least four eligible negative documents in its retrieved pool.

```bash
"$S/prepare_jsonl.sh" --input "$RAW" --data "$D" --model "$MODEL" \
    --instruction 'Retrieve the passage that answers the question.'
```

Continue with the same encode/freeze/train/evaluate commands. To supply existing rankings, replace encode/freeze with `"$S/import_rankings.sh" --data "$D" --rankings "$RAW/rankings.jsonl" --output "$R"`. Give each query one row such as `{"id":"q1","candidates":["d3","d1"],"scores":[0.8,0.6]}` in descending retrieval-score order. The shared instruction is stored in the manifest and used for retrieval and reranking. For graded labels, grouped candidate choices or different supervision units, follow the corresponding task modules below.

## Evidence Inference

Prepare the upstream Evidence Inference 2.0 annotation directory containing `prompts_merged.csv`, `annotations_merged.csv`, `splits/` and `txt_files/`. Reference implementation [jayded/evidence-inference](https://github.com/jayded/evidence-inference), revision `a661e8c14f973398380c8865cf2f27a535aaaf6d`.

```bash
S=scripts/reranker/evidence_inference
"$S/prepare.sh" --source "$RAW" --output "$D"
"$S/train.sh" --model "$MODEL" --data "$D" --output "$OUT"
"$S/evaluate.sh" --model "$MODEL" --data "$D" --adapter "$ADAPTER" --output "$EVAL"
```

Training runs the base scorer and the 2N/4N same-article negative recipes across three seeds. Each query contributes its mean sentence BCE. Validation runs each epoch and selects the adapter for test. Full article sentence ranking uses both character-overlap gold policies. Pass a saved epoch adapter directory for independent evaluation.

## IL-PCSR

Obtain authorized access to [Exploration-Lab/IL-PCSR](https://huggingface.co/datasets/Exploration-Lab/IL-PCSR), revision `28a7f9cdd7e9a33126e7025c9ae14bd2119bbc61`. Place the five parquet files named `{train,dev,test}_queries`, `statute_candidates` and `precedent_candidates` in the raw directory. Both recipes use masked judgments, separate statute/precedent corpora and the official splits.

```bash
S=scripts/reranker/il_pcsr
"$S/prepare.sh" --model "$MODEL" --raw "$RAW" --output "$D"
"$S/encode.sh" --model "$EMBED" --data "$D" --output "$R"
"$S/freeze.sh" --data "$D" --output "$R"
"$S/train.sh" --model "$MODEL" --data "$D" --retrieval "$R" --output "$OUT/whole"
"$S/evaluate.sh" --task lsr --model "$MODEL" --data "$D" --retrieval "$R" --output "$EVAL"
```

The whole-document recipe runs LSR/PCR × BCE/pairwise with one positive per query, BM25 training negatives and dense Top20/10 candidates. Its four single-GPU arms run in parallel with GPUS=4 and sequentially on one GPU otherwise. To evaluate a trained adapter pass `--adapter`. The original reporting protocol includes the full F1@1..10 scan.

For chunked retrieval reuse the prepared data and full token cache. Choose new directories `CHUNKS` and `CR`.

```bash
S=scripts/reranker/il_chunk
"$S/prepare.sh" --model "$MODEL" --source "$D" --output "$CHUNKS"
"$S/encode.sh" --model "$EMBED" --reranker "$MODEL" --data "$CHUNKS" --output "$CR"
"$S/freeze.sh" --data "$CHUNKS" --output "$CR"
"$S/assemble.sh" --source "$D" --data "$CHUNKS" --output "$CR"
for task in lsr pcr; do
    "$S/train.sh" --model "$MODEL" --data "$CHUNKS/$task" --retrieval "$CR/$task" --output "$OUT/chunk/$task"
done
"$S/evaluate.sh" --model "$MODEL" --data "$CHUNKS/lsr" --retrieval "$CR/lsr" --adapter "$ADAPTER" --output "$EVAL"
```

Chunk retrieval uses 2048-token windows with 256 overlap. Reranking packs selected spans within 8192 tokens. BCE covers all positive relations with one hard and two broad negatives per group. For each candidate depth, test uses the cutoff chosen on dev. Gold recall, MAP, MRR and retrieval F1 use the task's native qrels.

## JudicialMind and AU Tax

Use the AU and IN English parquet directories from `judicialmind/legal-training-dataset` at `8ed53a3527c78823ba12e979134dcdac152f8e55` under `$RAW`. Supply the normalized Australian Tax Guidance benchmark from `isaacus/australian-tax-guidance-retrieval` at `982aa11494f04dee16cacf18ae1fbf59f6669845` under `$TAX`. Its files are `documents.jsonl` (doc_id, text), `queries.jsonl` (query_id, query_text) and `qrels.jsonl` (query_id, doc_id, relevance). Use the source terms for each dataset.

```bash
S=scripts/reranker/judicialmind
"$S/prepare.sh" --raw "$RAW" --tax "$TAX" --output "$D"
"$S/cache_tokens.sh" --model "$MODEL" --data "$D"
"$S/encode.sh" --model "$EMBED" --data "$D" --output "$R"
"$S/freeze.sh" --data "$D" --output "$R" --top 100
"$S/train.sh" --model "$MODEL" --data "$D" --retrieval "$R" --output "$OUT/mixed"
"$S/train_au.sh" --model "$MODEL" --data "$D" --retrieval "$R" --output "$OUT/au"
```

The mixed recipe trains independent in-batch and hard-negative arms. AU-only repeats the AU subset for three epochs. For Top500, retain the Top100 retrieval directory, use a new retrieval directory for encode/freeze with `--top 500`, then run `train_top500.sh` against that directory. It uses two in-batch and one mined negative over 203 updates. Every saved checkpoint includes dev/test and external AU Tax results. Native country views include AU, IN and their union. For independent comparisons use `evaluate.sh --model "$MODEL" --data "$D" --retrieval "$R" --profile "$OUT/mixed/profile.json" --comparisons "$FILE" --output "$EVAL"`. The JSON mapping in FILE accepts model labels and adapter directories, e.g. `{"base":null,"trained":"/path/to/adapter"}`.

## CaseHOLD and NFCorpus

These workflows use the existing M1 model specifications under `configs/`. Supply the M1 model asset root as `--data-root`. Their existing GPU runner supports allocated Slurm GPUs or explicit exclusive-host operation as described in the runtime documentation.

```bash
S=scripts/reranker/casehold
"$S/prepare.sh" --raw "$RAW" --data-root "$ASSETS" --output "$D"
"$S/pilot.sh" --raw "$RAW" --data-root "$ASSETS" --preparation "$D/preparation.json" --output "$OUT/pilot"
GPUS=4 "$S/train.sh" --raw "$RAW" --data-root "$ASSETS" --baseline "$OUT/pilot" --output "$OUT/epoch"
"$S/evaluate.sh" --raw "$RAW" --split test --predictions "$OUT/epoch/test-3986" --output "$OUT/test-3986-f1.json"
```

Use the official `casehold/casehold` CSV files train.csv, val.csv and test.csv. Epoch training starts from the pretrained model and reuses the pilot's baseline predictions. Five-way group CE uses the native holdings. The epoch recipe preserves historical accuracy/MRR selection. Pass `--config configs/reranker/training/casehold-epoch-macro-f1.json` to select by dev Macro-F1, using a fresh output directory. The evaluation script recomputes option-level Macro-F1 from saved prediction chunks.

```bash
S=scripts/reranker/nfcorpus
"$S/unpack.sh" --archive "$ARCHIVE" --output "$D"
RAW="$D/raw/nfcorpus"
"$S/retrieve.sh" --raw "$RAW" --data-root "$ASSETS" --output "$R"
for stage in prepare train evaluate; do
    "$S/$stage.sh" --raw "$RAW" --retrieval "$R" --data-root "$ASSETS" --output "$OUT"
done
```

The dense NFCorpus recipe freezes Top100 candidates, trains the L0/L2 objectives, and evaluates saved checkpoints. For L0/L1 training on prepared groups use `scripts/reranker/groups/train.sh CONFIG --data-root "$ASSETS" --groups-root "$GROUPS" --output "$OUT"`. Group formats and objective conventions are documented in the existing [training contract](../contracts/reranker_training_v2.md).

To export saved JSON metrics to CSV use `scripts/reranker/groups/export_metrics.sh FILE [FILE ...] > metrics.csv`. The columns identify the source file, nested metric name and value. Pass the checkpoint metric files for the splits and candidate depths of interest.
