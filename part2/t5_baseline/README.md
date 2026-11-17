# T5-small text-to-SQL baseline for DS-GA 1011 HW4 Part 2

This is a complete runnable baseline implementation, **not a trained checkpoint**.
It starts from pretrained `google-t5/t5-small` and fine-tunes on your training pairs.
No text-to-SQL accuracy or F1 is claimed before you run it. It is an independent
training entry point: keep your original starter files for reference.

## Put the files in place

Extract this ZIP inside `/scratch/jnl9728/nyu-dsga1011-a4/part2`.
It creates a `t5_baseline/` subdirectory. From `part2`, the layout should be:

```
t5_baseline/
    train_baseline.py
    text_data.py
    sql_eval.py
    run.sh
    test_core.py
    test_model.py
    README.md
    VALIDATION.txt
data/
    train.nl
    train.sql
    dev.nl
    dev.sql
    test.nl
    flight_database.db
```

Use the Part 2 environment you already installed:
`/scratch/jnl9728/conda-envs/dsga1011-hw4-part2`.
`run.sh` initializes Conda, activates that environment, and directs caches and
model downloads to `/scratch/jnl9728`. It does not create an environment or
install anything. Run it on your allocated GPU node for training. Do not run
`conda create -n ...` again. The required database comes from the assignment;
it was not supplied here and is not included in this ZIP.

## 1. Small integration check

From `part2`:

```bash
bash t5_baseline/run.sh --experiment_name smoke --smoke_test
```

This trains on 32 examples for one epoch and evaluates 16 development examples.
Generation is capped at 128 tokens only in this smoke test to make it faster.
Its purpose is to find integration errors; a low F1 is expected and is not an
accuracy target. It still requires `data/flight_database.db`. Results and a toy
fine-tuned checkpoint are stored under `baseline_runs/smoke/`.

If you rerun the smoke test, choose a new experiment name or add `--resume` to
continue an interrupted one. A completed resumed smoke test does not train again.

Optional tests in the activated Part 2 environment:

```bash
python t5_baseline/test_core.py
python t5_baseline/test_model.py
```

The second command is a tiny CPU model-mechanics test; it requires your installed
PyTorch/Transformers but does not download any weights or tokenizer.

## 2. Full baseline

```bash
bash t5_baseline/run.sh --experiment_name baseline
```

Default configuration:

| Choice | Value |
|---|---|
| Initialization | Pretrained google-t5/t5-small |
| Trainable parameters | All parameters |
| Input preprocessing | Collapse whitespace and prepend `translate to SQL: ` |
| SQL preprocessing | Strip leading/trailing whitespace only |
| Tokenizer | T5TokenizerFast, google-t5/t5-small |
| Sequence truncation | None for training inputs or targets |
| EOS | Included by the tokenizer |
| Decoder start | Standard T5 pad token ID 0 |
| Decoder inputs | Model automatically shifts labels right (teacher forcing) |
| Padding | Dynamic, separately for inputs and labels |
| Label padding | -100; ignored by HF built-in cross-entropy loss |
| Optimizer | AdamW, lr 3e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| Microbatch | 4 examples |
| Gradient accumulation | 4 microbatches: nominal effective batch size 16 |
| Loss reduction | Token-weighted across each accumulation window |
| Evaluation batch size | 4 |
| Scheduler | Linear decay, 1 epoch of warmup |
| Maximum epochs | 10 |
| Early stopping | 3 consecutive epochs without improved dev Record F1 |
| Selection | Highest dev Record F1 |
| Gradient clipping | Global norm 1.0 |
| Gradient checkpointing | Enabled to reduce training memory |
| Precision | Float32 |
| Seed | 42 |
| Generation | Greedy, max_new_tokens=1024 |
| SQL execution | Read-only SQLite, 4 workers, 5-second progress deadline per prediction |
| Gold SQL execution | 30-second progress deadline; abort if any reference query fails |

The smaller microbatch with accumulation is a memory-conscious starting point.
It differs from the previously suggested physical batch of 16. Record the actual
settings above in Q5 if you use them. Gradient checkpointing saves memory by
recomputing intermediate activations. SQL targets are never silently truncated;
if memory is insufficient, try a *new* run with `--batch_size 2 --grad_accum_steps 8`.

The total training time depends on GPU type and SQL generation/execution. There
is no verified runtime estimate or verified score for this implementation.

## 3. Resume after an expired allocation

Use the same original arguments, plus `--resume`:

```bash
bash t5_baseline/run.sh --experiment_name baseline --resume
```

A full checkpoint is saved every 100 optimizer updates, before each dev evaluation,
and after each completed epoch. It contains weights, tokenizer, optimizer,
scheduler, RNG states, current epoch/batch offset, best score, patience, and history.
An interruption may lose work since the last successful save. Resume reconstructs
the same epoch permutation and skips the batches already saved. Results across
different GPU hardware are not guaranteed to be bit-identical.

Use the same settings/data/package versions on resume. For changed hyperparameters,
start a new experiment rather than changing the schedule of an existing run.
`last.previous` is a recovery fallback if interrupted during the final checkpoint
rename. `best/` contains model/tokenizer files for evaluation; `last/` additionally
contains optimizer state and is larger. Checkpoints are saved in scratch when you
run from your scratch `part2` directory.

## 4. Inspect development results

Training writes one pair of SQL/record outputs per epoch and evaluates with the
same mathematical metric definitions as your uploaded `utils.py`. Metrics are
fractions: 0.65 means 65 F1 on a 0-100 scale.

- `baseline_runs/baseline/run_config.json`: exact settings and data hashes for Q5.
- `baseline_runs/baseline/history.json`: completed epoch losses and dev metrics.
- `baseline_runs/baseline/data_diagnostics.json`: max token lengths, unknown-token
  counts, and targets exceeding the generation cap. Counts include EOS here;
  Q4 text statistics exclude it.
- `baseline_runs/baseline/best/`: selected model checkpoint and tokenizer.
- `baseline_runs/baseline/best_dev.sql` and `.pkl`: selected epoch's predictions.
- `baseline_runs/baseline/best_dev_metrics.json`: selected epoch's metrics.
- `baseline_runs/baseline/best_dev_examples.jsonl`: input, reference/predicted SQL,
  per-example scores and execution errors for Q6. Manually categorize at least
  three error types from real examples; no semantic categories are fabricated.

Check `generated_without_eos` in metrics: a positive value means some outputs
reached the generation cap. Inspect failures on dev before changing the cap.
An execution error may be a syntax error, missing table/column, timeout, etc.;
`execution_error_rate` must not be described as syntax errors alone. The original
assignment metrics count two empty result sets as matching, including when a
prediction failed; report execution errors as well to make this limitation visible.

The `.pkl` file contains `(records, error_messages)`, exactly as expected by the
provided evaluator. You can verify saved best-dev outputs using your original file:

```bash
python evaluate.py \
  --predicted_sql baseline_runs/baseline/best_dev.sql \
  --predicted_records baseline_runs/baseline/best_dev.pkl \
  --development_sql data/dev.sql \
  --development_records baseline_runs/baseline/dev_gold.pkl
```

To regenerate development predictions from the best checkpoint:

```bash
bash t5_baseline/run.sh --experiment_name baseline --mode dev
```

For a decoding-only experiment, you can add `--num_beams 4`. Generation settings
are saved in the resulting metrics file. This is a dev experiment, not a claim
that beam search will improve the score. If training used nondefault generation
settings, pass the same values explicitly here and for final test prediction.

## 5. Test predictions only after selecting your model

```bash
bash t5_baseline/run.sh --experiment_name baseline --mode test
```

This loads `best/`, reads `test.nl` in its original order, and creates:

- `baseline_runs/baseline/t5_ft_baseline_test.sql`
- `baseline_runs/baseline/t5_ft_baseline_test.pkl`

No test labels are needed. It does not report test F1 because the test references
are held out. Use the assignment's requested submission filenames when copying:

```bash
mkdir -p results records
cp baseline_runs/baseline/t5_ft_baseline_test.sql results/t5_ft_experiment_test.sql
cp baseline_runs/baseline/t5_ft_baseline_test.pkl records/t5_ft_experiment_test.pkl
```

Both files preserve test input order. For the report checkpoint link, upload the
entire `baseline_runs/baseline/best/` folder to your Google Drive yourself.

## Q4 statistics

```bash
bash t5_baseline/run.sh --mode stats
```

This reuses the exact preprocessing functions used in training. It excludes
padding and automatically added special tokens, as in your existing Q4 tables.
No database or GPU is needed for this command.

## Implementation notes

This uses HF built-in loss with -100 labels and the standard T5 pad start token.
Do not mix its collator with the original starter's `labels != 0` masking code.
The original starter suggests an extra-ID start token; this baseline instead uses
the pretrained model's default start token consistently for training/generation.
No train/test augmentation, custom tokenizer, SQL rewriting, or layer freezing is
applied. Pretrained T5 may not represent every rare SQL character perfectly;
inspect unknown-token counts and dev examples before changing the vocabulary.

The original starter files are not overwritten. This package has its own entry
point and SQL execution wrapper. The wrapper adds read-only connections and
per-query progress deadlines; metric formulas and output serialization match the
provided utilities. Timeouts can affect scores, so keep execution settings fixed
when comparing models and check reference queries all execute successfully.

## Validation status

See VALIDATION.txt. Code syntax, core tests, uploaded-data structure, and metric
parity were checked here. Model execution and full training could not be run in
the authoring workspace because PyTorch/Transformers and model downloads were
unavailable; run the integration check on your allocated cluster node first.
