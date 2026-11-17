# DS-GA 1011 Assignment 4: Fine-tuning Language Models

Code for both parts of the assignment. Model checkpoints are too large for GitHub and are
in the Google Drive folder linked in the written report.

## Part 1: BERT sentiment classification (`part1/`)

- `utils.py`: `custom_transform()` is the Q2a transformation (keyboard-neighbor vowel typos,
  terminal-punctuation changes, article deletion; the WordNet synonym step is present but
  disabled with `SYNONYM_PROB = 0.00`).
- `main.py`: training and evaluation (bert-base-cased, lr 5e-5, batch size 8, 3 epochs).

```bash
cd part1
python main.py --train --eval                        # Q1   -> out_original.txt
python main.py --eval_transformed                    # Q2b  -> out_transformed.txt
python main.py --train_augmented --eval --eval_transformed   # Q3 -> out_augmented_*.txt
```

## Part 2: T5 text-to-SQL (`part2/`)

All results in the report come from `part2/t5_baseline/`, not from the starter files
`train_t5.py`, `load_data.py` and `t5_utils.py`, which are left unfinished and are not used.

| File | Purpose |
|---|---|
| `t5_baseline/train_baseline.py` | Training, dev evaluation and test generation (fine-tuning and from scratch) |
| `t5_baseline/text_data.py` | Data loading and preprocessing (whitespace normalization, task prefix) |
| `t5_baseline/sql_eval.py` | SQL execution with a per-query timeout; metrics identical to `utils.py` |
| `t5_baseline/run.sh` | Sets up the environment and launches `train_baseline.py` |
| `t5_baseline/train_scratch.sbatch` | Slurm job script used for the from-scratch runs |
| `q4_stats.py` | Reproduces the Q4 data-statistics tables |

Environment: Python 3.11, PyTorch 2.1.2 (CUDA 12.1), transformers 4.40.0, tokenizers 0.19.
Run everything from `part2/`.

### Fine-tuned model (Q5 to Q7)

```bash
# Train: '<' placeholder, 25 epochs (best checkpoint: baseline_runs/lt_e25/best)
bash t5_baseline/run.sh --experiment_name lt_e25 --lt_placeholder --max_epochs 25 --patience 5

# Dev evaluation and test generation with 5-beam executable-beam selection
bash t5_baseline/run.sh --mode dev  --experiment_name lt_e25 --num_beams 5 --exec_select --eval_tag beam5_exec
bash t5_baseline/run.sh --mode test --experiment_name lt_e25 --num_beams 5 --exec_select
cp baseline_runs/lt_e25/t5_ft_lt_e25_test.sql t5_ft_experiment_test.sql
cp baseline_runs/lt_e25/t5_ft_lt_e25_test.pkl t5_ft_experiment_test.pkl
```

### Extra credit: T5 trained from scratch

```bash
COMMON="--from_scratch --batch_size 16 --grad_accum_steps 1 --no_gradient_checkpointing \
  --warmup_epochs 3 --max_epochs 80 --eval_every 4 --patience 5 \
  --eval_batch_size 32 --sql_timeout 10 --sql_workers 8"
sbatch --job-name=scr_bpe_lr5e-5 t5_baseline/train_scratch.sbatch \
  --experiment_name scr_bpe_lr5e-5 --tokenizer sql_bpe --bpe_vocab_size 1500 \
  --learning_rate 5e-5 --max_new_tokens 256 $COMMON

E=scr_bpe_lr5e-5
bash t5_baseline/run.sh --mode dev  --experiment_name $E --max_new_tokens 256 --eval_batch_size 16 --num_beams 5 --exec_select --eval_tag beam5_exec
bash t5_baseline/run.sh --mode test --experiment_name $E --max_new_tokens 256 --eval_batch_size 16 --num_beams 5 --exec_select
cp baseline_runs/$E/t5_ft_${E}_test.sql t5_ft_experiment_ec_test.sql
cp baseline_runs/$E/t5_ft_${E}_test.pkl t5_ft_experiment_ec_test.pkl
```

The other learning rates in the report's Table EC4 use the same command with
`--learning_rate` changed (and `--tokenizer t5 --lt_placeholder --max_new_tokens 600` for
the T5-tokenizer runs).

### Checking a dev result with the official metric
```bash
python evaluate.py --predicted_sql baseline_runs/lt_e25/t5_ft_lt_e25_dev_beam5_exec.sql \
  --predicted_records baseline_runs/lt_e25/t5_ft_lt_e25_dev_beam5_exec.pkl \
  --development_sql data/dev.sql --development_records records/ground_truth_dev.pkl
```
