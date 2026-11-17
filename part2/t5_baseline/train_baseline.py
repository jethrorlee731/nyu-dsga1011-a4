"""Fine-tune T5-small for the assignment; run from the existing part2 directory.

Uses standard HF labels=-100 padding and automatic decoder shifting. This is a
standalone alternative to the starter training loop, not a drop-in collate function
for the starter's manually masked labels=0 loss.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

# Configure downloads before importing Hugging Face. The launcher also sets these.
SCRATCH = Path(os.environ.get('T5_SCRATCH_ROOT', '/scratch/jnl9728'))
os.environ.setdefault('HF_HOME', str(SCRATCH / 'cache/huggingface'))
os.environ.setdefault('XDG_CACHE_HOME', str(SCRATCH / 'cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizerFast,
                          T5Config, T5ForConditionalGeneration, T5TokenizerFast,
                          get_linear_schedule_with_warmup)
from text_data import MODEL_NAME, PREFIX, make_features, read_split, token_stats, preprocess_nl, preprocess_sql
from sql_eval import execute_all, metrics, save_outputs, write_error_analysis

# '<' is not in the T5 vocabulary: it encodes to <unk>, which skip_special_tokens deletes,
# so the model can never emit '<' or '<='. Train on a placeholder made of pretrained
# pieces and map it back after decoding. 'less_than' never occurs in the data and
# round-trips every train/dev query exactly through this tokenizer.
LT_PLACEHOLDER = 'less_than'


def encode_lt(sql):
    return sql.replace('<', LT_PLACEHOLDER)


def decode_lt(sql):
    return sql.replace(LT_PLACEHOLDER, '<')


def build_sql_bpe_tokenizer(train_inputs, train_targets, vocab_size):
    """Joint BPE tokenizer for questions and SQL, trained on the training split only.

    Metaspace pre-tokenization splits only on spaces, so a schema item such as
    flight_1.departure_time can become one token, and decoding restores the exact
    single-space SQL string. The alphabet covers printable ASCII (the whole corpus,
    including '<'), so no training or dev text maps to <unk>. Special-token ids
    follow T5: <pad>=0 (also the decoder start token), </s>=1, <unk>=2.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    core = Tokenizer(models.BPE(unk_token='<unk>'))
    core.pre_tokenizer = pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always')
    core.decoder = decoders.Metaspace(replacement='▁', prepend_scheme='always')
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=False,
                                  special_tokens=['<pad>', '</s>', '<unk>'],
                                  initial_alphabet=[chr(i) for i in range(32, 127)])
    # Train on exactly the strings make_features will tokenize.
    core.train_from_iterator([preprocess_nl(x) for x in train_inputs] +
                             [preprocess_sql(x) for x in train_targets], trainer)
    if [core.token_to_id(t) for t in ['<pad>', '</s>', '<unk>']] != [0, 1, 2]:
        raise RuntimeError('Unexpected special-token ids')
    # Append </s> like the T5 tokenizer, so make_features works unchanged.
    core.post_processor = processors.TemplateProcessing(single='$A </s>', special_tokens=[('</s>', 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token='<pad>', eos_token='</s>',
                                        unk_token='<unk>', model_max_length=4096)
    bad = [q for q in train_targets if tokenizer.decode(
        tokenizer(preprocess_sql(q)).input_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False).strip() != q.strip()]
    if bad:
        raise RuntimeError(f'{len(bad)} training queries do not round-trip, e.g. {bad[0]!r}')
    return tokenizer


def balanced_parens(sql):
    depth = 0
    for ch in sql:
        depth += (ch == '(') - (ch == ')')
        if depth < 0:
            return False
    return depth == 0


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'dev', 'test', 'stats'], default='train')
    p.add_argument('--data_dir', type=Path, default=Path('data'))
    p.add_argument('--output_dir', type=Path, default=Path('baseline_runs'))
    p.add_argument('--experiment_name', default='baseline')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--grad_accum_steps', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--max_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--warmup_epochs', type=float, default=1.0)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--num_beams', type=int, default=1)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--save_every', type=int, default=100,
                   help='Save resumable state every this many optimizer updates')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--no_gradient_checkpointing', action='store_true')
    p.add_argument('--resume', action='store_true', help='Continue last completed optimizer update')
    p.add_argument('--smoke_test', action='store_true',
                   help='32 train / 16 dev examples, 1 epoch; separate experiment name required')
    # Was 5.0: a character-identical-to-gold dev query timed out and scored F1 = 0.
    p.add_argument('--sql_timeout', type=float, default=30.0)
    p.add_argument('--gold_sql_timeout', type=float, default=30.0)
    p.add_argument('--sql_workers', type=int, default=4)
    p.add_argument('--lt_placeholder', action='store_true',
                   help=f"Train on '{LT_PLACEHOLDER}' instead of '<' (fixed for dev/test by the run config)")
    p.add_argument('--exec_select', action='store_true',
                   help='Return all num_beams beams; keep the highest-ranked one that has balanced '
                        'parentheses and executes without error (falls back to beam 1)')
    p.add_argument('--from_scratch', action='store_true',
                   help='Random initialization from the t5-small config (extra credit)')
    p.add_argument('--tokenizer', choices=['t5', 'sql_bpe'], default='t5',
                   help='sql_bpe: train a joint BPE tokenizer on the training split (needs --from_scratch)')
    p.add_argument('--bpe_vocab_size', type=int, default=1500)
    p.add_argument('--eval_every', type=int, default=1,
                   help='Evaluate on dev every N epochs (and after the last); patience counts evaluations')
    p.add_argument('--eval_tag', default='',
                   help='Suffix for dev-mode output files, e.g. beam5_exec, so runs do not overwrite')
    args = p.parse_args()
    if not args.experiment_name or Path(args.experiment_name).name != args.experiment_name:
        p.error('experiment_name must be a single directory name')
    for name in ['batch_size', 'grad_accum_steps', 'eval_batch_size', 'max_epochs',
                 'patience', 'num_beams', 'max_new_tokens', 'save_every', 'sql_workers']:
        if getattr(args, name) <= 0:
            p.error(f'{name} must be positive')
    if args.learning_rate <= 0 or args.sql_timeout <= 0 or args.gold_sql_timeout <= 0:
        p.error('Learning rate and SQL timeouts must be positive')
    if args.weight_decay < 0 or args.max_grad_norm <= 0 or args.warmup_epochs < 0:
        p.error('Invalid weight decay, clipping, or warmup')
    if args.smoke_test:
        if args.mode != 'train':
            p.error('smoke_test is only for training')
        if args.experiment_name == 'baseline':
            p.error('Use --experiment_name smoke for a smoke test')
        args.max_epochs = 1
        args.warmup_epochs = 0
        args.max_new_tokens = min(args.max_new_tokens, 128)
    if args.warmup_epochs >= args.max_epochs:
        p.error('warmup_epochs must be less than max_epochs')
    if args.tokenizer == 'sql_bpe' and not args.from_scratch:
        p.error('--tokenizer sql_bpe has no pretrained embeddings; use it with --from_scratch')
    if args.tokenizer == 'sql_bpe' and args.lt_placeholder:
        p.error("--lt_placeholder is unnecessary with sql_bpe: '<' is in its vocabulary")
    if args.eval_every <= 0 or args.bpe_vocab_size < 200:
        p.error('eval_every must be positive and bpe_vocab_size at least 200')
    if args.exec_select and args.num_beams < 2:
        p.error('--exec_select needs --num_beams >= 2')
    if args.eval_tag and (args.mode != 'dev' or Path(args.eval_tag).name != args.eval_tag):
        p.error('--eval_tag is a plain name and only for --mode dev (test files keep the required name)')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(features, collator, batch_size, seed, epoch=None):
    generator = torch.Generator().manual_seed(seed + (epoch or 0))
    indices = list(range(len(features)))
    if epoch is not None:
        indices = torch.randperm(len(features), generator=generator).tolist()
    # An independent generator keeps loader construction out of dropout's RNG.
    return DataLoader(features, batch_size=batch_size, sampler=indices,
                      collate_fn=collator, num_workers=0, generator=generator)


def move_batch(batch, device):
    # Newer transformers collators emit labels=None for unlabeled (test) batches.
    return {key: value.to(device) for key, value in batch.items() if value is not None}


def directory_save(path, writer):
    """Write a complete new checkpoint before retiring the previous one."""
    path = Path(path)
    temporary, backup = path.with_name(path.name + '.tmp'), path.with_name(path.name + '.previous')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    writer(temporary)
    # If interrupted before final rename, --resume can recover .previous.
    if path.exists():
        if backup.exists():
            shutil.rmtree(backup)
        path.rename(backup)
    temporary.rename(path)
    if backup.exists():
        shutil.rmtree(backup)


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, state):
    def writer(directory):
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        payload = dict(state)
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(payload, directory / 'training_state.pt')
    directory_save(path, writer)


def restore_state(path, optimizer, scheduler):
    # Only load your own training checkpoint (contains Python RNG state).
    payload = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(payload.pop('optimizer'))
    scheduler.load_state_dict(payload.pop('scheduler'))
    random.setstate(payload.pop('python_rng'))
    np.random.set_state(payload.pop('numpy_rng'))
    torch.set_rng_state(payload.pop('torch_rng'))
    cuda_rng = payload.pop('cuda_rng')
    if torch.cuda.is_available() and cuda_rng is not None:
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError('Resume with the same number of visible GPUs')
        torch.cuda.set_rng_state_all(cuda_rng)
    return payload


def config_dict(args):
    result = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result.update(model=MODEL_NAME, prefix=PREFIX, input_truncation=False, target_truncation=False,
                  decoder_start_token='pad (0)', labels_padding=-100,
                  optimizer='AdamW', scheduler='linear', precision='float32',
                  transformers_version=__import__('transformers').__version__, torch_version=torch.__version__)
    result['data_sha256'] = {}
    for name in ['train.nl', 'train.sql', 'dev.nl', 'dev.sql', 'flight_database.db']:
        file = args.data_dir / name
        if file.exists():
            digest = hashlib.sha256()
            with file.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            result['data_sha256'][name] = digest.hexdigest()
    return result


def training_signature(config):
    ignored = {'resume', 'mode', 'device', 'data_dir', 'output_dir'}
    return {k: v for k, v in config.items() if k not in ignored}


def prepare_gold(args, gold_sql, run_dir):
    records, errors = execute_all(args.data_dir / 'flight_database.db', gold_sql,
                                 args.gold_sql_timeout, args.sql_workers)
    if any(errors):
        examples = [(i, e) for i, e in enumerate(errors) if e][:3]
        raise RuntimeError(f'Gold SQL execution failed; check database/timeout: {examples}')
    save_outputs(gold_sql, records, errors, run_dir / 'dev_gold.sql', run_dir / 'dev_gold.pkl')
    return records


def select_executable(args, candidates):
    """Per example, keep the best-scoring beam that is balanced and executes without error.

    Uses only the database, never gold SQL or gold records. Beams are tried rank by rank,
    so each round executes at most one query per still-unresolved example.
    """
    n, k = len(candidates), len(candidates[0])
    choice, pending = [None] * n, list(range(n))
    for rank in range(k):
        trial = [i for i in pending if balanced_parens(candidates[i][rank])]
        if not trial:
            continue
        print(f'Beam selection: executing rank {rank + 1} for {len(trial)} examples...', flush=True)
        _, errors = execute_all(args.data_dir / 'flight_database.db',
                                [candidates[i][rank] for i in trial], args.sql_timeout, args.sql_workers)
        for i, error in zip(trial, errors):
            if not error:
                choice[i] = rank
        pending = [i for i in pending if choice[i] is None]
        if not pending:
            break
    counts = {f'beam_{r + 1}': sum(c == r for c in choice) for r in range(k)}
    counts['fallback_no_valid_beam'] = len(pending)
    return [candidates[i][choice[i] or 0] for i in range(n)], counts


@torch.no_grad()
def evaluate(args, model, tokenizer, loader, device, inputs, targets, gold_records, run_dir, stem):
    model.eval()
    total_loss, count, predictions, no_eos = 0.0, 0, [], 0
    for batch in tqdm(loader, desc=stem):
        batch = move_batch(batch, device)
        if 'labels' in batch:
            out = model(**batch, use_cache=False)
            tokens = int((batch['labels'] != -100).sum().item())
            if not torch.isfinite(out.loss):
                raise RuntimeError('Non-finite development loss')
            total_loss += out.loss.item() * tokens
            count += tokens
        # Do not pass labels or teacher-forced decoder inputs to generation.
        k = args.num_beams if args.exec_select else 1
        generated = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                   num_beams=args.num_beams, num_return_sequences=k, do_sample=False,
                                   max_new_tokens=args.max_new_tokens, use_cache=True)
        top = generated.view(batch['input_ids'].size(0), k, -1)[:, 0]  # beams come sorted by score
        no_eos += int((~(top == tokenizer.eos_token_id).any(dim=1)).sum().item())
        texts = [text.strip() for text in tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        if args.lt_placeholder:
            texts = [decode_lt(text) for text in texts]
        predictions.extend(texts[i:i + k] for i in range(0, len(texts), k))
    if len(predictions) != len(inputs):
        raise RuntimeError('Prediction count does not match input count')
    selection = None
    if args.exec_select:
        predictions, selection = select_executable(args, predictions)
    else:
        predictions = [candidates[0] for candidates in predictions]
    if any('\n' in q or '\r' in q for q in predictions):
        raise RuntimeError('A generated query contains a newline; inspect before saving')
    print(f'Executing {len(predictions)} generated SQL queries...', flush=True)
    records, errors = execute_all(args.data_dir / 'flight_database.db', predictions,
                                 args.sql_timeout, args.sql_workers)
    save_outputs(predictions, records, errors, run_dir / f'{stem}.sql', run_dir / f'{stem}.pkl')
    result = {'examples': len(inputs), 'generated_without_eos': no_eos,
              'num_beams': args.num_beams, 'max_new_tokens': args.max_new_tokens,
              'sql_timeout_seconds': args.sql_timeout,
              'lt_placeholder': args.lt_placeholder, 'exec_select': args.exec_select}
    if selection is not None:
        result['selected_beam_counts'] = selection
    if targets is not None:
        result.update(metrics(targets, predictions, gold_records, records, errors))
        result['loss'] = total_loss / count
        write_error_analysis(run_dir / f'{stem}_examples.jsonl', inputs, targets,
                             predictions, gold_records, records, errors)
    else:
        result['execution_error_rate'] = sum(bool(e) for e in errors) / len(errors)
    (run_dir / f'{stem}_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if no_eos:
        print(f'NOTE: {no_eos} generations reached the limit without EOS; inspect on dev.', flush=True)
    return result


def main():
    args = arguments()
    set_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    run_dir = args.output_dir / args.experiment_name
    cache_dir = str(SCRATCH / 'cache/huggingface/hub')
    cache_kwargs = {'cache_dir': cache_dir}
    if args.mode == 'stats':
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME, **cache_kwargs)
        for processed in (False, True):
            print('\n' + ('After preprocessing' if processed else 'Before preprocessing'))
            for split in ['train', 'dev']:
                inputs, targets = read_split(args.data_dir, split)
                if processed:
                    inputs = [preprocess_nl(x) for x in inputs]
                    targets = [preprocess_sql(x) for x in targets]
                il, iv = token_stats(tokenizer, inputs)
                tl, tv = token_stats(tokenizer, targets)
                print(f'{split}: n={len(inputs)}, NL mean={il:.2f}, SQL mean={tl:.2f}, '
                      f'NL vocabulary={iv}, SQL vocabulary={tv}')
        return
    if not (args.data_dir / 'flight_database.db').is_file():
        raise FileNotFoundError(f'Keep the assignment database at {args.data_dir / "flight_database.db"}')
    if args.mode in ['dev', 'test']:
        best = run_dir / 'best'
        if not (best / 'config.json').exists():
            raise FileNotFoundError(f'No trained best checkpoint at {best}; train first')
        config = json.loads((run_dir / 'run_config.json').read_text())
        if config.get('smoke_test'):
            raise ValueError('A smoke checkpoint is for debugging, not full evaluation/submission')
        if config['prefix'] != PREFIX:
            raise ValueError('Input prefix differs from the training configuration')
        # Output post-processing must match how the checkpoint was trained, not the CLI.
        args.lt_placeholder = bool(config.get('lt_placeholder', False))
        print(f'lt_placeholder from run config: {args.lt_placeholder}', flush=True)
        # AutoTokenizer restores either the T5 tokenizer or a saved sql_bpe tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(best)
        model = T5ForConditionalGeneration.from_pretrained(best).to(device)
        print(f'Tokenizer: {type(tokenizer).__name__} ({len(tokenizer)} tokens); '
              f'from_scratch={config.get("from_scratch", False)}', flush=True)
        inputs, targets = read_split(args.data_dir, args.mode)
        # Only the dev loss sees these; metrics compare against the raw gold targets.
        feature_targets = None if targets is None else (
            [encode_lt(t) for t in targets] if args.lt_placeholder else targets)
        features = make_features(tokenizer, inputs, feature_targets)
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
        loader = make_loader(features, collator, args.eval_batch_size, args.seed)
        gold = prepare_gold(args, targets, run_dir) if targets is not None else None
        stem = f't5_ft_{args.experiment_name}_{args.mode}' + (f'_{args.eval_tag}' if args.eval_tag else '')
        evaluate(args, model, tokenizer, loader, device, inputs, targets, gold, run_dir, stem)
        return
    current_config = config_dict(args)
    checkpoint = run_dir / 'last'
    if args.resume and not (checkpoint / 'training_state.pt').exists():
        backup = run_dir / 'last.previous'
        if (backup / 'training_state.pt').exists():
            checkpoint = backup
        else:
            raise FileNotFoundError('No resumable checkpoint for this experiment')
    if args.resume:
        previous = json.loads((run_dir / 'run_config.json').read_text())
        if training_signature(previous) != training_signature(current_config):
            raise ValueError('Resume requires the original training settings and unchanged data')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Experiment already exists; use --resume or a new experiment_name')
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (run_dir / 'run_config.json').write_text(json.dumps(current_config, indent=2))
    source = checkpoint if args.resume else MODEL_NAME
    train_inputs, train_targets = read_split(args.data_dir, 'train', 32 if args.smoke_test else None)
    dev_inputs, dev_targets = read_split(args.data_dir, 'dev', 16 if args.smoke_test else None)
    if args.resume:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    elif args.tokenizer == 'sql_bpe':
        tokenizer = build_sql_bpe_tokenizer(train_inputs, train_targets, args.bpe_vocab_size)
        tokenizer.save_pretrained(run_dir / 'tokenizer')
        print(f'Trained sql_bpe tokenizer: {len(tokenizer)} tokens', flush=True)
    else:
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME, **cache_kwargs)
    # dev_targets stays raw: it is the gold for execution and SQL EM.
    lt = encode_lt if args.lt_placeholder else (lambda sql: sql)
    train_features = make_features(tokenizer, train_inputs, [lt(t) for t in train_targets])
    dev_features = make_features(tokenizer, dev_inputs, [lt(t) for t in dev_targets])
    print(f'Device: {device}; train={len(train_features)}, dev={len(dev_features)}', flush=True)
    print('Max encoder/target token lengths:', max(len(f['input_ids']) for f in train_features),
          max(len(f['labels']) for f in train_features), flush=True)
    diagnostics = {}
    for split, features in [('train', train_features), ('dev', dev_features)]:
        diagnostics[split] = {
            'examples': len(features),
            'max_input_tokens_with_eos': max(len(f['input_ids']) for f in features),
            'max_target_tokens_with_eos': max(len(f['labels']) for f in features),
            'targets_longer_than_generation_limit': sum(
                len(f['labels']) > args.max_new_tokens for f in features),
            'inputs_containing_unk': sum(tokenizer.unk_token_id in f['input_ids'] for f in features),
            'targets_containing_unk': sum(tokenizer.unk_token_id in f['labels'] for f in features),
        }
    (run_dir / 'data_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print('Data diagnostics:', json.dumps(diagnostics), flush=True)
    if args.from_scratch and not args.resume:
        # Same architecture as t5-small, randomly initialized (HF's T5 init scheme).
        model_config = T5Config.from_pretrained(MODEL_NAME, **cache_kwargs)
        model_config.vocab_size = len(tokenizer)
        model = T5ForConditionalGeneration(model_config).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(source, **cache_kwargs).to(device)
    origin = ('resumed checkpoint' if args.resume else
              'random init' if args.from_scratch else 'pretrained')
    print(f'Model: {origin}, '
          f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, '
          f'vocab {model.config.vocab_size}', flush=True)
    # Update every parameter. Gradient checkpointing trades compute for lower memory use.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    dev_loader = make_loader(dev_features, collator, args.eval_batch_size, args.seed)
    num_batches = math.ceil(len(train_features) / args.batch_size)
    steps_per_epoch = math.ceil(num_batches / args.grad_accum_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, eps=1e-8, betas=(0.9, 0.999))
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=round(args.warmup_epochs * steps_per_epoch),
        num_training_steps=args.max_epochs * steps_per_epoch)
    state = {'epoch': 0, 'next_batch': 0, 'global_step': 0, 'best_f1': -1.0,
             'bad_epochs': 0, 'loss_sum': 0.0, 'tokens': 0, 'history': []}
    if args.resume:
        state = restore_state(checkpoint, optimizer, scheduler)
        print(f'Resuming epoch {state["epoch"]+1}, batch {state["next_batch"]}, '
              f'optimizer step {state["global_step"]}', flush=True)
    gold_records = prepare_gold(args, dev_targets, run_dir)
    while state['epoch'] < args.max_epochs and state['bad_epochs'] < args.patience:
        epoch = state['epoch']
        loader = make_loader(train_features, collator, args.batch_size, args.seed, epoch)
        iterator = iter(loader)
        # Recreate the same epoch order and skip only batches already checkpointed.
        for _ in range(state['next_batch']):
            next(iterator)
        model.train()
        progress = tqdm(total=num_batches, initial=state['next_batch'], desc=f'Epoch {epoch+1}')
        while state['next_batch'] < num_batches:
            window_size = min(args.grad_accum_steps, num_batches-state['next_batch'])
            window = [next(iterator) for _ in range(window_size)]
            window_tokens = sum(int((b['labels'] != -100).sum().item()) for b in window)
            optimizer.zero_grad(set_to_none=True)
            # Weight microbatches by target-token count, including a short final window.
            for batch in window:
                tokens = int((batch['labels'] != -100).sum().item())
                batch = move_batch(batch, device)
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; last completed checkpoint is available')
                (loss * (tokens/window_tokens)).backward()
                state['loss_sum'] += loss.item() * tokens
                state['tokens'] += tokens
                progress.update(1)
                progress.set_postfix(loss=f'{loss.item():.4f}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            state['global_step'] += 1
            state['next_batch'] += window_size
            if state['global_step'] % args.save_every == 0:
                save_checkpoint(run_dir / 'last', model, tokenizer, optimizer, scheduler, state)
        progress.close()
        # Checkpoint before evaluation too: a timeout during SQL evaluation won't lose the epoch.
        save_checkpoint(run_dir / 'last', model, tokenizer, optimizer, scheduler, state)
        if (epoch + 1) % args.eval_every and epoch + 1 < args.max_epochs:
            print(f'Epoch {epoch+1}: train loss {state["loss_sum"]/state["tokens"]:.4f} '
                  f'(dev evaluation every {args.eval_every} epochs)', flush=True)
            state.update(epoch=epoch+1, next_batch=0, loss_sum=0.0, tokens=0)
            save_checkpoint(run_dir / 'last', model, tokenizer, optimizer, scheduler, state)
            continue
        stem = f'epoch_{epoch+1:02d}_dev'
        result = evaluate(args, model, tokenizer, dev_loader, device, dev_inputs, dev_targets,
                          gold_records, run_dir, stem)
        result.update(epoch=epoch+1, train_loss=state['loss_sum']/state['tokens'],
                      global_step=state['global_step'])
        if result['record_f1'] > state['best_f1']:
            state['best_f1'], state['bad_epochs'] = result['record_f1'], 0
            def save_best(directory):
                model.save_pretrained(directory)
                tokenizer.save_pretrained(directory)
                (directory / 'metrics.json').write_text(json.dumps(result, indent=2))
            directory_save(run_dir / 'best', save_best)
            for suffix in ['.sql', '.pkl', '_metrics.json', '_examples.jsonl']:
                shutil.copyfile(run_dir / f'{stem}{suffix}', run_dir / f'best_dev{suffix}')
        else:
            state['bad_epochs'] += 1
        state['history'].append(result)
        (run_dir / 'history.json').write_text(json.dumps(state['history'], indent=2))
        state.update(epoch=epoch+1, next_batch=0, loss_sum=0.0, tokens=0)
        save_checkpoint(run_dir / 'last', model, tokenizer, optimizer, scheduler, state)
    print(f'Done. Best dev Record F1: {100*state["best_f1"]:.2f}%. '
          f'Checkpoint: {run_dir / "best"}', flush=True)
    print('Test generation is explicit: rerun with --mode test after selecting your configuration.')


if __name__ == '__main__':
    main()