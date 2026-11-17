"""Shared text preparation for Q4 statistics and model training."""
from pathlib import Path

MODEL_NAME = 'google-t5/t5-small'
PREFIX = 'translate to SQL: '


def preprocess_nl(text):
    return PREFIX + ' '.join(text.split())


def preprocess_sql(text):
    # Never lowercase or normalize whitespace inside SQL string literals.
    return text.strip()


def read_split(data_dir, split, limit=None):
    data_dir = Path(data_dir)
    inputs = (data_dir / f'{split}.nl').read_text(encoding='utf-8').splitlines()
    targets = None
    if split != 'test':
        targets = (data_dir / f'{split}.sql').read_text(encoding='utf-8').splitlines()
        if len(inputs) != len(targets):
            raise ValueError(f'{split}: input and target counts differ')
    if not inputs or any(not x.strip() for x in inputs):
        raise ValueError(f'{split}: empty input split or blank input')
    if targets is not None and any(not x.strip() for x in targets):
        raise ValueError(f'{split}: blank SQL target')
    if limit is not None:
        inputs = inputs[:limit]
        targets = None if targets is None else targets[:limit]
    return inputs, targets


def make_features(tokenizer, inputs, targets=None):
    """No truncation: preserve the inputs/targets used for the Q4 statistics."""
    features = []
    for start in range(0, len(inputs), 128):
        encoded = tokenizer([preprocess_nl(x) for x in inputs[start:start+128]],
                            padding=False, truncation=False, verbose=False)
        labels = None
        if targets is not None:
            labels = tokenizer([preprocess_sql(x) for x in targets[start:start+128]],
                               padding=False, truncation=False, verbose=False)['input_ids']
        for j, ids in enumerate(encoded['input_ids']):
            feature = {'input_ids': ids, 'attention_mask': encoded['attention_mask'][j]}
            if labels is not None:
                feature['labels'] = labels[j]  # Includes EOS; padding is added later.
            features.append(feature)
    return features


def token_stats(tokenizer, texts):
    total, vocabulary = 0, set()
    for start in range(0, len(texts), 128):
        batches = tokenizer(texts[start:start+128], add_special_tokens=False,
                            padding=False, truncation=False, verbose=False)['input_ids']
        for ids in batches:
            total += len(ids)
            vocabulary.update(ids)
    return total / len(texts), len(vocabulary)
