from pathlib import Path
from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained(
    "google-t5/t5-small",
    cache_dir="/scratch/jnl9728/cache/huggingface/hub",
)

def token_stats(texts):
    total_tokens = 0
    vocabulary = set()

    for start in range(0, len(texts), 128):
        tokenized = tokenizer(
            texts[start:start + 128],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            verbose=False,
        )["input_ids"]

        for ids in tokenized:
            total_tokens += len(ids)
            vocabulary.update(ids)

    return total_tokens / len(texts), len(vocabulary)

def preprocess_nl(text):
    text = " ".join(text.split())
    return "translate to SQL: " + text


def preprocess_sql(text):
    return text.strip()

raw = {}
processed = {}

for split in ("train", "dev"):
    nl = Path(f"data/{split}.nl").read_text(encoding="utf-8").splitlines()
    sql = Path(f"data/{split}.sql").read_text(encoding="utf-8").splitlines()

    assert len(nl) == len(sql), f"Mismatched pairs in {split}"
    assert all(s.strip() for s in nl + sql), f"Blank entry in {split}"

    raw[split] = (nl, sql)

    # Proposed minimal preprocessing: match load_lines() in your starter.
    # Replace these with your actual preprocessing if you do more.
    processed[split] = (
        [preprocess_nl(s) for s in nl],
        [preprocess_sql(s) for s in sql],
    )

for title, data in [
    ("TABLE 1: Before preprocessing", raw),
    ("TABLE 2: After stripping leading/trailing whitespace", processed),
]:
    results = {}

    for split, (nl, sql) in data.items():
        nl_mean, nl_vocab = token_stats(nl)
        sql_mean, sql_vocab = token_stats(sql)

        results[split] = [
            len(nl),
            nl_mean,
            sql_mean,
            nl_vocab,
            sql_vocab,
        ]

    names = [
        "Number of examples",
        "Mean sentence length",
        "Mean SQL query length",
        "Vocabulary size (natural language)",
        "Vocabulary size (SQL)",
    ]

    print("\n" + title)
    print(f"{'Statistic':38s} {'Train':>12s} {'Dev':>12s}")

    for i, name in enumerate(names):
        train_value = results["train"][i]
        dev_value = results["dev"][i]
        if i in (1, 2):
            print(f"{name:38s} {train_value:12.2f} {dev_value:12.2f}")
        else:
            print(f"{name:38s} {train_value:12d} {dev_value:12d}")