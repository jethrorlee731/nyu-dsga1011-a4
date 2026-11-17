"""Assignment-compatible SQL outputs and metrics with bounded read-only execution.

The metric definitions reproduce the uploaded utils.py: literal SQL exact match,
set-based record EM, and mean per-example record F1 (including its 1e-8 epsilon).
The pickle format is exactly (records, error_messages), not just records.
"""
import json
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def execute_one(db_path, query, timeout):
    deadline = time.monotonic() + timeout
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    try:
        cursor = conn.execute(query)
        if cursor.description is None:
            raise ValueError('Expected a query returning records')
        return cursor.fetchall(), ''
    except Exception as exc:
        return [], f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()


def execute_all(db_path, queries, timeout=5.0, workers=4):
    if not Path(db_path).is_file():
        raise FileNotFoundError(f'Missing database: {db_path}')
    # executor.map preserves input order, even when queries finish out of order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(pool.map(lambda q: execute_one(db_path, q, timeout), queries))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def save_outputs(queries, records, errors, sql_path, record_path):
    if not (len(queries) == len(records) == len(errors)):
        raise ValueError('Queries, records, and errors must have identical counts')
    if any('\n' in q or '\r' in q for q in queries):
        raise ValueError('SQL output must have exactly one query per line')
    for path in (sql_path, record_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(sql_path).write_text('\n'.join(queries) + '\n', encoding='utf-8')
    with open(record_path, 'wb') as handle:
        pickle.dump((records, errors), handle)


def record_scores(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    overlap = len(gold & predicted)
    precision = overlap / len(predicted) if predicted else 1.0
    recall = overlap / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(gold == predicted), f1


def metrics(gold_sql, predicted_sql, gold_records, predicted_records, errors):
    count = len(gold_sql)
    if count == 0 or any(len(x) != count for x in
                         (predicted_sql, gold_records, predicted_records, errors)):
        raise ValueError('Metric inputs must be nonempty and have matching counts')
    scores = [record_scores(g, p) for g, p in zip(gold_records, predicted_records)]
    return {
        'sql_em': sum(g.strip() == p.strip() for g, p in zip(gold_sql, predicted_sql)) / count,
        'record_em': sum(s[0] for s in scores) / count,
        'record_f1': sum(s[1] for s in scores) / count,
        # Includes syntax, execution, and timeout errors; not just syntax errors.
        'execution_error_rate': sum(bool(e) for e in errors) / count,
    }


def write_error_analysis(path, inputs, gold_sql, predicted_sql, gold_records,
                         predicted_records, errors):
    """Per-example evidence for Q6; it does not invent semantic error categories."""
    with open(path, 'w', encoding='utf-8') as handle:
        for i, (text, gold, pred, gr, pr, error) in enumerate(zip(
                inputs, gold_sql, predicted_sql, gold_records, predicted_records, errors)):
            em, f1 = record_scores(gr, pr)
            row = {'index': i, 'input': text, 'gold_sql': gold, 'predicted_sql': pred,
                   'sql_em': gold.strip() == pred.strip(), 'record_em': em,
                   'record_f1': f1, 'execution_error': error,
                   'gold_record_count': len(set(gr)), 'predicted_record_count': len(set(pr))}
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
