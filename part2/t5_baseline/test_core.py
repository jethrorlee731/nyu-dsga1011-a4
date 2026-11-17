"""Fast dependency-free tests: python t5_baseline/test_core.py."""
import pickle
from pathlib import Path
import sqlite3
import tempfile
import unittest
from text_data import preprocess_nl, preprocess_sql, read_split
from sql_eval import execute_all, execute_one, metrics, save_outputs


class BaselineTests(unittest.TestCase):
    def test_preprocessing_preserves_sql_literals(self):
        self.assertEqual(preprocess_nl('  flights\tfrom  Boston  '),
                         'translate to SQL: flights from Boston')
        self.assertEqual(preprocess_sql("  SELECT 'NEW  YORK', 'MKE'; \n"),
                         "SELECT 'NEW  YORK', 'MKE';")

    def test_pair_validation_and_test_without_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'train.nl').write_text('one\ntwo\n')
            (p/'train.sql').write_text('SELECT 1\n')
            with self.assertRaises(ValueError):
                read_split(p, 'train')
            (p/'test.nl').write_text('first\nsecond\n')
            self.assertEqual(read_split(p, 'test'), (['first','second'], None))
            (p/'train.sql').write_text('SELECT 1\nSELECT 2\n')
            self.assertEqual(read_split(p, 'train', 1), (['one'], ['SELECT 1']))

    def test_metrics_partial_empty_and_order(self):
        result = metrics(['SELECT 1', 'SELECT 2', 'SELECT 3'],
                         ['SELECT 1', 'different SQL', 'SELECT 3'],
                         [[(1,), (2,)], [], [(3,)]],
                         [[(2,), (2,)], [], []], ['', '', 'bad query'])
        self.assertAlmostEqual(result['sql_em'], 2/3)
        self.assertAlmostEqual(result['record_em'], 1/3)
        self.assertAlmostEqual(result['record_f1'], (2/3 + 1)/3, places=7)
        self.assertAlmostEqual(result['execution_error_rate'], 1/3)
        with self.assertRaises(ValueError):
            metrics(['a'], [], [[]], [[]], [''])

    def test_execution_order_readonly_and_output_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            database = p/'test.db'
            with sqlite3.connect(database) as conn:
                conn.execute('CREATE TABLE flight (id INTEGER)')
                conn.executemany('INSERT INTO flight VALUES (?)', [(1,), (2,)])
            queries = ['SELECT id FROM flight ORDER BY id DESC', 'SELECT 42',
                       'SELECT nonexistent FROM flight', 'DELETE FROM flight']
            records, errors = execute_all(database, queries, workers=2)
            self.assertEqual(records[:2], [[(2,), (1,)], [(42,)]])
            self.assertEqual(errors[:2], ['', ''])
            self.assertTrue(errors[2] and errors[3])
            self.assertEqual(execute_one(database, 'SELECT COUNT(*) FROM flight', 1)[0], [(2,)])
            save_outputs(queries, records, errors, p/'pred.sql', p/'pred.pkl')
            self.assertEqual((p/'pred.sql').read_text().splitlines(), queries)
            with (p/'pred.pkl').open('rb') as f:
                self.assertEqual(pickle.load(f), (records, errors))

    def test_query_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.db'
            sqlite3.connect(path).close()
            records, error = execute_one(path,
                'WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x',
                0.01)
            self.assertEqual(records, [])
            self.assertIn('interrupted', error)


if __name__ == '__main__':
    unittest.main()
