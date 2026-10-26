"""Selection, signed backups, tape protection and standalone import boundaries."""
import ast
import unittest
from pathlib import Path

import numpy as np

import uct_16_questions_baseline as u
import plain_rl_16_cells_baseline as p
import uct_inspired_1024_cases_16_cells_train as b


class UCTBaselineTests(unittest.TestCase):
    def test_unvisited_priority_and_signed_value_scale(self):
        np.testing.assert_array_equal(u.uct_probabilities(np.array([[0., 0., 5.], [0., 0., 5.]])),
                                      [.5, .5, 0.])
        values = np.array([[1., 100., 100.], [0., 50., -100.]])
        self.assertEqual(u.uct_probabilities(values, success_value_scale=1).argmax(), 0)
        self.assertEqual(u.uct_probabilities(values).argmax(), 1)

    def test_backup_and_own_question_decay(self):
        stats = u.Statistics(.99)
        idx, values = stats.row(u.Tape(0).code, 0)
        stats.backup([(idx, 0), (idx, 0), (idx, 1)], True, False)
        np.testing.assert_array_equal(values[:, :2], [[2, 1], [2, 1]])
        stats.backup([(idx, 0), (idx, 2)], False, True)
        np.testing.assert_array_equal(values[:, :3], [[3, 1, 1], [2, 1, -1]])
        before = values.copy()
        stats.row(u.Tape(1).code, 100)
        np.testing.assert_array_equal(values, before)
        stats.row(u.Tape(0).code, 2)
        np.testing.assert_array_equal(values, before*np.float32(.99**2))
        self.assertEqual(stats.snapshot()['backup_rule'], u.SIGNED_BACKUP_RULE)

    def test_all_sixteen_questions_and_protected_writes(self):
        for qid in range(16):
            tape = u.Tape(qid)
            total = qid//4+qid%4
            for position, symbol in zip((6, 7, 8), (total//4, (total//2)%2, total%2)):
                self.assertEqual(tape.step(3*position+symbol), (False, False, False))
            self.assertEqual(tape.step(u.SUBMIT), (True, True, False))
            for position in u.PROTECTED_CELLS:
                tape = u.Tape(qid)
                before, code = tape.cells.copy(), tape.code
                symbol = int(tape.cells[position])
                self.assertEqual(tape.step(position*3+symbol), (False, False, False))
                self.assertEqual(tape.step(position*3+(1-symbol)), (False, False, True))
                np.testing.assert_array_equal(tape.cells, before)
                self.assertEqual(tape.code, code)

    def test_standalone_scripts_and_identical_binary_dataset(self):
        # No runtime import may depend on another project script, even lazily.
        allowed = {'argparse', 'collections', 'dataclasses', 'hashlib', 'json',
                   'math', 'pathlib', 'time', 'numpy', 'torch'}
        for module in (u, p, b):
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name.split('.')[0] in allowed for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module.split('.')[0], allowed)
        self.assertEqual(p.dataset_spec(), b.dataset_spec())
        self.assertEqual(p.load_dataset(p.DEFAULT_DATASET)[1], b.load_dataset(b.DEFAULT_DATASET)[1])


if __name__ == '__main__':
    unittest.main()
