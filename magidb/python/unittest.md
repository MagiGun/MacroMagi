---
title: Tests with unittest
summary: Classes of unittest.TestCase with test_* methods and self.assertEqual; run python -m unittest; it needs nothing installed.
tags: [python, tests, unittest, testing]
version: "3.12"
---
# Tests with unittest

`unittest` comes with Python: no install. pytest runs these tests too.

```python
# tests/test_stats.py
import unittest

from shop.stats import total


class TotalTests(unittest.TestCase):
    def test_adds_amounts(self):
        self.assertEqual(total([{"amount": 2}, {"amount": 3}], "amount"), 5)

    def test_nothing_is_zero(self):
        self.assertEqual(total([], "amount"), 0)

    def test_negative_is_refused(self):
        with self.assertRaises(ValueError):
            total([{"amount": -1}], "amount")
```

```
python -m unittest                      # every tests/test_*.py (run it in the project's folder)
python -m unittest -v                   # each test's name
python -m unittest tests.test_stats     # one file
```

`tests/` needs an empty `__init__.py`, or no tests are found ("Ran 0 tests").

## Reading a failure

```
FAIL: test_nothing_is_zero (tests.test_stats.TotalTests.test_nothing_is_zero)
...
AssertionError: 1 != 0
```

The first value is what the code gave, the second what the test expected. `ERROR` instead of `FAIL` means the code raised an exception: read its traceback's last lines. Fix the code, not the test, unless the test is wrong.

Other checks: `assertTrue(x)`, `assertIn(a, b)`, `assertAlmostEqual(a, b, places=2)`, `assertIsNone(x)`.
