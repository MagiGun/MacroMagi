---
title: Tests with pytest
summary: Functions named test_* with assert; run python -m pytest; -k picks tests, -x stops at the first failure.
tags: [python, tests, pytest, testing]
version: "3.12"
---
# Tests with pytest

```python
# test_totals.py
from totals import total

def test_total_adds_amounts():
    assert total([1, 2, 3]) == 6

def test_total_of_nothing_is_zero():
    assert total([]) == 0
```

```
python -m pytest            # every test_*.py
python -m pytest -x -q      # stop at the first failure, short output
python -m pytest -k total   # only tests with "total" in the name
```

## Reading a failure

pytest shows the failing test, the assert with both values (`assert 5 == 6`), and the file and line. Fix the code (not the test) unless the test is wrong.

Errors that something was expected:

```python
import pytest

def test_negative_is_refused():
    with pytest.raises(ValueError):
        deposit(-1)
```

`No module named pytest`: `python -m pip install pytest`.
