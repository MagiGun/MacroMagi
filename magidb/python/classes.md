---
title: Classes in Python
summary: A class with __init__ and methods; @dataclass writes __init__ for data; self is the object.
tags: [python, classes, dataclass, objects]
version: "3.12"
---
# Classes in Python

```python
class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        return self.balance

account = Account("Ana")
account.deposit(50)
```

## Data only

```python
from dataclasses import dataclass, field

@dataclass
class Item:
    name: str
    price: float
    tags: list[str] = field(default_factory=list)
```
