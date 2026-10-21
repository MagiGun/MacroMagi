---
title: Changing rows in SQL
summary: INSERT INTO t (cols) VALUES (...); UPDATE t SET col = ... WHERE ...; DELETE FROM t WHERE ...
tags: [sql, insert, update, delete]
---
# Changing rows in SQL

```sql
INSERT INTO people (name, age) VALUES ('Ana', 31);
UPDATE people SET age = 32 WHERE name = 'Ana';
DELETE FROM people WHERE age IS NULL;
```

Always give `WHERE` to UPDATE and DELETE – without it every row changes. Try it as a SELECT first.
