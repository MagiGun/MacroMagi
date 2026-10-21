---
title: SELECT in SQL
summary: SELECT columns FROM table WHERE … ORDER BY … LIMIT; GROUP BY with COUNT/SUM; JOIN … ON matches tables.
tags: [sql, select, join, group by]
---
# SELECT in SQL

```sql
SELECT name, total FROM orders WHERE total > 100 ORDER BY total DESC LIMIT 10;

SELECT city, COUNT(*) AS people, AVG(age) FROM people GROUP BY city HAVING COUNT(*) > 5;

SELECT o.id, c.name
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.created >= '2026-01-01';
```

`LEFT JOIN` keeps rows without a match. Text in single quotes; `LIKE '%rain%'` finds text in text.
