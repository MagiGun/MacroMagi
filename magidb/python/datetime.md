---
title: Dates and times in Python
summary: datetime.now(), strftime to write, strptime to read, timedelta to add; zoneinfo for time zones.
tags: [python, datetime, dates, time]
version: "3.12"
---
# Dates and times in Python

```python
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

now = datetime.now()
now.strftime("%Y-%m-%d %H:%M")                     # '2026-10-20 14:05'
datetime.strptime("20/10/2026", "%d/%m/%Y")        # read a date
date.today() + timedelta(days=7)
(date(2026, 12, 25) - date.today()).days
datetime.now(ZoneInfo("Europe/Lisbon"))
datetime.fromisoformat("2026-10-20T14:05:00")
```

`%Y` year, `%m` month, `%d` day, `%H` hour (24), `%M` minutes, `%S` seconds, `%A` the weekday's name.
