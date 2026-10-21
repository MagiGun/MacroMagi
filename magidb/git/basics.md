---
title: Saving work with Git
summary: git status shows changes; git add stages them; git commit -m saves; git log --oneline lists; git diff shows what changed.
tags: [git, commit, status, diff]
---
# Saving work with Git

```
git status                  # what changed
git diff                    # how (not yet staged)
git add file.py             # stage a file (git add -A: everything)
git commit -m "Fix the total for empty lists"
git log --oneline -10
git push                    # send to the remote
git pull                    # get others' work
```

Write the message as what the commit does, in a short line.
