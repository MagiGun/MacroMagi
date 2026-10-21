---
title: Branches in Git
summary: git switch -c name makes a branch; git switch name moves to it; git merge name brings it in.
tags: [git, branch, merge]
---
# Branches in Git

```
git switch -c fix-totals    # a new branch, and move to it
git switch main
git merge fix-totals
git branch -d fix-totals    # delete it once merged
git branch                  # list them
```

A merge conflict: open the files it names, keep the right lines between `<<<<<<<` and `>>>>>>>`, then `git add` them and `git commit`.
