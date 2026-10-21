---
title: Undoing in Git
summary: git restore file drops unsaved changes; git restore --staged unstages; git revert undoes a commit with a new one.
tags: [git, undo, restore, revert, reset]
---
# Undoing in Git

| To undo | Command |
| --- | --- |
| changes not yet staged, in a file | `git restore file.py` |
| staging | `git restore --staged file.py` |
| the last commit's message | `git commit --amend -m "Better message"` (not yet pushed) |
| a commit, keeping history | `git revert <commit>` |
| the last commit, keeping its changes | `git reset --soft HEAD~1` (not yet pushed) |

`git restore` and `git reset --hard` throw changes away for good: ask before running them.
