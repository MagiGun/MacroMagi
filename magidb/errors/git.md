---
title: Git errors
summary: not a git repository, rejected push, merge conflict, detached HEAD – what to do.
tags: [errors, git]
---
# Git errors

| Error | Do |
| --- | --- |
| `fatal: not a git repository` | run it in the project's folder (`folder` of run_command), or `git init` |
| `! [rejected] … (fetch first)` | `git pull`, settle any conflict, then `git push` |
| `CONFLICT (content): Merge conflict in x` | see [branches](../git/branches.md) |
| `You are in 'detached HEAD' state` | `git switch main` (or `git switch -c new-branch` to keep work) |
| `Please tell me who you are` | `git config user.name "…"` and `git config user.email "…"` – ask the user for them |
