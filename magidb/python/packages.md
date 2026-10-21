---
title: Packages, pip and virtual environments
summary: python -m pip install name; a virtual environment per project with python -m venv .venv; requirements.txt lists packages.
tags: [python, pip, packages, venv, install]
version: "3.12"
---
# Packages, pip and virtual environments

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate        (Linux, macOS)
python -m pip install requests
python -m pip install -r requirements.txt
python -m pip freeze > requirements.txt
```

- `python -m pip` installs into the Python you run, which `pip` alone may not.
- `ModuleNotFoundError: No module named 'x'`: install it – but the package name can differ: `yaml` is `pyyaml`, `cv2` is `opencv-python`, `PIL` is `pillow`, `docx` is `python-docx`, `sklearn` is `scikit-learn`.
