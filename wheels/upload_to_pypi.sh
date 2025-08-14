#! /bin/bash

set -e
pip install --upgrade build
pip install twine
python -m build
twine upload dist/* --verbose --repository-url https://test.pypi.org/legacy/ 