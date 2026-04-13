# Usage : make install | make test | make notebooks (nécessite jupyter)
.PHONY: install test notebooks pdf

install:
	pip install -r requirements.txt

test:
	pytest tests -v

notebooks:
	cd notebooks && jupyter nbconvert --execute --inplace 01_EDA.ipynb 02_Modeling.ipynb

pdf:
	py scripts/build_presentation_pdf.py
