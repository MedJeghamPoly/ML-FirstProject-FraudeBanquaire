# Usage : make install | make test | make notebooks | make streamlit
.PHONY: install test notebooks pdf streamlit

install:
	pip install -r requirements.txt

test:
	pytest tests -v

notebooks:
	cd notebooks && jupyter nbconvert --execute --inplace 01_EDA.ipynb 02_Modeling.ipynb

pdf:
	py scripts/build_presentation_pdf.py

streamlit:
	streamlit run streamlit_app.py
