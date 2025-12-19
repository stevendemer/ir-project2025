# IR PROJECT - SEARCH / ANALYTICS ENGINE

## How to run

* python -m venv venv
* source venv/bin/activate (linux/mac)
* pip install -r requirements.txt
* to run the streamlit frontend -> streamlit run frontend.py
* to run the analytics engine / fastapi backend (from the root folder - not inside the src/) uvicorn src.api:app --reload
* data/ folder contains the csv speeches and parquet files used for processing
