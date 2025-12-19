from fastapi import FastAPI, HTTPException
from .search_engine import SearchEngine
from contextlib import asynccontextmanager
import os

search_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Startup: initializing search engine')
    global search_engine

    current_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.dirname(current_dir)

    data_path = os.path.join(base_dir, "data", "cleaned_data.parquet")

    print(f'Trying to load from: {data_path}')

    if os.path.exists(data_path):
        search_engine = SearchEngine(data_path)
        print("Startup: Search engine loaded successfully")
    else:
        print(f'Warning: Data file {data_path} not found !')
     
    # differentiate the startup from shutdown section
    yield 

    print('Shutdown: Cleaning up resources...')
    search_engine = None

app = FastAPI(title="Greek Parliament Search Engine", description="Search speeches using the TF-IDF", lifespan=lifespan)

@app.get('/')
def read_root():
    return {
        "status": "Online",
        "message": "Go to /docs to test the search engine"
    }


@app.get("/search")
def search_speeches(query:str, limit: int = 5):
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine is not initialized")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    results = search_engine.search(query, top_k=limit)
    
    return {
        "query": query,
        "total_results": len(results),
        "results": results
    }

