from fastapi import FastAPI, UploadFile, File, Query
import pandas as pd
import io

app = FastAPI()

df = None  # Global variable to store the dataframe

@app.post("/load/")
async def load_data(file: UploadFile = File(...)):
    global df
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {"message": "File loaded successfully", "shape": df.shape}

@app.get("/duplicates/")
def identify_duplicates():
    if df is None:
        return {"error": "No data loaded"}
    duplicate_count = df.duplicated().sum()
    return {"duplicate_count": duplicate_count}

@app.get("/missing-values/")
def check_missing_values():
    if df is None:
        return {"error": "No data loaded"}
    missing_values = df.isna().sum().to_dict()
    return {"missing_values": missing_values}

@app.get("/datatypes/")
def get_data_types():
    if df is None:
        return {"error": "No data loaded"}
    data_types = df.dtypes.astype(str).to_dict()
    return {"data_types": data_types}

@app.get("/class-imbalance/")
def check_class_imbalance(columns: list[str] = Query(None)):
    if df is None:
        return {"error": "No data loaded"}
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    imbalance = {col: df[col].value_counts(normalize=True).to_dict() for col in columns if col in df.columns}
    
    return {"class_imbalance": imbalance}

@app.get("/unique-values/")
def get_unique_values():
    if df is None:
        return {"error": "No data loaded"}
    unique_values = df.nunique().to_dict()
    return {"unique_values": unique_values}
