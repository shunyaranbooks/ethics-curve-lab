from fastapi import FastAPI

app = FastAPI(title="Ethics Curve Lab API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}
