from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/")
async def print_json(request: Request):
    data = await request.json()   # Parse the incoming JSON
    print(data)                   # Print to console/logs
    return {"received": data}     # Return back the same JSON
