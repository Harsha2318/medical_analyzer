from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
from datetime import datetime

from main import process_medical_document

app = FastAPI()

static_dir = Path("static")
data_dir = Path("data")
for directory in [static_dir, data_dir]:
    directory.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-medical-document")
async def analyze_document(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Only PDF files are supported"}
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = data_dir / safe_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_medical_document(str(file_path))
        return JSONResponse(content={
            "status": "success",
            "analysis": result["analysis"],
            "summary": result["summary"],
            "validation": result["validation"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/cleanup")
async def cleanup_old_files():
    current_time = datetime.now()
    for file_path in data_dir.glob("*.pdf"):
        file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.days >= 1:
            file_path.unlink()
    return JSONResponse(content={"status": "success", "message": "Cleanup completed"})
