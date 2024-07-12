import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from modules.api_utils import load_models, perform_inference, create_XML

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load models at API startup
try:
    model_object, model_arrow = load_models()
    logger.info("Models loaded at startup")
except Exception as e:
    logger.error(f"Failed to load models at startup: {str(e)}")
    raise RuntimeError(f"Failed to load models: {str(e)}")

@app.get("/")
def read_root():
    return {"Welcome": "on the BPMN API!"}

@app.post("/test/")
def test_root(file: UploadFile = File(...)):
    logger.info("Test endpoint called")
    return {"Test": "successful!"}

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    score_threshold: float = 0.5,
    scale: float = 1.0,
    size_scale: float = 1.0
):
    try:
        logger.info("File received")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info("Image converted")

        prediction, text_mapping = perform_inference(model_object, model_arrow, image, score_threshold)
        logger.info("Inference performed")

        bpmn_xml = create_XML(prediction, text_mapping, scale, size_scale)
        logger.info("XML created")

        return {"bpmn_xml": bpmn_xml}
    except Exception as e:
        logger.error(f"Error during file upload and processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
