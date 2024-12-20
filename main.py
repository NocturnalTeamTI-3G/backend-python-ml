from app.core import config, model_predict, model_predict1, model_predict2
from fastapi import FastAPI, File, UploadFile
import shutil
import os

config = config.Config()

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # create folder if not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # check if the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"predict": "Invalid file type"}
    # Save the uploaded file to the tmp directory
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Perform prediction
    # predict = model_predict.predict_jerawat(file_location, 'svm_model.joblib')
    # predict = model_predict1.predict_jerawat(file_location, 'svm_model1.joblib')
    predict = model_predict2.predict_jerawat(file_location, 'svm_model.joblib', 'scaler.pkl')
    
    # Delete the file after prediction
    os.remove(file_location)
    
    return {"predict": predict}
