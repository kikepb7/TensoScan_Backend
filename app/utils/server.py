from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import pytesseract
from PIL import Image
import io
import uvicorn

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

@app.post("/process-images/")
async def process_images(files: List[UploadFile] = File(...)):
    if len(files) != 7:
        raise HTTPException(status_code=400, detail="Se deben enviar exactamente 7 imágenes.")

    digits = []
    for file in files:
        filename = file.filename
        parts = filename.split('_')
        extracted_digits = ""

        if len(parts) > 1:
            extracted_digits = ''.join(filter(str.isdigit, parts[1]))

        if len(extracted_digits) != 1:
            image = Image.open(io.BytesIO(await file.read()))
            text = pytesseract.image_to_string(image, config='--psm 10 -c tessedit_char_whitelist=0123456789')
            extracted_digits = ''.join(filter(str.isdigit, text))

            if len(extracted_digits) != 1:
                raise HTTPException(status_code=400, detail=f"No se pudo extraer un dígito ni del nombre del archivo ni de la imagen: {filename}")

        digits.append(extracted_digits)

    high_tension = ''.join(digits[:3])
    low_tension = ''.join(digits[3:5])
    pulse = ''.join(digits[5:])

    return {"high_tension": high_tension, "low_tension": low_tension, "pulse": pulse}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)