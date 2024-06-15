from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import uvicorn

app_desc = """<h2>Subida de Imagenes `predict/image`</h2>
<h2>BASURA DETECTION</h2>"""

app = FastAPI(title='PROYECTO BASURITA', description=app_desc)

# Carga del modelo entrenado
try:
    model = load_model("model2.keras")
    input_shape = model.input_shape[1:3]  # Obtener el tamaño de entrada del modelo (altura y ancho)
    print(f"Modelo cargado correctamente. Tamaño de entrada esperado: {input_shape}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

def prepare_image(img: Image.Image, target_size: tuple) -> np.ndarray:
    """Prepara la imagen para la predicción."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización
    return img_array

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Verifica que el archivo sea una imagen
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")

    try:
        # Lee la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Prepara la imagen
        img_array = prepare_image(img, target_size=input_shape)  # Usar el tamaño de entrada del modelo

        # Realiza la predicción
        prediction = model.predict(img_array)
        predicted_class = "trash" if prediction[0][0] > 0.5 else "clean"
        print(prediction)
        print(prediction[0][0])
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')