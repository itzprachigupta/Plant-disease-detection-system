from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import uvicorn
import tensorflow as tf

model = tf.keras.models.load_model("./Saved_model/final_model.h5")

solutions = {
    'Anthurium_Bacterial_Blight':'* Isolate the affected plant.\n*Prune and destroy infected plant parts.\n*Apply copper-based fungicides.\n*Reduce humidity and ensure good ventilation.\n*Avoid overhead watering.\n*Use well-draining soil.\n*Quarantine new plants.',
    'Anthurium_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Anthurium_Leaf_Spot':'*Prune and remove affected leaves.\n*Apply copper-based fungicides.\n*Maintain good air circulation.\n*Avoid overhead watering.\n*Manage water carefully.\n*Ensure proper spacing between plants.\n*Fertilize in moderation.',
    'Gerbera_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Gerbera_Leaf_Blight':'*Apply a fungicide with active ingredients like copper or mancozeb as a preventive measure.\n*Prune and remove infected leaves and debris.\n*Avoid overhead watering.\n*Maintain good spacing for air circulation.',
    'Gerbera_Leaf_Spot':'*Prune and remove affected leaves.\n*Improve air circulation by spacing plants adequately.\n*Water at the base to avoid wetting the leaves.',
    'Gerbera_Powdery_Mildew':'*Apply a fungicide such as neem oil or sulfur-based fungicides.\n*Prune and remove heavily infected leaves.\n*Avoid overhead watering; keep leaves dry.\n*Provide proper spacing for better ventilation.',
    'Gladiolus_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Gladiolus_Leaf_Spot':'*Prune and remove affected leaves.\n*Ensure good air circulation by spacing the gladiolus plants adequately.\n*Water at the base to keep foliage dry.\n*Fertilize in moderation.',
    'Gladiolus_Rust':'*Apply a fungicide suitable for rust control, typically containing copper or sulfur\n*Prune and remove heavily affected leaves.\n*Maintain proper spacing for air circulation.\n*Avoid overhead watering to prevent the spread of rust spores.',
    'Jasmine_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Jasmine_Leaf_Spot':'*Prune and dispose of affected leaves and branches.\n*Water at the base to keep leaves dry.\n*Apply a suitable fungicide\n*Ensure good air circulation and proper spacing.\n*Maintain a balanced fertilization schedule.',
    'Marigold_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Marigold_Leaf_Spot':'*Prune and dispose of affected leaves and branches.\n*Water at the base to keep leaves dry.\n*Apply a suitable fungicide\n*Ensure good air circulation and proper spacing.\n*Maintain a balanced fertilization schedule.',
    'Orchid_Black_Spot':'*Isolate the infected orchid from healthy ones.\n*Remove affected leaves.\n*Ensure proper ventilation and avoid overcrowding.\n*Apply a copper-based fungicide',
    'Orchid_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Orchid_Sun_Burn':'*Gradually acclimate the orchid to direct sunlight.\n*Provide filtered or indirect light.\n*Use shade cloth to protect orchids during hot periods.\n*Water and mist properly to avoid drying out.',
    'Orchid_Yellow_Leaf':'*Check for proper watering; avoid overwatering or underwatering.\n*Ensure the orchid is not exposed to direct sunlight or drafts.\n*Monitor the orchid for pests, as they can lead to yellowing leaves.\n*Adjust fertilizer application and light levels.',
    'Rose_Black_Spot':'*Spray with a fungicide containing myclobutanil, trifloxystrobin, or copper\n*Prune and remove infected leaves\n*Promote good air circulation.\n*Water at the base, avoiding wetting leaves.',
    'Rose_Downy_mildew':'*Spray with a fungicide containing mancozeb or copper\n*Prune and destroy infected leaves.\n*Provide proper spacing for air circulation.\n*Avoid overhead watering.',
    'Rose_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Tuberose_Fresh_Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.',
    'Tuberose_Leaf_blight':'*Isolate the affected tuberose from healthy plants.\n*Remove and destroy infected leaves.\n*Provide good air circulation and avoid overcrowding.\n*Apply a copper-based fungicide',
    'Tuberose_Leaf_spot':'*Prune and remove affected leaves.\n*Avoid overhead watering; water at the base.\n*Ensure proper spacing for air circulation.'
}

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((180, 180))
    image_array = np.array(image) / 179.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    classes = ['Anthurium_Bacterial_Blight',
               'Anthurium_Fresh_Leaf',
               'Anthurium_Leaf_Spot',
               'Gerbera_Fresh_Leaf',
               'Gerbera_Leaf_Blight',
               'Gerbera_Leaf_Spot',
               'Gerbera_Powdery_Mildew',
               'Gladiolus_Fresh_Leaf',
               'Gladiolus_Leaf_Spot',
               'Gladiolus_Rust',
               'Jasmine_Fresh_Leaf',
               'Jasmine_Leaf_Spot',
               'Marigold_Fresh_Leaf',
               'Marigold_Leaf_Spot',
               'Orchid_Black_Spot',
               'Orchid_Fresh_Leaf',
               'Orchid_Sun_Burn',
               'Orchid_Yellow_Leaf',
               'Rose_Black_Spot',
               'Rose_Downy_mildew',
               'Rose_Fresh_Leaf',
               'Tuberose_Fresh_Leaf',
               'Tuberose_Leaf_blight',
               'Tuberose_Leaf_spot']

    predicted_class = classes[np.argmax(prediction)]
    
    confidence = np.max(prediction[0])
    
    solution = solutions[predicted_class]

    print(predicted_class, confidence)
    print(solution)

    # Replace newline characters with HTML line breaks for proper display in the response
    solution = solution.replace('\n', '<br/>')
    
    if (confidence < 0.2):
        return {
            'class': 'Undefined',
            'confidence': float(confidence),
            'solution': "This does not seems correct input. Please provide a clearer image."
        }
    else:
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'solution': solution
        }
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
