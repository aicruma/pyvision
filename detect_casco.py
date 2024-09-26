import os
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Configuración de rutas
BASE_DIR = os.path.expanduser('~/helmet_detection')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Crear directorios si no existen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar el modelo preentrenado Faster R-CNN
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    """Carga y transforma la imagen."""
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Añadir dimensión de lote

def detect_objects(image_path):
    """Detecta objetos en la imagen y analiza la presencia de cascos."""
    image_tensor = load_image(image_path)
    original_image = cv2.imread(image_path)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Obtener las cajas de detección y etiquetas
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']
    
    # Umbral de confianza
    threshold = 0.5
    persons_detected = 0
    helmets_detected = 0
    
    for i, score in enumerate(scores):
        if score > threshold:
            label = labels[i].item()
            if label == 1:  # Clase 'persona' en COCO
                persons_detected += 1
                box = boxes[i].cpu().numpy().astype(int)
                
                # Extraer la región de la cabeza (asumimos que está en el 1/3 superior del cuerpo)
                head_top = box[1]
                head_bottom = box[1] + (box[3] - box[1]) // 3
                head_region = original_image[head_top:head_bottom, box[0]:box[2]]
                
                # Analizar la región de la cabeza para detectar cascos
                if detect_helmet_in_region(head_region):
                    helmets_detected += 1
                    cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(original_image, 'Persona con casco', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(original_image, 'Persona sin casco', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Guardar la imagen con las detecciones
    output_image_path = os.path.join(OUTPUT_DIR, f'detected_{os.path.basename(image_path)}')
    cv2.imwrite(output_image_path, original_image)
    
    return persons_detected, helmets_detected

def detect_helmet_in_region(region):
    """
    Analiza la región de la cabeza para detectar la presencia de un casco.
    Esta es una implementación simplificada y puede requerir ajustes.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral para detectar áreas claras (potenciales cascos)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Calcular el porcentaje de píxeles blancos
    white_pixel_percentage = (thresh > 0).mean()
    
    # Si más del 30% de los píxeles son blancos, asumimos que hay un casco
    return white_pixel_percentage > 0.3

def analyze_directory(data_dir):
    """Analiza todas las imágenes en un directorio."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(data_dir, filename)
            print(f"\nAnalizando imagen: {image_path}")
            
            persons, helmets = detect_objects(image_path)
            
            print(f"Personas detectadas: {persons}")
            print(f"Cascos detectados: {helmets}")
            
            # Guardar resultados en un archivo
            result_path = os.path.join(OUTPUT_DIR, f'resultado_{filename}.txt')
            with open(result_path, 'w') as f:
                f.write(f"Resultados de la detección para {filename}:\n")
                f.write(f"Personas detectadas: {persons}\n")
                f.write(f"Cascos detectados: {helmets}\n")

def main():
    analyze_directory(DATA_DIR)

if __name__ == "__main__":
    main()