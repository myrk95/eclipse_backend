import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2
import sys
import os

class MelanomaPredictor:
    def __init__(self, model_path=None, config_path=None):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            model_path = os.path.join(script_dir, 'modelo_melanoma_4datasets_final.h5')
        else:
            if not os.path.isabs(model_path):
                model_path = os.path.join(script_dir, model_path)
        
        if config_path is None:
            config_path = os.path.join(script_dir, 'config_modelo_4datasets.json')
        else:
            if not os.path.isabs(config_path):
                config_path = os.path.join(script_dir, config_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = tf.keras.models.load_model(model_path)
        
        self.img_size = self.config.get('img_size', (224, 224))
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)
        
        self.umbral_optimo = self.config.get('umbral_optimo', 0.2231)
        
        print(f"¡Modelo cargado!")
        print(f"Tamaño de entrada óptimo: {self.img_size}")
        print(f"Umbral óptimo: {self.umbral_optimo}")
    
    def preprocess_image(self, image_path, preserve_aspect=False):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
            
            if not preserve_aspect:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
            
            else:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"No se pudo leer la imagen: {image_path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                target_h, target_w = self.img_size
                aspect = w / h
                
                if w > h:
                    new_w = target_w
                    new_h = int(target_w / aspect)
                else:
                    new_h = target_h
                    new_w = int(target_h * aspect)
                
                resized = cv2.resize(img, (new_w, new_h))
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                img_array = canvas / 255.0
            
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Error al preprocesar imagen {image_path}: {e}")
            raise
    
    def predict(self, image_path, preserve_aspect=False):
        try:
            processed_img = self.preprocess_image(image_path, preserve_aspect)
            probabilidad = self.model.predict(processed_img, verbose=0)[0][0]
            
            if probabilidad > self.umbral_optimo:
                prediccion = "MALIGNO"
            else:
                prediccion = "BENIGNO"
            
            return probabilidad, prediccion
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None, None
    
    def batch_predict(self, image_paths, preserve_aspect=False):
        
        if not image_paths:
            return []
        
        batch_images = []
        valid_paths = []
        
        for img_path in image_paths:
            try:
                processed = self.preprocess_image(img_path, preserve_aspect)
                batch_images.append(processed)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Omitiendo {img_path}: {e}")
        
        if not batch_images:
            return []
        
        batch_array = np.concatenate(batch_images, axis=0)
        predictions = self.model.predict(batch_array, verbose=0)
        
        results = []
        for i, prob in enumerate(predictions.flatten()):
            if prob > self.umbral_optimo:
                prediccion = "MALIGNO"
            else:
                prediccion = "BENIGNO"
            
            results.append({
                'image': valid_paths[i],
                'probabilidad': float(prob),
                'prediccion': prediccion,
                'porcentaje': f"{prob:.2%}"
            })
        
        return results

def main():
    try:
        predictor = MelanomaPredictor()
        
        while True:
            print("1. Predecir una imagen")
            print("2. Predecir múltiples imágenes (batch)")
            print("3. Mostrar información del modelo")
            print("4. Salir")
            
            opcion = input("\nSelecciona una opción (1-4): ").strip()
            
            if opcion == "1":
                image_path = input("Introduce la ruta de la imagen: ").strip()
                
                if not os.path.exists(image_path):
                    print("Error: La imagen no existe.")
                    continue
                
                print("1. Redimensionar directamente (rápido)")
                print("2. Mantener relación de aspecto (recomendado)")
                
                metodo = input("Selecciona método (1-2): ").strip()
                preserve_aspect = (metodo == "2")
                
                try:
                    prob, pred = predictor.predict(image_path, preserve_aspect)
                    
                    if prob is not None:
                        print(f"\nRESULTADO:")
                        print(f"   Imagen: {os.path.basename(image_path)}")
                        print(f"   Probabilidad de melanoma: {prob:.2%}")
                        print(f"   Predicción: {pred}")
                        print(f"   Umbral: {predictor.umbral_optimo}")
                        
                        # Interpretación
                        if pred == "MALIGNO":
                            print(f"CONSULTA CON UN DERMATÓLOGO")
                        else:
                            print(f"✓ Probabilidad baja, pero sigue con revisiones periódicas")
                    else:
                        print("No se pudo obtener predicción.")
                        
                except Exception as e:
                    print(f"Error: {e}")
            
            elif opcion == "2":
                print("Introduce las rutas de las imágenes (una por línea).")
                print("Escribe 'fin' cuando termines.")
                
                image_paths = []
                while True:
                    path = input("Ruta: ").strip()
                    if path.lower() == 'fin':
                        break
                    if os.path.exists(path):
                        image_paths.append(path)
                    else:
                        print(f"La ruta no existe: {path}")
                
                if not image_paths:
                    print("No se ingresaron imágenes válidas.")
                    continue
                
                print("1. Redimensionar directamente (rápido)")
                print("2. Mantener relación de aspecto (recomendado)")
                
                metodo = input("Selecciona método (1-2): ").strip()
                preserve_aspect = (metodo == "2")
                
                try:
                    results = predictor.batch_predict(image_paths, preserve_aspect)
                    
                    print(f"\nRESULTADOS ({len(results)} imágenes):")
                    
                    malignas = 0
                    for result in results:
                        print(f"{os.path.basename(result['image'])}:")
                        print(f"→ {result['prediccion']} ({result['porcentaje']})")
                        
                        if result['prediccion'] == "MALIGNO":
                            malignas += 1
                    
                    print(f"   Total imágenes: {len(results)}")
                    print(f"   Benignas: {len(results) - malignas}")
                    print(f"   Malignas: {malignas}")
                    print(f"   Tasa de malignidad: {malignas/len(results):.1%}")
                    
                except Exception as e:
                    print(f"Error: {e}")
            
            elif opcion == "3":
                print(f"   Tamaño de entrada: {predictor.img_size}")
                print(f"   Umbral óptimo: {predictor.umbral_optimo}")
                print(f"   Arquitectura: MobileNetV2 con capas densas personalizadas")
                print(f"   Total parámetros: ~3 millones")
                print(f"   Precisión reportada: ~82% (validación)")
                print(f"   Sensibilidad (recall): 79.56%")
                print(f"   Precisión: 70.76%")
                print(f"   F1-score: 74.90%")
            
            elif opcion == "4":
                break
            
            else:
                print("Opción no válida.")
    
    except Exception as e:
        print(f"Error inicializando el predictor: {e}")
        print("Asegúrate de que los archivos están en la misma carpeta que inferencia.py")
        print("   Archivos requeridos:")
        print("   1. modelo_melanoma_4datasets_final.h5")
        print("   2. config_modelo_4datasets.json")

if __name__ == "__main__":
    main()