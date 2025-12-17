import os
import sys
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def _skin_ratio_ycrcb(rgb: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(mask.mean() / 255.0)

def _is_likely_skin_photo(rgb: np.ndarray, min_skin_ratio: float) -> Tuple[bool, float]:
    r = _skin_ratio_ycrcb(rgb)
    return (r >= float(min_skin_ratio)), r

def _photographic_gate(
    rgb: np.ndarray,
    min_photo_score: float,
) -> Tuple[bool, Dict[str, float]]:
    
    h, w = rgb.shape[:2]
    scale = 256.0 / max(h, w) if max(h, w) > 256 else 1.0
    if scale != 1.0:
        rgb_s = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        rgb_s = rgb

    gray = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)

    # 1) Edge fraction
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge_frac = float((mag > 25.0).mean())

    # 2) Textura local
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    local = cv2.absdiff(gray, blur)
    texture = float(local.mean() / 255.0)  # [0..1] aprox.

    # 3) Entropía de histograma (promedio en canales)
    entropies = []
    for c in range(3):
        hist = cv2.calcHist([rgb_s], [c], None, [32], [0, 256]).ravel().astype(np.float64)
        p = hist / (hist.sum() + 1e-12)
        ent = -float((p[p > 0] * np.log2(p[p > 0])).sum())  # [0..~5]
        entropies.append(ent)
    entropy = float(np.mean(entropies) / 5.0)  # normalizar ~[0..1]

    photo_score = 0.55 * entropy + 0.35 * texture - 0.25 * edge_frac

    ok = photo_score >= float(min_photo_score)
    return ok, {
        "photo_score": float(photo_score),
        "min_photo_score": float(min_photo_score),
        "entropy": float(entropy),
        "texture": float(texture),
        "edge_frac": float(edge_frac),
    }

def _color_domain_check(
    rgb: np.ndarray,
    max_blue_green_frac: float,
) -> Tuple[bool, Dict[str, float]]:
   
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    blue_green_frac = float(((h > 80) & (h < 140) & (s > 80)).mean())

    ok = blue_green_frac <= float(max_blue_green_frac)
    return ok, {"blue_green_frac": blue_green_frac, "max_blue_green_frac": float(max_blue_green_frac)}

def _center_square_crop(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s]

def _gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) + 1e-6
    mean = x.reshape(-1, 3).mean(axis=0)
    gray = float(mean.mean())
    gain = gray / mean
    x = x * gain
    return np.clip(x, 0, 255).astype(np.uint8)


def _image_quality_checks(rgb: np.ndarray) -> List[str]:
    warnings: List[str] = []

    # 1) Desenfoque (varianza del Laplaciano en luminancia)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fm = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if fm < 40.0:
        warnings.append(f"Imagen posiblemente borrosa (focus={fm:.1f}).")

    # 2) Reflejos/flash: muchos píxeles muy brillantes
    v = rgb.max(axis=2)
    highlight_frac = float((v >= 250).mean())
    if highlight_frac > 0.02:
        warnings.append(f"Posible flash/reflejos (highlights={highlight_frac:.1%}).")

    # 3) Sombras duras / iluminación muy desigual izquierda-derecha
    left = float(gray[:, : gray.shape[1] // 2].mean())
    right = float(gray[:, gray.shape[1] // 2 :].mean())
    if abs(left - right) > 35:
        warnings.append("Iluminación desigual (sombras duras).")

    return warnings


def _lesion_too_small_heuristic(rgb: np.ndarray) -> bool:
    h, w = rgb.shape[:2]
    cy0, cy1 = int(h * 0.33), int(h * 0.67)
    cx0, cx1 = int(w * 0.33), int(w * 0.67)
    center = rgb[cy0:cy1, cx0:cx1]
    gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)

    thr = np.percentile(gray, 20)
    dark_frac = float((gray <= thr).mean())
    return dark_frac < 0.08


class MelanomaPredictor:
    def __init__(self, model_path: str, img_size: Tuple[int, int] = (224, 224), threshold: float = 0.5):
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.img_size = tuple(img_size)
        self.threshold = float(threshold)

        self.apply_white_balance = True
        self.run_quality_checks = True
        self.strict_quality = False
        self.enable_photo_gate = True
        self.min_photo_score = 0.20
        self.enable_skin_gate = True
        self.min_skin_ratio = 0.12

        self.enable_color_gate = True
        self.max_blue_green_frac = 0.15
        print("¡Modelo cargado!")
        print(f"  Modelo: {model_path}")
        print(f"  Tamaño de entrada: {self.img_size}")
        print(f"  Umbral: {self.threshold}")

    def preprocess_image(
        self,
        image_path: str,
        preserve_aspect: bool = False,
        apply_white_balance: bool = True,
        run_quality_checks: bool = True,
        strict_quality: bool = False,
    ) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

        target_h, target_w = self.img_size

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


        if getattr(self, "enable_photo_gate", True):
            ok_photo, info = _photographic_gate(rgb, getattr(self, "min_photo_score", 0.20))
            if not ok_photo:
                raise ValueError(
                    f"No es piel o mala calidad de imagen (photo_score={info['photo_score']:.3f}, "
                    f"min={info['min_photo_score']:.3f}, entropy={info['entropy']:.3f}, "
                    f"texture={info['texture']:.3f}, edge_frac={info['edge_frac']:.3f})."
                )
        
        if getattr(self, "enable_skin_gate", True):
            ok_skin, skin_r = _is_likely_skin_photo(rgb, getattr(self, "min_skin_ratio", 0.20))
            if not ok_skin:
                raise ValueError(f"No es piel o mala calidad de imagen (skin_ratio={skin_r:.1%}).")

        if getattr(self, "enable_color_gate", True):
            ok_col, info = _color_domain_check(rgb, getattr(self, "max_blue_green_frac", 0.15))
            if not ok_col:
                raise ValueError(
                    f"No es piel o mala calidad de imagen "
                    f"(blue_green_frac={info['blue_green_frac']:.1%}, max={info['max_blue_green_frac']:.1%})."
                )

        if run_quality_checks:
            warnings = _image_quality_checks(rgb)
            if _lesion_too_small_heuristic(rgb):
                warnings.append("La lesión parece muy pequeña en el encuadre (acércate/recorta).")
            if warnings:
                msg = " | ".join(warnings)
                if strict_quality:
                    raise ValueError(f"QA falló: {msg}")
                else:
                    print(f"[AVISO QA] {os.path.basename(image_path)}: {msg}", file=sys.stderr)

        if preserve_aspect:
            h, w = rgb.shape[:2]
            aspect = (w / h) if h else 1.0
            if w > h:
                new_w = target_w
                new_h = int(round(target_w / aspect))
            else:
                new_h = target_h
                new_w = int(round(target_h * aspect))

            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_off = (target_h - new_h) // 2
            x_off = (target_w - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            img = canvas

            if apply_white_balance:
                resized_wb = _gray_world_white_balance(resized)
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized_wb
                img = canvas
        else:
            img = _center_square_crop(rgb)
            if apply_white_balance:
                img = _gray_world_white_balance(img)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        x = img.astype(np.float32)

        x = preprocess_input(x)  
        x = np.expand_dims(x, axis=0)
        return x

    def predict(self, image_path: str, preserve_aspect: bool = False) -> Dict[str, Any]:
        try:
            x = self.preprocess_image(
                image_path,
                preserve_aspect=preserve_aspect,
                apply_white_balance=self.apply_white_balance,
                run_quality_checks=self.run_quality_checks,
                strict_quality=self.strict_quality,
            )
            y = self.model.predict(x, verbose=0)

            prob = float(np.ravel(y)[0]) 
            label = "MALIGNO" if prob >= self.threshold else "BENIGNO"
            return {
                "image": image_path,
                "probabilidad": prob,
                "prediccion": label,
                "porcentaje": f"{prob:.2%}",
            }
        except Exception as e:
            return {"image": image_path, "error": str(e)}

    def predict_tuple(self, image_path: str, preserve_aspect: bool = False) -> Tuple[float, str]:
        r = self.predict(image_path, preserve_aspect=preserve_aspect)
        if "error" in r:
            return float("nan"), "ERROR"
        return float(r["probabilidad"]), str(r["prediccion"])

    def batch_predict(self, image_paths: List[str], preserve_aspect: bool = False) -> List[Dict[str, Any]]:
        """Predicción por lotes. Devuelve lista de dicts (mismo formato que predict())."""
        if not image_paths:
            return []

        batch = []
        valid = []
        out: List[Dict[str, Any]] = []

        for p in image_paths:
            try:
                batch.append(self.preprocess_image(
                    p,
                    preserve_aspect=preserve_aspect,
                    apply_white_balance=self.apply_white_balance,
                    run_quality_checks=self.run_quality_checks,
                    strict_quality=self.strict_quality,
                ))
                valid.append(p)
            except Exception as e:
                out.append({"image": p, "error": str(e)})

        if batch:
            X = np.concatenate(batch, axis=0)
            preds = np.ravel(self.model.predict(X, verbose=0))

            for p, prob in zip(valid, preds):
                prob = float(prob)
                label = "MALIGNO" if prob >= self.threshold else "BENIGNO"
                out.append({
                    "image": p,
                    "probabilidad": prob,
                    "prediccion": label,
                    "porcentaje": f"{prob:.2%}",
                })

        return out

def _cli():
    parser = argparse.ArgumentParser(description="Inferencia para modelo ISIC2019 (MobileNetV2, .keras)")
    parser.add_argument("--model", default="isic2019_mobilenetv2_best.keras", help="Ruta al modelo .keras")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para MALIGNO (default: 0.5)")
    parser.add_argument("--preserve_aspect", action="store_true", help="Mantener aspecto con padding negro (menos recomendable)")
    parser.add_argument("--image", help="Ruta a una imagen para predecir")
    parser.add_argument("--batch", nargs="*", help="Lista de rutas de imágenes para batch")

    parser.add_argument("--no_wb", action="store_true", help="Desactivar balance de blancos (white balance)")
    parser.add_argument("--no_qa", action="store_true", help="Desactivar chequeos de calidad (QA)")
    parser.add_argument("--strict_qa", action="store_true", help="Si QA falla, abortar en vez de avisar")
    parser.add_argument("--no_photo_gate", action="store_true", help="Desactivar gate fotográfico (capturas/dibujos)")
    parser.add_argument("--min_photo_score", type=float, default=0.20, help="Umbral de foto real (default: 0.20)")


    parser.add_argument("--no_skin_gate", action="store_true", help="Desactivar filtro rápido de 'parece piel'")
    parser.add_argument("--min_skin_ratio", type=float, default=0.12, help="Umbral skin ratio (default: 0.12)")

    parser.add_argument("--no_color_gate", action="store_true", help="Desactivar filtro de dominio de color (azul/verde saturado)")
    parser.add_argument("--max_blue_green_frac", type=float, default=0.15, help="Máx. fracción de azul/verde saturado (default: 0.15)")

    args = parser.parse_args()

    predictor = MelanomaPredictor(model_path=args.model, img_size=(224, 224), threshold=args.threshold)
    predictor.apply_white_balance = not args.no_wb
    predictor.run_quality_checks = not args.no_qa
    predictor.strict_quality = bool(args.strict_qa)

    predictor.enable_photo_gate = not args.no_photo_gate
    predictor.min_photo_score = float(args.min_photo_score)

    predictor.enable_skin_gate = not args.no_skin_gate
    predictor.min_skin_ratio = float(args.min_skin_ratio)

    predictor.enable_color_gate = not args.no_color_gate
    predictor.max_blue_green_frac = float(args.max_blue_green_frac)

    if args.image:
        r = predictor.predict(args.image, preserve_aspect=args.preserve_aspect)
        if "error" in r:
            print(f"ERROR: {r['error']}", file=sys.stderr)
            sys.exit(2)

        print("\nRESULTADO")
        print(f"  Imagen: {os.path.basename(r['image'])}")
        print(f"  Prob. maligno: {r['porcentaje']}")
        print(f"  Predicción: {r['prediccion']}")
        print(f"  Umbral: {predictor.threshold}")
        return

    if args.batch:
        results = predictor.batch_predict(args.batch, preserve_aspect=args.preserve_aspect)
        print(f"\nRESULTADOS ({len(results)} imágenes)")
        malignas = 0
        ok = 0
        for r in results:
            img = os.path.basename(r.get("image", ""))
            if "error" in r:
                print(f"{img}: ERROR - {r['error']}")
                continue
            print(f"{img}: {r['prediccion']} ({r['porcentaje']})")
            malignas += int(r["prediccion"] == "MALIGNO")
            ok += 1
        if ok:
            print(f"\n  Total OK: {ok} | Benignas: {ok - malignas} | Malignas: {malignas} | Tasa malignidad: {malignas/ok:.1%}")
        return

    while True:
        print("\n1. Predecir una imagen")
        print("2. Predecir múltiples imágenes (batch)")
        print("3. Salir")
        op = input("\nSelecciona una opción (1-3): ").strip()

        if op == "1":
            p = input("Ruta de la imagen: ").strip().strip('"')
            if not os.path.exists(p):
                print("Error: la imagen no existe.")
                continue
            r = predictor.predict(p, preserve_aspect=args.preserve_aspect)
            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue
            print(f"\n  Prob. maligno: {r['porcentaje']}")
            print(f"  Predicción: {r['prediccion']}")
            print(f"  Umbral: {predictor.threshold}")
            print("  Nota: salida orientativa; no es diagnóstico.")

        elif op == "2":
            print("Introduce rutas (una por línea). Escribe 'fin' para terminar.")
            paths = []
            while True:
                p = input("Ruta: ").strip().strip('"')
                if p.lower() == "fin":
                    break
                if os.path.exists(p):
                    paths.append(p)
                else:
                    print("No existe:", p)

            results = predictor.batch_predict(paths, preserve_aspect=args.preserve_aspect)
            print(f"\nRESULTADOS ({len(results)} imágenes)")
            for r in results:
                img = os.path.basename(r.get("image", ""))
                if "error" in r:
                    print(f"{img}: ERROR - {r['error']}")
                else:
                    print(f"{img}: {r['prediccion']} ({r['porcentaje']})")

        elif op == "3":
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    _cli()
