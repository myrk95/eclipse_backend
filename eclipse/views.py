import os
import uuid
import traceback
from django.conf import settings
from django.contrib.auth import get_user_model, authenticate
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Lunar, ResultatAnalisi, Historial

User = get_user_model()

# --------------------------------------------------
# Lazy loading del predictor IA
# --------------------------------------------------
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "inferencia",
    "isic2019_mobilenetv2_best.keras"
)

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from inferencia.inferencia import MelanomaPredictor
        _predictor = MelanomaPredictor(model_path=MODEL_PATH)
    return _predictor

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
@api_view(['POST'])
def login_view(request):
    try:
        email = request.data.get("email")
        password = request.data.get("password")
        if not email or not password:
            return Response({"error": "Email y contraseña requeridos"}, status=400)

        try:
            user_obj = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "Credenciales inválidas"}, status=400)

        user = authenticate(username=user_obj.username, password=password)
        if user is None:
            return Response({"error": "Credenciales inválidas"}, status=400)

        return Response({
            "status": "ok",
            "user_id": user.id,
            "username": user.username,
            "email": user.email
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en login_view", "detalle": str(e)}, status=500)


# --------------------------------------------------
# REGISTRO
# --------------------------------------------------
@api_view(['POST'])
def register_view(request):
    try:
        email = request.data.get("email")
        password = request.data.get("password")
        username = request.data.get("username", email.split("@")[0])
        if not email or not password:
            return Response({"error": "Email y contraseña requeridos"}, status=400)
        if User.objects.filter(email=email).exists():
            return Response({"error": "Usuario ya existe"}, status=400)

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        return Response({
            "status": "ok",
            "user_id": user.id,
            "email": user.email
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en register_view", "detalle": str(e)}, status=500)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
@api_view(['GET'])
def dashboard_view(request):
    try:
        user_id = request.query_params.get("user_id")
        if not user_id:
            return Response({"error": "user_id requerido"}, status=401)
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "Usuario inválido"}, status=401)

        ultimos_resultados = []
        lunars = Lunar.objects.filter(usuari=user).order_by("-data_pujada")[:5]
        for lunar in lunars:
            result = lunar.resultats.last()
            prob_str = f"{result.probabilitat:.2%}" if result and hasattr(result, 'probabilitat') else None
            ultimos_resultados.append({
                "lunar_id": lunar.id,
                "resultado": result.tipus if result else None,
                "probabilidad": prob_str,
                "imagen_url": request.build_absolute_uri(lunar.imatge.url)
            })

        return Response({
            "status": "ok",
            "mensaje": f"Bienvenido {user.username}",
            "ultimos_resultados": ultimos_resultados
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en dashboard_view", "detalle": str(e)}, status=500)


# --------------------------------------------------
# SUBIR IMAGEN
# --------------------------------------------------
@api_view(['POST'])
def upload_image(request):
    try:
        user_id = request.data.get("user_id")
        image_file = request.FILES.get("image")
        nom = request.data.get("nom") or "Lunar sin nombre"
        descripcio = request.data.get("descripcio") or ""

        if not user_id or not image_file:
            return Response({"error": "Datos incompletos"}, status=400)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "Usuario inválido"}, status=401)

        lunar = Lunar.objects.create(
            usuari=user,
            imatge=image_file,
            name=nom,
            descripcio=descripcio
        )

        imagen_url = None
        try:
            imagen_url = request.build_absolute_uri(lunar.imatge.url)
        except Exception:
            pass

        return Response({
            "status": "ok",
            "lunar_id": lunar.id,
            "imagen_url": imagen_url,
            "nombre": lunar.name,
            "descripcion": lunar.descripcio
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en upload_image", "detalle": str(e)}, status=500)


# --------------------------------------------------
# ANALYSIS RESULT
# --------------------------------------------------
@api_view(['POST'])
def analysis_result(request):
    try:
        user_id = request.data.get("user_id")
        lunar_id = request.data.get("lunar_id")
        image_file = request.FILES.get("image")

        if not user_id:
            return Response({"error": "user_id requerido"}, status=400)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "Usuario inválido"}, status=401)

        temp_path = None
        temp_file_created = False

        try:
            if lunar_id:
                try:
                    lunar = Lunar.objects.get(id=lunar_id, usuari=user)
                except Lunar.DoesNotExist:
                    return Response({"error": "Lunar inválido"}, status=400)
                temp_path = lunar.imatge.path
            elif image_file:
                lunar = Lunar.objects.create(usuari=user, imatge=image_file)
                temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_filename = f"{uuid.uuid4()}.jpg"
                temp_path = os.path.join(temp_dir, temp_filename)
                with open(temp_path, "wb+") as f:
                    for chunk in image_file.chunks():
                        f.write(chunk)
                temp_file_created = True
            else:
                return Response({"error": "Se requiere lunar_id o imagen"}, status=400)
        except Exception as e:
            return Response({"error": f"No se pudo procesar la imagen: {str(e)}"}, status=500)

        # Predicción con control de errores
        probabilidad = None
        prediccion = None
        warnings = []

        try:
            predictor = get_predictor()
            try:
                resultado = predictor.predict(temp_path)
            except Exception as e:
                return Response({"error": f"Error al ejecutar la IA: {str(e)}"}, status=500)

            if "error" in resultado:
                return Response({"error": resultado["error"]}, status=400)

            probabilidad = resultado.get("probabilidad")
            prediccion = resultado.get("prediccion")
            warnings = resultado.get("warnings", [])

            if probabilidad is None or prediccion is None:
                return Response({"error": "Resultado inválido del predictor"}, status=500)

        except Exception as e:
            return Response({"error": f"Error durante la inicialización de la IA: {str(e)}"}, status=500)

        finally:
            if temp_file_created and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        # Guardar resultado y historial
        try:
            ResultatAnalisi.objects.create(
                lunar=lunar,
                tipus=prediccion,
                probabilitat=probabilidad,
                descripcio=f"Resultado del análisis: {prediccion}"
            )
            Historial.objects.create(usuari=user, lunar=lunar)
        except Exception as e:
            return Response({"error": f"No se pudo guardar el resultado/historial: {str(e)}"}, status=500)

        imagen_url = None
        try:
            imagen_url = request.build_absolute_uri(lunar.imatge.url)
        except Exception:
            pass

        return Response({
            "status": "ok",
            "lunar_id": lunar.id,
            "resultado": prediccion,
            "probabilidad": f"{probabilidad:.2%}" if probabilidad is not None else None,
            "imagen_url": imagen_url,
            "warnings": warnings
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en analysis_result", "detalle": str(e)}, status=500)


# --------------------------------------------------
# HISTORIAL
# --------------------------------------------------
@api_view(['GET'])
def history_view(request):
    try:
        user_id = request.query_params.get("user_id")
        if not user_id:
            return Response({"error": "user_id requerido"}, status=401)

        # Obtener el usuario
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "Usuario inválido"}, status=401)

        historial = []
        # Obtener los registros del historial del usuario, ordenados por fecha descendente
        registros = Historial.objects.filter(usuari=user).order_by("-data").select_related('lunar')

        for h in registros:
            lunar = h.lunar
            # Último resultado del lunar
            result = lunar.resultats.last()
            prob_str = f"{result.probabilitat:.2%}" if result else None

            historial.append({
                "lunar_id": lunar.id,
                "nombre": lunar.name,
                "descripcion": lunar.descripcio,
                "imagen": request.build_absolute_uri(lunar.imatge.url) if lunar.imatge else None,
                "resultado": result.tipus if result else None,
                "descripcion_resultado": result.descripcio if result else None,
                "probabilidad": prob_str,
                "fecha": h.data
            })

        return Response({"status": "ok", "historial": historial})

    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en history_view", "detalle": str(e)}, status=500)



# --------------------------------------------------
# PROFILE
# --------------------------------------------------
@api_view(['GET'])
def profile_view(request):
    try:
        user_id = request.query_params.get("user_id")
        if not user_id:
            return Response({"error": "user_id requerido"}, status=401)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "Usuario inválido"}, status=401)

        return Response({
            "status": "ok",
            "profile": {
                "username": user.username,
                "email": user.email
            }
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en profile_view", "detalle": str(e)}, status=500)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
@api_view(['GET', 'POST'])
def settings_view(request):
    try:
        return Response({"status": "ok"})
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en settings_view", "detalle": str(e)}, status=500)


# --------------------------------------------------
# SUPPORT
# --------------------------------------------------
@api_view(['GET', 'POST'])
def support_view(request):
    try:
        return Response({"status": "ok"})
    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Ocurrió un error en support_view", "detalle": str(e)}, status=500)
