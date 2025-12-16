import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate, login
from django.contrib.auth.hashers import make_password
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from inferencia.inferencia import MelanomaPredictor
from .models import Lunar, ResultatAnalisi, Historial, Configuracio, Suport

User = get_user_model()

# -----------------------------
# Inicializar predictor IA con ruta absoluta
# -----------------------------
MODEL_PATH = os.path.join(settings.BASE_DIR, "inferencia", "isic2019_mobilenetv2_best.keras")
predictor = MelanomaPredictor(model_path=MODEL_PATH)

# -----------------------------
# LOGIN
# -----------------------------
@csrf_exempt
@api_view(['POST'])
def login_view(request):
    email = request.data.get('email')
    password = request.data.get('password')

    if not email or not password:
        return Response({"error": "Email y contraseña requeridos"}, status=400)

    try:
        user_obj = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({"error": "Credenciales inválidas"}, status=400)

    user = authenticate(username=user_obj.username, password=password)
    if user is None:
        return Response({"error": "Credenciales inválidas"}, status=400)

    login(request, user)
    return Response({"status": "ok", "user_id": user.id, "username": user.username, "email": user.email})


# -----------------------------
# REGISTRO
# -----------------------------
@csrf_exempt
@api_view(['POST'])
def register_view(request):
    email = request.data.get('email')
    password = request.data.get('password')
    username = request.data.get('username', email.split("@")[0])

    if not email or not password:
        return Response({"error": "Email y contraseña requeridos"}, status=400)

    if User.objects.filter(email=email).exists():
        return Response({"error": "Usuario ya existe"}, status=400)

    user = User.objects.create(
        username=username,
        email=email,
        password=make_password(password)
    )
    return Response({"status": "ok", "user_id": user.id, "email": user.email})


# -----------------------------
# DASHBOARD
# -----------------------------
@csrf_exempt
@api_view(['GET'])
def dashboard_view(request):
    if request.user.is_anonymous:
        return Response({"error": "Usuario no autenticado"}, status=401)

    ultimos_resultados = []
    lunars = Lunar.objects.filter(usuari=request.user).order_by('-data_pujada')[:5]
    for lunar in lunars:
        result = lunar.resultats.last()
        ultimos_resultados.append({
            "lunar_id": lunar.id,
            "resultado": result.tipus if result else None,
            "probabilidad": f"{result.probabilitat:.2%}" if result else None,
            "imagen_url": request.build_absolute_uri(lunar.imatge.url)
        })

    return Response({"status": "ok", "mensaje": f"Bienvenido {request.user.username}", "ultimos_resultados": ultimos_resultados})


# -----------------------------
# SUBIR IMAGEN
# -----------------------------
@csrf_exempt
@api_view(['POST'])
def upload_image(request):
    if request.user.is_anonymous:
        return Response({"error": "Usuario no autenticado"}, status=401)

    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"error": "No se subió imagen"}, status=400)

    lunar = Lunar.objects.create(usuari=request.user, imatge=image_file)

    return Response({
        "status": "ok",
        "lunar_id": lunar.id,
        "imagen_url": request.build_absolute_uri(lunar.imatge.url)
    })


# -----------------------------
# ANALYSIS RESULT
# -----------------------------
@csrf_exempt
@api_view(['POST'])
def analysis_result(request):
    if request.user.is_anonymous:
        return Response({"error": "Usuario no autenticado"}, status=401)

    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"error": "No se subió imagen"}, status=400)

    lunar = Lunar.objects.create(usuari=request.user, imatge=image_file)

    # Guardar temporal para IA
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, image_file.name)
    with open(temp_path, "wb+") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    # Predicción IA
    try:
        resultado = predictor.predict(temp_path)
        if "error" in resultado:
            raise Exception(resultado["error"])
        probabilidad = resultado["probabilidad"]
        prediccion = resultado["prediccion"]
    except Exception as e:
        os.remove(temp_path)
        return Response({"error": str(e)}, status=500)

    # Guardar resultado en DB
    ResultatAnalisi.objects.create(
        lunar=lunar,
        tipus=prediccion,
        probabilitat=probabilidad,
        descripcio=f"Resultado del análisis: {prediccion}"
    )

    # Historial
    Historial.objects.create(usuari=request.user, lunar=lunar)

    os.remove(temp_path)

    return Response({
        "status": "ok",
        "lunar_id": lunar.id,
        "resultado": prediccion,
        "probabilidad": f"{probabilidad:.2%}" if probabilidad is not None else None,
        "imagen_url": request.build_absolute_uri(lunar.imatge.url)
    })
@csrf_exempt
@api_view(['GET'])
def history_view(request):
    return Response({"status": "ok", "historial": []})

@csrf_exempt
@api_view(['GET'])
def profile_view(request):
    return Response({"status": "ok", "profile": {}})

@csrf_exempt
@api_view(['GET', 'POST'])
def settings_view(request):
    return Response({"status": "ok", "settings": {}})

@csrf_exempt
@api_view(['GET', 'POST'])
def support_view(request):
    return Response({"status": "ok", "support": {}})
