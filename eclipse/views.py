import random
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate, login
from django.contrib.auth.hashers import make_password
from inferencia.inferencia import MelanomaPredictor  # <-- el teu model IA

User = get_user_model()
predictor = MelanomaPredictor()  # Instància global per a crides ràpides

# -----------------------------
# Login
# -----------------------------
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
    return Response({"status": "ok", "user_id": user.id, "email": user.email})


# -----------------------------
# Registro
# -----------------------------
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
# Dashboard
# -----------------------------
@api_view(['GET'])
def dashboard_view(request):
    return Response({
        "status": "ok",
        "mensaje": "Bienvenido al dashboard",
        "ultimos_resultados": [
            {"id": 1, "resultado": "negativo"},
            {"id": 2, "resultado": "positivo"},
        ]
    })


# -----------------------------
# Upload image (simulado)
# -----------------------------
@api_view(['POST'])
def upload_image(request):
    image_path = request.data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return Response({"error": f"No se encontró la imagen: {image_path}"}, status=400)
    
    image_id = random.randint(1000, 9999)
    return Response({
        "status": "ok",
        "image_id": image_id,
        "image_path": image_path
    })


# -----------------------------
# Analysis result
# -----------------------------
# -----------------------------
# Analysis result (form-data)
# -----------------------------
@api_view(['POST'])
def analysis_result(request):
    # Obtiene la imagen desde form-data
    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"error": "No se subió imagen"}, status=400)

    # Guardar temporalmente la imagen
    temp_path = f"/tmp/{image_file.name}"
    with open(temp_path, "wb+") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    # Hacer predicción
    try:
        probabilidad, prediccion = predictor.predict(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return Response({"error": str(e)}, status=500)

    # Eliminar archivo temporal
    os.remove(temp_path)

    # Devolver resultado
    return Response({
        "status": "ok",
        "resultado": prediccion,
        "probabilidad": f"{probabilidad:.2%}" if probabilidad is not None else None
    })


# -----------------------------
# Historial
# -----------------------------
@api_view(['GET'])
def history_view(request):
    historial = [
        {"id": 1, "resultado": "negativo"},
        {"id": 2, "resultado": "positivo"},
        {"id": 3, "resultado": "negativo"},
    ]
    return Response({"status": "ok", "historial": historial})


# -----------------------------
# Perfil
# -----------------------------
@api_view(['GET', 'PUT'])
def profile_view(request):
    if request.method == 'GET':
        return Response({
            "status": "ok",
            "usuario": {
                "username": "usuario_demo",
                "email": "demo@test.com"
            }
        })
    elif request.method == 'PUT':
        username = request.data.get("username", "usuario_demo")
        return Response({
            "status": "ok",
            "usuario_actualizado": {
                "username": username,
                "email": "demo@test.com"
            }
        })


# -----------------------------
# Configuración
# -----------------------------
@api_view(['GET', 'PUT'])
def settings_view(request):
    if request.method == 'GET':
        return Response({
            "status": "ok",
            "settings": {
                "notificaciones": True,
                "tema": "claro"
            }
        })
    elif request.method == 'PUT':
        notificaciones = request.data.get("notificaciones", True)
        tema = request.data.get("tema", "claro")
        return Response({
            "status": "ok",
            "settings_actualizados": {
                "notificaciones": notificaciones,
                "tema": tema
            }
        })


# -----------------------------
# Soporte
# -----------------------------
@api_view(['POST'])
def support_view(request):
    mensaje = request.data.get("mensaje", "")
    return Response({
        "status": "ok",
        "mensaje_recibido": mensaje
    })
