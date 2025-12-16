import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate, login
from django.contrib.auth.hashers import make_password
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from inferencia.inferencia import MelanomaPredictor
from .models import Lunar, ResultatAnalisi, Historial, Configuracio, Suport

User = get_user_model()
predictor = MelanomaPredictor()  # Instancia global del modelo IA

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
@api_view(['GET'])
def dashboard_view(request):
    ultimos_resultados = []

    # Traer últimos 5 análisis del usuario
    lunars = Lunar.objects.filter(usuari=request.user).order_by('-data_pujada')[:5]
    for lunar in lunars:
        result = lunar.resultats.last()
        ultimos_resultados.append({
            "lunar_id": lunar.id,
            "resultado": result.tipus if result else None,
            "probabilidad": f"{result.probabilitat:.2%}" if result else None,
            "imagen_url": request.build_absolute_uri(lunar.imatge.url)
        })

    return Response({
        "status": "ok",
        "mensaje": f"Bienvenido {request.user.username}",
        "ultimos_resultados": ultimos_resultados
    })


# -----------------------------
# SUBIR IMAGEN
# -----------------------------
@csrf_exempt
@api_view(['POST'])
def upload_image(request):
    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"error": "No se subió imagen"}, status=400)

    lunar = Lunar.objects.create(
        usuari=request.user,
        imatge=image_file
    )

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
    image_file = request.FILES.get("image")
    if not image_file:
        return Response({"error": "No se subió imagen"}, status=400)

    # Guardar imagen en Lunar
    lunar = Lunar.objects.create(usuari=request.user, imatge=image_file)

    # Guardar temporal para IA
    temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', image_file.name)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb+") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    # Predicción IA
    try:
        probabilidad, prediccion = predictor.predict(temp_path)
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
    Historial.objects.create(
        usuari=request.user,
        lunar=lunar
    )

    # Eliminar archivo temporal
    os.remove(temp_path)

    return Response({
        "status": "ok",
        "lunar_id": lunar.id,
        "resultado": prediccion,
        "probabilidad": f"{probabilidad:.2%}" if probabilidad is not None else None,
        "imagen_url": request.build_absolute_uri(lunar.imatge.url)
    })


# -----------------------------
# HISTORIAL
# -----------------------------
@csrf_exempt
@api_view(['GET'])
def history_view(request):
    historial_list = []
    historial = Historial.objects.filter(usuari=request.user).order_by('-data')
    
    for h in historial:
        result = h.lunar.resultats.last()
        historial_list.append({
            "lunar_id": h.lunar.id,
            "name": h.lunar.name,
            "descripcion": h.lunar.descripcio,
            "imagen_url": request.build_absolute_uri(h.lunar.imatge.url),
            "porcentaje": f"{h.lunar.porcentaje:.2%}" if h.lunar.porcentaje is not None else None,
            "resultado": result.tipus if result else None,  # maligno o benigno
            "probabilidad": f"{result.probabilitat:.2%}" if result else None,
            "fecha": h.data
        })

    return Response({"status": "ok", "historial": historial_list})


# -----------------------------
# PERFIL
# -----------------------------
@csrf_exempt
@api_view(['GET', 'PUT'])
def profile_view(request):
    if request.method == 'GET':
        return Response({
            "status": "ok",
            "usuario": {
                "username": request.user.username,
                "email": request.user.email
            }
        })
    elif request.method == 'PUT':
        username = request.data.get("username", request.user.username)
        request.user.username = username
        request.user.save()
        return Response({
            "status": "ok",
            "usuario_actualizado": {
                "username": request.user.username,
                "email": request.user.email
            }
        })


# -----------------------------
# CONFIGURACION
# -----------------------------
@csrf_exempt
@api_view(['GET', 'PUT'])
def settings_view(request):
    try:
        config = request.user.configuracio
    except Configuracio.DoesNotExist:
        config = Configuracio.objects.create(usuari=request.user)

    if request.method == 'GET':
        return Response({
            "status": "ok",
            "settings": {
                "notificaciones": config.notificacions,
                "tema": config.tema,
                "privacitat": config.privacitat
            }
        })
    elif request.method == 'PUT':
        config.notificacions = request.data.get("notificaciones", config.notificacions)
        config.tema = request.data.get("tema", config.tema)
        config.privacitat = request.data.get("privacitat", config.privacitat)
        config.save()
        return Response({
            "status": "ok",
            "settings_actualizados": {
                "notificaciones": config.notificacions,
                "tema": config.tema,
                "privacitat": config.privacitat
            }
        })


# -----------------------------
# SOPORTE
# -----------------------------
@csrf_exempt
@api_view(['POST'])
def support_view(request):
    mensaje = request.data.get("mensaje", "")
    s = Suport.objects.create(
        usuari=request.user,
        missatge=mensaje
    )
    return Response({
        "status": "ok",
        "mensaje_recibido": s.missatge,
        "fecha": s.data,
        "estado": s.estat
    })
