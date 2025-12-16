import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate, login
from django.contrib.auth.hashers import make_password

from inferencia.inferencia import MelanomaPredictor
from .models import Usuari, Lunar, ResultatAnalisi, Historial

User = get_user_model()
predictor = MelanomaPredictor()

# -----------------------------
# Login
# -----------------------------
@api_view(['POST'])
def login_view(request):
    email = request.data.get("email")
    password = request.data.get("password")

    if not email or not password:
        return Response({"error": "Email y contraseña requeridos"}, status=400)

    try:
        user_obj = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({"error": "Credenciales inválidas"}, status=400)

    user = authenticate(username=user_obj.username, password=password)
    if not user:
        return Response({"error": "Credenciales inválidas"}, status=400)

    login(request, user)
    return Response({
        "status": "ok",
        "user_id": user.id,
        "email": user.email
    })


# -----------------------------
# Registro
# -----------------------------
@api_view(['POST'])
def register_view(request):
    email = request.data.get("email")
    password = request.data.get("password")
    username = request.data.get("username", email.split("@")[0])

    if not email or not password:
        return Response({"error": "Email y contraseña requeridos"}, status=400)

    if User.objects.filter(email=email).exists():
        return Response({"error": "Usuario ya existe"}, status=400)

    user = User.objects.create(
        username=username,
        email=email,
        password=make_password(password)
    )

    return Response({
        "status": "ok",
        "user_id": user.id,
        "email": user.email
    })


# -----------------------------
# Analysis result (FORM-DATA)
# -----------------------------
@api_view(['POST'])
def analysis_result(request):
    user_id = request.data.get("user_id")
    image_file = request.FILES.get("image")

    if not user_id:
        return Response({"error": "user_id requerido"}, status=400)

    if not image_file:
        return Response({"error": "No se subió imagen"}, status=400)

    try:
        usuari = Usuari.objects.get(id=user_id)
    except Usuari.DoesNotExist:
        return Response({"error": "Usuario no existe"}, status=404)

    # Guardar imagen real
    lunar = Lunar.objects.create(
        usuari=usuari,
        imatge=image_file
    )

    # Archivo temporal para IA
    temp_path = f"/tmp/{image_file.name}"
    with open(temp_path, "wb+") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    try:
        probabilitat, prediccio = predictor.predict(temp_path)
    except Exception as e:
        os.remove(temp_path)
        lunar.delete()
        return Response({"error": str(e)}, status=500)

    os.remove(temp_path)

    resultat = ResultatAnalisi.objects.create(
        lunar=lunar,
        tipus=prediccio,
        probabilitat=probabilitat,
        descripcio="Análisis automático IA"
    )

    Historial.objects.create(
        usuari=usuari,
        lunar=lunar
    )

    return Response({
        "status": "ok",
        "user_id": usuari.id,
        "lunar_id": lunar.id,
        "resultado_id": resultat.id,
        "tipo": prediccio,
        "probabilidad": f"{probabilitat:.2%}"
    })


# -----------------------------
# Historial por USER_ID
# -----------------------------
@api_view(['GET'])
def history_view(request):
    user_id = request.GET.get("user_id")

    if not user_id:
        return Response({"error": "user_id requerido"}, status=400)

    historials = Historial.objects.filter(
        usuari_id=user_id
    ).select_related("lunar")

    data = []
    for h in historials:
        data.append({
            "lunar_id": h.lunar.id,
            "imagen": h.lunar.imatge.url,
            "fecha": h.data
        })

    return Response({
        "status": "ok",
        "user_id": user_id,
        "historial": data
    })
