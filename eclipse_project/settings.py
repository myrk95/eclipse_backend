from pathlib import Path
import os

# -----------------------------
# Base del proyecto
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-fallback-key')
DEBUG = os.environ.get('DJANGO_DEBUG', 'False') == 'True'

ALLOWED_HOSTS = [
    'eclipse-backend-m8zi.onrender.com',  # producción Render
    'localhost',                           # desarrollo
    '127.0.0.1',                           # desarrollo
]

# -----------------------------
# Aplicaciones
# -----------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',
    'corsheaders',

    'eclipse',  # tu app principal
]

# -----------------------------
# Middleware
# -----------------------------
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Debe ir primero
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'eclipse_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'eclipse_project.wsgi.application'

# -----------------------------
# Base de datos
# -----------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # Para producción recomiendo PostgreSQL
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# -----------------------------
# Validación de contraseñas
# -----------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

# -----------------------------
# Internacionalización
# -----------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# -----------------------------
# Archivos estáticos y media
# -----------------------------
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# -----------------------------
# Usuario personalizado
# -----------------------------
AUTH_USER_MODEL = 'eclipse.Usuari'

# -----------------------------
# Default auto field
# -----------------------------
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# -----------------------------
# REST Framework
# -----------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
}

# -----------------------------
# CORS
# -----------------------------
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # React web dev
    "http://localhost:5173",  # Vite dev
    "https://tu-frontend-produccion.com",  # React web prod
]
CORS_ALLOW_HEADERS = ["*"]
CORS_ALLOW_METHODS = ["GET","POST","PUT","PATCH","DELETE","OPTIONS"]

# -----------------------------
# CSRF
# -----------------------------
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "https://eclipse-backend-m8zi.onrender.com",
]

# -----------------------------
# TensorFlow: carga perezosa para no matar workers
# -----------------------------
# Ejemplo de uso en tu código:
# from inferencia.inferencia import MelanomaPredictor
# predictor = MelanomaPredictor(MODEL_PATH)
#
# En inferencia/inferencia.py:
# class MelanomaPredictor:
#     _model = None
#     def __init__(self, model_path):
#         if MelanomaPredictor._model is None:
#             import tensorflow as tf
#             MelanomaPredictor._model = tf.keras.models.load_model(model_path)
#         self.model = MelanomaPredictor._model

# -----------------------------
# Variables de entorno opcionales
# -----------------------------
MODEL_PATH = os.environ.get(
    'MODEL_PATH', str(BASE_DIR / 'inferencia/isic2019_mobilenetv2_best.keras')
)