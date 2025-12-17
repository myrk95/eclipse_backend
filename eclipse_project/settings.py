from pathlib import Path
import os
# RUNNEAR python manage.py runserver 0.0.0.0:8000
# -----------------------------
# Base del proyecto
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-fallback-key'
DEBUG = True

# Permitir conexiones desde cualquier host en desarrollo
ALLOWED_HOSTS = ['*']

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
    'corsheaders.middleware.CorsMiddleware',  # debe ir primero
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
        'ENGINE': 'django.db.backends.sqlite3',
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
MEDIA_ROOT = BASE_DIR / 'media'  # aquí se guardarán las imágenes localmente

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
CORS_ALLOW_ALL_ORIGINS = True  # para desarrollo
CORS_ALLOW_HEADERS = ["*"]
CORS_ALLOW_METHODS = ["GET","POST","PUT","PATCH","DELETE","OPTIONS"]

# -----------------------------
# CSRF
# -----------------------------
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://192.168.1.82:8081",  # tu IP local y puerto metro bundler
]

# -----------------------------
# TensorFlow: ruta del modelo
# -----------------------------
MODEL_PATH = BASE_DIR / 'inferencia/isic2019_mobilenetv2_best.keras'
