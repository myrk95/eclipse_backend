from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Health check (para Render)
    path('health/', views.profile_view, name='health'),

    # Autenticación
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),

    # Dashboard
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # Subir imagen
    path('upload_image/', views.upload_image, name='upload_image'),

    # Resultado de análisis con IA
    path('analysis_result/', views.analysis_result, name='analysis_result'),

    # Historial
    path('history/', views.history_view, name='history'),

    # Perfil
    path('profile/', views.profile_view, name='profile'),

    # Configuración
    path('settings/', views.settings_view, name='settings'),

    # Soporte
    path('support/', views.support_view, name='support'),
]
