from django.urls import path
from . import views

urlpatterns = [
    # Health check
    path('health/', views.dashboard_view, name='health'),  # puedes usar dashboard_view o crear un health específico

    # Autenticación
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),

    # Dashboard / Home
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # Subir imagen
    path('upload_image/', views.upload_image, name='upload_image'),

    # Resultado de análisis
    path('analysis_result/', views.analysis_result, name='analysis_result'),  # ← tu modelo

    # Historial
    path('history/', views.history_view, name='history'),

    # Perfil
    path('profile/', views.profile_view, name='profile'),

    # Configuración
    path('settings/', views.settings_view, name='settings'),

    # Soporte
    path('support/', views.support_view, name='support'),
]
