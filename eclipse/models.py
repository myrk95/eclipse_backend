from django.db import models
from django.contrib.auth.models import AbstractUser

# Clase Usuari
class Usuari(AbstractUser):
    email = models.EmailField(unique=True)
    data_registre = models.DateTimeField(auto_now_add=True)
    # Ya hereda username y password de AbstractUser

# Clase Lunar
class Lunar(models.Model):
    usuari = models.ForeignKey(Usuari, on_delete=models.CASCADE, related_name="lunars")
    imatge = models.ImageField(upload_to="lunars/")
    data_pujada = models.DateTimeField(auto_now_add=True)

# Clase ResultatAnalisi
class ResultatAnalisi(models.Model):
    lunar = models.ForeignKey(Lunar, on_delete=models.CASCADE, related_name="resultats")
    tipus = models.CharField(max_length=100)
    probabilitat = models.FloatField()
    descripcio = models.TextField()

# Clase Historial
class Historial(models.Model):
    usuari = models.ForeignKey(Usuari, on_delete=models.CASCADE, related_name="historials")
    lunar = models.ForeignKey(Lunar, on_delete=models.CASCADE, related_name="historials")
    data = models.DateTimeField(auto_now_add=True)

# Clase Configuracio
class Configuracio(models.Model):
    usuari = models.OneToOneField(Usuari, on_delete=models.CASCADE, related_name="configuracio")
    notificacions = models.BooleanField(default=True)
    tema = models.CharField(max_length=50, default="clar")
    privacitat = models.CharField(max_length=50, default="privat")

# Clase Suport
class Suport(models.Model):
    usuari = models.ForeignKey(Usuari, on_delete=models.CASCADE, related_name="suports")
    missatge = models.TextField()
    data = models.DateTimeField(auto_now_add=True)
    estat = models.CharField(max_length=50, default="pendent")
