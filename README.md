# Eclipse Project

Proyecto Django para gestión y análisis de lunares, con backend en Django y app principal `eclipse`.

---

## 🛠 Estructura del proyecto



eclipse_project/ # Carpeta raíz del proyecto
├── manage.py # Archivo principal de Django
├── eclipse_project/ # Configuración del proyecto (settings, urls, wsgi)
└── eclipse/ # App principal con modelos, vistas y lógica de lunares


---

## ⚡ Tecnologías usadas

- Backend: Django 5.2.8  
- Base de datos: SQLite (por defecto, se puede cambiar a PostgreSQL/MySQL)  
- Frontend: React (opcional, carpeta aparte `lunar_frontend`)  
- Control de versiones: Git + GitHub

---

## 🚀 Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/myrk95/eclipse_project.git
cd eclipse_project


Crear un entorno virtual:

python3 -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows


Instalar dependencias:

pip install django
# Opcional:
pip install djangorestframework django-cors-headers


Ejecutar migraciones:

python manage.py makemigrations
python manage.py migrate


Crear superusuario (opcional):

python manage.py createsuperuser


Levantar el servidor:

python manage.py runserver


Accede en tu navegador: http://127.0.0.1:8000/

📝 Uso

La app eclipse gestiona usuarios y lunares.

Puedes añadir modelos de análisis, vistas y endpoints de API según se avance en el proyecto.

Frontend (React) se conectará a la API para mostrar la información.

🌿 Buenas prácticas de Git

Crear ramas por funcionalidad:

git checkout -b 0-1-backend


Hacer commits claros:

git add .
git commit -m "Añadidos modelos de lunares y vistas iniciales"


Subir ramas al remoto:

git push -u origin 0-1-backend


Siempre trabajar en ramas separadas y luego hacer Pull Requests para integrarlas a main.

👥 Colaboradores

myrk95 (propietario del proyecto)

Otros colaboradores añadidos desde GitHub
