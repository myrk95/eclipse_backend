"""
WSGI config for eclipse_project project.
"""

import os
import sys

# Añade la carpeta raíz del proyecto al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eclipse_project.settings')

application = get_wsgi_application()
