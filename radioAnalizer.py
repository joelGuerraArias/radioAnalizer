# -*- coding: utf-8 -*-
"""
RadioAnalizer — Análisis de audios en vivo / grabaciones
========================================================
Aplicación independiente: monitorea carpetas de audio, transcribe, detecta términos,
genera clips en audio y notifica (Telegram, correo, webhooks, etc.).

Ejecutar siempre con el Python del entorno virtual del proyecto (venv_new).
"""

# === IMPORTS ESTÁNDAR ===
import os
import sys

# Raíz del proyecto: imports locales y cwd solo desde esta carpeta (evita mezclar otros proyectos en PYTHONPATH)
_RADIO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _RADIO_ROOT not in sys.path:
    sys.path.insert(0, _RADIO_ROOT)
try:
    os.chdir(_RADIO_ROOT)
except OSError:
    pass

# === CONFIGURAR DLLs NVIDIA (cuDNN + cuBLAS) para CUDA antes de cualquier import ===
_nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
for _subpkg in ("cudnn", "cublas", "cuda_runtime"):
    _bin_dir = os.path.join(_nvidia_base, _subpkg, "bin")
    if os.path.isdir(_bin_dir) and _bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_bin_dir)
# CUDA 12.6 (ctranslate2/faster-whisper necesita cublas64_12.dll) - prioridad sobre CUDA 13
_cuda126_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_cuda126_bin) and _cuda126_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _cuda126_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_cuda126_bin)
        except OSError:
            pass
import glob
import subprocess
import json
import time
import re
import base64
import logging
import traceback
import threading
import shutil
import socket
import io
import smtplib
from datetime import datetime, timedelta, date
from pathlib import Path
from urllib.parse import urlparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import html as html_module
import unicodedata

# === IMPORTS DE TERCEROS ===
import requests
import pandas as pd
import streamlit as st
import openai
from mistralai import Mistral
from faster_whisper import WhisperModel
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from supabase import create_client, Client

# Google Gemini AI
from google import genai

# Google Drive imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from googleapiclient.errors import HttpError

# === IMPORTS LOCALES ===
from intro_coincidencia_tts import (
    extraer_info_medio_hora,
    construir_texto_intro_coincidencia,
)
from coincidencias_logger import coincidencias_logger
from coincidencias_logger import (
    log_coincidencia_detectada, log_api_request, log_api_response, log_api_error,
    log_gdrive_upload_start, log_gdrive_upload_success, log_gdrive_upload_error,
    log_error_critico, log_proceso_completado
)

# Importar configuración centralizada
try:
    from config import get_settings
    settings = get_settings()
    USAR_CONFIG_CENTRALIZADA = True
except ImportError:
    USAR_CONFIG_CENTRALIZADA = False
    settings = None

# Importar utilidades
try:
    from utils import con_reintentos, ReintentoExhausto
    USAR_RETRY_DECORATOR = True
except ImportError:
    USAR_RETRY_DECORATOR = False

from google_sheet_index_utils import siguiente_indice_columna_a, titulo_hoja_desde_range_a1

# Importar modelos de datos
try:
    from models import Coincidencia, ClipInfo, ResultadoProcesamiento
    USAR_DATACLASSES = True
except ImportError:
    USAR_DATACLASSES = False

RADIO_ANALIZER_VERSION = "v4.5"

# === SETUP STREAMLIT (DEBE SER LO PRIMERO) ===
st.set_page_config(
    page_title=f"🎙️ RadioAnalizer {RADIO_ANALIZER_VERSION} - Análisis de Audios",
    layout="wide",
)

# === PREVENIR EJECUCIÓN AL IMPORTAR ===
# El código de Streamlit se ejecuta solo cuando se ejecuta directamente
# No se ejecuta cuando se importa como módulo

# === SISTEMA DE LOGGING ===
def configurar_logging():
    """
    Configura el sistema completo de logging
    """
    # Crear directorio de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formato de logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Logger principal
    logger = logging.getLogger('RadioAnalizer')
    logger.setLevel(logging.DEBUG)
    
    # Limpiar handlers existentes
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para archivo de errores
    error_handler = logging.FileHandler(
        log_dir / f'errors_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Handler para archivo general
    info_handler = logging.FileHandler(
        log_dir / f'app_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Handler para archivo debug
    debug_handler = logging.FileHandler(
        log_dir / f'debug_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    # Agregar handlers
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(console_handler)
    
    return logger

# Configurar logging al inicio
logger = configurar_logging()

def log_exception(func_name, exception, extra_info=""):
    """
    Registra excepciones con información completa
    """
    error_msg = f"ERROR en {func_name}: {str(exception)}"
    if extra_info:
        error_msg += f" | Info adicional: {extra_info}"
    
    logger.error(error_msg)
    logger.debug(f"Traceback completo:\n{traceback.format_exc()}")

def log_info(message, func_name=""):
    """
    Registra información general
    """
    if func_name:
        message = f"[{func_name}] {message}"
    logger.info(message)

def log_debug(message, func_name=""):
    """
    Registra información de debug
    """
    if func_name:
        message = f"[{func_name}] {message}"
    logger.debug(message)

def log_warning(message, func_name=""):
    """
    Registra advertencias
    """
    if func_name:
        message = f"[{func_name}] {message}"
    logger.warning(message)

def verificar_conectividad():
    """
    Verifica conectividad a internet probando múltiples servicios
    """
    import socket
    
    servicios_test = [
        ("8.8.8.8", 53),      # Google DNS
        ("1.1.1.1", 53),      # Cloudflare DNS
        ("api.telegram.org", 443),  # Telegram
    ]
    
    for host, port in servicios_test:
        try:
            socket.create_connection((host, port), timeout=5)
            log_debug(f"Conectividad OK con {host}:{port}", "verificar_conectividad")
            return True
        except Exception:
            continue
    
    log_info("Sin conectividad a internet detectada", "verificar_conectividad")
    return False

def esperar_con_backoff(intento, max_espera=60):
    """
    Implementa backoff exponencial para reintentos
    """
    import random
    
    # Backoff exponencial con jitter
    espera = min(max_espera, (2 ** intento) + random.uniform(0, 1))
    log_debug(f"Esperando {espera:.1f}s antes del reintento {intento}", "esperar_con_backoff")
    time.sleep(espera)
    return espera

def test_api_connectivity():
    """
    Prueba la conectividad con las APIs de OpenAI y Mistral
    """
    resultados = {
        'openai': False,
        'mistral': False,
        'internet': False
    }
    
    # Test conectividad general
    resultados['internet'] = verificar_conectividad()
    
    if not resultados['internet']:
        log_warning("Sin conectividad a internet - APIs no disponibles", "test_api_connectivity")
        return resultados
    
    # Test OpenAI API
    try:
        response = requests.get("https://api.openai.com/v1/models", 
                              headers={"Authorization": f"Bearer {openai_client.api_key}"}, 
                              timeout=10)
        if response.status_code == 200:
            resultados['openai'] = True
            log_info("OpenAI API conectividad OK", "test_api_connectivity")
        else:
            log_warning(f"OpenAI API error: {response.status_code}", "test_api_connectivity")
    except Exception as e:
        log_warning(f"OpenAI API no disponible: {e}", "test_api_connectivity")
    
    # Test Mistral API
    try:
        response = requests.get("https://api.mistral.ai/v1/models", 
                              headers={"Authorization": f"Bearer {mistral_api_key}"}, 
                              timeout=10)
        if response.status_code == 200:
            resultados['mistral'] = True
            log_info("Mistral API conectividad OK", "test_api_connectivity")
        else:
            log_warning(f"Mistral API error: {response.status_code}", "test_api_connectivity")
    except Exception as e:
        log_warning(f"Mistral API no disponible: {e}", "test_api_connectivity")
    
    return resultados

def diagnosticar_conectividad():
    """
    Diagnóstico completo de conectividad y APIs
    """
    log_info("Iniciando diagnóstico de conectividad...", "diagnosticar_conectividad")
    
    # Test DNS
    try:
        socket.gethostbyname("api.openai.com")
        log_info("DNS OpenAI: OK", "diagnosticar_conectividad")
    except Exception as e:
        log_warning(f"DNS OpenAI falló: {e}", "diagnosticar_conectividad")
    
    try:
        socket.gethostbyname("api.mistral.ai")
        log_info("DNS Mistral: OK", "diagnosticar_conectividad")
    except Exception as e:
        log_warning(f"DNS Mistral falló: {e}", "diagnosticar_conectividad")
    
    # Test APIs
    resultados = test_api_connectivity()
    
    if not resultados['internet']:
        log_warning("❌ Sin conectividad a internet", "diagnosticar_conectividad")
    elif not resultados['openai'] and not resultados['mistral']:
        log_warning("❌ APIs no disponibles - verificar configuración", "diagnosticar_conectividad")
    else:
        log_info("✅ Conectividad parcial disponible", "diagnosticar_conectividad")
    
    return resultados



# === CONFIGURACIÓN DE API KEYS (desde variables de entorno) ===
# Las credenciales se cargan desde archivo .env o variables de entorno del sistema
# NUNCA hardcodear API keys en el código!

# Cargar variables de entorno (.env en la raíz de este proyecto)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(_RADIO_ROOT, ".env"), override=False)
except ImportError:
    pass  # Si no está instalado, usar solo variables de entorno del sistema

# OpenAI
_openai_api_key = os.getenv('OPENAI_API_KEY', '')
if not _openai_api_key:
    _openai_api_key = os.getenv('OPENAI_API_KEY_BACKUP', '')

openai_client = openai.OpenAI(api_key=_openai_api_key) if _openai_api_key else None

# DeepSeek (API compatible OpenAI; enriquecimiento de motivos tangenciales con transcripción)
# En .env puede usarse DEEPSEEK_API_KEY o DEEPSEEK_KEY (cualquiera de las dos).
DEEPSEEK_API_KEY = (
    os.getenv('DEEPSEEK_API_KEY', '')
    or os.getenv('DEEPSEEK_KEY', '')
    or os.getenv('DeepSeek_KEY', '')
    or os.getenv('DeepSeek KEY', '')
)
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com').rstrip('/')
deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        deepseek_client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            timeout=120.0,
        )
    except Exception as _deepseek_init_err:
        logging.warning(f"[WARN] Cliente DeepSeek no inicializado: {_deepseek_init_err}")
        deepseek_client = None

# Mistral
mistral_api_key = os.getenv('MISTRAL_API_KEY', '')

# Gemini 3.0
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("[OK] Cliente Gemini 3.0 inicializado correctamente")
    except Exception as e:
        logging.warning(f"[WARN] Error inicializando Gemini: {e}")
        gemini_client = None
else:
    logging.warning("[WARN] Gemini no configurado. Configura GEMINI_API_KEY en .env")

# ElevenLabs (intro de voz para coincidencias)
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '')
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', 'eleven_multilingual_v2')

# === CONFIGURACIÓN GOOGLE DRIVE (desde variables de entorno) ===
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
GOOGLE_REFRESH_TOKEN = os.getenv('GOOGLE_REFRESH_TOKEN', '')
GOOGLE_DRIVE_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', '')
# Carpeta raíz fija para tangenciales (subcarpetas: tangenciales radio-{nombre cliente})
GOOGLE_DRIVE_TANGENCIALES_PARENT_ID = os.getenv(
    'GOOGLE_DRIVE_TANGENCIALES_PARENT_ID',
    '1Jzsrkx2YRYfyqgxKQ1oW1PHmYuZCKKXf',
)
GOOGLE_SHEETS_ID_EDESUR = os.getenv(
    'GOOGLE_SHEETS_ID_EDESUR',
    '1y1wYUf9SZf8RVPJt0PfGU69QFlBsd7f5we_FBGQRGYg',
)
GOOGLE_SHEETS_ID_INTRANT = os.getenv(
    'GOOGLE_SHEETS_ID_INTRANT',
    '1zEhGX9aauTboEO8H7qEyqlCMzX8RKod6EEy81ZQpLhA',
)
GOOGLE_SHEETS_RANGE_COINCIDENCIAS = os.getenv('GOOGLE_SHEETS_RANGE', 'Hoja 1!A:G')

# === ELIMINA DLLs de CUDA inválidas de Torch ===
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
for dll in glob.glob(os.path.join(torch_lib, "torch_cuda*.dll")):
    try:
        os.remove(dll)
    except OSError:
        pass

# === CONFIGURACIÓN ===
# Carpetas: entrada de audios + carpeta AUDIOCHECKS (logs, caché, configs, clips, evidencias).
# Override: CARPETA_VIDEOS / CARPETA_PROCESADOS o RADIO_CARPETA_*.
_DIR_SCRIPT = Path(_RADIO_ROOT)
_DEFAULT_AUDIOS = _DIR_SCRIPT / "audios_entrada"
# CARPETA_VIDEOS: nombre heredado del motor; aquí es la carpeta de entrada de audios a escanear
CARPETA_VIDEOS = os.getenv(
    "CARPETA_VIDEOS",
    os.getenv("RADIO_CARPETA_AUDIOS", str(_DEFAULT_AUDIOS)),
)
# Por defecto: carpeta AUDIOCHECKS junto a la entrada (ej. grabaciones/AUDIOS → grabaciones/AUDIOCHECKS)
_default_procesados = os.path.join(os.path.dirname(os.path.abspath(CARPETA_VIDEOS)), "AUDIOCHECKS")
CARPETA_PROCESADOS = os.getenv(
    "CARPETA_PROCESADOS",
    os.getenv("RADIO_CARPETA_PROCESADOS", _default_procesados),
)
INFORME_GENERAL_RADIO_PATH = os.getenv(
    "RADIO_INFORME_GENERAL_PATH",
    os.path.join(os.path.expanduser("~"), "Desktop", "informes", "informe_general.md"),
)
# Subcarpeta dentro de AUDIOCHECKS: copias + TXT de evidencia (registrar_audio_check)
CARPETA_AUDIOCHECKS_EVIDENCIAS = os.path.join(CARPETA_PROCESADOS, "audioChecks")
ULTIMA_COINCIDENCIA_REENVIO_JSON = os.path.join(CARPETA_PROCESADOS, "ultima_coincidencia_reenvio.json")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".wma")

# === CONFIGURACIÓN SUPABASE (desde variables de entorno) ===
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY', '')

# Inicializar cliente de Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        logging.warning(f"[WARN] Error inicializando Supabase: {e}")
        supabase = None
else:
    logging.warning("[WARN] Supabase no configurado. Configura SUPABASE_URL y SUPABASE_ANON_KEY en .env")

# Crear carpetas necesarias (entrada de audios, AUDIOCHECKS y subcarpeta de evidencias)
os.makedirs(CARPETA_VIDEOS, exist_ok=True)
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)
os.makedirs(CARPETA_AUDIOCHECKS_EVIDENCIAS, exist_ok=True)
PROCESADOS_LOG = os.path.join(CARPETA_PROCESADOS, "procesados.log")
TERMINOS_CONFIG = "terminos_guardados.json"  # Archivo para términos (en la raíz de la app)
WEBHOOK_CONFIG = os.path.join(CARPETA_PROCESADOS, "webhook_config.json")  # Configuración del webhook
TELEGRAM_CONFIG = os.path.join(CARPETA_PROCESADOS, "telegram_config.json")  # Configuración de Telegram
CLOUDINARY_CONFIG = os.path.join(CARPETA_PROCESADOS, "cloudinary_config.json")  # Configuración de Cloudinary
CACHE_ESCANEO = os.path.join(CARPETA_PROCESADOS, "cache_escaneo.json")  # Caché de archivos escaneados
CLIENTES_CONFIG = "clientes_config.json"  # Configuración de clientes (en raíz para no perderse)
TAMANO_MINIMO_BYTES = 8 * 1024 * 1024  # 8 MB: tamaño mínimo antes de procesar

# Prefijo en asuntos Brevo (coincidencias / tangenciales)
EMAIL_ASUNTO_PREFIJO_RADIO = "Radio — "

# === SISTEMA MULTI-CLIENTE ===
def generar_id_cliente():
    """Genera un ID único para un nuevo cliente"""
    import uuid
    return str(uuid.uuid4())[:8]

def cargar_clientes():
    """Carga la lista de clientes configurados"""
    try:
        if os.path.exists(CLIENTES_CONFIG):
            with open(CLIENTES_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('clientes', [])
    except Exception as e:
        log_exception("cargar_clientes", e)
    return []

def guardar_clientes(clientes):
    """Guarda la lista de clientes"""
    try:
        data = {
            'clientes': clientes,
            'fecha_actualizacion': datetime.now().isoformat(),
            'total_clientes': len(clientes)
        }
        with open(CLIENTES_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_exception("guardar_clientes", e)
        return False

def obtener_cliente_por_id(cliente_id):
    """Obtiene un cliente por su ID"""
    clientes = cargar_clientes()
    for cliente in clientes:
        if cliente.get('id') == cliente_id:
            return cliente
    return None

def obtener_cliente_default():
    """Retorna la configuración por defecto (sistema actual)"""
    return {
        'id': 'default',
        'nombre': 'Sistema Principal (EDESUR)',
        'activo': True,
        'envios_habilitados': True,
        'color': '#1E88E5',
        'webhook': {
            'enabled': True,
            'url': 'https://hook.us1.make.com/1nk48toiy2c64f9966yue8bwhzqnosny',
            'url_secundario': '',
            'url_terciario': ''
        },
        'telegram': {
            'enabled': True,
            'bot_token': '',
            'chat_id': '',
            'send_clips': True,
            'send_summary': True,
            'use_cloudinary': True
        },
        'brevo': {
            'enabled': True,
            'api_key': '',
            'sender_email': '',
            'sender_name': 'Sistema de Análisis de Audios',
            'correos_destinatarios': []
        },
        'google_drive': {
            'enabled': True,
            'folder_id': GOOGLE_DRIVE_FOLDER_ID if 'GOOGLE_DRIVE_FOLDER_ID' in dir() else ''
        },
        'cloudinary': {
            'enabled': True,
            'cloud_name': '',
            'api_key': '',
            'api_secret': '',
            'folder': 'radio_analyzer_audios'
        },
        'supabase': {
            'enabled': True,
            'url': SUPABASE_URL if SUPABASE_URL else '',
            'anon_key': SUPABASE_ANON_KEY if SUPABASE_ANON_KEY else ''
        }
    }

def crear_cliente_nuevo(nombre, color='#4CAF50'):
    """Crea un nuevo cliente con configuración vacía"""
    return {
        'id': generar_id_cliente(),
        'nombre': nombre,
        'activo': True,
        'envios_habilitados': True,
        'color': color,
        'fecha_creacion': datetime.now().isoformat(),
        'webhook': {
            'enabled': False,
            'url': '',
            'url_secundario': '',
            'url_terciario': ''
        },
        'telegram': {
            'enabled': False,
            'bot_token': '',
            'chat_id': '',
            'send_clips': True,
            'send_summary': True,
            'use_cloudinary': True
        },
        'brevo': {
            'enabled': False,
            'api_key': '',
            'sender_email': '',
            'sender_name': f'Sistema {nombre}',
            'correos_destinatarios': []
        },
        'google_drive': {
            'enabled': False,
            'folder_id': '',
            'client_id': '',
            'client_secret': '',
            'refresh_token': ''
        },
        'cloudinary': {
            'enabled': False,
            'cloud_name': '',
            'api_key': '',
            'api_secret': '',
            'folder': f'clips_{nombre.lower().replace(" ", "_")}'
        },
        'supabase': {
            'enabled': False,
            'url': '',
            'anon_key': ''
        }
    }

def agregar_cliente(cliente):
    """Agrega un nuevo cliente a la lista"""
    clientes = cargar_clientes()
    # Verificar que no exista un cliente con el mismo nombre
    for c in clientes:
        if c.get('nombre', '').lower() == cliente.get('nombre', '').lower():
            return False, "Ya existe un cliente con ese nombre"
    clientes.append(cliente)
    if guardar_clientes(clientes):
        return True, f"Cliente '{cliente['nombre']}' agregado exitosamente"
    return False, "Error guardando cliente"

def actualizar_cliente(cliente_id, datos_actualizados):
    """Actualiza un cliente existente"""
    clientes = cargar_clientes()
    for i, cliente in enumerate(clientes):
        if cliente.get('id') == cliente_id:
            clientes[i].update(datos_actualizados)
            clientes[i]['fecha_actualizacion'] = datetime.now().isoformat()
            if guardar_clientes(clientes):
                return True, "Cliente actualizado exitosamente"
            return False, "Error guardando cambios"
    return False, "Cliente no encontrado"

def eliminar_cliente(cliente_id):
    """Elimina un cliente de la lista"""
    if cliente_id == 'default':
        return False, "No se puede eliminar el cliente por defecto"
    clientes = cargar_clientes()
    clientes_filtrados = [c for c in clientes if c.get('id') != cliente_id]
    if len(clientes_filtrados) == len(clientes):
        return False, "Cliente no encontrado"
    if guardar_clientes(clientes_filtrados):
        return True, "Cliente eliminado exitosamente"
    return False, "Error eliminando cliente"

def obtener_clientes_activos():
    """Retorna lista de clientes activos incluyendo el default"""
    clientes = cargar_clientes()
    # Agregar cliente default si no existe
    tiene_default = any(c.get('id') == 'default' for c in clientes)
    if not tiene_default:
        # Migrar configuración actual al cliente default
        cliente_default = migrar_config_actual_a_cliente()
        clientes.insert(0, cliente_default)
        guardar_clientes(clientes)
    return [c for c in clientes if c.get('activo', True)]

def migrar_config_actual_a_cliente():
    """Migra la configuración actual del sistema al formato de cliente"""
    # Cargar configuraciones actuales
    webhook_config = cargar_webhook_config()
    telegram_config = cargar_telegram_config()
    brevo_config = cargar_brevo_config()
    cloudinary_config = cargar_cloudinary_config()
    correos = cargar_correos_guardados()
    
    return {
        'id': 'default',
        'nombre': 'Sistema Principal (EDESUR)',
        'activo': True,
        'envios_habilitados': True,
        'color': '#1E88E5',
        'fecha_creacion': datetime.now().isoformat(),
        'webhook': {
            'enabled': webhook_config.get('enabled', True),
            'url': webhook_config.get('url', ''),
            'url_secundario': webhook_config.get('url_secundario', ''),
            'url_terciario': webhook_config.get('url_terciario', ''),
            'enviar_makecom': webhook_config.get('enviar_makecom', True),
            'enviar_n8n': webhook_config.get('enviar_n8n', True),
            'enviar_n8n_test': webhook_config.get('enviar_n8n_test', True)
        },
        'telegram': {
            'enabled': telegram_config.get('enabled', False),
            'bot_token': telegram_config.get('bot_token', ''),
            'chat_id': telegram_config.get('chat_id', ''),
            'send_clips': telegram_config.get('send_clips', True),
            'send_summary': telegram_config.get('send_summary', True),
            'use_cloudinary': telegram_config.get('use_cloudinary', True)
        },
        'brevo': {
            'enabled': brevo_config.get('enabled', False),
            'api_key': brevo_config.get('api_key', ''),
            'sender_email': brevo_config.get('sender_email', ''),
            'sender_name': brevo_config.get('sender_name', 'Sistema de Análisis de Audio'),
            'correos_destinatarios': [c.get('email', '') for c in correos if c.get('activo', True)]
        },
        'google_drive': {
            'enabled': True,
            'folder_id': GOOGLE_DRIVE_FOLDER_ID if 'GOOGLE_DRIVE_FOLDER_ID' in dir() else ''
        },
        'cloudinary': {
            'enabled': cloudinary_config.get('cloud_name', '') != '',
            'cloud_name': cloudinary_config.get('cloud_name', ''),
            'api_key': cloudinary_config.get('api_key', ''),
            'api_secret': cloudinary_config.get('api_secret', ''),
            'folder': cloudinary_config.get('folder', 'video_analyzer_clips')
        },
        'supabase': {
            'enabled': SUPABASE_URL is not None and SUPABASE_ANON_KEY is not None,
            'url': SUPABASE_URL if SUPABASE_URL else '',
            'anon_key': SUPABASE_ANON_KEY if SUPABASE_ANON_KEY else ''
        }
    }

def obtener_terminos_por_cliente(cliente_id):
    """Obtiene los términos asociados a un cliente específico"""
    try:
        if os.path.exists(TERMINOS_CONFIG):
            with open(TERMINOS_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                terminos = data.get('terminos', [])
                # Filtrar términos por cliente
                if isinstance(terminos, list) and len(terminos) > 0:
                    if isinstance(terminos[0], dict):
                        return [t for t in terminos if t.get('cliente_id', 'default') == cliente_id]
                    else:
                        # Términos legacy (solo strings) - asignar a default
                        if cliente_id == 'default':
                            return [{'termino': t, 'cliente_id': 'default'} for t in terminos]
                        return []
    except Exception as e:
        log_exception("obtener_terminos_por_cliente", e)
    return []

def obtener_cliente_por_termino(termino):
    """Dado un término, retorna el cliente asociado"""
    try:
        if os.path.exists(TERMINOS_CONFIG):
            with open(TERMINOS_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                terminos = data.get('terminos', [])
                for t in terminos:
                    if isinstance(t, dict):
                        if t.get('termino', '').lower() == termino.lower():
                            cliente_id = t.get('cliente_id', 'default')
                            return obtener_cliente_por_id(cliente_id) or obtener_cliente_default()
                    elif isinstance(t, str):
                        if t.lower() == termino.lower():
                            return obtener_cliente_default()
    except Exception as e:
        log_exception("obtener_cliente_por_termino", e)
    return obtener_cliente_default()


def etiqueta_entidad_por_cliente_id(cliente_id):
    """Marca estable para texto visible (no altera ids internos)."""
    cid = str(cliente_id or "").strip().lower()
    if cid == "default":
        return "EDESUR"
    if cid == "intrant":
        return "Intrant"
    if cid == "minerd":
        return "MINERD"
    return None


def capitalizar_marcas_medios_rd_en_texto(texto):
    """EDESUR, Intrant y MINERD bien escritos en texto para humanos (evita URLs/correos @dominio.intrant)."""
    if not texto:
        return texto
    t = str(texto)
    # (?<![@\w]) evita cambiar intrant dentro de xxx@intrant.gob / identificadores pegados con _
    t = re.sub(r"(?<![@\w])\bedesur\b(?!\.)", "EDESUR", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<![@\w])\bminerd\b(?!\.)", "MINERD", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<![@\w])\bintrant\b(?!\.)", "Intrant", t, flags=re.IGNORECASE)
    return t


def nombre_cliente_mostrar_para_ui(cliente, cliente_id_fallback=None):
    """Nombre coherentemente capitalizado para UI, correos y payloads legibles («cliente»)."""
    if cliente:
        cid = str(cliente.get("id") or "").strip().lower()
        marca = etiqueta_entidad_por_cliente_id(cid)
        base_nombre = (cliente.get("nombre") or "").strip()
        if cid in ("intrant", "minerd"):
            return marca or base_nombre or (cliente_id_fallback or "Cliente")
        if cid == "default":
            if base_nombre:
                return capitalizar_marcas_medios_rd_en_texto(base_nombre)
            return marca or "EDESUR"
        if base_nombre:
            return capitalizar_marcas_medios_rd_en_texto(base_nombre)
        return marca or capitalizar_marcas_medios_rd_en_texto(str(cliente_id_fallback or "Cliente"))
    marca = etiqueta_entidad_por_cliente_id(cliente_id_fallback)
    if marca:
        return marca
    if cliente_id_fallback:
        return capitalizar_marcas_medios_rd_en_texto(str(cliente_id_fallback))
    return "Desconocido"


# === FUNCIONES DE ENVÍO POR CLIENTE ===
def enviar_webhook_cliente(cliente, video_path, resumen, terminos, data_extra=None):
    """Envía a los webhooks configurados para un cliente específico"""
    func_name = "enviar_webhook_cliente"
    
    webhook_config = cliente.get('webhook', {})
    if not webhook_config.get('enabled', False):
        return False, "Webhook deshabilitado para este cliente"
    
    # Verificar conectividad
    if not verificar_conectividad():
        return False, "Sin conectividad a internet"
    
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    log_info(f"Enviando webhook para cliente: {cliente_nombre}", func_name)
    
    # Datos para el webhook (resumen ejecutivo completo, sin truncar)
    terminos_c = []
    if terminos:
        terminos_c = [capitalizar_marcas_medios_rd_en_texto(str(t)) for t in terminos]
    data = {
        'evento': 'video_analizado',
        'timestamp': datetime.now().isoformat(),
        'video': os.path.basename(video_path) if video_path else 'N/A',
        'terminos': terminos_c,
        'resumen': capitalizar_marcas_medios_rd_en_texto(resumen if resumen else ''),
        'servidor': 'analizador_audio_ia_v2',
        'cliente': cliente_nombre
    }
    
    if data_extra:
        data.update(data_extra)
    
    mensajes = []
    exitos = []
    
    # Enviar a webhook principal
    if webhook_config.get('url'):
        exito, mensaje = enviar_a_webhook_individual(
            webhook_config['url'], data, func_name, f"Webhook {cliente_nombre}"
        )
        exitos.append(exito)
        mensajes.append(f"{'✅' if exito else '❌'} Principal: {mensaje}")
    
    # Enviar a webhook secundario
    if webhook_config.get('url_secundario'):
        exito, mensaje = enviar_a_webhook_individual(
            webhook_config['url_secundario'], data, func_name, f"Webhook2 {cliente_nombre}"
        )
        exitos.append(exito)
        mensajes.append(f"{'✅' if exito else '❌'} Secundario: {mensaje}")
    
    # Enviar a webhook terciario
    if webhook_config.get('url_terciario'):
        exito, mensaje = enviar_a_webhook_individual(
            webhook_config['url_terciario'], data, func_name, f"Webhook3 {cliente_nombre}"
        )
        exitos.append(exito)
        mensajes.append(f"{'✅' if exito else '❌'} Terciario: {mensaje}")
    
    alguno_exitoso = any(exitos) if exitos else False
    return alguno_exitoso, " | ".join(mensajes)

def enviar_telegram_cliente(cliente, mensaje, video_path=None, parse_mode='Markdown', video_url=None):
    """Envía mensaje y/o clip (Telegram sendVideo) a Telegram usando credenciales del cliente"""
    func_name = "enviar_telegram_cliente"
    
    telegram_config = cliente.get('telegram', {})
    if not telegram_config.get('enabled', False):
        return False, "Telegram deshabilitado para este cliente"
    
    bot_token = telegram_config.get('bot_token', '')
    chat_id = telegram_config.get('chat_id', '')
    
    if not bot_token or not chat_id:
        return False, "Telegram no configurado (falta token o chat_id)"
    
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    log_info(f"Enviando Telegram para cliente: {cliente_nombre}", func_name)

    # Función de escape para Telegram (mejorada para preservar puntuación)
    def escape_telegram_text(text):
        import re
        if not text: return ""
        # Solo eliminar caracteres que rompen el Markdown de Telegram si no se desea parse_mode
        # o que causan ruidos visuales innecesarios, pero PRESERVAR puntuación básica.
        # Eliminamos: * _ ` [ ] ( ) ~ > + = | { } ! \ (algunas veces causan errores de parseo)
        # PRESERVAMOS: . , : - / # (son esenciales para la legibilidad)
        text = re.sub(r'[*_`\[\]()~>+=|{}!\\#]', '', text)
        # Limpiar triples saltos de línea
        text = text.replace('\n\n\n', '\n\n').strip()
        return text
    
    resultados = []
    
    # Enviar mensaje de texto
    if mensaje and telegram_config.get('send_summary', True):
        # Limpiar caracteres Markdown problemáticos usando función robusta
        mensaje_limpio = escape_telegram_text(mensaje)
        exito, msg = enviar_mensaje_telegram(mensaje_limpio, chat_id, bot_token, None)  # Sin parse_mode para evitar errores
        resultados.append(('Mensaje', exito, msg))
    
    # Enviar clip si existe
    if video_path and os.path.exists(video_path) and telegram_config.get('send_clips', True):
        use_cloudinary = telegram_config.get('use_cloudinary', True)
        
        # 🆕 Si ya tenemos el URL de Cloudinary, usarlo directamente (evita subir dos veces)
        if video_url and use_cloudinary:
            log_info(f"Usando URL de Cloudinary existente: {video_url}", func_name)
            # 🚀 MEJORA: El caption ahora incluye el resumen completo (hasta 1024 caracteres)
            caption_limpio = escape_telegram_text(mensaje)[:1024]
            # Enviar clip usando URL existente (sin parse_mode para evitar errores)
            exito, msg, _ = enviar_video_telegram_url(video_url, caption_limpio, chat_id, bot_token, None)
            resultados.append(('Clip', exito, msg))
        else:
            if use_cloudinary:
                # ... (resto omitido) ...
                # Configurar Cloudinary del cliente si está habilitado
                cloudinary_config = cliente.get('cloudinary', {})
                if cloudinary_config.get('enabled') and cloudinary_config.get('cloud_name'):
                    try:
                        cloudinary.config(
                            cloud_name=cloudinary_config['cloud_name'],
                            api_key=cloudinary_config['api_key'],
                            api_secret=cloudinary_config['api_secret']
                        )
                    except:
                        pass
            
            # 🚀 MEJORA: El caption ahora incluye el resumen completo (hasta 1024 caracteres)
            caption_limpio = escape_telegram_text(mensaje)[:1024]
            exito, msg, _ = enviar_video_telegram(video_path, caption_limpio, chat_id, bot_token, use_cloudinary)
            resultados.append(('Clip', exito, msg))
    
    exitos = [r[1] for r in resultados]
    mensajes = [f"{'✅' if r[1] else '❌'} {r[0]}: {r[2]}" for r in resultados]
    
    return any(exitos) if exitos else False, " | ".join(mensajes)

def enviar_brevo_cliente(cliente, termino_encontrado, resumen_completo, nombre_video, video_path=None, info_medio="", terminos_detectados=[], video_url=None, transcripcion_segmento=""):
    """Envía correo usando Brevo SMTP con credenciales del cliente (igual que transmistral2.py)"""
    func_name = "enviar_brevo_cliente"
    
    brevo_config = cliente.get('brevo', {})
    if not brevo_config.get('enabled', False):
        return False, "Brevo deshabilitado para este cliente"
    
    api_key = brevo_config.get('api_key', '')
    sender_email = brevo_config.get('sender_email', '')
    sender_name = brevo_config.get(
        'sender_name', f'Sistema {nombre_cliente_mostrar_para_ui(cliente)}'
    )
    correos_destinatarios, correos_normalizados = obtener_destinatarios_activos_cliente(cliente)
    # Persistir normalización de estructura sin alterar comportamiento
    if brevo_config.get('correos_destinatarios', []) != correos_normalizados:
        brevo_config['correos_destinatarios'] = correos_normalizados
        cliente['brevo'] = brevo_config
        actualizar_cliente(cliente.get('id', ''), cliente)
    
    # Configuración SMTP (igual que transmistral2.py)
    smtp_user = brevo_config.get('smtp_user', sender_email)
    smtp_server = brevo_config.get('smtp_server', 'smtp-relay.brevo.com')
    smtp_port = brevo_config.get('smtp_port', 587)
    
    if not api_key or not sender_email:
        return False, "Configuración de Brevo incompleta"
    
    if not correos_destinatarios:
        return False, "No hay destinatarios configurados"
    
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    sender_name = capitalizar_marcas_medios_rd_en_texto(sender_name)
    termino_asunto = capitalizar_marcas_medios_rd_en_texto(str(termino_encontrado)).strip()
    log_info(f"Enviando correo Brevo para cliente: {cliente_nombre} a {len(correos_destinatarios)} destinatarios", func_name)
    
    try:
        # Crear mensaje
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"{EMAIL_ASUNTO_PREFIJO_RADIO}🎯 Coincidencia: {termino_asunto}"
        msg['From'] = f"{sender_name} <{sender_email}>"
        
        # Configurar destinatarios: El primero va en 'To', el resto en 'Bcc'
        # Esto mejora la entrega a Gmail y otros proveedores que a veces rechazan correos solo con Bcc.
        msg['To'] = correos_destinatarios[0]
        if len(correos_destinatarios) > 1:
            msg['Bcc'] = ', '.join(correos_destinatarios[1:])
        
        # Generar HTML del correo
        html_content = crear_plantilla_email_html(
            termino_encontrado, resumen_completo, nombre_video, 
            info_medio, terminos_detectados, video_url,
            transcripcion_segmento=transcripcion_segmento
        )
        
        # Texto plano como alternativa (mismas marcas que el HTML)
        text_content = capitalizar_marcas_medios_rd_en_texto(
            f"""
Coincidencia detectada: {termino_encontrado}

Archivo de audio: {nombre_video}
Medio: {info_medio if info_medio else ""}

{resumen_completo}

---
Sistema de Análisis de Audios - {cliente_nombre}
        """.strip()
        )
        
        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))
        
        # NO adjuntar audios grandes - solo usar URLs (igual que transmistral2.py)
        if video_url:
            log_info(f"Usando URL para audio en correo: {video_url}", func_name)
        
        # Enviar usando SMTP de Brevo (configuración igual a transmistral2.py)
        log_info(f"Conectando a SMTP: {smtp_server}:{smtp_port} con usuario {smtp_user}", func_name)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, api_key)
            server.send_message(msg)
        
        log_info(f"✅ Correo enviado a {len(correos_destinatarios)} destinatarios", func_name)
        return True, f"Correo enviado a {len(correos_destinatarios)} destinatarios"
        
    except Exception as e:
        log_exception(func_name, e)
        return False, f"Error: {str(e)}"


def formatear_nombre_medio_desde_ruta(relative_path):
    """
    Primer segmento de la ruta relativa como nombre de medio legible.
    Quita [ID], sustituye _ por espacios.
    """
    if not relative_path or not str(relative_path).strip():
        return "Medio desconocido"
    first = str(relative_path).replace("\\", "/").split("/")[0].strip()
    first = re.sub(r"\[[^\]]*\]", "", first)
    first = first.replace("_", " ").strip()
    first = first if first else "Medio desconocido"
    return capitalizar_marcas_medios_rd_en_texto(first)


# Patrón: "La 91 720p 2026-05-01 06-52-11 seg000.m4a"
_RE_EMISION_RADIO_720 = re.compile(
    r"^(.+?)\s+720p\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})\s+seg\d+",
    re.IGNORECASE,
)

_MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
    7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


def parsear_emision_radio_desde_archivo(relative_path):
    """
    Extrae emisora, fecha y hora de inicio de segmento desde el nombre de archivo típico
    («… 720p YYYY-MM-DD HH-mm-ss segNNN.ext»). Retorna dict o None.
    """
    if not relative_path:
        return None
    base = os.path.basename(str(relative_path).replace("\\", "/")).strip()
    m = _RE_EMISION_RADIO_720.match(base)
    if not m:
        return None
    raw_em, y, mo, d, hh, mi, ss = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss))
    except ValueError:
        return None
    return {"emisora_raw": raw_em.strip(), "dt_emision": dt, "basename": base}


def normalizar_nombre_emisora_correo(emisora_raw):
    """Nombre de emisora para listados/correo (mayúscula inicial; casos raros del grabador)."""
    if not emisora_raw or not str(emisora_raw).strip():
        return "Emisora desconocida"
    s = str(emisora_raw).strip()
    slug_compact = "".join(c for c in unicodedata.normalize("NFD", s.lower())
                           if unicodedata.category(c) != "Mn" and (c.isalnum() or c in "_"))
    if "independencia" in slug_compact:
        return "Independencia FM"
    if slug_compact.startswith("la91") or re.match(r"^la\s*91\b", s, re.IGNORECASE):
        return "La 91"
    if "alofoke" in slug_compact:
        return "Alofoke"
    if "estrella" in slug_compact and "905" in slug_compact:
        return "Estrella 905 FM"
    if slug_compact.startswith("rumba"):
        return "Rumba 985 FM"
    if "superq" in slug_compact or ("super" in slug_compact and "1009" in slug_compact):
        return "Super Q 1009"
    if "super7" in slug_compact.replace(" ", "") or "super7fm" in slug_compact.replace(" ", ""):
        return "Super 7 FM 107.7"
    if "super" in slug_compact and "1077" in slug_compact:
        return "Super 7 FM 107.7"
    if slug_compact.startswith("zol"):
        return "ZOL FM República Dominicana"
    tokens = []
    for w in re.split(r"\s+", s):
        if not w:
            continue
        if w.upper() in ("FM", "RD", "TV", "Q", "AM", "ZOL"):
            tokens.append(w.upper())
            continue
        if re.fullmatch(r"\d+[a-zA-Z]?|[\d\-]+", w):
            tokens.append(w)
            continue
        if "-" in w:
            tokens.append("-".join(p.capitalize() for p in w.split("-") if p))
            continue
        tokens.append(w[:1].upper() + w[1:].lower() if len(w) > 1 else w.upper())
    return " ".join(tokens)


def formato_linea_emision_legible(relative_path):
    """
    Una línea legible tipo: «Independencia FM, 1 de mayo de 2026 a las 7:42» (sin sufijo técnico 720p).
    Si no coincide el patrón, usa formatear_nombre_medio_desde_ruta.
    """
    info = parsear_emision_radio_desde_archivo(relative_path)
    if info:
        nombre = normalizar_nombre_emisora_correo(info["emisora_raw"])
        dt = info["dt_emision"]
        fecha_txt = f"{dt.day} de {_MESES_ES.get(dt.month, str(dt.month))} de {dt.year}"
        return f"{nombre}, {fecha_txt} a las {dt.hour}:{dt.minute:02d}"
    return formatear_nombre_medio_desde_ruta(relative_path)


def nombre_base_tangencial_normalizado(rel, termino, momento_termino):
    """
    Nombre de archivo legible: emisora + fecha/hora de emisión (del nombre de archivo) + término + posición en audio.
    """
    term_slug = re.sub(r'[^\w\-]', '_', (termino or '').strip(), flags=re.UNICODE).strip('_')[:30] or 'termino'
    try:
        mt = float(momento_termino or 0)
    except (TypeError, ValueError):
        mt = 0.0
    m, s = divmod(int(mt), 60)
    pos = f"{m}m{s:02d}s"
    info = parsear_emision_radio_desde_archivo(rel)
    if info:
        em = normalizar_nombre_emisora_correo(info["emisora_raw"])
        em_slug = re.sub(r'[^\w]+', '', em.replace(' ', ''))[:40] or 'Medio'
        dt = info["dt_emision"]
        fh = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}_{dt.hour:02d}-{dt.minute:02d}"
        base = f"{em_slug}_{fh}_{term_slug}_{pos}"
    else:
        med = formatear_nombre_medio_desde_ruta(rel)
        med_slug = re.sub(r'[^\w\-]+', '_', med, flags=re.UNICODE).strip('_')[:35] or 'medio'
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f"{med_slug}_{ts}_{term_slug}_{pos}"
    base = re.sub(r'[\r\n\t<>:"/\\|?*]+', '_', base)
    base = re.sub(r'_+', '_', base).strip('_')
    if len(base) > 180:
        base = base[:180].rstrip('_')
    return base or 'tangencial'


def crear_item_tangencial(rel, termino, motivo, momento_termino, texto_evidencia=""):
    """Un registro homogéneo para UI, Analisishoy y correo Brevo."""
    hora_dt = datetime.now()
    item = {
        'archivo': rel,
        'termino': termino,
        'motivo': motivo,
        'timestamp': momento_termino,
        'hora_deteccion': hora_dt.isoformat(timespec='seconds'),
        'medio': formatear_nombre_medio_desde_ruta(rel),
        'texto_evidencia': (texto_evidencia or '').strip(),
        'clip_path': None,
        'transcripcion_path': None,
        'gdrive_url_audio': None,
        'gdrive_url_txt': None,
    }
    return item


def motivo_display_tangencial(item):
    """Texto de motivo seguro para UI y correo (nunca vacío)."""
    if not isinstance(item, dict):
        return 'Mención tangencial sin desarrollo'
    m = (item.get('motivo') or '').strip()
    return m or 'Mención tangencial sin desarrollo'


def motivo_tangencial_una_frase(item):
    """Frase corta para correo (tras DeepSeek, motivo suele ser una sola oración)."""
    raw = motivo_display_tangencial(item)
    raw = ' '.join((raw or '').split())
    if not raw:
        return 'No tiene peso suficiente como alerta con clip: mención tangencial o sin desarrollo del tema.'
    for end in ('. ', '! ', '? '):
        i = raw.find(end)
        if 20 <= i <= 520:
            return raw[: i + 1].strip()
    for end in '.!?':
        i = raw.find(end)
        if 20 <= i <= 520:
            return raw[: i + 1].strip()
    if len(raw) <= 360:
        return raw
    cut = raw[:360]
    sp = cut.rfind(' ')
    if sp > 140:
        cut = cut[:sp]
    return cut.strip() + '…'


def deepseek_tangenciales_activo():
    """True si hay cliente DeepSeek y no está desactivado por RADIO_DEEPSEEK_TANGENCIALES."""
    if not deepseek_client:
        return False
    v = os.getenv('RADIO_DEEPSEEK_TANGENCIALES', '1').strip().lower()
    return v not in ('0', 'false', 'no', 'off')


def extraer_texto_transcripcion_ventana(segments_timestamps, centro_seg, ventana_seg=120, duracion_audio=None):
    """
    Concatena textos de segmentos que solapan la ventana [centro_seg - ventana_seg, centro_seg + ventana_seg].
    """
    if not segments_timestamps:
        return ""
    try:
        c = float(centro_seg)
    except (TypeError, ValueError):
        return ""
    lo = c - float(ventana_seg)
    hi = c + float(ventana_seg)
    if duracion_audio is not None:
        try:
            d = float(duracion_audio)
            lo = max(0.0, lo)
            hi = min(d, hi)
        except (TypeError, ValueError):
            pass
    parts = []
    for seg in segments_timestamps:
        try:
            start = float(seg.get('start', 0))
        except (TypeError, ValueError):
            continue
        end_raw = seg.get('end', start)
        try:
            end = float(end_raw)
        except (TypeError, ValueError):
            end = start
        if end < start:
            end = start
        if end >= lo and start <= hi:
            t = (seg.get('text') or '').strip()
            if t:
                parts.append(t)
    return " ".join(parts).strip()


def _tangencial_motivo_una_frase(texto):
    """Recorta a la primera oración si el modelo devolvió más de una."""
    texto = (texto or "").strip()
    if not texto:
        return texto
    for sep in ('. ', '? ', '! ', '.\n', '?\n', '!\n'):
        i = texto.find(sep)
        if i != -1:
            return texto[: i + 1].strip()
    return texto


def enriquecer_motivos_tangenciales_deepseek(items_list, func_name='enriquecer_motivos_tangenciales_deepseek'):
    """
    Sustituye `motivo` por una sola frase vía DeepSeek a partir de la transcripción y del motivo técnico.
    Conserva el motivo técnico en `motivo_sistema` cuando tiene éxito. Los ítems que ya tienen
    `motivo_sistema` no se reenvían (p. ej. enriquecidos antes del correo inmediato).
    """
    if not items_list:
        return
    for it in items_list:
        if isinstance(it, dict) and not (it.get('texto_evidencia') or '').strip():
            it['texto_evidencia'] = (it.get('transcripcion_extracto') or '').strip()[:12000]

    if not deepseek_tangenciales_activo():
        if items_list and not deepseek_client:
            log_info('DeepSeek tangenciales omitido: sin DEEPSEEK_API_KEY o cliente no inicializado', func_name)
        elif items_list:
            log_info('DeepSeek tangenciales desactivado (RADIO_DEEPSEEK_TANGENCIALES)', func_name)
        return

    pend_orig_idx = []
    for i, it in enumerate(items_list):
        if isinstance(it, dict) and not ((it.get('motivo_sistema') or '').strip()):
            pend_orig_idx.append(i)
    if not pend_orig_idx:
        log_info(
            f'DeepSeek tangenciales: omitido (todos ya con motivo_sistema — {len(items_list)} ítems)',
            func_name,
        )
        return

    MAX_CH = 7000
    system_batch = (
        'Eres analista de monitoreo de medios. Recibirás un JSON con ítems: idx, termino, motivo_sistema, transcripcion.\n'
        '"motivo_sistema" es el motivo técnico que ya dio el clasificador; "transcripcion" es lo que se dijo en el medio (~audio).\n'
        'Para cada ítem, redacta UNA SOLA FRASE en español, tono profesional para el cliente:\n'
        'integrando transcripcion y motivo_sistema — qué ocurre en el contenido Y por qué el sistema lo marcó así.\n'
        'Prioriza parafrasear la transcripción; no inventes hechos que no puedan inferirse de transcripcion+motivo_sistema.\n'
        'Si transcripcion está vacía o muy breve, resume motivo_sistema sin inventar diálogo.\n'
        'Una sola oración por ítem (sin punto aparte ni listas).\n'
        'Responde SOLO con un JSON array: [{"idx": número, "explicacion": "texto"}, ...], un objeto por ítem; '
        'idx es el número recibido en cada ítem.'
    )
    system_one = (
        'Mismo rol. Responde SOLO JSON: {"explicacion": "..."} '
        '(una sola frase en español que una transcripcion + motivo_sistema técnico; sin inventar; '
        'si no hay transcripcion, aclara motivo_sistema sin inventar diálogo).'
    )

    prepared = []
    for loc_idx, orig_idx in enumerate(pend_orig_idx):
        it = items_list[orig_idx]
        ev = ''
        if isinstance(it, dict):
            ev = (it.get('texto_evidencia') or '').strip()
        if len(ev) > MAX_CH:
            ev = ev[:MAX_CH] + '…'
        term = ''
        ms_tecnico = ''
        if isinstance(it, dict):
            term = (it.get('termino') or '').strip()
            ms_tecnico = ((it.get('motivo_sistema') or '').strip() or (it.get('motivo') or '').strip())
        prepared.append({'idx': loc_idx, 'termino': term, 'motivo_sistema': ms_tecnico, 'transcripcion': ev})

    def _parse_json_content(raw):
        raw = (raw or '').strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
            raw = re.sub(r'\s*```\s*$', '', raw)
        return raw

    try:
        user_payload = json.dumps(prepared, ensure_ascii=False)
        resp = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {'role': 'system', 'content': system_batch},
                {'role': 'user', 'content': user_payload},
            ],
            temperature=0.3,
        )
        raw = _parse_json_content(resp.choices[0].message.content)
        arr = json.loads(raw)
        if not isinstance(arr, list):
            raise ValueError('Respuesta DeepSeek no es lista')
        by_loc = {}
        for x in arr:
            if isinstance(x, dict) and 'idx' in x:
                try:
                    by_loc[int(x['idx'])] = (x.get('explicacion') or '').strip()
                except (TypeError, ValueError):
                    continue
        n_ok = 0
        for loc_idx, orig_idx in enumerate(pend_orig_idx):
            it = items_list[orig_idx]
            expl = _tangencial_motivo_una_frase(by_loc.get(loc_idx, ''))
            if expl:
                it['motivo_sistema'] = it.get('motivo', '')
                it['motivo'] = expl
                n_ok += 1
        log_info(f'DeepSeek tangenciales: enriquecidos {n_ok}/{len(pend_orig_idx)} pendientes de {len(items_list)} (lote)', func_name)
    except Exception as e_batch:
        log_warning(f'DeepSeek tangenciales lote falló ({e_batch}); ítem por ítem', func_name)
        n_ok = 0
        for loc_idx, orig_idx in enumerate(pend_orig_idx):
            it = items_list[orig_idx]
            try:
                user_one = json.dumps(prepared[loc_idx], ensure_ascii=False)
                resp = deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {'role': 'system', 'content': system_one},
                        {'role': 'user', 'content': user_one},
                    ],
                    temperature=0.3,
                )
                raw = _parse_json_content(resp.choices[0].message.content)
                obj = json.loads(raw)
                expl = _tangencial_motivo_una_frase(obj.get('explicacion') or '')
                if expl:
                    it['motivo_sistema'] = it.get('motivo', '')
                    it['motivo'] = expl
                    n_ok += 1
            except Exception as e_one:
                log_warning(f'DeepSeek tangencial ítem orig_idx={orig_idx} sin enriquecer: {e_one}', func_name)
        log_info(f'DeepSeek tangenciales: enriquecidos {n_ok}/{len(pend_orig_idx)} pendientes de {len(items_list)} (individual)', func_name)


def escanear_emisoras_entrada(carpeta_base=None, segundos_grabacion=None):
    """
    Lista emisoras (carpetas de primer nivel bajo la entrada de audios) y estima si
    hay grabación activa: última modificación de algún audio reciente (excluye c_clip_* y carpetas PROCESADO).

    Returns:
        list[dict]: nombre, ruta_completa, n_audios, ultima_actividad_ts, ultima_actividad_str,
                    grabando (bool), inactiva_desde_seg (float|None)
    """
    if carpeta_base is None:
        carpeta_base = CARPETA_VIDEOS
    if segundos_grabacion is None:
        try:
            segundos_grabacion = int(os.getenv("RADIO_GRABACION_SEGUNDOS", "120"))
        except ValueError:
            segundos_grabacion = 120

    def stats_audios_sin_clips_generados(carpeta):
        """max mtime y conteo de audios, ignorando subcarpetas de clips del analizador."""
        max_mtime = 0.0
        n = 0
        if not os.path.isdir(carpeta):
            return 0, 0.0
        for dirpath, dirnames, filenames in os.walk(carpeta):
            dirnames[:] = [d for d in dirnames if not d.startswith("c_clip_")]
            if os.path.exists(os.path.join(dirpath, "PROCESADO.txt")):
                continue
            for f in filenames:
                if not f.lower().endswith(AUDIO_EXTENSIONS):
                    continue
                full = os.path.join(dirpath, f)
                try:
                    st = os.stat(full)
                    max_mtime = max(max_mtime, st.st_mtime)
                    n += 1
                except OSError:
                    continue
        return n, max_mtime

    def stats_solo_raiz(carpeta):
        """Archivos de audio directamente en `carpeta` (sin subcarpetas)."""
        max_mtime = 0.0
        n = 0
        try:
            for f in os.listdir(carpeta):
                p = os.path.join(carpeta, f)
                if not os.path.isfile(p):
                    continue
                if not f.lower().endswith(AUDIO_EXTENSIONS):
                    continue
                try:
                    st = os.stat(p)
                    max_mtime = max(max_mtime, st.st_mtime)
                    n += 1
                except OSError:
                    continue
        except OSError:
            pass
        return n, max_mtime

    out = []
    base = os.path.abspath(carpeta_base)
    ahora = time.time()

    if not os.path.isdir(base):
        return out

    # Archivos sueltos en la raíz de entrada (emisora virtual)
    n_raiz, mt_raiz = stats_solo_raiz(base)
    if n_raiz > 0:
        delta = ahora - mt_raiz if mt_raiz else 999999
        out.append({
            "nombre": "(Raíz de entrada)",
            "ruta_completa": base,
            "n_audios": n_raiz,
            "ultima_actividad_ts": mt_raiz,
            "ultima_actividad_str": datetime.fromtimestamp(mt_raiz).strftime("%Y-%m-%d %H:%M:%S") if mt_raiz else "—",
            "grabando": mt_raiz > 0 and delta <= segundos_grabacion,
            "inactiva_desde_seg": delta if mt_raiz else None,
        })

    try:
        subdirs = sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
        )
    except OSError:
        subdirs = []

    for d in subdirs:
        path = os.path.join(base, d)
        n, mt = stats_audios_sin_clips_generados(path)
        nombre_legible = formatear_nombre_medio_desde_ruta(d)
        delta = ahora - mt if mt else 999999
        out.append({
            "nombre": nombre_legible,
            "nombre_carpeta": d,
            "ruta_completa": path,
            "n_audios": n,
            "ultima_actividad_ts": mt,
            "ultima_actividad_str": datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S") if mt and n > 0 else "—",
            "grabando": n > 0 and mt > 0 and delta <= segundos_grabacion,
            "inactiva_desde_seg": delta if n > 0 and mt else None,
        })

    return out


def formato_posicion_en_audio_segundos(ts_seg):
    """Posición en la línea de tiempo del archivo: XmYYs (audio)."""
    ts = float(ts_seg or 0)
    mm = int(ts // 60)
    ss = int(ts % 60)
    return f"{mm}m{ss:02d}s"


def tangencial_formato_momento_mencion_termino(item):
    """(texto legible, segundos exactos) para el instante de la mención en el archivo."""
    if not isinstance(item, dict):
        return ('—', 0.0)
    try:
        t = float(item.get('timestamp') or 0)
    except (TypeError, ValueError):
        t = 0.0
    if t <= 0:
        return ('—', 0.0)
    return (formato_posicion_en_audio_segundos(t), t)


def tangencial_momento_audio_texto_correo(item):
    """
    Texto único para correos: cuándo ocurre la mención del término dentro del archivo de audio
    (desde el inicio de la pista), p. ej. «12m34s — 754.0 s desde el inicio del archivo».
    """
    compact, tsec = tangencial_formato_momento_mencion_termino(item if isinstance(item, dict) else {})
    if tsec and tsec > 0:
        return f"{compact} — {tsec:.1f} s desde el inicio del archivo"
    return compact if compact != '—' else "momento en audio no disponible"


def _hora_deteccion_formateada(item):
    """Una sola convención: YYYY-MM-DD HH:MM:SS."""
    raw = item.get("hora_deteccion")
    if not raw:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if isinstance(raw, datetime):
            return raw.strftime("%Y-%m-%d %H:%M:%S")
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(raw)


def construir_tangenciales_narrativo(items_list):
    """
    Agrupa por término: lista compacta por **emisora** (sin enumerar cada clip)
    y remite el detalle a la tabla del correo.
    Devuelve (texto_plano, markdown, html_fragmento_cuerpo).
    """
    if not items_list:
        return "", "", ""

    from collections import defaultdict

    intro_md = (
        "### ⚠️ Menciones tangenciales (sin desarrollo)\n\n"
        "Estos términos se detectaron pero fueron rechazados para generación de clip por falta de profundidad temática:\n\n"
    )
    intro_txt = (
        "MENCIONES TANGENCIALES (sin desarrollo)\n\n"
        "Estos términos se detectaron pero fueron rechazados para generación de clip por falta de profundidad temática:\n\n"
    )
    intro_html = (
        '<p style="margin:0 0 16px 0;color:#4a5568;font-size:15px;line-height:1.65;">'
        "Estos términos se detectaron pero fueron <strong>rechazados</strong> para generación de clip por falta de "
        "profundidad temática:</p>"
    )

    def _nombre_emisor_agrupada(it):
        arch = it.get("archivo") or ""
        p = parsear_emision_radio_desde_archivo(arch)
        if p:
            return normalizar_nombre_emisora_correo(p["emisora_raw"])
        return formatear_nombre_medio_desde_ruta(arch)

    by_term = defaultdict(list)
    for it in items_list:
        t = (it.get("termino") or "").strip() or "(sin término)"
        by_term[t].append(it)

    bloques_md = []
    bloques_txt = []
    bloques_html = []

    for termino in sorted(by_term.keys(), key=lambda x: x.lower()):
        termino_disp = capitalizar_marcas_medios_rd_en_texto(str(termino))
        occ = by_term[termino]
        emisoras_orden = []
        visto = set()
        for it in occ:
            nm = _nombre_emisor_agrupada(it)
            k = nm.strip().lower()
            if k not in visto:
                visto.add(k)
                emisoras_orden.append(nm)
        emisoras_orden.sort(key=lambda x: x.lower())
        lista_union = ", ".join(emisoras_orden)

        bloques_md.append(
            f"- **{termino_disp}**: tangencial en **{len(emisoras_orden)}** emisoras: {lista_union} "
            f"(**{len(occ)}** menciones sin clip). Detalle por emisión normalizada y motivo en la tabla.\n"
        )
        bloques_txt.append(
            f"{termino_disp}:\n"
            f"Emisoras ({len(emisoras_orden)}): {lista_union}. Total menciones sin clip: {len(occ)}.\n"
            f"Consulte la tabla para fecha/hora de cada segmento y motivos.\n\n"
        )
        lista_esc = html_module.escape(capitalizar_marcas_medios_rd_en_texto(lista_union))
        bloques_html.append(
            f'<p style="margin:14px 0;line-height:1.65;font-size:15px;color:#2d3748;">'
            f'<strong style="font-weight:600;color:#2d3748;">{html_module.escape(termino_disp)}</strong>: '
            f"tangencial en <strong>{len(emisoras_orden)}</strong> emisoras: {lista_esc} "
            f"(<strong>{len(occ)}</strong> menciones; fecha/hora y motivos en la tabla).</p>"
        )

    md = intro_md + "".join(bloques_md) + "\n*Implicaciones:* revisar si el término merece monitoreo más estricto.\n"
    txt = intro_txt + "".join(bloques_txt) + "Implicaciones: revisar si el término merece monitoreo más estricto.\n"
    html_body = intro_html + "".join(bloques_html)
    return txt, md, html_body


def _destinatarios_tangencial_fallback_global():
    """Destinatarios si el cliente no tiene lista: correos_guardados + recipient de brevo global."""
    out = []
    for c in cargar_correos_guardados():
        if not c.get('activo', True):
            continue
        addr = (c.get('email') or c.get('correo') or '').strip()
        if addr:
            out.append(addr)
    rec = (cargar_brevo_config().get('recipient_email') or '').strip()
    if rec and rec not in out:
        out.append(rec)
    return out


def brevo_config_para_tangencial(cliente):
    """
    Configuración efectiva para correos de tangenciales (obligatorios por producto):
    fusiona brevo del cliente con brevo_config.json y destinatarios globales.

    Se envía si existen API key, remitente y al menos un destinatario, aunque el toggle
    «Brevo habilitado» del cliente esté desactivado. Si faltan datos, retorna None.
    """
    cliente = cliente if isinstance(cliente, dict) else {}
    g = cargar_brevo_config()
    bc = dict(cliente.get('brevo') or {})

    def _pick_str(*candidatos):
        for v in candidatos:
            if v is None:
                continue
            s = v.strip() if isinstance(v, str) else str(v).strip()
            if s:
                return s
        return ''

    api_key = _pick_str(bc.get('api_key', ''), g.get('api_key', ''))
    sender_email = _pick_str(bc.get('sender_email', ''), g.get('sender_email', ''))
    sender_name = _pick_str(
        bc.get('sender_name', ''),
        g.get('sender_name', ''),
        f"Sistema {nombre_cliente_mostrar_para_ui(cliente)}",
    )
    smtp_server = _pick_str(bc.get('smtp_server', ''), g.get('smtp_server', ''), 'smtp-relay.brevo.com')
    smtp_user = _pick_str(bc.get('smtp_user', ''), bc.get('sender_email', ''), g.get('smtp_user', ''), sender_email)
    try:
        smtp_port = int(bc.get('smtp_port') or g.get('smtp_port') or 587)
    except (TypeError, ValueError):
        smtp_port = 587

    correos_destinatarios, _ = obtener_destinatarios_activos_cliente(cliente)
    if not correos_destinatarios:
        correos_destinatarios = _destinatarios_tangencial_fallback_global()

    if not api_key or not sender_email or not correos_destinatarios:
        return None

    return {
        'api_key': api_key,
        'sender_email': sender_email,
        'sender_name': sender_name,
        'smtp_user': smtp_user,
        'smtp_server': smtp_server,
        'smtp_port': smtp_port,
        'correos_destinatarios': correos_destinatarios,
    }


def crear_plantilla_email_tangenciales_html(cliente, items_list, hora_ciclo, aviso_inmediato=False):
    """
    Plantilla HTML: resumen narrativo por término + tabla de apoyo (medio, motivo, tiempos).
    Si aviso_inmediato, añade nota de resumen al cierre del ciclo y columna de enlaces Drive.
    """
    primary = cliente.get('color') or '#667eea'
    secondary = '#764ba2'
    nombre_cliente = html_module.escape(nombre_cliente_mostrar_para_ui(cliente))
    subtitulo_plain = (
        f"Aviso inmediato — {hora_ciclo}"
        if aviso_inmediato
        else f"Cierre de ciclo: {hora_ciclo}"
    )
    nota_inmediata = ''
    if aviso_inmediato:
        nota_inmediata = (
            '<p style="margin:0 0 16px 0;padding:12px 14px;background:#fff3cd;border-radius:10px;'
            'border-left:4px solid #ffc107;color:#856404;font-size:13px;line-height:1.55;">'
            'Al cierre del ciclo de escaneo recibirá un correo adicional con el <strong>resumen de todas</strong> '
            'las tangenciales acumuladas en el lote.</p>'
        )
    _, _, cuerpo_narrativo_html = construir_tangenciales_narrativo(items_list)

    th_enlaces = ''
    if aviso_inmediato:
        th_enlaces = (
            f'<th style="padding:12px;text-align:left;border-bottom:2px solid {primary};color:#343a40;">'
            'Enlaces (Drive)</th>'
        )

    filas = []
    for i, tang in enumerate(items_list, 1):
        momento_audio_correo = tangencial_momento_audio_texto_correo(tang)
        tiempo_audio_html = html_module.escape(momento_audio_correo)
        medio = html_module.escape(capitalizar_marcas_medios_rd_en_texto(formato_linea_emision_legible(tang.get('archivo', ''))))
        term = html_module.escape(capitalizar_marcas_medios_rd_en_texto(str(tang.get('termino', ''))))
        motivo_pri = capitalizar_marcas_medios_rd_en_texto(str(tang.get('motivo', '')))
        ms = (tang.get('motivo_sistema') or '').strip()
        if ms:
            ms_disp = capitalizar_marcas_medios_rd_en_texto(ms)
            if ms_disp.strip() and ms_disp.strip() != motivo_pri.strip():
                motivo = (
                    f"{html_module.escape(motivo_pri)}<br>"
                    f"<span style=\"font-size:12px;color:#6c757d;line-height:1.4;\">"
                    f"Ref. técnica (clasificador): {html_module.escape(ms_disp)}</span>"
                )
            else:
                motivo = html_module.escape(motivo_pri)
        else:
            motivo = html_module.escape(motivo_pri)
        hd = html_module.escape(_hora_deteccion_formateada(tang))
        bg = '#f8f9fa' if i % 2 == 1 else '#ffffff'
        td_enlaces = ''
        if aviso_inmediato:
            ua = (tang.get('gdrive_url_audio') or '').strip()
            ut = (tang.get('gdrive_url_txt') or '').strip()
            parts = []
            if ua:
                parts.append(
                    f'<a href="{html_module.escape(ua, quote=True)}" style="color:{primary};">Audio</a>'
                )
            if ut:
                parts.append(
                    f'<a href="{html_module.escape(ut, quote=True)}" style="color:{primary};">Transcripción</a>'
                )
            enl = ' · '.join(parts) if parts else '—'
            td_enlaces = f'<td style="padding:14px 12px;border-bottom:1px solid #e9ecef;font-size:13px;">{enl}</td>'
        filas.append(f"""
            <tr style="background:{bg};">
                <td style="padding:14px 12px;border-bottom:1px solid #e9ecef;font-size:13px;">{medio}</td>
                <td style="padding:14px 12px;border-bottom:1px solid #e9ecef;"><span style="background:linear-gradient(135deg,{primary}22 0%,{secondary}22 100%);color:#333;padding:6px 12px;border-radius:20px;font-weight:600;">{term}</span></td>
                <td style="padding:14px 12px;border-bottom:1px solid #e9ecef;font-size:13px;color:#495057;">{motivo}</td>
                <td style="padding:14px 12px;border-bottom:1px solid #e9ecef;font-family:Consolas,monospace;font-size:13px;line-height:1.45;color:#212529;">{tiempo_audio_html}</td>
                <td style="padding:14px 12px;border-bottom:1px solid #e9ecef;white-space:nowrap;font-size:13px;color:#495057;">{hd}</td>{td_enlaces}
            </tr>""")

    tbody = "".join(filas)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Menciones tangenciales — {nombre_cliente}</title>
</head>
<body style="margin:0;padding:0;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#eef1f5;">
    <div style="max-width:820px;margin:0 auto;padding:24px;">
        <div style="background:#fff;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,0.08);overflow:hidden;">
            <div style="background:linear-gradient(135deg, {primary} 0%, {secondary} 100%);color:#fff;padding:28px 32px;text-align:center;">
                <p style="margin:0 0 8px 0;font-size:13px;letter-spacing:0.08em;opacity:0.95;text-transform:uppercase;">Monitoreo de audio</p>
                <h1 style="margin:0;font-size:26px;font-weight:600;">Menciones tangenciales</h1>
                <p style="margin:12px 0 0 0;font-size:16px;opacity:0.95;">{nombre_cliente}</p>
                <p style="margin:8px 0 0 0;font-size:14px;opacity:0.88;">{html_module.escape(subtitulo_plain)}</p>
            </div>
            <div style="padding:28px 24px 8px 24px;">
                {nota_inmediata}
                <p style="margin:0 0 20px 0;color:#6c757d;font-size:14px;line-height:1.6;">
                    Segmentos <strong>no enviados como alerta con clip</strong> (relevancia baja o término sin desarrollo suficiente en el audio).
                </p>
                <div style="margin-bottom:24px;">{cuerpo_narrativo_html}</div>
                <p style="margin:0 0 12px 0;color:#343a40;font-size:14px;font-weight:600;">Detalle por ocurrencia</p>
                <div style="overflow-x:auto;border-radius:12px;border:1px solid #e9ecef;">
                    <table style="width:100%;border-collapse:collapse;font-size:14px;">
                        <thead>
                            <tr style="background:linear-gradient(180deg,#f8f9fa 0%,#eef1f5 100%);">
                                <th style="padding:12px;text-align:left;border-bottom:2px solid {primary};color:#343a40;">Emisión</th>
                                <th style="padding:12px;text-align:left;border-bottom:2px solid {primary};color:#343a40;">Término</th>
                                <th style="padding:12px;text-align:left;border-bottom:2px solid {primary};color:#343a40;">Motivo</th>
                                <th style="padding:12px;text-align:left;border-bottom:2px solid {primary};color:#343a40;">Momento del término en el audio</th>
                                <th style="padding:12px;text-align:left;border-bottom:2px solid {primary};color:#343a40;">Hora detección</th>{th_enlaces}
                            </tr>
                        </thead>
                        <tbody>
                            {tbody}
                        </tbody>
                    </table>
                </div>
            </div>
            <div style="background:#343a40;color:#ced4da;padding:20px 28px;text-align:center;font-size:12px;line-height:1.6;">
                <p style="margin:0;">Correo automático del sistema de análisis de audios. No responda a este mensaje.</p>
                <p style="margin:8px 0 0 0;opacity:0.85;">Generado {html_module.escape(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}</p>
            </div>
        </div>
    </div>
</body>
</html>"""


def enviar_brevo_menciones_tangenciales_cliente(cliente, items_list, hora_ciclo=None):
    """
    Envía un correo solo con menciones tangenciales del ciclo para un cliente (Brevo SMTP).
    Obligatorio cuando hay ítems: usa credenciales del cliente fusionadas con brevo_config.json;
    no se omite por «Brevo deshabilitado» ni por envíos pausados del cliente.
    """
    func_name = "enviar_brevo_menciones_tangenciales_cliente"
    if not items_list:
        return False, "Sin ítems"

    brevo_config = brevo_config_para_tangencial(cliente)
    if not brevo_config:
        return (
            False,
            "Tangenciales: correo obligatorio pero Brevo incompleto (api_key, remitente o destinatarios). "
            "Revise brevo_config.json, correos guardados o el cliente en la app.",
        )

    api_key = brevo_config['api_key']
    sender_email = brevo_config['sender_email']
    sender_name = brevo_config['sender_name']
    correos_destinatarios = brevo_config['correos_destinatarios']

    smtp_user = brevo_config['smtp_user']
    smtp_server = brevo_config['smtp_server']
    smtp_port = brevo_config['smtp_port']

    if hora_ciclo is None:
        hora_ciclo = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    sender_name = capitalizar_marcas_medios_rd_en_texto(sender_name)
    html_content = crear_plantilla_email_tangenciales_html(cliente, items_list, hora_ciclo)

    txt_narr, _, _ = construir_tangenciales_narrativo(items_list)
    lineas_txt = [
        f"Menciones tangenciales (monitoreo) — {cliente_nombre}",
        f"Cierre de ciclo: {hora_ciclo}",
        "",
        txt_narr,
        "",
        "Detalle por ocurrencia:",
        "",
    ]
    for i, tang in enumerate(items_list, 1):
        medio = formato_linea_emision_legible(tang.get('archivo', ''))
        hd = _hora_deteccion_formateada(tang)
        mt = (tang.get('motivo') or '').strip()
        ms = (tang.get('motivo_sistema') or '').strip()
        det_motivo = f"motivo: {mt}"
        if ms and ms != mt:
            det_motivo += f" | ref. técnica: {ms}"
        momento_arch = tangencial_momento_audio_texto_correo(tang)
        lineas_txt.append(
            f"{i}. emisión: {medio} | término: {tang.get('termino', '')} | "
            f"MOMENTO DEL TÉRMINO EN EL AUDIO: {momento_arch} | {det_motivo} | hora detección (sistema): {hd}"
        )
    lineas_txt.extend(["", f"— {cliente_nombre} | Sistema de análisis de audios"])
    text_content = capitalizar_marcas_medios_rd_en_texto("\n".join(lineas_txt))

    asunto = f"{EMAIL_ASUNTO_PREFIJO_RADIO}Menciones tangenciales (monitoreo) — {cliente_nombre} — {hora_ciclo}"

    try:
        msg = MIMEMultipart('mixed')
        msg['Subject'] = asunto
        msg['From'] = f"{sender_name} <{sender_email}>"
        msg['To'] = correos_destinatarios[0]
        if len(correos_destinatarios) > 1:
            msg['Bcc'] = ', '.join(correos_destinatarios[1:])

        alt = MIMEMultipart('alternative')
        alt.attach(MIMEText(text_content, 'plain', 'utf-8'))
        alt.attach(MIMEText(html_content, 'html', 'utf-8'))
        msg.attach(alt)

        n_adj = 0
        for tang in items_list:
            cp = tang.get('clip_path')
            if not cp or not isinstance(cp, str):
                continue
            if not os.path.isfile(cp) or not cp.lower().endswith('.mp3'):
                continue
            try:
                with open(cp, 'rb') as fh:
                    part_audio = MIMEBase('audio', 'mpeg')
                    part_audio.set_payload(fh.read())
                encoders.encode_base64(part_audio)
                part_audio.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=os.path.basename(cp),
                )
                msg.attach(part_audio)
                n_adj += 1
            except Exception as e_att:
                log_warning(f"Tangencial: no se adjuntó {cp}: {e_att}", func_name)

        log_info(
            f"Enviando correo tangenciales Brevo: {cliente_nombre} → {len(correos_destinatarios)} dest., adjuntos MP3: {n_adj}",
            func_name,
        )
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, api_key)
            server.send_message(msg)

        log_info(f"Correo tangenciales enviado a {cliente_nombre}", func_name)
        return True, f"Correo enviado a {len(correos_destinatarios)} destinatarios"
    except Exception as e:
        log_exception(func_name, e)
        return False, f"Error: {str(e)}"


def enviar_correos_tangenciales_fin_ciclo(menciones_tangenciales_data):
    """
    Agrupa menciones tangenciales por cliente (obtener_cliente_por_termino) y envía un correo por entidad.
    Retorna lista de tuplas (nombre_cliente, ok, mensaje).
    """
    func_name = "enviar_correos_tangenciales_fin_ciclo"
    if not menciones_tangenciales_data:
        return []

    hora_ciclo = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    grupos = {}
    for item in menciones_tangenciales_data:
        cliente = obtener_cliente_por_termino(item.get('termino', ''))
        cid = cliente.get('id', 'default')
        if cid not in grupos:
            grupos[cid] = {'cliente': cliente, 'items': []}
        grupos[cid]['items'].append(item)

    resultados = []
    for cid, data in grupos.items():
        c = data['cliente']
        nombre = nombre_cliente_mostrar_para_ui(c, cid)
        # Tangenciales y motivos se envían siempre por correo si Brevo está configurado (no se omiten por envíos pausados).
        ok, msg = enviar_brevo_menciones_tangenciales_cliente(c, data['items'], hora_ciclo)
        resultados.append((nombre, ok, msg))
        if ok:
            log_info(f"✅ Correo tangenciales: {nombre} — {msg}", func_name)
        else:
            log_info(f"⏭️ Correo tangenciales omitido o error ({nombre}): {msg}", func_name)

    return resultados


def enviar_brevo_tangencial_inmediato(cliente, item_tangencial):
    """
    Envía un correo Brevo en el acto con una sola tangencial (motivo + transcripción + Drive).
    Al cierre del ciclo sigue enviándose el resumen vía enviar_correos_tangenciales_fin_ciclo.
    Obligatorio: fusiona credenciales con brevo global; no se bloquea por toggle «enabled» del cliente.
    """
    func_name = 'enviar_brevo_tangencial_inmediato'
    if not item_tangencial:
        return False, 'Sin ítem'

    brevo_config = brevo_config_para_tangencial(cliente)
    if not brevo_config:
        return (
            False,
            'Tangencial inmediata: Brevo incompleto (api_key, remitente o destinatarios). '
            'Configure brevo_config.json o el cliente.',
        )

    api_key = brevo_config['api_key']
    sender_email = brevo_config['sender_email']
    sender_name = brevo_config['sender_name']
    correos_destinatarios = brevo_config['correos_destinatarios']
    smtp_user = brevo_config['smtp_user']
    smtp_server = brevo_config['smtp_server']
    smtp_port = brevo_config['smtp_port']

    hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    term_label = str(item_tangencial.get('termino') or '?')[:120]

    html_content = crear_plantilla_email_tangenciales_html(cliente, [item_tangencial], hora, aviso_inmediato=True)

    linea_momento = (
        f"Momento del término en el archivo de audio: {tangencial_momento_audio_texto_correo(item_tangencial)}"
    )
    ua = (item_tangencial.get('gdrive_url_audio') or '').strip()
    ut = (item_tangencial.get('gdrive_url_txt') or '').strip()
    ext = ((item_tangencial.get('texto_evidencia') or '')[:600]).strip()
    frase_corr = motivo_tangencial_una_frase(item_tangencial)
    ref_tecnico = ((item_tangencial.get('motivo_sistema') or '').strip() or motivo_display_tangencial(item_tangencial))
    emision_legible = formato_linea_emision_legible(item_tangencial.get('archivo', ''))

    lineas_txt = [
        f'Tangencial (aviso inmediato) — {cliente_nombre}',
        f'Detectado por el sistema: {hora}',
        '',
        '═══ Ubicación en el audio ═══',
        linea_momento,
        '(Tiempo desde el inicio del archivo hasta donde se detectó la mención del término.)',
        '',
        'Emisión:',
        emision_legible,
        '',
        '¿POR QUÉ ES TANGENCIAL? (una frase)',
        frase_corr,
        '',
        '— Motivo de referencia (análisis técnico):',
        ref_tecnico,
        '',
        f"Archivo: {item_tangencial.get('archivo', '')}",
        f'Término: {term_label}',
    ]
    if ua:
        lineas_txt.append(f'Audio (Drive): {ua}')
    if ut:
        lineas_txt.append(f'Transcripción (Drive): {ut}')
    if ext:
        lineas_txt.append(f'Extracto transcripción: {ext}')
    lineas_txt.extend([
        '',
        'Al cierre del ciclo de escaneo recibirá un correo adicional con el resumen de todas las tangenciales del lote.',
        '',
        f'— {cliente_nombre} | Sistema de análisis de audios',
    ])
    text_content = capitalizar_marcas_medios_rd_en_texto('\n'.join(lineas_txt))
    _ta_c, _ta_sec = tangencial_formato_momento_mencion_termino(item_tangencial)
    sufijo_audio_asunto = f' — en audio {_ta_c} (~{_ta_sec:.0f}s)' if (_ta_sec and _ta_sec > 0) else ''
    asunto = (
        f"{EMAIL_ASUNTO_PREFIJO_RADIO}Tangencial — {cliente_nombre} — «{term_label}»{sufijo_audio_asunto} — {hora}"
    )
    sender_name = capitalizar_marcas_medios_rd_en_texto(sender_name)

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = asunto
        msg['From'] = f'{sender_name} <{sender_email}>'
        msg['To'] = correos_destinatarios[0]
        if len(correos_destinatarios) > 1:
            msg['Bcc'] = ', '.join(correos_destinatarios[1:])

        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))

        log_info(
            f'Enviando correo tangencial INMEDIATO Brevo: {cliente_nombre} — término «{term_label}» → {len(correos_destinatarios)} dest.',
            func_name,
        )
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, api_key)
            server.send_message(msg)

        log_info(f'Correo tangencial inmediato enviado a {cliente_nombre}', func_name)
        return True, f'Correo inmediato a {len(correos_destinatarios)} destinatarios'
    except Exception as e:
        log_exception(func_name, e)
        return False, f'Error: {str(e)}'


def notificar_brevo_tangencial_inmediato_si(cliente, item_tangencial, caller_func_name):
    """Intenta enviar Brevo inmediato sin interrumpir el bucle de análisis.

    Tangenciales deben notificarse por correo si hay configuración SMTP; no se omiten por envíos pausados.

    Returns:
        None si no hubo intento (sin ítem o sin cliente).
        (True, msg) si el envío se realizó correctamente.
        (False, msg) si Brevo incompleto, error SMTP u otra causa.
    """
    if not item_tangencial or not cliente:
        return None
    try:
        ok, msg = enviar_brevo_tangencial_inmediato(cliente, item_tangencial)
        if ok:
            log_info(f'📧 Tangencial inmediata Brevo OK ({nombre_cliente_mostrar_para_ui(cliente)}): {msg}', caller_func_name)
        else:
            log_warning(f'⚠️ Tangencial inmediata Brevo omitida/error: {msg}', caller_func_name)
        return ok, msg
    except Exception as e:
        log_warning(f'⚠️ Tangencial inmediata Brevo excepción: {e}', caller_func_name)
        return False, str(e)


def enviar_gdrive_cliente(cliente, archivo_path, nombre_archivo=None):
    """Sube archivo a Google Drive usando la carpeta del cliente"""
    func_name = "enviar_gdrive_cliente"
    
    gdrive_config = cliente.get('google_drive', {})
    if not gdrive_config.get('enabled', False):
        return False, "Google Drive deshabilitado para este cliente", None
    
    folder_id = gdrive_config.get('folder_id', '')
    if not folder_id:
        # Usar carpeta por defecto si el cliente no tiene una específica
        folder_id = GOOGLE_DRIVE_FOLDER_ID
    
    if not folder_id:
        return False, "No hay carpeta de Google Drive configurada", None
    
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    log_info(f"Subiendo a Google Drive para cliente: {cliente_nombre} (carpeta: {folder_id})", func_name)
    
    try:
        resultado, mensaje = subir_archivo_google_drive(archivo_path, nombre_archivo, folder_id=folder_id)
        if resultado:
            return True, mensaje, resultado.get('id')
        return False, mensaje, None
    except Exception as e:
        log_exception(func_name, e)
        return False, f"Error: {str(e)}", None

def verificar_crear_tabla_supabase(cliente_supabase, cliente_nombre, tabla_nombre="alertasmediosintrant"):
    """
    Verifica si la tabla existe y la crea si no existe.
    Retorna True si la tabla existe o fue creada exitosamente.
    
    Args:
        cliente_supabase: Cliente de Supabase
        cliente_nombre: Nombre del cliente para el campo default
        tabla_nombre: Nombre de la tabla a verificar/crear
    """
    func_name = "verificar_crear_tabla_supabase"
    
    try:
        # Intentar hacer una consulta simple para verificar si la tabla existe
        result = cliente_supabase.table(tabla_nombre).select('id').limit(1).execute()
        log_info(f"✅ Tabla '{tabla_nombre}' existe para cliente: {cliente_nombre}", func_name)
        return True, "Tabla existente"
    except Exception as e:
        error_str = str(e).lower()
        
        # Si el error indica que la tabla no existe, intentar crearla
        if 'does not exist' in error_str or 'relation' in error_str or '42P01' in str(e):
            log_info(f"⚠️ Tabla '{tabla_nombre}' no existe para {cliente_nombre}, intentando crear...", func_name)
            
            try:
                # Crear la tabla usando SQL directo via RPC o la API de Supabase
                # Nota: Esto requiere que el usuario tenga permisos para crear tablas
                # o que la tabla se cree manualmente en Supabase Dashboard
                
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS {tabla_nombre} (
                    id BIGSERIAL PRIMARY KEY,
                    termino_detectado TEXT,
                    nombre_archivo TEXT,
                    tipo_archivo TEXT,
                    contexto TEXT,
                    resumen_ejecutivo TEXT,
                    fecha_detencion TIMESTAMPTZ DEFAULT NOW(),
                    url_video TEXT,
                    enlace_directo TEXT,
                    cliente TEXT DEFAULT '{cliente_nombre}',
                    transcripcion TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
                
                # Intentar crear via RPC si está disponible
                try:
                    cliente_supabase.rpc('exec_sql', {'sql': create_sql}).execute()
                    log_info(f"✅ Tabla '{tabla_nombre}' creada exitosamente para {cliente_nombre}", func_name)
                    return True, "Tabla creada"
                except:
                    # Si no hay RPC, mostrar instrucciones
                    log_warning(f"No se pudo crear la tabla automáticamente. Crear manualmente en Supabase Dashboard.", func_name)
                    return False, f"Crear tabla '{tabla_nombre}' manualmente en Supabase Dashboard"
                    
            except Exception as create_error:
                log_exception(func_name, create_error, f"Error creando tabla para {cliente_nombre}")
                return False, f"Error creando tabla: {str(create_error)}"
        else:
            # Otro tipo de error
            log_exception(func_name, e, f"Error verificando tabla para {cliente_nombre}")
            return False, f"Error: {str(e)}"

def enviar_supabase_cliente(cliente, coincidencias_items, nombre_archivo, tipo_archivo, resumen_archivo="", transcripcion_completa="", url_video=None, enlace_directo=None):
    """
    Envía coincidencias a Supabase usando credenciales del cliente.
    Usa la tabla configurada en el cliente (por defecto 'alertasmediosintrant').
    """
    func_name = "enviar_supabase_cliente"
    
    # Obtener nombre de tabla del cliente (default: alertasmediosintrant)
    supabase_config = cliente.get('supabase', {})
    tabla_nombre = supabase_config.get('tabla_nombre', 'alertasmediosintrant')
    
    # Limpiar nombre de tabla por si tiene caracteres especiales
    if tabla_nombre.startswith("📝") or tabla_nombre == "alertasmediosintrant":
        tabla_nombre = "alertasmediosintrant"
    
    supabase_config = cliente.get('supabase', {})
    if not supabase_config.get('enabled', False):
        return False, "Supabase deshabilitado para este cliente"
    
    supabase_url = supabase_config.get('url', '')
    supabase_key = supabase_config.get('anon_key', '')
    
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    resumen_archivo = capitalizar_marcas_medios_rd_en_texto(resumen_archivo or "")
    transcripcion_completa = capitalizar_marcas_medios_rd_en_texto(transcripcion_completa or "")
    nombre_archivo = capitalizar_marcas_medios_rd_en_texto(nombre_archivo or "")

    # Si no hay credenciales del cliente, usar las globales
    if not supabase_url or not supabase_key:
        if supabase:
            log_info(f"Usando Supabase global para cliente: {cliente_nombre} (tabla: {tabla_nombre})", func_name)
            # Verificar/crear tabla en Supabase global
            tabla_ok, tabla_msg = verificar_crear_tabla_supabase(supabase, cliente_nombre, tabla_nombre)
            if not tabla_ok:
                st.warning(f"⚠️ {tabla_msg}")
            
            # Insertar en la tabla alertasmediosintrant
            try:
                for item in coincidencias_items:
                    data = {
                        'termino_detectado': capitalizar_marcas_medios_rd_en_texto(str(item.get('termino', ''))),
                        'nombre_archivo': nombre_archivo,
                        'tipo_archivo': tipo_archivo,
                        'contexto': capitalizar_marcas_medios_rd_en_texto(str(item.get('contexto', '')))[:500],
                        'resumen_ejecutivo': resumen_archivo[:1000] if resumen_archivo else '',
                        'fecha_detencion': datetime.now().isoformat(),
                        'url_video': url_video,
                        'enlace_directo': enlace_directo,
                        'cliente': cliente_nombre,
                        'transcripcion': transcripcion_completa[:2000] if transcripcion_completa else ''
                    }
                    log_info(f"📤 Insertando en Supabase (Global) - Tabla: {tabla_nombre} | Clip: {url_video}", func_name)
                    supabase.table(tabla_nombre).insert(data).execute()
                
                return True, f"Insertadas {len(coincidencias_items)} coincidencias en '{tabla_nombre}'"
            except Exception as e:
                log_exception(func_name, e)
                return False, f"Error insertando: {str(e)}"
        return False, "Supabase no configurado"
    
    try:
        # Crear cliente de Supabase específico para este cliente
        cliente_supabase = create_client(supabase_url, supabase_key)
        
        log_info(f"Enviando a Supabase para cliente: {cliente_nombre}", func_name)
        
        # Verificar/crear tabla automáticamente
        tabla_ok, tabla_msg = verificar_crear_tabla_supabase(cliente_supabase, cliente_nombre, tabla_nombre)
        if not tabla_ok:
            st.warning(f"⚠️ Tabla '{tabla_nombre}': {tabla_msg}")
            # Continuar de todos modos, puede que la tabla exista pero con otro esquema
        
        # Obtener las columnas reales de la tabla para enviar solo lo que existe
        columnas_reales = []
        try:
            res_col = cliente_supabase.table(tabla_nombre).select("*").limit(1).execute()
            if res_col.data:
                columnas_reales = list(res_col.data[0].keys())
                log_info(f"📋 Columnas detectadas en '{tabla_nombre}': {columnas_reales}", func_name)
        except Exception as col_err:
            log_warning(f"No se pudieron detectar columnas de '{tabla_nombre}': {col_err}", func_name)
        
        # Insertar coincidencias en la tabla alertasmediosintrant
        insertadas = 0
        for item in coincidencias_items:
            try:
                data = {
                    'termino_detectado': capitalizar_marcas_medios_rd_en_texto(str(item.get('termino', ''))),
                    'nombre_archivo': nombre_archivo,
                    'tipo_archivo': tipo_archivo,
                    'contexto': capitalizar_marcas_medios_rd_en_texto(str(item.get('contexto', '')))[:500],
                    'resumen_ejecutivo': resumen_archivo[:1000] if resumen_archivo else '',
                    'fecha_detencion': datetime.now().isoformat(),
                    'url_video': url_video,
                    'enlace_directo': enlace_directo,
                    'cliente': cliente_nombre,
                    'transcripcion': transcripcion_completa[:2000] if transcripcion_completa else ''
                }
                
                try:
                    # Filtrar data para enviar solo columnas que existen en la tabla
                    if columnas_reales:
                        data_filtrada = {k: v for k, v in data.items() if k in columnas_reales}
                        
                        # Mapeos especiales si faltan columnas pero hay alternativas
                        if 'cliente' not in data_filtrada and 'nombre_medio' in columnas_reales:
                            data_filtrada['nombre_medio'] = cliente_nombre
                            
                        data_final = data_filtrada
                    else:
                        data_final = data
                        
                    log_info(f"📤 Insertando en Supabase (Cliente) - Tabla: {tabla_nombre} | Datos: {list(data_final.keys())}", func_name)
                    cliente_supabase.table(tabla_nombre).insert(data_final).execute()
                except Exception as e:
                    log_warning(f"Error final en inserción Supabase: {e}", func_name)
                    raise e
                
                insertadas += 1
                log_info(f"✅ Coincidencia insertada en '{tabla_nombre}' para {cliente_nombre}", func_name)
                
            except Exception as insert_error:
                log_warning(f"Error insertando coincidencia: {insert_error}", func_name)
                continue
        
        if insertadas > 0:
            return True, f"Insertadas {insertadas}/{len(coincidencias_items)} coincidencias en '{tabla_nombre}'"
        else:
            return False, "No se insertaron coincidencias"
        
    except Exception as e:
        log_exception(func_name, e)
        return False, f"Error: {str(e)}"

def enviar_coincidencia_a_cliente(cliente, nombre_archivo, termino_encontrado, contexto_termino, tipo_archivo, clip_path=None, transcripcion_completa="", timestamp=None, idea_general=None, video_url=None, video_path=None, transcripcion_segmento=""):
    """
    Envía una coincidencia a TODOS los destinos configurados para un cliente específico.
    Esta es la función principal que orquesta el envío multi-destino por cliente.
    """
    func_name = "enviar_coincidencia_a_cliente"
    
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    cliente_color = cliente.get('color', '#4CAF50')
    
    log_info(f"📤 Iniciando envío para cliente: {cliente_nombre} | Término: {termino_encontrado}", func_name)
    
    # Mostrar en UI a qué cliente se está enviando
    st.markdown(f"""
    <div style="background-color: {cliente_color}20; padding: 10px; border-radius: 10px; border-left: 4px solid {cliente_color}; margin: 10px 0;">
        <h4 style="margin: 0; color: {cliente_color};">📤 Enviando a: {cliente_nombre}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    resultados = {}

    # === RENOMBRAR CLIP CON PREFIJO cc{Termino}_ PARA IDENTIFICAR COINCIDENCIAS ===
    # Formato nuevo: ccTermino_CanalMedio_YYYYMMDD_HHMMSS_Xm00s.mp4
    # El canal/medio NUNCA se elimina del nombre.
    if clip_path and os.path.exists(clip_path):
        try:
            termino_prefijo = re.sub(r'[^\w]', '', termino_encontrado.strip())  # solo alfanumérico
            carpeta = os.path.dirname(clip_path)
            nombre_base = os.path.basename(clip_path)
            prefijo = f"cc{termino_prefijo}_"
            if not nombre_base.startswith(prefijo):
                # Extraer el medio/canal del nombre del archivo original
                # El nombre original viene como: CanalNombre_resolucion_fecha_seg###.mp4
                # O como: ts_termino_Xm00s.mp4 (clips ya generados)
                # Intentamos extraer el medio desde nombre_archivo (variable del scope padre)
                try:
                    info_medio = extraer_info_medio_hora(nombre_archivo)
                    canal = re.sub(r'[^\w]', '', info_medio.get('medio', 'Medio').strip())
                except Exception:
                    # Fallback: tomar la primera parte del nombre antes del primer '_'
                    canal = re.sub(r'[^\w]', '', nombre_base.split('_')[0])
                ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Extraer duración del nombre base si está en formato Xm00s
                duracion_match = re.search(r'(\d+m\d+s)', nombre_base)
                duracion_str = duracion_match.group(1) if duracion_match else ""
                nuevo_nombre = f"{prefijo}{canal}_{ts_now}{('_' + duracion_str) if duracion_str else ''}.mp4"
                nuevo_clip_path = os.path.join(carpeta, nuevo_nombre)
                os.rename(clip_path, nuevo_clip_path)
                clip_path = nuevo_clip_path
                log_info(f"✅ Clip renombrado: {nuevo_nombre}", func_name)
                st.success(f"🏷️ Archivo marcado: {nuevo_nombre}")
        except Exception as e_rename:
            log_warning(f"⚠️ No se pudo renombrar el clip: {e_rename}", func_name)

    # Extraer información del medio
    info_medio_hora = extraer_info_medio_hora(nombre_archivo)
    
    # === GENERAR RESUMEN EJECUTIVO COMPLETO Y ESTRUCTURADO ===
    # El resumen se envía TAL CUAL por todos los medios (Telegram, Email, Drive, etc.)
    resumen_ejecutivo = ""
    if idea_general:
        # idea_general ya viene estructurado con: Tema principal, Contexto, Puntos clave, Relevancia
        resumen_ejecutivo = idea_general
    elif transcripcion_completa and len(transcripcion_completa.strip()) > 100:
        try:
            resumen_ejecutivo = generar_resumen_archivo(nombre_archivo, [termino_encontrado], transcripcion_completa, tipo_archivo)
        except:
            resumen_ejecutivo = f"Tema principal: Detección de término \"{termino_encontrado}\"\n\nContexto: {contexto_termino[:300]}"
    else:
        resumen_ejecutivo = f"Tema principal: Detección de término \"{termino_encontrado}\"\n\nContexto: {contexto_termino[:300]}"

    termino_mostrar = capitalizar_marcas_medios_rd_en_texto(str(termino_encontrado))
    resumen_ejecutivo = capitalizar_marcas_medios_rd_en_texto(resumen_ejecutivo)
    if contexto_termino:
        contexto_termino = capitalizar_marcas_medios_rd_en_texto(contexto_termino)
    if transcripcion_completa and transcripcion_completa.strip():
        transcripcion_completa = capitalizar_marcas_medios_rd_en_texto(transcripcion_completa)
    if transcripcion_segmento and str(transcripcion_segmento).strip():
        transcripcion_segmento = capitalizar_marcas_medios_rd_en_texto(transcripcion_segmento)

    log_info(f"📤 Contenido listo para destinos externos | Término (presentación): {termino_mostrar}", func_name)

    # === CONSTRUIR REPORTE COMPLETO (ANÁLISIS COMPLETO) ===
    # El resumen_ejecutivo ya contiene las secciones formateadas (## 🎯 Resumen Ejecutivo,
    # ## 📈 Estadísticas, etc.), así que solo añadimos el encabezado principal.
    mensaje_completo = f"# 📊 ANÁLISIS COMPLETO: {nombre_archivo}\n\n{resumen_ejecutivo}"
    mensaje_completo = capitalizar_marcas_medios_rd_en_texto(mensaje_completo)
    
    # Para el correo: incluir resumen + transcripción completa sin truncar
    resumen_para_email = f"**RESUMEN EJECUTIVO:**\n{resumen_ejecutivo}\n\n**TRANSCRIPCIÓN DEL CONTENIDO:**\n{transcripcion_completa}" if transcripcion_completa and len(transcripcion_completa.strip()) > 50 else resumen_ejecutivo
    resumen_para_email = capitalizar_marcas_medios_rd_en_texto(resumen_para_email)
    
    # === 1. ENVIAR A TELEGRAM ===
    if cliente.get('telegram', {}).get('enabled'):
        try:
            with st.spinner(f"📱 Enviando a Telegram de {cliente_nombre}..."):
                exito, msg = enviar_telegram_cliente(cliente, mensaje_completo, clip_path, video_url=video_url)
                resultados['telegram'] = (exito, msg)
                if exito:
                    st.success(f"📱 Telegram: {msg}")
                else:
                    st.warning(f"📱 Telegram: {msg}")
        except Exception as e:
            log_error_critico(func_name, f"Error inesperado en Telegram: {e}")
            resultados['telegram'] = (False, f"Error: {e}")
    
    # === 2. ENVIAR A WEBHOOK ===
    if cliente.get('webhook', {}).get('enabled'):
        try:
            with st.spinner(f"🌐 Enviando a Webhook de {cliente_nombre}..."):
                exito, msg = enviar_webhook_cliente(cliente, clip_path or nombre_archivo, mensaje_completo, [termino_mostrar])
                resultados['webhook'] = (exito, msg)
                if exito:
                    st.success(f"🌐 Webhook: {msg}")
                else:
                    st.warning(f"🌐 Webhook: {msg}")
        except Exception as e:
            log_error_critico(func_name, f"Error inesperado en Webhook: {e}")
            resultados['webhook'] = (False, f"Error: {e}")
    
    # === 3. ENVIAR A BREVO (EMAIL) ===
    if cliente.get('brevo', {}).get('enabled'):
        try:
            with st.spinner(f"📧 Enviando correo de {cliente_nombre}..."):
                exito, msg = enviar_brevo_cliente(
                    cliente, termino_mostrar, resumen_para_email, 
                    nombre_archivo, clip_path, info_medio_hora, 
                    [termino_mostrar], video_url,
                    transcripcion_segmento=transcripcion_segmento
                )
                resultados['brevo'] = (exito, msg)
                if exito:
                    st.success(f"📧 Brevo: {msg}")
                else:
                    st.warning(f"📧 Brevo: {msg}")
        except Exception as e:
            log_error_critico(func_name, f"Error inesperado en Brevo: {e}")
            resultados['brevo'] = (False, f"Error: {e}")
    
    # === 4. ENVIAR A GOOGLE DRIVE ===
    if cliente.get('google_drive', {}).get('enabled'):
        with st.spinner(f"☁️ Subiendo a Google Drive de {cliente_nombre}..."):
            # Generar nombre base compartido para que el TXT y el clip hagan match
            gdrive_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            nombre_video_limpio = os.path.splitext(os.path.basename(nombre_archivo))[0]
            nombre_base_gdrive = f"{termino_mostrar}_{nombre_video_limpio}_{gdrive_timestamp}"
            
            # Subir resumen como TXT (incluye resumen ejecutivo + transcripción completa)
            try:
                txt_path = f"temp_resumen_{cliente_nombre.replace(' ', '_')}.txt"
                contenido_drive = f"COINCIDENCIA DETECTADA\n{'='*50}\n\n"
                contenido_drive += f"Medio: {info_medio_hora}\n"
                contenido_drive += f"Término: {termino_mostrar}\n"
                contenido_drive += f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                contenido_drive += f"Archivo: {nombre_archivo}\n\n"
                contenido_drive += f"{'='*50}\nRESUMEN EJECUTIVO:\n{'='*50}\n\n{resumen_ejecutivo}\n\n"
                if contexto_termino:
                    contenido_drive += f"{'='*50}\nCONTEXTO:\n{'='*50}\n\n{contexto_termino}\n\n"
                if transcripcion_completa and len(transcripcion_completa.strip()) > 50:
                    contenido_drive += f"{'='*50}\nTRANSCRIPCIÓN COMPLETA:\n{'='*50}\n\n{transcripcion_completa}\n"
                contenido_drive = capitalizar_marcas_medios_rd_en_texto(contenido_drive)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(contenido_drive)
                exito_txt, msg_txt, _ = enviar_gdrive_cliente(cliente, txt_path, f"{nombre_base_gdrive}.txt")
                os.remove(txt_path)
            except:
                exito_txt, msg_txt = False, "Error creando TXT"
            
            # Subir clip con el mismo nombre base
            exito_video = False
            msg_video = "Sin clip"
            if clip_path and os.path.exists(clip_path):
                exito_video, msg_video, _ = enviar_gdrive_cliente(cliente, clip_path, f"{nombre_base_gdrive}.mp4")
            
            resultados['google_drive'] = (exito_txt or exito_video, f"TXT: {msg_txt} | Clip: {msg_video}")
            if exito_txt or exito_video:
                st.success(f"☁️ Google Drive: TXT={'✅' if exito_txt else '❌'} Clip={'✅' if exito_video else '❌'}")
            else:
                st.warning(f"☁️ Google Drive: {msg_txt} | {msg_video}")
    
    # === 5. ENVIAR A SUPABASE ===
    if cliente.get('supabase', {}).get('enabled'):
        with st.spinner(f"🗄️ Guardando en Supabase de {cliente_nombre}..."):
            coincidencia_item = {
                'termino': termino_mostrar,
                'contexto': contexto_termino,
                'timestamp': timestamp
            }
            exito, msg = enviar_supabase_cliente(
                cliente, [coincidencia_item], nombre_archivo, tipo_archivo,
                resumen_ejecutivo, transcripcion_completa, video_url, video_url
            )
            resultados['supabase'] = (exito, msg)
            if exito:
                st.success(f"🗄️ Supabase: {msg}")
            else:
                st.warning(f"🗄️ Supabase: {msg}")
    
    
    # === 6. GOOGLE SHEETS (coincidencias por cliente: JSON o EDESUR/Intrant por .env) ===
    sheet_id, sheet_range = spreadsheet_y_rango_coincidencias_cliente(cliente)
    if sheet_id:
        try:
            with st.spinner(f"📊 Enviando fila a Google Sheets ({cliente_nombre})..."):
                log_info(
                    f"Google Sheets: cliente={cliente_nombre} spreadsheet={sheet_id[:20]}… rango={sheet_range}",
                    func_name,
                )
                fecha_gs = extraer_fecha_ddmmyyyy_desde_archivo(nombre_archivo)
                medio_gs = extraer_nombre_medio_corto_desde_archivo(nombre_archivo)
                texto_gs = (transcripcion_segmento or contexto_termino or "").strip() or "Sin texto"
                sent_gs = analizar_sentimiento_mencion_heuristica(texto_gs)
                url_gs = (video_url or "").strip()
                cliente_id_gs = str((cliente or {}).get('id') or '').strip().lower()
                if cliente_id_gs == 'intrant':
                    # Hoja Intrant: fecha | periodista | titulo | texto | medio | sentimiento | url
                    fila = [fecha_gs, "Redaccion", termino_mostrar, texto_gs, medio_gs, sent_gs, url_gs]
                    incluir_indice_gs = False
                else:
                    fila = [fecha_gs, "Radio", texto_gs, medio_gs, sent_gs, url_gs]
                    incluir_indice_gs = True
                ok_gs, msg_gs = append_fila_google_sheet(
                    sheet_id, sheet_range, fila, incluir_indice=incluir_indice_gs
                )
                resultados["google_sheets"] = (ok_gs, msg_gs)
                if ok_gs:
                    st.success(f"📊 Google Sheets: {msg_gs}")
                else:
                    st.warning(f"📊 Google Sheets: {msg_gs}")
        except Exception as e_gs:
            log_error_critico(func_name, f"Error inesperado Google Sheets: {e_gs}")
            resultados["google_sheets"] = (False, str(e_gs))
            st.warning(f"📊 Google Sheets: {e_gs}")
    
    # === GENERAR MD ANALISISHOY ===
    # Guardar todas las coincidencias de la sesión en Analisishoy_YYYYMMDD.md
    try:
        ok_md, ruta_o_error = generar_analisishoy_md(
            nombre_archivo=nombre_archivo,
            termino_encontrado=termino_mostrar,
            contexto_termino=contexto_termino,
            resumen_ejecutivo=resumen_ejecutivo,
            transcripcion_completa=transcripcion_completa,
            video_url=video_url,
            info_medio=info_medio_hora
        )
        if ok_md:
            st.success(f"📄 AnalisisHoy MD actualizado: {os.path.basename(ruta_o_error)}")
        else:
            st.warning(f"⚠️ No se pudo escribir AnalisisHoy MD: {ruta_o_error}")
            log_warning(f"No se pudo escribir AnalisisHoy MD: {ruta_o_error}", "enviar_coincidencia_a_cliente")
    except Exception as e_md:
        log_warning(f"⚠️ No se pudo generar Analisishoy MD: {e_md}", "enviar_coincidencia_a_cliente")
        st.warning(f"⚠️ AnalisisHoy MD: {e_md}")
    
    # Resumen final

    total_exitosos = sum(1 for v in resultados.values() if v[0])
    total_destinos = len(resultados)
    
    if total_exitosos == total_destinos:
        st.success(f"✅ **{cliente_nombre}**: Enviado a {total_exitosos}/{total_destinos} destinos")
    elif total_exitosos > 0:
        st.warning(f"⚠️ **{cliente_nombre}**: Enviado a {total_exitosos}/{total_destinos} destinos")
    else:
        st.error(f"❌ **{cliente_nombre}**: Error en todos los destinos")
    
    return resultados

# === FUNCIONES DE WEBHOOK ===
def cargar_webhook_config():
    """Carga configuración del webhook"""
    default_config = {
        'enabled': True,  # Habilitado por defecto
        'url': 'https://hook.us1.make.com/1nk48toiy2c64f9966yue8bwhzqnosny',  # Tu webhook configurado
        'url_secundario': 'https://meny.app.n8n.cloud/webhook/edesurbot',  # Segundo webhook
        'url_terciario': 'https://meny.app.n8n.cloud/webhook-test/edesurbot',  # Tercer webhook de prueba
        'enviar_makecom': True,  # Switch para Make.com
        'enviar_n8n': True,  # Switch para N8N
        'enviar_n8n_test': True,  # Switch para N8N-Test
        'method': 'POST',
        'headers': {
            'Content-Type': 'application/json'
        },
        'send_video': True,
        'send_clips': True,
        'max_file_size_mb': 8,  # Reducido para evitar error 400 en Make.com
        'timeout': 30
    }
    
    try:
        if os.path.exists(WEBHOOK_CONFIG):
            with open(WEBHOOK_CONFIG, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Fusionar con defaults para compatibilidad
                default_config.update(config)
    except Exception as e:
        st.warning(f"⚠️ Error cargando configuración webhook: {e}")
    
    return default_config

def guardar_webhook_config(config):
    """Guarda configuración del webhook"""
    try:
        with open(WEBHOOK_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ Error guardando configuración webhook: {e}")
        return False

def enviar_clips_a_webhook(clips_generados, resumen, terminos_detectados, video_origen):
    """Envía clips específicos donde se encontraron coincidencias + SIEMPRE CON RESUMEN"""
    func_name = "enviar_clips_a_webhook"
    log_info(f"Iniciando envío de {len(clips_generados)} clips a webhook. Audio: {video_origen}", func_name)
    
    config = cargar_webhook_config()
    
    if not config['enabled'] or not config['url']:
        log_info("Webhook no configurado o deshabilitado", func_name)
        return False, "Webhook no configurado o deshabilitado"
    
    try:
        # Datos básicos SIEMPRE con resumen ejecutivo (textos con marcas RD para externos)
        resumen_out = capitalizar_marcas_medios_rd_en_texto(resumen or '')
        terminos_out = [
            capitalizar_marcas_medios_rd_en_texto(str(x)) for x in (terminos_detectados or [])
        ]
        data = {
            'evento': 'video_analizado_con_coincidencias',
            'timestamp': datetime.now().isoformat(),
            'video_origen': video_origen,
            'terminos_detectados': terminos_out,
            'total_terminos_encontrados': len(terminos_out),
            'resumen_ejecutivo': resumen_out,
            'clips_enviados': [],
            'metodo_envio': 'WH',
            'servidor': 'analizador_audio_ia_v2'
        }
        
        # Enviar cada clip donde se encontró una coincidencia
        for clip in clips_generados:
            clip_path = clip.get('path', '')
            
            if os.path.exists(clip_path):
                clip_size_mb = os.path.getsize(clip_path) / (1024*1024)
                
                # Verificar que el clip no sea muy grande para Make.com
                if clip_size_mb <= config['max_file_size_mb']:
                    try:
                        with open(clip_path, 'rb') as f:
                            clip_content = base64.b64encode(f.read()).decode('utf-8')
                            
                        clip_data = {
                            'termino_encontrado': capitalizar_marcas_medios_rd_en_texto(str(clip.get('termino', ''))),
                            'tiempo_en_video': clip.get('tiempo', ''),
                            'contexto': capitalizar_marcas_medios_rd_en_texto(str(clip.get('contexto', ''))),
                            'nombre_archivo': f"[WH] {os.path.basename(clip_path)}",
                            'tamaño_mb': round(clip_size_mb, 2),
                            'video_base64': clip_content,
                            'cloudinary_url': None  # Se podría agregar URL de Cloudinary aquí
                        }
                        
                        data['clips_enviados'].append(clip_data)
                    except Exception as e:
                        # Error leyendo archivo, enviar solo metadatos
                        clip_data = {
                            'termino_encontrado': capitalizar_marcas_medios_rd_en_texto(str(clip.get('termino', ''))),
                            'tiempo_en_video': clip.get('tiempo', ''),
                            'contexto': capitalizar_marcas_medios_rd_en_texto(str(clip.get('contexto', ''))),
                            'nombre_archivo': f"[WH] {os.path.basename(clip_path)}",
                            'tamaño_mb': round(clip_size_mb, 2),
                            'video_base64': None,
                            'error_lectura': str(e)[:100],
                            'razon_no_enviado': f"Error leyendo archivo: {str(e)[:50]}"
                        }
                        data['clips_enviados'].append(clip_data)
                else:
                    # Clip muy grande para Make.com, enviar solo metadatos
                    clip_data = {
                        'termino_encontrado': capitalizar_marcas_medios_rd_en_texto(str(clip.get('termino', ''))),
                        'tiempo_en_video': clip.get('tiempo', ''),
                        'contexto': capitalizar_marcas_medios_rd_en_texto(str(clip.get('contexto', ''))),
                        'nombre_archivo': f"[WH] {os.path.basename(clip_path)}",
                        'tamaño_mb': round(clip_size_mb, 2),
                        'video_base64': None,
                        'cloudinary_url': None,  # Aquí se podría poner la URL si se sube a Cloudinary
                        'razon_no_enviado': f"Muy grande para Make.com ({clip_size_mb:.1f}MB > {config['max_file_size_mb']}MB)",
                        'recomendacion': "Clip enviado solo a Telegram vía Cloudinary"
                    }
                    data['clips_enviados'].append(clip_data)
        
        # Agregar resumen de lo enviado
        data['total_clips'] = len(clips_generados)
        data['clips_con_video'] = len([c for c in data['clips_enviados'] if c.get('video_base64')])
        
        # Enviar al webhook con configuración mejorada
        headers_mejorados = config.get('headers', {}).copy()
        headers_mejorados.update({
            'User-Agent': 'RadioAnalizer/2.1',
            'Connection': 'close'
        })
        
        response = requests.post(
            config['url'], 
            json=data, 
            headers=headers_mejorados, 
            timeout=config.get('timeout', 30)
        )
        
        if response.status_code == 200:
            clips_enviados = data['clips_con_video']
            return True, f"✅ Enviados {clips_enviados} clips al webhook"
        elif response.status_code == 400:
            return False, f"❌ Error HTTP 400 (Petición muy grande): Reducir tamaño de clips o enviar solo metadatos"
        elif response.status_code == 413:
            return False, f"❌ Error HTTP 413 (Payload muy grande): Archivos demasiado grandes para Make.com"
        else:
            return False, f"❌ Error HTTP {response.status_code}: {response.text[:100]}"
            
    except requests.exceptions.Timeout:
        return False, "⏰ Timeout del webhook"
    except requests.exceptions.ConnectionError:
        return False, "🔌 Error de conexión"
    except Exception as e:
        return False, f"❌ Error: {str(e)[:100]}"

def enviar_clips_individuales_webhook(clips_generados, resumen, terminos_detectados, video_origen):
    """Envía clips uno por uno con pausas de 60 segundos entre cada uno"""
    func_name = "enviar_clips_individuales_webhook"
    log_info(f"Iniciando envío individual de {len(clips_generados)} clips con pausas de 60s", func_name)
    
    config = cargar_webhook_config()
    
    if not config['enabled'] or not config['url']:
        log_info("Webhook no configurado o deshabilitado", func_name)
        return False, "Webhook no configurado o deshabilitado"
    
    clips_enviados_exitosamente = 0
    clips_fallidos = 0
    
    for i, clip in enumerate(clips_generados, 1):
        try:
            st.info(f"📹 Enviando Clip {i}/{len(clips_generados)}: {clip.get('termino', '')} ({clip.get('tiempo', '')}) - {os.path.getsize(clip.get('path', '')) / (1024*1024):.1f}MB")
            
            terminos_out = [
                capitalizar_marcas_medios_rd_en_texto(str(x)) for x in (terminos_detectados or [])
            ]
            resumen_out = capitalizar_marcas_medios_rd_en_texto(resumen or '')

            # Crear datos para este clip específico
            data = {
                'evento': 'clip_individual_analizado',
                'timestamp': datetime.now().isoformat(),
                'video_origen': video_origen,
                'terminos_detectados': terminos_out,
                'resumen_ejecutivo': resumen_out,
                'clip_numero': i,
                'total_clips': len(clips_generados),
                'clip_data': None,
                'servidor': 'analizador_audio_ia_v2'
            }
            
            clip_path = clip.get('path', '')
            
            if os.path.exists(clip_path):
                clip_size_mb = os.path.getsize(clip_path) / (1024*1024)
                
                # Verificar tamaño del clip
                if clip_size_mb <= config['max_file_size_mb']:
                    try:
                        with open(clip_path, 'rb') as f:
                            clip_content = base64.b64encode(f.read()).decode('utf-8')
                            
                        clip_data = {
                            'termino_encontrado': capitalizar_marcas_medios_rd_en_texto(str(clip.get('termino', ''))),
                            'tiempo_en_video': clip.get('tiempo', ''),
                            'contexto': capitalizar_marcas_medios_rd_en_texto(str(clip.get('contexto', ''))),
                            'nombre_archivo': os.path.basename(clip_path),
                            'tamaño_mb': round(clip_size_mb, 2),
                            'video_base64': clip_content
                        }
                        
                        data['clip_data'] = clip_data
                    except Exception as e:
                        # Error leyendo archivo, enviar solo metadatos
                        clip_data = {
                            'termino_encontrado': capitalizar_marcas_medios_rd_en_texto(str(clip.get('termino', ''))),
                            'tiempo_en_video': clip.get('tiempo', ''),
                            'contexto': capitalizar_marcas_medios_rd_en_texto(str(clip.get('contexto', ''))),
                            'nombre_archivo': os.path.basename(clip_path),
                            'tamaño_mb': round(clip_size_mb, 2),
                            'video_base64': None,
                            'error_lectura': str(e)[:100]
                        }
                        data['clip_data'] = clip_data
                else:
                    # Clip muy grande, enviar solo metadatos
                    clip_data = {
                        'termino_encontrado': capitalizar_marcas_medios_rd_en_texto(str(clip.get('termino', ''))),
                        'tiempo_en_video': clip.get('tiempo', ''),
                        'contexto': capitalizar_marcas_medios_rd_en_texto(str(clip.get('contexto', ''))),
                        'nombre_archivo': os.path.basename(clip_path),
                        'tamaño_mb': round(clip_size_mb, 2),
                        'video_base64': None,
                        'razon_no_enviado': f"Muy grande ({clip_size_mb:.1f}MB > {config['max_file_size_mb']}MB)"
                    }
                    data['clip_data'] = clip_data
            
            # Enviar este clip individual a AMBOS webhooks
            st.info(f"🌐 Enviando clip {i} a ambos webhooks...")
            
            # Enviar solo a webhooks seleccionados
            exito_principal = False
            exito_secundario = False
            exito_terciario = False
            mensaje_principal = "No seleccionado"
            mensaje_secundario = "No seleccionado"
            mensaje_terciario = "No seleccionado"
            
            # Enviar a webhook principal (Make.com) si está habilitado
            if config.get('enviar_makecom', True):
                exito_principal, mensaje_principal = enviar_a_webhook_individual(
                    config['url'], data, func_name, f"Make.com-Clip{i}"
                )
            
            # Enviar a webhook secundario (N8N) si está habilitado - COMENTADO TEMPORALMENTE
            # if config.get('enviar_n8n', True):
            #     exito_secundario, mensaje_secundario = enviar_a_webhook_individual(
            #         config['url_secundario'], data, func_name, f"N8N-Clip{i}"
            #     )
            
            # Enviar a webhook terciario (N8N-Test) si está habilitado - COMENTADO TEMPORALMENTE
            # if config.get('enviar_n8n_test', True):
            #     exito_terciario, mensaje_terciario = enviar_a_webhook_individual(
            #         config['url_terciario'], data, func_name, f"N8N-Test-Clip{i}"
            #     )
            
            # Mostrar resultados
            if config.get('enviar_makecom', True):
                if exito_principal:
                    st.success(f"✅ Clip {i} enviado exitosamente a Make.com")
                else:
                    st.warning(f"⚠️ Clip {i} falló en Make.com: {mensaje_principal}")
                    
            # if config.get('enviar_n8n', True):
            #     if exito_secundario:
            #         st.success(f"✅ Clip {i} enviado exitosamente a N8N")
            #     else:
            #         st.warning(f"⚠️ Clip {i} falló en N8N: {mensaje_secundario}")
                    
            # if config.get('enviar_n8n_test', True):
            #     if exito_terciario:
            #         st.success(f"✅ Clip {i} enviado exitosamente a N8N-Test")
            #     else:
            #         st.warning(f"⚠️ Clip {i} falló en N8N-Test: {mensaje_terciario}")
            
            # Contar éxitos (solo Make.com activo por ahora)
            alguno_exitoso = (config.get('enviar_makecom', True) and exito_principal)
            # (config.get('enviar_n8n', True) and exito_secundario) or \
            # (config.get('enviar_n8n_test', True) and exito_terciario)
            
            if alguno_exitoso:
                clips_enviados_exitosamente += 1
                st.success(f"✅ Clip {i} enviado a al menos un webhook seleccionado")
            else:
                clips_fallidos += 1
                st.error(f"❌ Clip {i} falló en todos los webhooks seleccionados - CONTINUANDO con siguiente clip")
            
            # Pausa de 60 segundos entre clips (excepto después del último)
            if i < len(clips_generados):
                log_info(f"Esperando 60 segundos antes del siguiente clip ({i+1}/{len(clips_generados)})", func_name)
                with st.spinner(f"⏳ Esperando 60s antes del siguiente clip ({i+1}/{len(clips_generados)})..."):
                    time.sleep(60)
                st.info(f"✅ Listo para enviar clip {i+1}")
            
        except Exception as e:
            st.error(f"❌ Error procesando clip {i}: {str(e)[:100]}")
            clips_fallidos += 1
            log_exception(func_name, e, f"Clip {i}: {clip.get('path', '')}")
    
    # Resultado final
    if clips_enviados_exitosamente > 0:
        mensaje = f"✅ {clips_enviados_exitosamente} clips enviados exitosamente"
        if clips_fallidos > 0:
            mensaje += f", {clips_fallidos} fallaron"
        return True, mensaje
    else:
        return False, f"❌ Todos los clips fallaron ({clips_fallidos} errores)"

def registrar_envio_exitoso(webhook_nombre, data, intento):
    """Registra un envío exitoso en procesados.log"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_origen = data.get('video_origen', 'desconocido')
        clip_info = data.get('clip_data', {})
        
        if clip_info:
            # Es un clip individual
            termino = clip_info.get('termino_encontrado', 'desconocido')
            tiempo = clip_info.get('tiempo_en_video', 'desconocido')
            entrada = f"[{timestamp}] ✅ CLIP_ENVIADO: {webhook_nombre} | Audio: {video_origen} | Término: {termino} | Tiempo: {tiempo} | Intento: {intento}/3"
        else:
            # Es resumen general
            entrada = f"[{timestamp}] ✅ RESUMEN_ENVIADO: {webhook_nombre} | Audio: {video_origen} | Intento: {intento}/3"
        
        with open(PROCESADOS_LOG, "a", encoding="utf-8") as f:
            f.write(entrada + "\n")
            
    except Exception as e:
        log_exception(f"Error registrando envío exitoso: {e}", "registrar_envio_exitoso")

def registrar_envio_fallido(webhook_nombre, data, error_mensaje, intentos_totales):
    """Registra un envío fallido en procesados.log"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_origen = data.get('video_origen', 'desconocido')
        clip_info = data.get('clip_data', {})
        
        if clip_info:
            # Es un clip individual
            termino = clip_info.get('termino_encontrado', 'desconocido')
            tiempo = clip_info.get('tiempo_en_video', 'desconocido')
            entrada = f"[{timestamp}] ❌ CLIP_FALLIDO: {webhook_nombre} | Audio: {video_origen} | Término: {termino} | Tiempo: {tiempo} | Error: {error_mensaje} | Intentos: {intentos_totales}/3"
        else:
            # Es resumen general
            entrada = f"[{timestamp}] ❌ RESUMEN_FALLIDO: {webhook_nombre} | Audio: {video_origen} | Error: {error_mensaje} | Intentos: {intentos_totales}/3"
        
        with open(PROCESADOS_LOG, "a", encoding="utf-8") as f:
            f.write(entrada + "\n")
            
    except Exception as e:
        log_exception(f"Error registrando envío fallido: {e}", "registrar_envio_fallido")

def registrar_video_procesado(nombre_video, coincidencias_items, resumen_video):
    """Registra un archivo de audio procesado con detalles de clips y resumen en procesados.log"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Entrada principal del audio procesado
        entrada_principal = f"[{timestamp}] 🎬 AUDIO_PROCESADO: {nombre_video}"
        
        # Agregar información de coincidencias si las hay
        if coincidencias_items:
            terminos_encontrados = list(set([item['termino'] for item in coincidencias_items]))
            total_clips = len(coincidencias_items)
            entrada_principal += f" | Términos: {', '.join(terminos_encontrados)} | Clips: {total_clips}"
        else:
            entrada_principal += " | Sin coincidencias"
        
        # Escribir entrada principal
        with open(PROCESADOS_LOG, "a", encoding="utf-8") as f:
            f.write(entrada_principal + "\n")
            
            # Agregar línea simple para compatibilidad con sistema anterior
            f.write(nombre_video + "\n")
            
            # Agregar detalles de cada clip si los hay
            if coincidencias_items:
                for i, clip_item in enumerate(coincidencias_items, 1):
                    termino = clip_item.get('termino', 'desconocido')
                    tiempo = clip_item.get('tiempo', 'desconocido')
                    contexto = clip_item.get('contexto', '')[:50] + "..." if len(clip_item.get('contexto', '')) > 50 else clip_item.get('contexto', '')
                    
                    entrada_clip = f"[{timestamp}] 📹 SUBCLIP_{i}: {nombre_video} | Término: {termino} | Tiempo: {tiempo} | Contexto: {contexto}"
                    f.write(entrada_clip + "\n")
            
            # Agregar línea de separación
            f.write("=" * 80 + "\n")
            
    except Exception as e:
        log_exception(f"Error registrando video procesado: {e}", "registrar_video_procesado")
        # Fallback: usar el método simple original
        try:
            with open(PROCESADOS_LOG, "a", encoding="utf-8") as f:
                f.write(nombre_video + "\n")
        except:
            pass

def enviar_a_webhook_individual(url, data, func_name, nombre_webhook):
    """Envía datos a un webhook específico con reintentos y logging detallado"""
    log_info(f"Iniciando envío a {nombre_webhook}: {url}", func_name)
    
    # Intentos con retry para conexiones inestables (3 intentos con pausa de 30s)
    for intento in range(3):
        try:
            log_info(f"Intento {intento + 1}/3 para {nombre_webhook}", func_name)
            
            # Configuración mejorada para conexiones problemáticas
            response = requests.post(
                url, 
                json=data, 
                timeout=15,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'RadioAnalizer/2.1',
                    'Connection': 'close'  # Evitar keep-alive
                }
            )
            
            if response.status_code == 200:
                mensaje_exito = f"{nombre_webhook}: HTTP {response.status_code} (intento {intento + 1})"
                log_info(f"✅ ÉXITO - {mensaje_exito}", func_name)
                
                # Registrar en procesados.log el envío exitoso
                registrar_envio_exitoso(nombre_webhook, data, intento + 1)
                
                return True, mensaje_exito
            else:
                mensaje_error = f"{nombre_webhook}: HTTP {response.status_code}"
                log_info(f"❌ Error HTTP - {mensaje_error}", func_name)
                
                if intento == 2:  # Último intento
                    registrar_envio_fallido(nombre_webhook, data, mensaje_error, 3)
                    return False, mensaje_error
                    
                log_info(f"⏳ Esperando 30 segundos antes del siguiente intento...", func_name)
                time.sleep(30)  # PAUSA DE 30 SEGUNDOS
                
        except requests.exceptions.ConnectionError as e:
            mensaje_error = f"{nombre_webhook}: Error de conexión: {str(e)[:100]}"
            log_info(f"❌ Error conexión - {mensaje_error}", func_name)
            
            if intento == 2:
                registrar_envio_fallido(nombre_webhook, data, mensaje_error, 3)
                return False, mensaje_error
                
            log_info(f"⏳ Esperando 30 segundos antes del siguiente intento...", func_name)
            time.sleep(30)  # PAUSA DE 30 SEGUNDOS
            
        except requests.exceptions.Timeout:
            mensaje_error = f"{nombre_webhook}: Timeout"
            log_info(f"❌ Timeout - {mensaje_error}", func_name)
            
            if intento == 2:
                registrar_envio_fallido(nombre_webhook, data, mensaje_error, 3)
                return False, mensaje_error
                
            log_info(f"⏳ Esperando 30 segundos antes del siguiente intento...", func_name)
            time.sleep(30)  # PAUSA DE 30 SEGUNDOS
            
        except Exception as e:
            mensaje_error = f"{nombre_webhook}: Error: {str(e)[:100]}"
            log_info(f"❌ Error general - {mensaje_error}", func_name)
            
            if intento == 2:
                registrar_envio_fallido(nombre_webhook, data, mensaje_error, 3)
                return False, mensaje_error
                
            log_info(f"⏳ Esperando 30 segundos antes del siguiente intento...", func_name)
            time.sleep(30)  # PAUSA DE 30 SEGUNDOS
    
    mensaje_final = f"{nombre_webhook}: Falló después de 3 intentos"
    registrar_envio_fallido(nombre_webhook, data, mensaje_final, 3)
    return False, mensaje_final

def webhook_notification_simple(video_path, resumen, terminos):
    """Notificación simple a AMBOS webhooks - VERSIÓN ROBUSTA"""
    func_name = "webhook_notification_simple"
    config = cargar_webhook_config()
    
    if not config['enabled'] or not config['url']:
        log_info("Webhook no configurado o deshabilitado", func_name)
        return False, "Webhook no configurado"
    
    # Verificar conectividad antes de intentar
    if not verificar_conectividad():
        log_info("Sin conectividad - saltando webhook", func_name)
        return False, "Sin conectividad a internet"
    
    log_info(f"Iniciando envío de webhook para: {os.path.basename(video_path)}", func_name)
    
    # Datos básicos para notificación rápida
    resumen_out = capitalizar_marcas_medios_rd_en_texto((resumen or '')[:500])
    terminos_out = [capitalizar_marcas_medios_rd_en_texto(str(t)) for t in (terminos or [])]
    data = {
        'evento': 'video_analizado',
        'timestamp': datetime.now().isoformat(),
        'video': os.path.basename(video_path),
        'terminos': terminos_out,
        'resumen': resumen_out,
        'servidor': 'analizador_audio_ia_v2'
    }
    
    # Enviar solo a webhooks seleccionados
    mensajes = []
    exitos = []
    
    # Enviar a webhook principal (Make.com) si está habilitado
    if config.get('enviar_makecom', True):
        exito_principal, mensaje_principal = enviar_a_webhook_individual(
            config['url'], data, func_name, "Make.com"
        )
        exitos.append(exito_principal)
        if exito_principal:
            mensajes.append(f"✅ {mensaje_principal}")
        else:
            mensajes.append(f"❌ {mensaje_principal}")
    
    # Enviar a webhook secundario (N8N) si está habilitado - COMENTADO TEMPORALMENTE
    # if config.get('enviar_n8n', True):
    #     exito_secundario, mensaje_secundario = enviar_a_webhook_individual(
    #         config['url_secundario'], data, func_name, "N8N"
    #     )
    #     exitos.append(exito_secundario)
    #     if exito_secundario:
    #         mensajes.append(f"✅ {mensaje_secundario}")
    #     else:
    #         mensajes.append(f"❌ {mensaje_secundario}")
    
    # Enviar a webhook terciario (N8N-Test) si está habilitado - COMENTADO TEMPORALMENTE
    # if config.get('enviar_n8n_test', True):
    #     exito_terciario, mensaje_terciario = enviar_a_webhook_individual(
    #         config['url_terciario'], data, func_name, "N8N-Test"
    #     )
    #     exitos.append(exito_terciario)
    #     if exito_terciario:
    #         mensajes.append(f"✅ {mensaje_terciario}")
    #     else:
    #         mensajes.append(f"❌ {mensaje_terciario}")
    
    # Retornar éxito si al menos uno funcionó
    alguno_exitoso = any(exitos) if exitos else False
    mensaje_final = " | ".join(mensajes) if mensajes else "No hay webhooks seleccionados"
    
    log_info(f"Resultado webhooks: {mensaje_final}", func_name)
    return alguno_exitoso, mensaje_final


def transcripcion_completa_path_desde_clip(clip_path):
    if not clip_path:
        return None
    try:
        main = os.path.dirname(os.path.dirname(os.path.abspath(clip_path)))
        p = os.path.join(main, "TRANSCRIPCION_COMPLETA.txt")
        return p if os.path.isfile(p) else None
    except Exception:
        return None


def extraer_cuerpo_transcripcion_completa_txt(ruta_txt: str) -> str:
    """Devuelve solo el bloque de transcripción (sin cabeceras/estadísticas) si el TXT es el estándar del analizador."""
    if not ruta_txt or not os.path.isfile(ruta_txt):
        return ""
    try:
        with open(ruta_txt, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return ""
    for marker in ("TRANSCRIPCIÓN:", "TRANSCRIPCION:"):
        idx = raw.find(marker)
        if idx != -1:
            body = raw[idx + len(marker) :]
            stat = body.find("📊 ESTADÍSTICAS:")
            if stat != -1:
                body = body[:stat]
            sep = "=" * 20
            if sep in body:
                body = body.split(sep, 1)[-1]
            return body.strip()
    return raw.strip()


def guardar_snapshot_ultima_coincidencia_reenvio(
    nombre_archivo,
    termino_encontrado,
    contexto_termino,
    tipo_archivo,
    clip_path,
    trans_completa_path_hint,
    timestamp,
    idea_general,
    video_url_final,
    transcripcion_segmento,
):
    func_name = "guardar_snapshot_ultima_coincidencia_reenvio"
    try:
        os.makedirs(CARPETA_PROCESADOS, exist_ok=True)
        tpath = trans_completa_path_hint or transcripcion_completa_path_desde_clip(clip_path)
        payload = {
            "nombre_archivo": nombre_archivo,
            "termino_encontrado": termino_encontrado,
            "contexto_termino": contexto_termino or "",
            "tipo_archivo": tipo_archivo or "Audio",
            "clip_path": os.path.abspath(clip_path) if clip_path and os.path.isfile(clip_path) else None,
            "transcripcion_completa_path": os.path.abspath(tpath) if tpath and os.path.isfile(tpath) else None,
            "timestamp": float(timestamp) if timestamp is not None else None,
            "idea_general": idea_general or "",
            "video_url": (video_url_final or "").strip() or None,
            "transcripcion_segmento": (transcripcion_segmento or "").strip(),
            "saved_at": datetime.now().isoformat(),
        }
        with open(ULTIMA_COINCIDENCIA_REENVIO_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log_info(f"Snapshot reenvío guardado: {ULTIMA_COINCIDENCIA_REENVIO_JSON}", func_name)
    except Exception as e:
        log_warning(f"No se pudo guardar snapshot reenvío: {e}", func_name)


def cargar_snapshot_ultima_coincidencia_reenvio():
    if not os.path.isfile(ULTIMA_COINCIDENCIA_REENVIO_JSON):
        return None
    try:
        with open(ULTIMA_COINCIDENCIA_REENVIO_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def descubrir_ultima_coincidencia_en_procesados_para_reenvio():
    """Último clip de coincidencia real (subcarpeta c_clip_* sin tangencial) por fecha de archivo."""
    func_name = "descubrir_ultima_coincidencia_en_procesados_para_reenvio"
    if not os.path.isdir(CARPETA_PROCESADOS):
        log_warning("CARPETA_PROCESADOS no existe", func_name)
        return None

    best = None
    exts = (".mp3", ".mp4", ".m4a", ".wav")
    try:
        for d in os.listdir(CARPETA_PROCESADOS):
            if not d.startswith("c_"):
                continue
            dp = os.path.join(CARPETA_PROCESADOS, d)
            if not os.path.isdir(dp):
                continue
            for sub in os.listdir(dp):
                if not sub.startswith("c_clip_") or sub.startswith("c_clip_tangencial"):
                    continue
                sp = os.path.join(dp, sub)
                if not os.path.isdir(sp):
                    continue
                try:
                    subfiles = [os.path.join(sp, fn) for fn in os.listdir(sp)]
                except OSError:
                    continue
                for fp in subfiles:
                    if not os.path.isfile(fp) or not fp.lower().endswith(exts):
                        continue
                    try:
                        mt = os.path.getmtime(fp)
                    except OSError:
                        continue
                    termino = sub[len("c_clip_") :]
                    if best is None or mt > best[0]:
                        best = (mt, fp, dp, termino)
    except Exception as e:
        log_warning(f"Error escaneando procesados: {e}", func_name)
        return None

    if not best:
        return None

    _, clip_path, main_dir, termino = best
    nombre_archivo = os.path.basename(clip_path)
    tipo_archivo = "Audio"
    proc = os.path.join(main_dir, "PROCESADO.txt")
    if os.path.isfile(proc):
        try:
            with open(proc, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "Archivo origen:" in line:
                        rest = line.split("Archivo origen:", 1)[-1].strip()
                        if " (" in rest:
                            nombre_archivo = rest.rsplit(" (", 1)[0].strip()
                            tipo_archivo = rest.rsplit(" (", 1)[-1].rstrip(")").strip() or "Audio"
                        else:
                            nombre_archivo = rest
                        break
        except OSError:
            pass

    transc_path = os.path.join(main_dir, "TRANSCRIPCION_COMPLETA.txt")
    transc_path = transc_path if os.path.isfile(transc_path) else None
    transcripcion_completa = extraer_cuerpo_transcripcion_completa_txt(transc_path) if transc_path else ""
    contexto = (
        (transcripcion_completa[:800] + "…")
        if len(transcripcion_completa) > 800
        else transcripcion_completa
    )
    idea_general = (
        f"Tema principal: {termino}\n\n"
        f"Reenvío desde carpeta AUDIOCHECKS (descubierto automáticamente).\n\n"
        f"Contexto: {contexto[:600]}"
    )

    return {
        "nombre_archivo": nombre_archivo,
        "termino_encontrado": termino,
        "contexto_termino": contexto,
        "tipo_archivo": tipo_archivo,
        "clip_path": os.path.abspath(clip_path),
        "transcripcion_completa_path": os.path.abspath(transc_path) if transc_path else None,
        "timestamp": None,
        "idea_general": idea_general,
        "video_url": None,
        "transcripcion_segmento": (transcripcion_completa[:2000] if transcripcion_completa else contexto),
        "saved_at": None,
    }


def enviar_coincidencia_inmediata(nombre_archivo, termino_encontrado, contexto_termino, tipo_archivo, clip_path=None, transcripcion_completa="", timestamp=None, idea_general=None, video_url=None, transcripcion_segmento=""):
    """
    Envía una coincidencia inmediatamente tan pronto se encuentra.
    
    🆕 SISTEMA MULTI-CLIENTE:
    - Detecta automáticamente el cliente asociado al término
    - Envía a TODOS los destinos configurados para ese cliente
    - Cada cliente puede tener diferentes credenciales y destinos
    
    Flujo de envío:
    1. Obtener cliente asociado al término
    2. Enviar a todos los destinos del cliente (Telegram, Webhook, Brevo, GDrive, Supabase)
    
    Args:
        idea_general: Idea principal del segmento extraída por GPT-4o (nueva)
    """
    func_name = "enviar_coincidencia_inmediata"
    
    # Log inicio del proceso
    log_coincidencia_detectada(nombre_archivo, termino_encontrado, datetime.now().strftime('%H:%M:%S'), 0)
    
    try:
        # 🆕 OBTENER CLIENTE ASOCIADO AL TÉRMINO
        cliente = obtener_cliente_por_termino(termino_encontrado)
        cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)

        # Coincidencias con clip verificado se notifican siempre (coherente con tangenciales:
        # no bloquear por envios_habilitados; el toggle sigue en JSON/UI por compatibilidad).

        st.info(f"📤 **Enviando coincidencia a cliente: {cliente_nombre}**")
        log_info(f"Cliente detectado para término '{termino_encontrado}': {cliente_nombre}", func_name)

        # La intro TTS ya se aplica en el flujo principal antes de subir a Cloudinary (evita URL sin intro).

        # 🆕 SUBIR AUDIO A CLOUDINARY (solo si no se pasó ya o si el cliente tiene su propia cuenta)
        video_url_cloudinary = None
        if clip_path and os.path.exists(clip_path):
            cloudinary_config = cliente.get('cloudinary', {})
            # Solo subir si el cliente tiene Cloudinary habilitado
            if cloudinary_config.get('enabled'):
                try:
                    # Si ya tenemos un url y el cloud_name es el mismo que el global, no hace falta re-subir
                    global_cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME', '')
                    if video_url and cloudinary_config.get('cloud_name') == global_cloud_name and global_cloud_name != '':
                        log_info(f"⏭️ Usando URL global de Cloudinary para cliente {cliente_nombre}", func_name)
                        video_url_cloudinary = video_url
                    else:
                        # Configurar Cloudinary del cliente (puede ser otra cuenta)
                        cloudinary.config(
                            cloud_name=cloudinary_config['cloud_name'],
                            api_key=cloudinary_config['api_key'],
                            api_secret=cloudinary_config['api_secret']
                        )
                        
                        st.info("☁️ Subiendo clip a Cloudinary (Client Specific)...")
                        video_url_cloudinary, upload_msg = subir_video_cloudinary(clip_path, termino_encontrado)
                        
                        if video_url_cloudinary:
                            log_info(f"✅ Clip subido a Cloudinary Cliente: {video_url_cloudinary}", func_name)
                            st.success(f"☁️ Clip subido a Cloudinary del cliente")
                        else:
                            log_warning(f"⚠️ Error subiendo a Cloudinary: {upload_msg}", func_name)
                except Exception as e:
                    log_warning(f"⚠️ Error configurando/subiendo a Cloudinary: {e}", func_name)
        
        # 🆕 ENVIAR A TODOS LOS DESTINOS DEL CLIENTE (con el URL de Cloudinary)
        resultados = enviar_coincidencia_a_cliente(
            cliente=cliente,
            nombre_archivo=nombre_archivo,
            termino_encontrado=termino_encontrado,
            contexto_termino=contexto_termino,
            tipo_archivo=tipo_archivo,
            clip_path=clip_path,
            transcripcion_completa=transcripcion_completa,
            timestamp=timestamp,
            idea_general=idea_general,
            video_url=video_url_cloudinary or video_url,
            transcripcion_segmento=transcripcion_segmento
        )
        
        # Mostrar resumen de envíos del cliente
        st.success(f"✅ Envío completado para cliente: {cliente_nombre}")
        
        # 🆕 SIEMPRE retornar después de enviar a través del cliente
        # Esto evita duplicados cuando el cliente default también tiene servicios configurados
        log_info(f"✅ Envío multi-cliente completado para: {cliente_nombre}", func_name)
        final_url = (video_url_cloudinary or video_url) or None
        guardar_snapshot_ultima_coincidencia_reenvio(
            nombre_archivo,
            termino_encontrado,
            contexto_termino,
            tipo_archivo,
            clip_path,
            None,
            timestamp,
            idea_general,
            final_url,
            transcripcion_segmento,
        )
        return True, f"Enviado a cliente: {cliente_nombre}", video_url_cloudinary
        
        # ========== CÓDIGO COMENTADO - YA NO SE USA ==========
        # El siguiente código causaba DUPLICADOS para el cliente default
        # porque enviar_coincidencia_a_cliente() ya envía a todos los servicios
        # =====================================================
        webhook_config = cargar_webhook_config()
        telegram_config = cargar_telegram_config()
        
        # Extraer información del medio y hora
        info_medio_hora = extraer_info_medio_hora(nombre_archivo)
        
        # PASO 0: Usar idea general del segmento si está disponible (GPT-4o)
        resumen_ejecutivo = ""
        if idea_general:
            # ✨ USAR IDEA GENERAL EXTRAÍDA POR GPT-4o DEL SEGMENTO ESPECÍFICO
            resumen_ejecutivo = f"""🤖 Análisis del segmento:

{idea_general}

Término detectado: "{termino_encontrado}"
"""
            log_info(f"✅ Usando idea general extraída por GPT-4o para envío", func_name)
        elif transcripcion_completa and len(transcripcion_completa.strip()) > 100:
            try:
                resumen_ejecutivo = generar_resumen_archivo(nombre_archivo, [termino_encontrado], transcripcion_completa, tipo_archivo)
            except Exception as e:
                log_warning(f"Error generando resumen ejecutivo: {e}", func_name)
                # Resumen básico si falla la IA
                resumen_ejecutivo = f"""Tema principal: Se detectó una mención del término "{termino_encontrado}" en el contenido.

Contexto: {contexto_termino[:300]}{'...' if len(contexto_termino) > 300 else ''}

Puntos clave: El término "{termino_encontrado}" fue identificado en el contexto del programa, indicando relevancia informativa.

Relevancia: Esta mención es significativa para el monitoreo de contenido y puede requerir seguimiento adicional."""
        else:
            # Resumen básico cuando no hay transcripción completa
            resumen_ejecutivo = f"""Tema principal: Detección de término relevante "{termino_encontrado}" en contenido audiovisual.

Contexto: {contexto_termino[:300]}{'...' if len(contexto_termino) > 300 else ''}

Puntos clave: Se identificó una coincidencia directa con el término buscado en el momento específico del contenido.

Relevancia: La mención detectada es importante para el monitoreo continuo y análisis de contenido."""
        
        # Crear mensaje de RESUMEN EJECUTIVO (formato específico solicitado)
        mensaje_coincidencia = f"📺 Medio: {info_medio_hora}\n\n"
        mensaje_coincidencia += f"TÉRMINOS DETECTADOS: {termino_encontrado}\n\n"
        mensaje_coincidencia += f"{resumen_ejecutivo}"
        
        # === MOSTRAR RESUMEN EJECUTIVO COMPLETO EN LA INTERFAZ ===
        st.success("📋 **RESUMEN EJECUTIVO GENERADO:**")
        st.markdown("---")
        st.markdown(mensaje_coincidencia)
        st.markdown("---")
        
        if clip_path and os.path.exists(clip_path):
            mensaje_coincidencia += f"\n\n🎬 Clip generado: {os.path.basename(clip_path)}\n"
            mensaje_coincidencia += f"📤 Clip de audio a continuación en 30 segundos..."
        
        # === PASO 1: ENVIAR RESUMEN EJECUTIVO A TELEGRAM PRIMERO ===
        st.info("📝 **PASO 1: Enviando RESUMEN EJECUTIVO a Telegram...**")
        
        # Enviar resumen ejecutivo a Telegram PRIMERO
        if telegram_config['enabled'] and telegram_config['bot_token'] and telegram_config['chat_id']:
            with st.spinner("📱 Enviando resumen a Telegram..."):
                # Función de escape para Telegram (resumen completo; se envía en varias partes si supera 4096 caracteres)
                def escape_telegram_text(text):
                    import re
                    text = re.sub(r'[*_`\[\]()~>#+=|{}.!\\-]', '', text)
                    text = text.replace('"', '').replace("'", '').replace('\n\n\n', '\n\n')
                    return text
                
                mensaje_telegram_limpio = escape_telegram_text(mensaje_coincidencia)
                
                exito_resumen, mensaje_resumen_tg = enviar_mensaje_telegram(
                    mensaje_telegram_limpio,
                    telegram_config['chat_id'],
                    telegram_config['bot_token'],
                    parse_mode='Markdown'  # Mismo formato que el clip adjunto
                )
                
                if exito_resumen:
                    log_info(f"✅ Resumen ejecutivo enviado a Telegram: {mensaje_resumen_tg}", func_name)
                    st.success("📝 ✅ **RESUMEN EJECUTIVO enviado a Telegram**")
                else:
                    log_warning(f"⚠️ Error enviando resumen ejecutivo a Telegram: {mensaje_resumen_tg}", func_name)
                    st.warning(f"⚠️ **Error resumen ejecutivo**: {mensaje_resumen_tg}")
        else:
            st.warning("📱 **Telegram no configurado** - Saltando envío de resumen")
        
        # === PASO 2: ENVIAR A WEBHOOK (DESACTIVADO TEMPORALMENTE) ===
        # st.info("🌐 **PASO 2: Enviando a webhook...**")
        
        # # Enviar al webhook si está configurado
        # if webhook_config['enabled'] and webhook_config['url']:
        #     with st.spinner("🌐 Enviando a webhook..."):
        #         # Limpiar mensaje para webhook (sin caracteres problemáticos)
        #         mensaje_webhook_limpio = mensaje_coincidencia.replace('**', '').replace('*', '').replace('`', '').replace('\n', ' ')
        #         
        #         data = {
        #             "tipo": "coincidencia_inmediata_texto",
        #         "archivo": nombre_archivo,
        #         "termino": termino_encontrado,
        #         "contexto": contexto_termino[:500],  # Limitar contexto
        #         "info_medio_hora": info_medio_hora,
        #         "tipo_archivo": tipo_archivo,
        #         "mensaje": mensaje_webhook_limpio,
        #         "timestamp": datetime.now().isoformat(),
        #         "fuente": "Radio Analyzer IA - Detección Inmediata (Texto)",
        #         "paso": "1_resumen_texto"
        #     }
        #     
        #     exito_webhook, mensaje_webhook = enviar_a_webhook_individual(
        #         webhook_config['url'], 
        #         data, 
        #         func_name, 
        #         "Webhook Coincidencia Texto"
        #     )
        #     
        #     if exito_webhook:
        #         log_info(f"Resumen de coincidencia enviado al webhook: {mensaje_webhook}", func_name)
        #         st.success("🌐 Resumen enviado al webhook exitosamente")
        #     else:
        #         log_warning(f"Error enviando resumen al webhook: {mensaje_webhook}", func_name)
        #         st.warning(f"⚠️ Error webhook: {mensaje_webhook}")
        
        # Enviar a Telegram si está configurado (COMENTADO - Ya se envió en PASO 1)
        # if telegram_config['enabled'] and telegram_config['bot_token'] and telegram_config['chat_id']:
        #     # Limpiar mensaje para Telegram (sin markdown problemático)
        #     mensaje_telegram_limpio = mensaje_coincidencia.replace('**', '').replace('*', '').replace('`', '')
        #     
        #     exito_telegram, mensaje_telegram = enviar_mensaje_telegram(
        #         mensaje_telegram_limpio,
        #         telegram_config['chat_id'],
        #         telegram_config['bot_token']
        #     )
        #     
        #     if exito_telegram:
        #         log_info(f"Resumen de coincidencia enviado a Telegram: {mensaje_telegram}", func_name)
        #         st.success("📱 Resumen enviado a Telegram exitosamente")
        #     else:
        #         log_warning(f"Error enviando resumen a Telegram: {mensaje_telegram}", func_name)
        #         st.warning(f"⚠️ Error Telegram: {mensaje_telegram}")
        
        # === PASO 2.5: ENVIAR CORREO BREVO ===
        st.info("📧 PASO 2.5: Enviando correo con Brevo...")
        
        # Enviar correo si está configurado
        try:
            brevo_config = cargar_brevo_config()
            correos_destinatarios = obtener_correos_activos()
            if brevo_config['enabled'] and brevo_config['api_key'] and brevo_config['sender_email'] and correos_destinatarios:
                # Ya no usamos URL de Cloudinary en flujo de Telegram
                video_url_para_correo = None
                
                # Combinar transcripción y resumen ejecutivo mejorado
                contenido_completo_correo = f"""**TRANSCRIPCIÓN DEL CONTENIDO:**

{transcripcion_completa if transcripcion_completa else "Transcripción no disponible"}

---

**RESUMEN EJECUTIVO:**

{resumen_ejecutivo}"""
                
                exito_correo, mensaje_correo = enviar_correo_brevo(
                    termino_encontrado,
                    contenido_completo_correo,  # Transcripción + resumen
                    nombre_archivo,
                    clip_path,  # Archivo local para adjunto
                    info_medio_hora,  # Información del medio
                    [termino_encontrado],  # Lista de términos detectados
                    video_url_para_correo  # URL de Cloudinary para player incrustado
                )
                
                if exito_correo:
                    log_info(f"✅ Correo enviado exitosamente: {mensaje_correo}", func_name)
                    st.success("📧 ✅ Correo enviado exitosamente con Brevo")
                else:
                    log_warning(f"⚠️ Error enviando correo: {mensaje_correo}", func_name)
                    st.warning(f"⚠️ Error correo: {mensaje_correo}")
            else:
                log_info("Correo Brevo no configurado o deshabilitado", func_name)
                st.info("📧 Correo Brevo no configurado")
        except Exception as e:
            log_exception(func_name, e, "Error en envío de correo")
            st.error(f"❌ Error inesperado en correo: {str(e)[:100]}")
        
        # === PASO 3: PAUSA OBLIGATORIA DE 30 SEGUNDOS ===
        if clip_path and os.path.exists(clip_path):
            st.info("⏸️ PASO 3: Esperando 30 segundos antes de enviar el clip de audio...")
            log_info("Esperando 30 segundos antes de enviar el clip de audio", func_name)
            
            with st.spinner("⏳ Esperando 30s antes del clip de audio..."):
                time.sleep(30)
            
            st.success("✅ PASO 3 completado - Procediendo a enviar clip de audio")
            
            # === PASO 4: ENVIAR CLIP DE AUDIO ===
            st.info("🎬 PASO 4: Enviando clip de audio...")
            status_tg = st.empty()
            # Variables para resumen JSON por clip
            telegram_ok = False
            telegram_msg = ""
            drive_ok = False
            drive_link = None
            drive_msg = ""
            
            # Calcular tamaño del archivo si existe (definir antes de usar)
            file_size_mb = 0
            if clip_path and os.path.exists(clip_path):
                file_size_mb = os.path.getsize(clip_path) / (1024 * 1024)
            
            if telegram_config['enabled'] and telegram_config['bot_token'] and telegram_config['chat_id']:
                
                caption_clip = f"🎯 **CLIP DE COINCIDENCIA**\n\n"
                caption_clip += f"📺 **Medio**: {info_medio_hora}\n"
                caption_clip += f"⏰ **Generado**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                caption_clip += f"🔍 **TÉRMINOS DETECTADOS**: {termino_encontrado}\n\n"
                caption_clip += f"🏷️ **Término específico**: {termino_encontrado}\n"
                caption_clip += f"📝 **Contexto**: {contexto_termino[:200]}{'...' if len(contexto_termino) > 200 else ''}\n\n"
                caption_clip += f"📋 **INFORMACIÓN DEL ARCHIVO**:\n"
                caption_clip += f"📄 **Tipo de archivo**: {tipo_archivo}\n"
                caption_clip += f"📏 **Tamaño**: {file_size_mb:.1f}MB\n\n"
                caption_clip += f"━━━━━━━━━━━━━━━━━━━━━\n"
                
                # Reportar tamaño y forzar uso exclusivo de API directa (sin URL/Cloudinary)
                st.info(f"📏 Tamaño detectado para envío a Telegram: {file_size_mb:.2f} MB")
                max_mb = telegram_config.get('max_file_size_mb', 8)
                if file_size_mb <= max_mb:
                    st.info(f"📤 **Envío por API directa** ({file_size_mb:.1f}MB ≤ {max_mb}MB)")
                    caption_clip += f"🌐 **Vía**: API directa Telegram\n"
                else:
                    st.warning(f"🚫 **No se enviará**: {file_size_mb:.1f}MB > {max_mb}MB (solo API directa)")
                    caption_clip += f"🚫 **Vía**: Omitido (excede {max_mb}MB; se requiere compresión)\n"
                
                # ========== CONTROL DE DUPLICADOS ANTES DE ENVIAR ==========
                if 'clips_enviados_telegram' not in st.session_state:
                    st.session_state.clips_enviados_telegram = []
                
                # Verificar si ya fue enviado
                if clip_path in st.session_state.clips_enviados_telegram:
                    exito_clip = True
                    mensaje_clip = "✅ Ya enviado (duplicado evitado)"
                    st.info(f"⏭️ Clip ya enviado a Telegram: {os.path.basename(clip_path)}")
                    log_info(f"Clip duplicado evitado: {clip_path}", func_name)
                else:
                    with st.spinner("🎬 Enviando clip de audio a Telegram (API directa)..."):
                        if file_size_mb <= 50:
                            # Envío directo sin parse_mode para evitar fallos de formato
                            exito_clip, mensaje_clip, _ = enviar_video_telegram_directo(
                                clip_path,
                                caption_clip,
                                telegram_config['chat_id'],
                                telegram_config['bot_token'],
                                parse_mode=None
                            )
                        else:
                            exito_clip, mensaje_clip, _ = (False, f"Archivo demasiado grande ({file_size_mb:.1f}MB > {max_mb}MB).", None)
                
                if exito_clip:
                    log_info(f"Clip de audio enviado a Telegram OK | Tamaño={file_size_mb:.2f}MB | Detalle={mensaje_clip}", func_name)
                    status_tg.success("📱 Enviado a Telegram ✅")
                    st.success("🎬 Clip de audio enviado a Telegram exitosamente")
                    telegram_ok = True
                    telegram_msg = mensaje_clip
                    
                    # ========== REGISTRAR EN SESIÓN PARA EVITAR DUPLICADOS ==========
                    if 'clips_enviados_telegram' not in st.session_state:
                        st.session_state.clips_enviados_telegram = []
                    if clip_path not in st.session_state.clips_enviados_telegram:
                        st.session_state.clips_enviados_telegram.append(clip_path)
                    try:
                        coincidencias_logger.coincidencias_logger.info(
                            f"📱 TELEGRAM | OK | Archivo: {os.path.basename(clip_path)} | Tamaño: {file_size_mb:.2f}MB | Mensaje: {mensaje_clip}"
                        )
                    except Exception:
                        pass
                else:
                    log_warning(f"Fallo envío Telegram | Tamaño={file_size_mb:.2f}MB | Motivo={mensaje_clip}", func_name)
                    status_tg.error("📱 Envío a Telegram ❌")
                    st.warning(f"⚠️ Error enviando clip: {mensaje_clip}")
                    telegram_ok = False
                    telegram_msg = mensaje_clip
                    try:
                        coincidencias_logger.coincidencias_logger.error(
                            f"📱 TELEGRAM | ERROR | Archivo: {os.path.basename(clip_path)} | Tamaño: {file_size_mb:.2f}MB | Motivo: {mensaje_clip}"
                        )
                    except Exception:
                        pass
            
            # También enviar clip al webhook si está configurado (DESACTIVADO TEMPORALMENTE)
            # if webhook_config['enabled'] and webhook_config['url']:
            #     # Crear data para el clip
            #     clip_data = {
            #         "tipo": "coincidencia_inmediata_clip",
            #         "archivo": nombre_archivo,
            #         "termino": termino_encontrado,
            #         "contexto": contexto_termino,
            #         "info_medio_hora": info_medio_hora,
            #         "tipo_archivo": tipo_archivo,
            #         "clip_filename": os.path.basename(clip_path),
            #         "clip_size_mb": round(os.path.getsize(clip_path) / (1024*1024), 2),
            #         "timestamp": datetime.now().isoformat(),
            #         "fuente": "Radio Analyzer IA - Detección Inmediata (Clip)",
            #         "paso": "3_video_clip"
            #     }
            #     
            #     # Intentar enviar clip como base64 si es pequeño
            #     try:
            #         clip_size_mb = os.path.getsize(clip_path) / (1024*1024)
            #         if clip_size_mb <= webhook_config.get('max_file_size_mb', 8):
            #             with open(clip_path, 'rb') as f:
            #                 clip_content = base64.b64encode(f.read()).decode('utf-8')
            #             clip_data['video_base64'] = clip_content
            #             st.info(f"📤 Enviando clip al webhook ({clip_size_mb:.1f}MB)")
            #         else:
            #             clip_data['video_base64'] = None
            #             clip_data['razon_no_enviado'] = f"Muy grande ({clip_size_mb:.1f}MB > {webhook_config.get('max_file_size_mb', 8)}MB)"
            #             st.info(f"📋 Enviando solo metadatos del clip al webhook (muy grande: {clip_size_mb:.1f}MB)")
            #             
            #         exito_webhook_clip, mensaje_webhook_clip = enviar_a_webhook_individual(
            #             webhook_config['url'], 
            #             clip_data, 
            #             func_name, 
            #             "Webhook Clip Inmediato"
            #         )
            #         
            #         if exito_webhook_clip:
            #             log_info(f"Clip enviado al webhook: {mensaje_webhook_clip}", func_name)
            #             st.success("🌐 Clip enviado al webhook exitosamente")
            #         else:
            #             log_warning(f"Error enviando clip al webhook: {mensaje_webhook_clip}", func_name)
            #             st.warning(f"⚠️ Error webhook clip: {mensaje_webhook_clip}")
            #             
            #     except Exception as e:
            #         log_warning(f"Error preparando clip para webhook: {e}", func_name)
            #         st.warning(f"⚠️ Error preparando clip: {e}")
            
            st.success("✅ PASO 4 completado - Clip de audio enviado")
        
        # === PASO 5: ENVIAR A GOOGLE DRIVE ===
        st.info("☁️ **PASO 5: Enviando coincidencia a Google Drive...**")
        
        try:
            # Crear nombre único para el archivo de coincidencia
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_base_archivo = os.path.splitext(nombre_archivo)[0]
            nombre_coincidencia_txt = f"COINCIDENCIA_{termino_encontrado}_{timestamp}_{nombre_base_archivo}.txt"
            
            # Crear contenido del archivo de coincidencia
            contenido_coincidencia = f"""🎯 COINCIDENCIA DETECTADA INMEDIATAMENTE
===============================================

📺 MEDIO: {info_medio_hora}
🔍 TÉRMINO DETECTADO: {termino_encontrado}
📄 TIPO DE ARCHIVO: {tipo_archivo}
📝 CONTEXTO: {contexto_termino}
⏰ HORA DE DETECCIÓN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎬 CLIP GENERADO: {os.path.basename(clip_path) if clip_path and os.path.exists(clip_path) else 'No generado'}

===============================================
DETALLES DE LA COINCIDENCIA:
===============================================

- Archivo origen: {nombre_archivo}
- Término encontrado: {termino_encontrado}
- Contexto completo: {contexto_termino}
- Información del medio: {info_medio_hora}
- Timestamp de detección: {datetime.now().isoformat()}
- Enviado inmediatamente tras detección

===============================================
ESTADO DE ENVÍOS:
===============================================

✅ Webhook: Enviado
✅ Telegram: Enviado (texto + audio)
✅ Google Drive: Enviando...

===============================================
GENERADO POR: Radio Analyzer IA v2.0
TIPO DE ENVÍO: Coincidencia Inmediata
===============================================
"""
            
            # Enviar archivo de texto a Google Drive
            with st.spinner("☁️ Subiendo coincidencia a Google Drive..."):
                resultado_txt, mensaje_txt = subir_texto_google_drive(
                    contenido_coincidencia, 
                    nombre_coincidencia_txt
                )
                
                if resultado_txt:
                    st.success(f"☁️ ✅ **COINCIDENCIA enviada a Google Drive**: {resultado_txt.get('name')}")
                    log_info(f"Coincidencia inmediata enviada a Google Drive: {resultado_txt.get('name')}", func_name)
                
                # ENVIAR TRANSCRIPCIÓN COMPLETA si está disponible
                if transcripcion_completa and len(transcripcion_completa.strip()) > 50:
                    nombre_transcripcion_completa = f"TRANSCRIPCION_COMPLETA_{termino_encontrado}_{timestamp}_{nombre_base_archivo}.txt"
                    
                    contenido_transcripcion_completa = f"""TRANSCRIPCIÓN COMPLETA DEL AUDIO - COINCIDENCIA INMEDIATA
===============================================

AUDIO ORIGEN: {nombre_archivo}
TÉRMINO DETECTADO: {termino_encontrado}
FECHA TRANSCRIPCIÓN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ENVIADO INMEDIATAMENTE TRAS DETECCIÓN

===============================================
TRANSCRIPCIÓN COMPLETA:
===============================================

{transcripcion_completa}

===============================================
CONTEXTO DE LA COINCIDENCIA:
===============================================

{contexto_termino}

===============================================
GENERADO POR: Radio Analyzer IA v2.0
TIPO DE ENVÍO: Coincidencia Inmediata
===============================================
"""
                    
                    resultado_transcripcion, mensaje_transcripcion = subir_texto_google_drive(
                        contenido_transcripcion_completa, 
                        nombre_transcripcion_completa
                    )
                    
                    if resultado_transcripcion:
                        st.success(f"📝 ✅ TRANSCRIPCIÓN COMPLETA enviada a Google Drive: {resultado_transcripcion.get('name')}")
                        log_info(f"Transcripción completa de coincidencia inmediata enviada a Google Drive: {resultado_transcripcion.get('name')}", func_name)
                    else:
                        st.warning(f"⚠️ Error enviando transcripción completa a Google Drive: {mensaje_transcripcion}")
                        log_warning(f"Error enviando transcripción completa a Google Drive: {mensaje_transcripcion}", func_name)
                else:
                    st.info("ℹ️ No hay transcripción completa disponible para enviar")
                
                # Si hay clip, enviarlo también a Google Drive y capturar URL
                video_url_gdrive = None
                video_url_cloudinary = None
                if clip_path and os.path.exists(clip_path):
                    status_gd = st.empty()
                    with st.spinner("🎬 Subiendo clip de audio a Google Drive..."):
                        nombre_clip_gdrive = f"CLIP_{termino_encontrado}_{timestamp}_{os.path.basename(clip_path)}"
                        
                        resultado_clip, mensaje_clip_gdrive = subir_archivo_google_drive(
                            clip_path, 
                            nombre_clip_gdrive
                        )
                        
                        if resultado_clip:
                            # Capturar URL del clip para usar en el correo
                            video_url_gdrive = resultado_clip.get('webViewLink')
                            status_gd.success(f"☁️ Subido a Drive ✅")
                            st.success(f"🎬 ✅ **CLIP DE AUDIO enviado a Google Drive**: {resultado_clip.get('name')} | [Abrir]({resultado_clip.get('webViewLink')})")
                            log_info(f"Clip de coincidencia inmediata enviado a Google Drive: {resultado_clip.get('name')} - URL: {video_url_gdrive}", func_name)
                    
                    # Intentar subir también a Cloudinary para obtener URL directa
                    with st.spinner("☁️ Subiendo clip de audio a Cloudinary..."):
                        try:
                            cloudinary_configurado = configurar_cloudinary()
                            if cloudinary_configurado:
                                video_url_cloudinary, mensaje_cloudinary = subir_video_cloudinary(clip_path, termino_encontrado)
                                if video_url_cloudinary:
                                    st.success(f"☁️ ✅ **CLIP DE AUDIO subido a Cloudinary**: {video_url_cloudinary}")
                                    log_info(f"Clip de coincidencia inmediata subido a Cloudinary: {video_url_cloudinary}", func_name)
                                else:
                                    st.warning(f"⚠️ Error subiendo a Cloudinary: {mensaje_cloudinary}")
                                    log_warning(f"Error subiendo clip a Cloudinary: {mensaje_cloudinary}", func_name)
                            else:
                                st.warning("⚠️ Cloudinary no está configurado")
                                log_warning("Cloudinary no está configurado para subir clip", func_name)
                        except Exception as e:
                            st.warning(f"⚠️ Error subiendo a Cloudinary: {e}")
                            log_warning(f"Error subiendo clip a Cloudinary: {e}", func_name)
                            drive_ok = True
                            drive_link = resultado_clip.get('webViewLink')
                            
                            # ========== REGISTRAR EN SESIÓN PARA EVITAR DUPLICADOS ==========
                            if 'clips_enviados_drive' not in st.session_state:
                                st.session_state.clips_enviados_drive = []
                            if clip_path not in st.session_state.clips_enviados_drive:
                                st.session_state.clips_enviados_drive.append(clip_path)
                            try:
                                coincidencias_logger.coincidencias_logger.info(
                                    f"☁️ DRIVE | OK | Archivo: {resultado_clip.get('name')} | Link: {resultado_clip.get('webViewLink')}"
                                )
                            except Exception:
                                pass
                        else:
                            status_gd.error("☁️ Drive ❌")
                            st.warning(f"⚠️ **Error enviando clip a Google Drive**: {mensaje_clip_gdrive}")
                            log_warning(f"Error enviando clip a Google Drive: {mensaje_clip_gdrive}", func_name)
                            drive_ok = False
                            drive_msg = mensaje_clip_gdrive
                            try:
                                coincidencias_logger.coincidencias_logger.error(
                                    f"☁️ DRIVE | ERROR | Archivo: {os.path.basename(clip_path)} | Motivo: {mensaje_clip_gdrive}"
                                )
                            except Exception:
                                pass

                # === Registrar resumen JSON por clip ===
                try:
                    resumen_clip = {
                        "timestamp": datetime.now().isoformat(),
                        "clip_filename": os.path.basename(clip_path) if clip_path else None,
                        "clip_path": clip_path,
                        "size_mb": round(file_size_mb, 2),
                        "termino": termino_encontrado,
                        "video_origen": nombre_archivo,
                        "telegram": {"ok": telegram_ok, "message": telegram_msg},
                        "drive": {"ok": drive_ok, "link": drive_link, "message": drive_msg}
                    }
                    logs_dir = os.path.join(os.getcwd(), "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    date_str = datetime.now().strftime("%Y%m%d")
                    jsonl_path = os.path.join(logs_dir, f"clips_summary_{date_str}.jsonl")
                    with open(jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(resumen_clip, ensure_ascii=False) + "\n")
                except Exception as e:
                    log_warning(f"No se pudo escribir resumen JSON del clip: {e}", func_name)
                
                else:
                    st.warning(f"⚠️ Error enviando coincidencia a Google Drive: {mensaje_txt}")
                    log_warning(f"Error enviando coincidencia a Google Drive: {mensaje_txt}", func_name)
                
        except Exception as e:
            st.warning(f"⚠️ Error en envío a Google Drive: {e}")
            log_warning(f"Error enviando coincidencia inmediata a Google Drive: {e}", func_name)
        
        st.success("✅ **PASO 5 completado** - Coincidencia enviada a Google Drive")
        
        # Resumen final del proceso
        st.markdown("---")
        st.subheader("🎉 **PROCESO COMPLETADO**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("📝 Resumen enviado")
        with col2:
            st.success("🎬 Clip de audio enviado")
        with col3:
            st.success("☁️ Google Drive actualizado")
        
        # === PASO 6: RESUMEN FINAL DE ENVÍOS ===
        st.success("🎉 **ENVÍO COMPLETO A TODOS LOS DESTINOS**")
        
        # Mostrar resumen detallado de envíos
        st.markdown("### 📊 **RESUMEN DE ENVÍOS:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🌐 **Webhook (Make.com)**")
            if webhook_config.get('enabled', False):
                st.success("✅ Texto enviado")
                st.success("✅ Clip enviado")
            else:
                st.info("⚪ Deshabilitado")
        
        with col2:
            st.markdown("#### 📱 **Telegram**")
            if telegram_config.get('enabled', False):
                st.success("✅ Texto enviado")
                st.success("✅ Clip enviado")
            else:
                st.info("⚪ Deshabilitado")
        
        with col3:
            st.markdown("#### ☁️ **Google Drive**")
            st.success("✅ Archivo TXT subido")
            st.success("✅ Clip de audio subido")
        
        # Agregar información del archivo MD
        st.markdown("#### 📝 **Archivo de Coincidencias**")
        st.success("✅ Archivo MD actualizado: coincidencias.md")
        
        # === PASO 7: GENERAR ARCHIVO MD DE COINCIDENCIAS ===
        st.info("📝 PASO 7: Generando archivo MD de coincidencias...")
        
        try:
            # Combinar transcripción completa con resumen ejecutivo para el MD
            contenido_completo_md = f"""**TRANSCRIPCIÓN DEL CONTENIDO:**

{transcripcion_completa if transcripcion_completa else "Transcripción no disponible"}

---

**RESUMEN EJECUTIVO:**

{resumen_ejecutivo}"""
            
            # Generar archivo MD - usar URL de Cloudinary si está disponible, sino Google Drive
            video_url_para_md = video_url_cloudinary if video_url_cloudinary else video_url_gdrive
            exito_md, mensaje_md = generar_archivo_coincidencias_md(
                termino_encontrado,
                contenido_completo_md,
                nombre_archivo,
                info_medio_hora,
                [termino_encontrado],
                video_url_para_md,
                clip_path
            )
            
            if exito_md:
                st.success("📝 ✅ Archivo MD actualizado: coincidencias.md")
                log_info(f"Archivo MD actualizado: {mensaje_md}", func_name)
            else:
                st.warning(f"⚠️ Error generando archivo MD: {mensaje_md}")
                log_warning(f"Error generando archivo MD: {mensaje_md}", func_name)
                
        except Exception as e:
            st.warning(f"⚠️ Error en generación de archivo MD: {e}")
            log_warning(f"Error generando archivo MD: {e}", func_name)
        
        # === PASO 7: ENVIAR A SUPABASE ===
        st.info("🗄️ **PASO 7: Enviando a Supabase...**")
        try:
            # Preparar item de coincidencia para Supabase
            coincidencia_item = {
                'termino': termino_encontrado,
                'texto': contexto_termino,
                'contexto': contexto_termino,
                'timestamp': timestamp if timestamp is not None else '0.0',  # Agregar timestamp para control de duplicados
                'url_cloudinary': video_url_cloudinary if 'video_url_cloudinary' in locals() else None
            }
            
            supabase_success, supabase_msg = enviar_coincidencias_a_supabase(
                [coincidencia_item],  # Lista con un item
                nombre_archivo,
                tipo_archivo,
                resumen_ejecutivo,
                transcripcion_completa,
                video_url_cloudinary if 'video_url_cloudinary' in locals() else None,
                None  # enlace_directo
            )
            
            if supabase_success:
                st.success(f"🗄️ ✅ {supabase_msg}")
                log_info(f"Coincidencia enviada a Supabase: {supabase_msg}", func_name)
            else:
                st.warning(f"⚠️ Supabase: {supabase_msg}")
                log_warning(f"Supabase falló: {supabase_msg}", func_name)
        except Exception as e:
            st.warning(f"⚠️ Error Supabase: {e}")
            log_warning(f"Error enviando a Supabase: {e}", func_name)
        
        st.markdown("---")
        st.success(f"🎯 **Coincidencia procesada completamente:** {termino_encontrado}")
        
        return True, "Coincidencia enviada: resumen + clip de audio + Google Drive + archivo MD + Supabase (con pausas)", video_url_cloudinary
        
    except Exception as e:
        log_exception(func_name, e, f"Error enviando coincidencia inmediata para {nombre_archivo}")
        return False, f"Error: {str(e)}", None

# === FUNCIONES DE TELEGRAM Y CLOUDINARY ===
def cargar_telegram_config():
    """Carga configuración de Telegram"""
    default_config = {
        'enabled': True,  # Habilitado por defecto
        'bot_token': '8017408973:AAGlIaN5mxOKilTmhEWtCcCBxbw-J69sLWk',  # Tu nuevo bot token
        'chat_id': '@edesuralertas',  # Tu canal de Telegram actualizado
        'send_clips': True,
        'send_summary': True,
        'use_cloudinary': True,
        'max_file_size_mb': 50,
        'timeout': 30
    }
    
    try:
        if os.path.exists(TELEGRAM_CONFIG):
            with open(TELEGRAM_CONFIG, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_config.update(config)
    except Exception as e:
        st.warning(f"⚠️ Error cargando configuración Telegram: {e}")
    
    return default_config

def guardar_telegram_config(config):
    """Guarda configuración de Telegram"""
    try:
        with open(TELEGRAM_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ Error guardando configuración Telegram: {e}")
        return False

def cargar_cloudinary_config():
    """Carga configuración de Cloudinary (prioriza variables de entorno)"""
    default_config = {
        'cloud_name': os.getenv('CLOUDINARY_CLOUD_NAME', ''),
        'api_key': os.getenv('CLOUDINARY_API_KEY', ''),
        'api_secret': os.getenv('CLOUDINARY_API_SECRET', ''),
        'folder': 'video_analyzer_clips',
        'resource_type': 'video',
        'quality': 'auto',
        'format': 'mp4'
    }
    
    try:
        if os.path.exists(CLOUDINARY_CONFIG):
            with open(CLOUDINARY_CONFIG, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_config.update(config)
    except Exception as e:
        st.warning(f"⚠️ Error cargando configuración Cloudinary: {e}")
    
    return default_config

def guardar_cloudinary_config(config):
    """Guarda configuración de Cloudinary"""
    try:
        with open(CLOUDINARY_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ Error guardando configuración Cloudinary: {e}")
        return False

# === CONFIGURACIÓN BREVO (EMAIL) ===
BREVO_CONFIG = "brevo_config.json"
CORREOS_GUARDADOS = "correos_guardados.json"

def cargar_brevo_config():
    """Carga configuración de Brevo"""
    default_config = {
        'enabled': False,
        'api_key': '',
        'sender_email': '',
        'sender_name': 'Sistema de Análisis de Audio',
        'recipient_email': '',
        'recipient_name': '',
        'smtp_server': 'smtp-relay.sendinblue.com',
        'smtp_port': 587
    }
    
    try:
        if os.path.exists(BREVO_CONFIG):
            with open(BREVO_CONFIG, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Fusionar con defaults para agregar nuevos campos si los hay
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
    except Exception as e:
        log_exception("cargar_brevo_config", e)
    
    return default_config

def guardar_brevo_config(config):
    """Guarda configuración de Brevo"""
    try:
        with open(BREVO_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_exception("guardar_brevo_config", e)
        return False

def cargar_correos_guardados():
    """Carga lista de correos guardados"""
    try:
        if os.path.exists(CORREOS_GUARDADOS):
            with open(CORREOS_GUARDADOS, 'r', encoding='utf-8') as f:
                data = json.load(f)
                correos = data.get('correos', [])
                # Normalizar formato: convertir 'email' a 'correo' si es necesario
                for correo in correos:
                    if 'email' in correo and 'correo' not in correo:
                        correo['correo'] = correo['email']
                return correos
    except Exception as e:
        log_exception("cargar_correos_guardados", e)
    return []

def guardar_correos_lista(correos_lista):
    """Guarda lista de correos"""
    try:
        # Normalizar formato: asegurar que cada correo tenga el formato correcto
        correos_normalizados = []
        for correo in correos_lista:
            if isinstance(correo, dict):
                # Si ya es un dict, normalizar los campos
                correo_norm = {
                    'correo': correo.get('correo') or correo.get('email', ''),
                    'email': correo.get('correo') or correo.get('email', ''),  # Mantener ambos para compatibilidad
                    'nombre': correo.get('nombre', ''),
                    'fecha_agregado': correo.get('fecha_agregado', datetime.now().isoformat()),
                    'activo': correo.get('activo', True)
                }
                correos_normalizados.append(correo_norm)
            else:
                # Si es solo un string, crear el formato completo
                correos_normalizados.append({
                    'correo': correo,
                    'email': correo,  # Mantener ambos para compatibilidad
                    'nombre': '',
                    'fecha_agregado': datetime.now().isoformat(),
                    'activo': True
                })
        
        data = {
            'correos': correos_normalizados,
            'total_correos': len(correos_normalizados),
            'fecha_actualizacion': datetime.now().isoformat()
        }
        with open(CORREOS_GUARDADOS, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_exception("guardar_correos_lista", e)
        return False

def agregar_correo_a_lista(nuevo_correo, nombre=""):
    """Agrega un correo a la lista guardada"""
    correos = cargar_correos_guardados()
    
    # Verificar si ya existe
    for correo_data in correos:
        if correo_data['email'].lower() == nuevo_correo.lower():
            return False, "El correo ya existe en la lista"
    
    # Agregar nuevo correo
    correo_obj = {
        'email': nuevo_correo.strip(),
        'nombre': nombre.strip() if nombre else nuevo_correo.split('@')[0],
        'fecha_agregado': datetime.now().isoformat(),
        'activo': True
    }
    
    correos.append(correo_obj)
    
    if guardar_correos_lista(correos):
        return True, f"Correo {nuevo_correo} agregado exitosamente"
    else:
        return False, "Error guardando la lista"

def eliminar_correo_de_lista(correo_a_eliminar):
    """Elimina un correo de la lista"""
    correos = cargar_correos_guardados()
    correos_filtrados = [c for c in correos if c['email'].lower() != correo_a_eliminar.lower()]
    
    if len(correos_filtrados) != len(correos):
        if guardar_correos_lista(correos_filtrados):
            return True, f"Correo {correo_a_eliminar} eliminado"
        else:
            return False, "Error guardando la lista"
    else:
        return False, "Correo no encontrado"

def obtener_correos_activos():
    """Obtiene solo los correos activos de la lista"""
    correos = cargar_correos_guardados()
    out = []
    for c in correos:
        if not c.get('activo', True):
            continue
        addr = (c.get('email') or c.get('correo') or '').strip()
        if addr:
            out.append(addr)
    return out

def obtener_destinatarios_activos_cliente(cliente):
    """
    Normaliza y retorna solo destinatarios activos para un cliente.
    Soporta listas antiguas de strings y formato nuevo de dicts con 'activo'.
    """
    brevo_cfg = (cliente or {}).get('brevo', {}) if isinstance(cliente, dict) else {}
    destinatarios = brevo_cfg.get('correos_destinatarios', [])
    activos = []
    normalizados = []

    for item in destinatarios:
        if isinstance(item, dict):
            email = (item.get('email') or item.get('correo') or '').strip()
            if not email:
                continue
            nombre = item.get('nombre', '')
            activo = item.get('activo', True)
            normalizados.append({
                'email': email,
                'correo': email,
                'nombre': nombre,
                'activo': activo
            })
            if activo:
                activos.append(email)
        else:
            email = str(item).strip()
            if not email:
                continue
            normalizados.append({
                'email': email,
                'correo': email,
                'nombre': email.split('@')[0] if '@' in email else email,
                'activo': True
            })
            activos.append(email)

    return activos, normalizados

def configurar_cloudinary():
    """Configura Cloudinary con las credenciales guardadas"""
    config = cargar_cloudinary_config()
    
    if config['cloud_name'] and config['api_key'] and config['api_secret']:
        cloudinary.config(
            cloud_name=config['cloud_name'],
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            secure=True
        )
        return True
    return False

def subir_video_cloudinary(video_path, termino="", timestamp=""):
    """Sube un archivo (clip) a Cloudinary y retorna la URL"""
    try:
        if not configurar_cloudinary():
            return None, "Cloudinary no configurado"
        
        config = cargar_cloudinary_config()
        
        # Crear nombre único para el archivo
        nombre_base = os.path.splitext(os.path.basename(video_path))[0]
        public_id = f"{config['folder']}/{termino}_{timestamp}_{nombre_base}" if termino else f"{config['folder']}/{nombre_base}"
        
        # Subir video
        result = cloudinary.uploader.upload(
            video_path,
            resource_type=config['resource_type'],
            public_id=public_id,
            folder=config['folder'],
            quality=config['quality'],
            format=config['format'],
            overwrite=True,
            invalidate=True
        )
        
        return result['secure_url'], "Archivo subido exitosamente"
        
    except Exception as e:
        return None, f"Error subiendo a Cloudinary: {str(e)[:100]}"

# === FUNCIONES DE GOOGLE DRIVE ===
def obtener_credenciales_google_drive():
    """Obtiene credenciales de Google Drive usando refresh token"""
    try:
        # Crear credenciales usando el refresh token
        creds = Credentials(
            None,  # No access token inicial
            refresh_token=GOOGLE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=[
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets',
            ],
        )
        
        # Refrescar el token si es necesario
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
        
        return creds
    except Exception as e:
        log_exception("obtener_credenciales_google_drive", e)
        return None

def crear_servicio_google_drive():
    """Crea el servicio de Google Drive"""
    try:
        creds = obtener_credenciales_google_drive()
        if not creds:
            return None, "No se pudieron obtener credenciales de Google Drive"
        
        service = build('drive', 'v3', credentials=creds)
        return service, "Servicio creado exitosamente"
    except Exception as e:
        log_exception("crear_servicio_google_drive", e)
        return None, f"Error creando servicio: {str(e)[:100]}"


def crear_servicio_google_sheets():
    """Cliente Google Sheets API v4 (mismas credenciales OAuth que Drive)."""
    try:
        creds = obtener_credenciales_google_drive()
        if not creds:
            return None, 'No se pudieron obtener credenciales de Google (Sheets)'
        service = build('sheets', 'v4', credentials=creds)
        return service, 'Servicio Sheets creado'
    except Exception as e:
        log_exception('crear_servicio_google_sheets', e)
        return None, f'Error creando servicio Sheets: {str(e)[:100]}'


def spreadsheet_id_coincidencias_radio(cliente):
    """ID de hoja para coincidencias: default → EDESUR, intrant → Intrant (variables de entorno)."""
    cid = str(cliente.get('id') or '').strip().lower()
    if cid == 'default':
        return (GOOGLE_SHEETS_ID_EDESUR or '').strip() or None
    if cid == 'intrant':
        return (GOOGLE_SHEETS_ID_INTRANT or '').strip() or None
    return None


def spreadsheet_y_rango_coincidencias_cliente(cliente):
    """
    Spreadsheet y rango A1 para filas de coincidencia.
    Si el cliente define google_sheets, respeta enabled / spreadsheet_id / range.
    Si no hay id en JSON pero el cliente es default/intrant, usa variables de entorno.
    Retorna (spreadsheet_id, range_a1) o (None, None).
    """
    gs = (cliente or {}).get('google_sheets')
    if isinstance(gs, dict):
        if gs.get('enabled', True) is not True:
            return None, None
        sid = (gs.get('spreadsheet_id') or '').strip()
        if sid:
            rng = (gs.get('range') or '').strip() or GOOGLE_SHEETS_RANGE_COINCIDENCIAS
            return sid, rng
    legacy_id = spreadsheet_id_coincidencias_radio(cliente)
    if legacy_id:
        return legacy_id, GOOGLE_SHEETS_RANGE_COINCIDENCIAS
    return None, None


def extraer_fecha_ddmmyyyy_desde_archivo(nombre_archivo):
    """DD/MM/YYYY desde patrón de fecha en el nombre del archivo."""
    try:
        m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', nombre_archivo or '')
        if m:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.strftime('%d/%m/%Y')
    except Exception:
        pass
    return datetime.now().strftime('%d/%m/%Y')


def extraer_nombre_medio_corto_desde_archivo(nombre_archivo):
    """Nombre corto del medio desde nombre de archivo (antes de fecha 720p)."""
    try:
        nombre_sin_ext = os.path.splitext(nombre_archivo or '')[0]
        patron_fh = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})'
        match = re.search(patron_fh, nombre_sin_ext)
        if match:
            raw = nombre_sin_ext[: match.start()].strip().rstrip('_')
            raw = re.sub(r'_\d+p$', '', raw, flags=re.IGNORECASE)
            out = raw.replace('_', ' ').strip()
            return out if out else 'Medio'
        return (nombre_sin_ext.replace('_', ' ')[:120] or 'Medio').strip()
    except Exception:
        return 'Medio'


def analizar_sentimiento_mencion_heuristica(texto):
    """Sentimiento por palabras clave en español (rápido, sin API)."""
    if not texto or not str(texto).strip():
        return 'Neutral'
    text_lower = str(texto).lower()
    positive_words = [
        'excelente', 'bueno', 'gracias', 'perfecto', 'genial', 'increíble', 'fantástico',
        'rápido', 'eficiente', 'solucionado', 'arreglado', 'funciona', 'mejor', 'felicidades',
    ]
    negative_words = [
        'malo', 'terrible', 'problema', 'falla', 'error', 'sin luz', 'apagón', 'corte',
        'queja', 'reclamo', 'mal servicio', 'no funciona', 'pésimo',
    ]
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg and pos > 0:
        return 'Positivo'
    if neg > pos and neg > 0:
        return 'Negativo'
    return 'Neutral'


def append_fila_google_sheet(spreadsheet_id, range_a1, fila_valores, incluir_indice=True):
    """
    Añade una fila a Google Sheets.
    - incluir_indice=True: [# automático] + [fecha, origen, texto, medio, sentimiento, url].
    - incluir_indice=False: fila_valores se envía tal cual (Intrant: fecha, periodista, titulo, texto, medio, sentimiento, url).
    """
    func_name = 'append_fila_google_sheet'
    try:
        if not spreadsheet_id:
            return False, 'Sheets: spreadsheet_id vacío'
        if not GOOGLE_REFRESH_TOKEN or not GOOGLE_CLIENT_ID:
            return False, 'Sheets: credenciales Google no configuradas (.env)'
        esperados = 6 if incluir_indice else 7
        if len(fila_valores) != esperados:
            return False, f'Sheets: se esperan {esperados} valores, llegaron {len(fila_valores)}'
        service, msg = crear_servicio_google_sheets()
        if not service:
            return False, msg
        if incluir_indice:
            titulo = titulo_hoja_desde_range_a1(range_a1)
            siguiente = siguiente_indice_columna_a(service, spreadsheet_id, titulo)
            fila_sheet = [siguiente] + list(fila_valores)
        else:
            siguiente = None
            fila_sheet = list(fila_valores)
        body = {'values': [fila_sheet]}
        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_a1,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body,
        ).execute()
        if incluir_indice:
            log_info(
                f'Fila índice={siguiente} añadida a Sheet id={spreadsheet_id} rango={range_a1}',
                func_name,
            )
            return True, f'Fila añadida a Google Sheet (#{siguiente})'
        log_info(f'Fila sin índice añadida a Sheet id={spreadsheet_id} rango={range_a1}', func_name)
        return True, 'Fila añadida a Google Sheet'
    except HttpError as e:
        raw = e.content.decode(errors='replace') if getattr(e, 'content', None) else str(e)
        log_exception(func_name, e, raw[:400])
        return False, f'Sheets HTTP: {raw[:200]}'
    except Exception as e:
        log_exception(func_name, e)
        return False, f'Sheets: {str(e)[:200]}'


def enviar_tangenciales_a_google_sheets(menciones_tangenciales_data):
    """Registra menciones tangenciales en la hoja Google Sheets configurada para cada cliente."""
    func_name = "enviar_tangenciales_a_google_sheets"
    resultados = []
    if not menciones_tangenciales_data:
        return resultados
    for tang in menciones_tangenciales_data:
        try:
            termino = tang.get('termino') or ''
            cliente = obtener_cliente_por_termino(termino)
            if not cliente:
                continue
            cliente_id = str((cliente or {}).get('id') or '').strip().lower()
            sheet_id, sheet_range = spreadsheet_y_rango_coincidencias_cliente(cliente)
            if not sheet_id:
                continue
            cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
            archivo = tang.get('archivo', '')
            fecha_gs = extraer_fecha_ddmmyyyy_desde_archivo(archivo)
            medio_gs = (
                tang.get('medio')
                or extraer_nombre_medio_corto_desde_archivo(archivo)
                or formatear_nombre_medio_desde_ruta(archivo)
            )
            texto_gs = (
                tang.get('texto_evidencia')
                or tang.get('motivo_sistema')
                or tang.get('motivo')
                or ''
            ).strip() or 'Mención tangencial sin desarrollo'
            sent_gs = analizar_sentimiento_mencion_heuristica(texto_gs)
            url_gs = (
                tang.get('gdrive_url_audio')
                or tang.get('gdrive_url_txt')
                or ''
            ).strip()
            termino_titulo = capitalizar_marcas_medios_rd_en_texto(str(termino or 'Tangencial'))
            if cliente_id == 'intrant':
                fila = [fecha_gs, 'Redaccion', f'Tangencial - {termino_titulo}', texto_gs, medio_gs, sent_gs, url_gs]
                incluir_indice = False
            else:
                texto_tangencial = f"Tangencial: {termino_titulo} | Motivo: {texto_gs}"
                fila = [fecha_gs, 'Radio (Tangencial)', texto_tangencial, medio_gs, sent_gs, url_gs]
                incluir_indice = True
            ok, msg = append_fila_google_sheet(sheet_id, sheet_range, fila, incluir_indice=incluir_indice)
            resultados.append((cliente_nombre, ok, msg))
            if ok:
                log_info(f"Tangencial enviada a Sheets ({cliente_nombre}): {msg}", func_name)
            else:
                log_warning(f"Tangencial no enviada a Sheets ({cliente_nombre}): {msg}", func_name)
        except Exception as e:
            log_warning(f"Error enviando tangencial a Sheets: {e}", func_name)
            resultados.append(('desconocido', False, str(e)[:200]))
    return resultados

def subir_archivo_google_drive(archivo_path, nombre_archivo=None, mime_type=None, folder_id=None):
    """Sube un archivo a Google Drive en la carpeta especificada"""
    func_name = "subir_archivo_google_drive"
    
    # Usar carpeta del cliente o la global por defecto
    target_folder = folder_id or GOOGLE_DRIVE_FOLDER_ID
    
    try:
        if not os.path.exists(archivo_path):
            error_msg = f"Archivo no existe: {archivo_path}"
            log_error_critico(func_name, error_msg, archivo_path)
            return None, error_msg
        
        service, mensaje = crear_servicio_google_drive()
        if not service:
            log_error_critico(func_name, mensaje, archivo_path)
            return None, mensaje
        
        # Usar nombre del archivo si no se especifica
        if not nombre_archivo:
            nombre_archivo = os.path.basename(archivo_path)
        
        # Determinar MIME type si no se especifica
        if not mime_type:
            low = archivo_path.lower()
            if low.endswith('.mp4'):
                mime_type = 'video/mp4'
            elif low.endswith('.mp3'):
                mime_type = 'audio/mpeg'
            elif low.endswith('.txt'):
                mime_type = 'text/plain'
            elif low.endswith('.md'):
                mime_type = 'text/markdown'
            else:
                mime_type = 'application/octet-stream'
        
        # Obtener tamaño del archivo para logging
        file_size = os.path.getsize(archivo_path)
        
        # Log inicio de subida
        log_gdrive_upload_start(nombre_archivo, file_size, mime_type)
        
        # Crear metadata del archivo
        file_metadata = {
            'name': nombre_archivo,
            'parents': [target_folder]
        }
        
        # Crear media para subida
        media = MediaFileUpload(archivo_path, mimetype=mime_type, resumable=True)
        
        # Subir archivo
        log_info(f"Subiendo {nombre_archivo} a Google Drive...", func_name)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,webViewLink'
        ).execute()
        
        # Log éxito
        log_gdrive_upload_success(nombre_archivo, file.get('id'), file.get('webViewLink'))
        log_info(f"Archivo subido exitosamente: {file.get('name')} (ID: {file.get('id')})", func_name)
        return file, "Archivo subido exitosamente"
        
    except HttpError as e:
        error_msg = f"Error HTTP de Google Drive: {e.resp.status} {e.content.decode()}"
        log_gdrive_upload_error(nombre_archivo or os.path.basename(archivo_path), error_msg, e.resp.status)
        log_exception(func_name, e, error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error subiendo archivo: {str(e)[:100]}"
        log_gdrive_upload_error(nombre_archivo or os.path.basename(archivo_path), error_msg)
        log_exception(func_name, e, error_msg)
        return None, error_msg


def _sanitizar_nombre_cliente_carpeta_drive(nombre):
    """Nombre seguro para carpeta Drive bajo tangenciales radio-…"""
    n = (nombre or 'Cliente').strip()
    n = re.sub(r'[\r\n\t<>:"/\\|?*]+', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return (n[:200] if n else '') or 'Cliente'


def obtener_o_crear_carpeta_tangenciales_cliente(nombre_cliente):
    """
    Dentro de GOOGLE_DRIVE_TANGENCIALES_PARENT_ID, obtiene o crea la carpeta
    'tangenciales radio-{nombre_cliente}'.
    Retorna (folder_id, None) o (None, mensaje_error).
    """
    func_name = "obtener_o_crear_carpeta_tangenciales_cliente"
    parent = (GOOGLE_DRIVE_TANGENCIALES_PARENT_ID or '').strip()
    if not parent:
        return None, "GOOGLE_DRIVE_TANGENCIALES_PARENT_ID vacío"
    nombre_limpio = _sanitizar_nombre_cliente_carpeta_drive(nombre_cliente)
    nombre_carpeta = f"tangenciales radio-{nombre_limpio}"
    try:
        service, msg = crear_servicio_google_drive()
        if not service:
            return None, msg or "Sin servicio Google Drive"
        esc = nombre_carpeta.replace("\\", "\\\\").replace("'", "\\'")
        q = (
            f"'{parent}' in parents and name = '{esc}' and "
            f"mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        resp = service.files().list(q=q, spaces='drive', fields='files(id,name)', pageSize=5).execute()
        files = resp.get('files', [])
        if files:
            return files[0]['id'], None
        file_metadata = {
            'name': nombre_carpeta,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent],
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        fid = folder.get('id')
        log_info(f"Carpeta tangenciales en Drive creada: {nombre_carpeta} → {fid}", func_name)
        return fid, None
    except HttpError as e:
        err = f"Drive HttpError: {e.resp.status}"
        log_exception(func_name, e, err)
        return None, err
    except Exception as e:
        log_exception(func_name, e)
        return None, str(e)[:200]


def enriquecer_tangencial_clip_transcripcion_drive(
    item,
    archivo_path,
    rel,
    archivo_main_dir,
    momento_termino,
    buffer_anterior,
    duracion_clip,
    dur_total,
    clip_path_existente=None,
    func_name='enriquecer_tangencial_clip_transcripcion_drive',
):
    """
    Genera clip MP3 (si no se pasa uno existente), escribe TXT con evidencia/transcripción
    y sube ambos a la subcarpeta del cliente bajo GOOGLE_DRIVE_TANGENCIALES_PARENT_ID.
    Actualiza item: clip_path, transcripcion_path, gdrive_url_audio, gdrive_url_txt.
    """
    termino = item.get('termino') or ''
    cliente = obtener_cliente_por_termino(termino)
    # Tangenciales: generar clip/TXT y subir a Drive aunque el cliente tenga «envíos» pausados
    # (webhook/Telegram/etc.); el correo tangencial ya es obligatorio y el archivo en Drive es parte del mismo flujo.
    nombre_cliente = _sanitizar_nombre_cliente_carpeta_drive(
        nombre_cliente_mostrar_para_ui(cliente)
    )

    evidencia = (item.get('texto_evidencia') or '').strip()
    ms_extra = (item.get('motivo_sistema') or '').strip()
    motivo = (item.get('motivo') or '').strip()

    termino_safe = re.sub(r'[^\w\- ]', '_', termino, flags=re.UNICODE).strip('_')[:40] or 'termino'
    clip_dir = os.path.join(archivo_main_dir, f"c_clip_tangencial_{termino_safe}")
    os.makedirs(clip_dir, exist_ok=True)

    try:
        mt = float(momento_termino or 0)
    except (TypeError, ValueError):
        mt = 0.0

    base_new = nombre_base_tangencial_normalizado(rel, termino, momento_termino)
    clip_final = os.path.join(clip_dir, f"{base_new}.mp3")

    if clip_path_existente and os.path.isfile(clip_path_existente):
        try:
            if os.path.abspath(clip_path_existente) != os.path.abspath(clip_final):
                shutil.copy2(clip_path_existente, clip_final)
        except OSError as e_copy:
            log_warning(f"Tangencial: no se pudo copiar clip existente a nombre normalizado: {e_copy}", func_name)
            clip_final = clip_path_existente
    else:
        try:
            buf = float(buffer_anterior)
        except (TypeError, ValueError):
            buf = 30.0
        try:
            dur = float(duracion_clip)
        except (TypeError, ValueError):
            dur = 90.0
        inicio = max(0.0, mt - buf)
        dur_real = dur
        fin = inicio + dur_real
        try:
            dtot = float(dur_total)
        except (TypeError, ValueError):
            dtot = fin
        if fin > dtot:
            if dur_real <= dtot:
                inicio = max(0.0, dtot - dur_real)
                fin = dtot
            else:
                inicio = 0.0
                fin = dtot
                dur_real = min(dur_real, dtot)
        cmd = [
            "ffmpeg", "-y", "-ss", str(inicio),
            "-t", str(dur_real), "-i", archivo_path,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
            clip_final,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            log_warning(f"Tangencial: ffmpeg no generó clip ({e})", func_name)
            return
        if not os.path.isfile(clip_final):
            log_warning("Tangencial: archivo MP3 no creado", func_name)
            return

    base_file = os.path.splitext(os.path.basename(clip_final))[0]
    txt_path = os.path.join(os.path.dirname(clip_final), f"{base_file}_transcripcion.txt")
    try:
        bloque_tecnico = f"Motivo (técnico): {ms_extra}\n" if ms_extra else ""
        contenido_txt = (
            f"Mención tangencial — {rel}\n"
            f"Término: {termino}\n"
            f"Medio: {item.get('medio', '')}\n"
            f"Posición en audio (s): {mt}\n"
            f"Motivo (cliente / resumen): {motivo}\n"
            f"{bloque_tecnico}"
            f"--- Transcripción / evidencia ---\n{evidencia}\n"
        )
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(contenido_txt)
    except OSError as e:
        log_warning(f"Tangencial: error escribiendo TXT: {e}", func_name)
        txt_path = None

    folder_id, err = obtener_o_crear_carpeta_tangenciales_cliente(nombre_cliente)
    if err or not folder_id:
        log_warning(f"Tangencial Drive: carpeta cliente ({nombre_cliente}): {err}", func_name)
        item['clip_path'] = clip_final
        item['transcripcion_path'] = txt_path
        return

    nombre_mp3 = os.path.basename(clip_final)
    res_mp3, _ = subir_archivo_google_drive(
        clip_final, nombre_mp3, mime_type='audio/mpeg', folder_id=folder_id
    )
    if res_mp3:
        item['gdrive_url_audio'] = res_mp3.get('webViewLink')

    if txt_path and os.path.isfile(txt_path):
        nombre_txt = os.path.basename(txt_path)
        res_txt, _ = subir_archivo_google_drive(
            txt_path, nombre_txt, mime_type='text/plain', folder_id=folder_id
        )
        if res_txt:
            item['gdrive_url_txt'] = res_txt.get('webViewLink')

    item['clip_path'] = clip_final
    item['transcripcion_path'] = txt_path


def subir_texto_google_drive(contenido_texto, nombre_archivo, mime_type='text/plain', folder_id=None):
    """Sube contenido de texto directamente a Google Drive"""
    func_name = "subir_texto_google_drive"
    
    # Usar carpeta del cliente o la global por defecto
    target_folder = folder_id or GOOGLE_DRIVE_FOLDER_ID
    
    try:
        service, mensaje = crear_servicio_google_drive()
        if not service:
            log_error_critico(func_name, mensaje)
            return None, mensaje
        
        # Obtener tamaño del contenido para logging
        content_size = len(contenido_texto.encode('utf-8'))
        
        # Log inicio de subida
        log_gdrive_upload_start(nombre_archivo, content_size, mime_type)
        
        # Crear metadata del archivo
        file_metadata = {
            'name': nombre_archivo,
            'parents': [target_folder]
        }
        
        # Crear media para contenido de texto
        media = MediaIoBaseUpload(
            io.BytesIO(contenido_texto.encode('utf-8')),
            mimetype=mime_type,
            resumable=True
        )
        
        # Subir archivo
        log_info(f"Subiendo texto {nombre_archivo} a Google Drive...", func_name)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,webViewLink'
        ).execute()
        
        # Log éxito
        log_gdrive_upload_success(nombre_archivo, file.get('id'), file.get('webViewLink'))
        log_info(f"Texto subido exitosamente: {file.get('name')} (ID: {file.get('id')})", func_name)
        return file, "Texto subido exitosamente"
        
    except HttpError as e:
        error_msg = f"Error HTTP de Google Drive: {e.resp.status} {e.content.decode()}"
        log_gdrive_upload_error(nombre_archivo, error_msg, e.resp.status)
        log_exception(func_name, e, error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error subiendo texto: {str(e)[:100]}"
        log_gdrive_upload_error(nombre_archivo, error_msg)
        log_exception(func_name, e, error_msg)
        return None, error_msg

def enviar_clips_a_google_drive(clips_generados, resumen, terminos_detectados, video_origen, transcripcion_completa=""):
    """Envía clips, resumen Y TRANSCRIPCIÓN COMPLETA a Google Drive"""
    func_name = "enviar_clips_a_google_drive"
    log_info(f"Iniciando envío de {len(clips_generados)} clips + transcripción completa a Google Drive. Audio: {video_origen}", func_name)
    
    # ========== CONTROL DE DUPLICADOS ==========
    # Verificar si ya se subieron estos clips individualmente
    clips_ya_subidos = 0
    clips_pendientes = []
    
    for clip in clips_generados:
        clip_path = clip.get('path', '')
        if clip_path in st.session_state.get('clips_enviados_drive', []):
            clips_ya_subidos += 1
            st.info(f"⏭️ Clip ya subido a Drive individualmente: {os.path.basename(clip_path)}")
        else:
            clips_pendientes.append(clip)
    
    if clips_ya_subidos == len(clips_generados):
        st.success(f"✅ Todos los clips ya fueron subidos a Drive individualmente ({clips_ya_subidos}/{len(clips_generados)})")
        return True, f"✅ Todos los clips ya subidos a Drive individualmente"
    
    if clips_pendientes:
        st.info(f"📤 Subiendo {len(clips_pendientes)} clips pendientes a Drive (de {len(clips_generados)} total)")
        clips_generados = clips_pendientes  # Usar solo los clips pendientes
    
    try:
        # Crear nombre de carpeta para este audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_video_limpio = os.path.splitext(os.path.basename(video_origen))[0]
        carpeta_video = f"c_Analisis_{nombre_video_limpio}_{timestamp}"
        
        # Subir resumen ejecutivo como TXT
        nombre_resumen = f"RESUMEN_{nombre_video_limpio}_{timestamp}.txt"
        resumen_completo = f"""ANÁLISIS DE AUDIO - RESUMEN EJECUTIVO
===============================================

AUDIO ORIGEN: {video_origen}
FECHA ANÁLISIS: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
TÉRMINOS DETECTADOS: {', '.join(terminos_detectados)}
TOTAL CLIPS GENERADOS: {len(clips_generados)}

===============================================
RESUMEN EJECUTIVO:
===============================================

{resumen}

===============================================
DETALLES DE CLIPS:
===============================================

"""
        
        # Agregar detalles de cada clip
        for i, clip in enumerate(clips_generados, 1):
            resumen_completo += f"""
CLIP {i}/{len(clips_generados)}:
- Término encontrado: {clip.get('termino', 'N/A')}
- Tiempo en audio: {clip.get('tiempo', 'N/A')}
- Contexto: {clip.get('contexto', 'N/A')[:200]}...
- Archivo: {os.path.basename(clip.get('path', 'N/A'))}
"""
        
        # Subir resumen a Google Drive
        resultado_resumen, mensaje_resumen = subir_texto_google_drive(
            resumen_completo, 
            nombre_resumen
        )
        
        if resultado_resumen:
            log_info(f"✅ Resumen subido: {resultado_resumen.get('name')}", func_name)
        else:
            log_warning(f"⚠️ Error subiendo resumen: {mensaje_resumen}", func_name)
        
        # Subir TRANSCRIPCIÓN COMPLETA a Google Drive
        if transcripcion_completa and len(transcripcion_completa.strip()) > 50:
            nombre_transcripcion = f"TRANSCRIPCION_COMPLETA_{nombre_video_limpio}_{timestamp}.txt"
            
            contenido_transcripcion = f"""TRANSCRIPCIÓN COMPLETA DEL AUDIO
===============================================

AUDIO ORIGEN: {video_origen}
FECHA TRANSCRIPCIÓN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
TÉRMINOS DETECTADOS: {', '.join(terminos_detectados)}
TOTAL CLIPS GENERADOS: {len(clips_generados)}

===============================================
TRANSCRIPCIÓN COMPLETA:
===============================================

{transcripcion_completa}

===============================================
GENERADO POR: Radio Analyzer IA v2.0
===============================================
"""
            
            resultado_transcripcion, mensaje_transcripcion = subir_texto_google_drive(
                contenido_transcripcion, 
                nombre_transcripcion
            )
            
            if resultado_transcripcion:
                log_info(f"✅ Transcripción completa subida: {resultado_transcripcion.get('name')}", func_name)
                st.success(f"📝 Transcripción completa enviada a Google Drive")
            else:
                log_warning(f"⚠️ Error subiendo transcripción completa: {mensaje_transcripcion}", func_name)
        
        # Subir cada clip y su transcripción TXT
        clips_subidos = 0
        clips_fallidos = 0
        transcripciones_subidas = 0
        transcripciones_fallidas = 0
        
        for i, clip in enumerate(clips_generados, 1):
            clip_path = clip.get('path', '')
            txt_path = os.path.splitext(clip_path)[0] + '.txt'
            
            # Subir clip de audio
            if os.path.exists(clip_path):
                # Crear nombre único para el clip
                nombre_clip = f"CLIP_{i:02d}_{clip.get('termino', 'termino')}_{clip.get('tiempo', 'tiempo')}_{os.path.basename(clip_path)}"
                
                resultado_clip, mensaje_clip = subir_archivo_google_drive(
                    clip_path, 
                    nombre_clip
                )
                
                if resultado_clip:
                    clips_subidos += 1
                    log_info(f"✅ Clip {i} subido: {resultado_clip.get('name')}", func_name)
                else:
                    clips_fallidos += 1
                    log_warning(f"⚠️ Error subiendo clip {i}: {mensaje_clip}", func_name)
            else:
                clips_fallidos += 1
                log_warning(f"⚠️ Archivo de audio no existe: {clip_path}", func_name)
            
            # Subir transcripción TXT
            if os.path.exists(txt_path):
                # Crear nombre único para la transcripción
                nombre_txt = f"TRANSCRIPCION_{i:02d}_{clip.get('termino', 'termino')}_{clip.get('tiempo', 'tiempo')}_{os.path.basename(txt_path)}"
                
                resultado_txt, mensaje_txt = subir_archivo_google_drive(
                    txt_path, 
                    nombre_txt
                )
                
                if resultado_txt:
                    transcripciones_subidas += 1
                    log_info(f"✅ Transcripción {i} subida: {resultado_txt.get('name')}", func_name)
                else:
                    transcripciones_fallidas += 1
                    log_warning(f"⚠️ Error subiendo transcripción {i}: {mensaje_txt}", func_name)
            else:
                transcripciones_fallidas += 1
                log_warning(f"⚠️ Archivo de transcripción no existe: {txt_path}", func_name)
        
        # Crear resumen de envío
        resumen_envio = f"""ENVÍO COMPLETADO A GOOGLE DRIVE
===============================================

AUDIO: {video_origen}
FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTADOS:
- Resumen ejecutivo: {'✅ Subido' if resultado_resumen else '❌ Falló'}
- Clips de audio subidos: {clips_subidos}/{len(clips_generados)}
- Clips de audio fallidos: {clips_fallidos}
- Transcripciones TXT subidas: {transcripciones_subidas}/{len(clips_generados)}
- Transcripciones TXT fallidas: {transcripciones_fallidas}

CARPETA DESTINO: {GOOGLE_DRIVE_FOLDER_ID}
"""
        
        log_info(f"Envío a Google Drive completado: {clips_subidos} clips subidos, {clips_fallidos} fallidos, {transcripciones_subidas} transcripciones subidas, {transcripciones_fallidas} fallidas", func_name)
        
        return True, f"✅ {clips_subidos} clips, {transcripciones_subidas} transcripciones TXT y resumen enviados a Google Drive"
        
    except Exception as e:
        error_msg = f"Error en envío a Google Drive: {str(e)[:100]}"
        log_exception(func_name, e, error_msg)
        return False, error_msg

def test_google_drive_connection():
    """Prueba la conexión con Google Drive"""
    try:
        service, mensaje = crear_servicio_google_drive()
        if service:
            # Intentar listar archivos en la carpeta
            results = service.files().list(
                pageSize=1,
                q=f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents",
                fields="files(id, name)"
            ).execute()
            
            return True, f"✅ Conexión exitosa. Archivos en carpeta: {len(results.get('files', []))}"
        else:
            return False, f"❌ Error: {mensaje}"
    except Exception as e:
        return False, f"❌ Error probando conexión: {str(e)[:100]}"

def enviar_webhook_simple(url, data):
    """
    Envía un webhook simple para pruebas de conexión
    """
    func_name = "enviar_webhook_simple"
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RadioAnalizer/2.1'
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, f"Webhook respondió correctamente (HTTP {response.status_code})"
        else:
            return False, f"Webhook respondió con error (HTTP {response.status_code})"
            
    except requests.exceptions.Timeout:
        return False, "Timeout: Webhook no respondió en 10 segundos"
    except requests.exceptions.ConnectionError:
        return False, "Error de conexión: No se pudo conectar al webhook"
    except Exception as e:
        return False, f"Error inesperado: {str(e)[:100]}"

def enviar_coincidencias_a_supabase(coincidencias_items, nombre_archivo, tipo_archivo, resumen_archivo="", transcripcion_completa="", url_video=None, enlace_directo=None):
    """
    Envía las coincidencias encontradas a la tabla 'alertas_medios' en Supabase
    
    Args:
        coincidencias_items: Lista de diccionarios con las coincidencias
        nombre_archivo: Nombre del archivo procesado
        tipo_archivo: Tipo de archivo (video, audio, etc.)
        resumen_archivo: Resumen del archivo (opcional)
        transcripcion_completa: Transcripción completa (opcional)
        url_video: URL del video en Cloudinary (opcional)
        enlace_directo: Enlace directo al video (opcional)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    func_name = "enviar_coincidencias_a_supabase"
    
    if not supabase:
        return False, "❌ Cliente de Supabase no inicializado"
    
    if not coincidencias_items:
        return True, "ℹ️ No hay coincidencias para enviar"
    
    # ========== CONTROL DE DUPLICADOS PARA SUPABASE ==========
    # Filtrar coincidencias que ya fueron enviadas a Supabase en esta sesión
    coincidencias_no_duplicadas = []
    duplicados_detectados = 0
    
    for item in coincidencias_items:
        # Crear clave única para esta coincidencia (término + timestamp + archivo)
        termino = item.get('termino', '')
        timestamp = item.get('timestamp', '')
        clave_supabase = f"{termino}_{timestamp}_{nombre_archivo}"
        
        # Verificar si ya fue enviada a Supabase
        if clave_supabase in st.session_state.coincidencias_enviadas_supabase:
            duplicados_detectados += 1
            log_info(f"⏭️ DUPLICADO SUPABASE EVITADO: '{termino}' en {timestamp}s para {nombre_archivo}", func_name)
            continue
        
        # Agregar a la lista de no duplicados
        coincidencias_no_duplicadas.append(item)
        # Marcar como enviada
        st.session_state.coincidencias_enviadas_supabase.add(clave_supabase)
    
    if duplicados_detectados > 0:
        log_info(f"🛡️ CONTROL DUPLICADOS SUPABASE: {duplicados_detectados} duplicados evitados", func_name)
        st.info(f"🛡️ **Control Duplicados**: {duplicados_detectados} coincidencias duplicadas evitadas para Supabase")
    
    # Si no hay coincidencias nuevas, retornar éxito
    if not coincidencias_no_duplicadas:
        return True, f"ℹ️ Todas las coincidencias ya fueron enviadas a Supabase ({duplicados_detectados} duplicados evitados)"
    
    # Usar solo las coincidencias no duplicadas
    coincidencias_items = coincidencias_no_duplicadas
    
    try:
        resumen_archivo = capitalizar_marcas_medios_rd_en_texto(resumen_archivo or "")
        transcripcion_completa = capitalizar_marcas_medios_rd_en_texto(transcripcion_completa or "")
        nombre_archivo = capitalizar_marcas_medios_rd_en_texto(nombre_archivo or "")
        # Extraer información del medio y hora del nombre del archivo
        info_medio_hora = extraer_info_medio_hora(nombre_archivo)
        
        # Intentar extraer nombre del medio y hora/fecha del programa
        nombre_medio = "Medio de Comunicación"
        hora_programa = None
        fecha_programa = None
        
        # Parsear info_medio_hora (formato típico: "MEDIO - HH:MM" o "MEDIO HH:MM")
        if info_medio_hora:
            partes = info_medio_hora.split('-')
            if len(partes) >= 2:
                nombre_medio = partes[0].strip()
                hora_str = partes[1].strip()
                
                # Intentar parsear la hora
                try:
                    from datetime import datetime
                    # Extraer hora si está en formato HH:MM o HH:MM:SS
                    import re
                    match_hora = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', hora_str)
                    if match_hora:
                        hora = int(match_hora.group(1))
                        minuto = int(match_hora.group(2))
                        segundo = int(match_hora.group(3)) if match_hora.group(3) else 0
                        
                        # Crear objeto time
                        from datetime import time
                        hora_programa = time(hora, minuto, segundo)
                except Exception as e:
                    log_warning(f"No se pudo parsear hora del programa: {e}", func_name)
            elif len(partes) == 1:
                nombre_medio = partes[0].strip()
        
        # Intentar extraer fecha del nombre del archivo
        # Formato común: YYYY-MM-DD o YYYYMMDD
        try:
            import re
            # Buscar patrón de fecha en el nombre del archivo
            match_fecha = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', nombre_archivo)
            if match_fecha:
                from datetime import date
                año = int(match_fecha.group(1))
                mes = int(match_fecha.group(2))
                dia = int(match_fecha.group(3))
                fecha_programa = date(año, mes, dia)
        except Exception as e:
            log_warning(f"No se pudo parsear fecha del programa: {e}", func_name)
        
        nombre_medio = capitalizar_marcas_medios_rd_en_texto(nombre_medio)

        # Preparar datos para Supabase - SIN NULLS
        datos_supabase = []
        
        # Usar fecha/hora actual si no se pudieron extraer
        from datetime import datetime as dt
        fecha_actual = dt.now().date()
        hora_actual = dt.now().time()
        
        for item in coincidencias_items:
            # Obtener URL del clip de Cloudinary desde el item
            url_clip = item.get('url_cloudinary', url_video)
            
            # ASEGURAR QUE NO HAYA NULLS - Usar valores por defecto
            # Timestamp actual en formato ISO 8601 para PostgreSQL
            timestamp_actual = dt.now().isoformat()
            
            registro = {
                'fecha_detencion': timestamp_actual,  # Timestamp de cuando se detectó
                'termino_detectado': capitalizar_marcas_medios_rd_en_texto(str(item.get('termino', 'termino_desconocido'))),
                'nombre_medio': nombre_medio if nombre_medio else 'Medio de Comunicacion',
                'hora_programa': hora_programa.isoformat() if hora_programa else hora_actual.isoformat(),
                'fecha_programa': fecha_programa.isoformat() if fecha_programa else fecha_actual.isoformat(),
                'url_video': url_clip if url_clip else '',  # URL de Cloudinary del CLIP
                'nombre_archivo': nombre_archivo if nombre_archivo else 'archivo_desconocido',
                'enlace_directo': enlace_directo if enlace_directo else '',
                'contexto': capitalizar_marcas_medios_rd_en_texto(
                    item.get('texto', '') or item.get('contexto', '') or 'Sin contexto disponible'
                ),
                'resumen_ejecutivo': resumen_archivo if resumen_archivo else 'Resumen no disponible',
                'transcripcion': transcripcion_completa if transcripcion_completa else 'Transcripcion no disponible',
                'relevancia': 'Alta'
            }
            
            # Validar que NO haya None en ningún campo
            for key, value in registro.items():
                if value is None:
                    if key in ['hora_programa']:
                        registro[key] = hora_actual.isoformat()
                    elif key in ['fecha_programa']:
                        registro[key] = fecha_actual.isoformat()
                    elif key in ['fecha_detencion']:
                        registro[key] = dt.now().isoformat()
                    elif key in ['url_video', 'enlace_directo']:
                        registro[key] = ''
                    else:
                        registro[key] = f'{key}_no_disponible'
            
            datos_supabase.append(registro)
            
            # Log de lo que se va a enviar
            log_info(f"""
📤 Preparando envío a Supabase (SIN NULLS):
   - Timestamp Detección: {registro['fecha_detencion']}
   - Término: {registro['termino_detectado']}
   - Medio: {registro['nombre_medio']}
   - Hora Programa: {registro['hora_programa']}
   - Fecha Programa: {registro['fecha_programa']}
   - URL Clip: {registro['url_video'] or 'VACIO'}
   - Archivo: {registro['nombre_archivo']}
   - Contexto: {len(registro['contexto'])} caracteres
   - Resumen: {len(registro['resumen_ejecutivo'])} caracteres
   - Transcripción: {len(registro['transcripcion'])} caracteres
""", func_name)
        
        # Mostrar en UI lo que se va a enviar con detalles
        st.info(f"📊 **Enviando a Supabase:** {len(datos_supabase)} coincidencia(s)")
        
        # Mostrar resumen de datos que se enviarán
        with st.expander("📋 Ver datos que se enviarán (sin NULLs)", expanded=False):
            for i, reg in enumerate(datos_supabase, 1):
                st.markdown(f"""
**Registro {i}:**
- ⏱️ Timestamp Detección: `{reg['fecha_detencion']}`
- 🎯 Término: `{reg['termino_detectado']}`
- 📺 Medio: `{reg['nombre_medio']}`
- ⏰ Hora Programa: `{reg['hora_programa']}`
- 📅 Fecha Programa: `{reg['fecha_programa']}`
- 🔗 URL: `{'✅ SI' if reg['url_video'] else '❌ NO'}`
- 📄 Archivo: `{reg['nombre_archivo']}`
- 💬 Contexto: `{len(reg['contexto'])} chars`
- 📝 Resumen: `{len(reg['resumen_ejecutivo'])} chars`
- 📜 Transcripción: `{len(reg['transcripcion'])} chars`
""")
                st.markdown("---")
        
        # Insertar en Supabase
        result = supabase.table('alertas_medios').insert(datos_supabase).execute()
        
        if result.data:
            log_info(f"✅ Enviadas {len(datos_supabase)} coincidencias a Supabase para archivo: {nombre_archivo}", func_name)
            st.success(f"✅ Supabase: {len(result.data)} registro(s) insertado(s)")
            return True, f"✅ Enviadas {len(datos_supabase)} coincidencias a Supabase"
        else:
            log_error_critico(func_name, f"❌ Error al insertar en Supabase: {result}")
            st.error(f"❌ Supabase: No se insertaron datos")
            return False, f"❌ Error al insertar en Supabase: {result}"
            
    except Exception as e:
        error_msg = f"Error enviando a Supabase: {str(e)}"
        log_error_critico(func_name, error_msg)
        st.error(f"❌ Supabase: {error_msg}")
        
        # Mostrar traceback completo para debugging
        import traceback
        log_error_critico(func_name, f"Traceback completo:\n{traceback.format_exc()}")
        
        return False, f"❌ {error_msg}"

def verificar_todas_las_apis():
    """Verifica el estado de todas las APIs antes del procesamiento"""
    func_name = "verificar_todas_las_apis"
    resultados = {}
    
    st.info("🔍 **VERIFICANDO ESTADO DE TODAS LAS APIs**")
    st.markdown("---")
    
    # 1. Verificar Google Drive
    with st.spinner("☁️ Verificando Google Drive..."):
        try:
            gdrive_ok, gdrive_msg = test_google_drive_connection()
            resultados['google_drive'] = {
                'activo': gdrive_ok,
                'mensaje': gdrive_msg,
                'icono': '✅' if gdrive_ok else '❌'
            }
            if gdrive_ok:
                st.success(f"☁️ **Google Drive**: {gdrive_msg}")
            else:
                st.error(f"☁️ **Google Drive**: {gdrive_msg}")
        except Exception as e:
            resultados['google_drive'] = {
                'activo': False,
                'mensaje': f"Error verificando Google Drive: {str(e)[:100]}",
                'icono': '❌'
            }
            st.error(f"☁️ **Google Drive**: Error verificando conexión")
    
    # 2. Verificar Telegram
    with st.spinner("📱 Verificando Telegram..."):
        try:
            telegram_config = cargar_telegram_config()
            if telegram_config['enabled'] and telegram_config['bot_token'] and telegram_config['chat_id']:
                # Probar envío de mensaje de prueba
                mensaje_prueba = f"🧪 Prueba de conexión - {datetime.now().strftime('%H:%M:%S')}"
                telegram_ok, telegram_msg = enviar_mensaje_telegram(
                    mensaje_prueba,
                    telegram_config['chat_id'],
                    telegram_config['bot_token']
                )
                resultados['telegram'] = {
                    'activo': telegram_ok,
                    'mensaje': telegram_msg,
                    'icono': '✅' if telegram_ok else '❌'
                }
                if telegram_ok:
                    st.success(f"📱 **Telegram**: {telegram_msg}")
                else:
                    st.error(f"📱 **Telegram**: {telegram_msg}")
            else:
                resultados['telegram'] = {
                    'activo': False,
                    'mensaje': "Telegram no configurado",
                    'icono': '⚠️'
                }
                st.warning("📱 **Telegram**: No configurado")
        except Exception as e:
            resultados['telegram'] = {
                'activo': False,
                'mensaje': f"Error verificando Telegram: {str(e)[:100]}",
                'icono': '❌'
            }
            st.error(f"📱 **Telegram**: Error verificando conexión")
    
    # 3. Verificar Webhook
    with st.spinner("🌐 Verificando Webhook..."):
        try:
            webhook_config = cargar_webhook_config()
            if webhook_config['enabled'] and webhook_config['url']:
                # Probar webhook con mensaje de prueba
                data_prueba = {
                    "tipo": "prueba_conexion",
                    "mensaje": f"Prueba de conexión - {datetime.now().strftime('%H:%M:%S')}",
                    "timestamp": datetime.now().isoformat()
                }
                webhook_ok, webhook_msg = enviar_webhook_simple(webhook_config['url'], data_prueba)
                resultados['webhook'] = {
                    'activo': webhook_ok,
                    'mensaje': webhook_msg,
                    'icono': '✅' if webhook_ok else '❌'
                }
                if webhook_ok:
                    st.success(f"🌐 **Webhook**: {webhook_msg}")
                else:
                    st.error(f"🌐 **Webhook**: {webhook_msg}")
            else:
                resultados['webhook'] = {
                    'activo': False,
                    'mensaje': "Webhook no configurado",
                    'icono': '⚠️'
                }
                st.warning("🌐 **Webhook**: No configurado")
        except Exception as e:
            resultados['webhook'] = {
                'activo': False,
                'mensaje': f"Error verificando Webhook: {str(e)[:100]}",
                'icono': '❌'
            }
            st.error(f"🌐 **Webhook**: Error verificando conexión")
    
    # 4. Verificar Brevo
    with st.spinner("📧 Verificando Brevo..."):
        try:
            brevo_config = cargar_brevo_config()
            if brevo_config['enabled'] and brevo_config['api_key'] and brevo_config['sender_email']:
                # Probar envío de correo de prueba
                correo_ok, correo_msg = enviar_correo_brevo(
                    "PRUEBA DE CONEXIÓN",
                    "Este es un correo de prueba para verificar la conexión con Brevo.",
                    "Prueba de conexión",
                    info_medio="Sistema de Verificación"
                )
                resultados['brevo'] = {
                    'activo': correo_ok,
                    'mensaje': correo_msg,
                    'icono': '✅' if correo_ok else '❌'
                }
                if correo_ok:
                    st.success(f"📧 **Brevo**: {correo_msg}")
                else:
                    st.error(f"📧 **Brevo**: {correo_msg}")
            else:
                resultados['brevo'] = {
                    'activo': False,
                    'mensaje': "Brevo no configurado",
                    'icono': '⚠️'
                }
                st.warning("📧 **Brevo**: No configurado")
        except Exception as e:
            resultados['brevo'] = {
                'activo': False,
                'mensaje': f"Error verificando Brevo: {str(e)[:100]}",
                'icono': '❌'
            }
            st.error(f"📧 **Brevo**: Error verificando conexión")
    
    # 5. Verificar Cloudinary
    with st.spinner("☁️ Verificando Cloudinary..."):
        try:
            cloudinary_config = cargar_cloudinary_config()
            if cloudinary_config['enabled'] and cloudinary_config['cloud_name'] and cloudinary_config['api_key']:
                # Probar subida de archivo de prueba
                archivo_prueba = "test_cloudinary.txt"
                with open(archivo_prueba, 'w', encoding='utf-8') as f:
                    f.write("Prueba de conexión con Cloudinary")
                
                cloudinary_ok, cloudinary_msg, cloudinary_url = subir_video_cloudinary(
                    archivo_prueba,
                    "Prueba de conexión"
                )
                
                # Limpiar archivo de prueba
                if os.path.exists(archivo_prueba):
                    os.remove(archivo_prueba)
                
                resultados['cloudinary'] = {
                    'activo': cloudinary_ok,
                    'mensaje': cloudinary_msg,
                    'icono': '✅' if cloudinary_ok else '❌'
                }
                if cloudinary_ok:
                    st.success(f"☁️ **Cloudinary**: {cloudinary_msg}")
                else:
                    st.error(f"☁️ **Cloudinary**: {cloudinary_msg}")
            else:
                resultados['cloudinary'] = {
                    'activo': False,
                    'mensaje': "Cloudinary no configurado",
                    'icono': '⚠️'
                }
                st.warning("☁️ **Cloudinary**: No configurado")
        except Exception as e:
            resultados['cloudinary'] = {
                'activo': False,
                'mensaje': f"Error verificando Cloudinary: {str(e)[:100]}",
                'icono': '❌'
            }
            st.error(f"☁️ **Cloudinary**: Error verificando conexión")
    
    # Resumen final
    st.markdown("---")
    st.subheader("📊 **RESUMEN DE VERIFICACIÓN**")
    
    # Contar APIs activas
    apis_activas = sum(1 for api in resultados.values() if api['activo'])
    total_apis = len(resultados)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("APIs Activas", f"{apis_activas}/{total_apis}")
    with col2:
        st.metric("Estado General", "✅ Listo" if apis_activas > 0 else "❌ Sin APIs")
    with col3:
        st.metric("Google Drive", "✅ Activo" if resultados.get('google_drive', {}).get('activo') else "❌ Inactivo")
    
    # Mostrar estado de cada API
    for api_name, estado in resultados.items():
        nombre_display = api_name.replace('_', ' ').title()
        st.write(f"{estado['icono']} **{nombre_display}**: {estado['mensaje']}")
    
    log_info(f"Verificación de APIs completada: {apis_activas}/{total_apis} activas", func_name)
    return resultados

# === GENERAR MD DE COINCIDENCIAS EN CARPETA PROCESADOS ===

def escribir_informe_general_radio(contenido_md):
    """Actualiza el informe general visible fuera de la carpeta de procesados."""
    func_name = "escribir_informe_general_radio"
    try:
        informe_path = INFORME_GENERAL_RADIO_PATH
        os.makedirs(os.path.dirname(informe_path), exist_ok=True)
        with open(informe_path, "w", encoding="utf-8") as f:
            f.write(contenido_md)
        log_info(f"✅ Informe general actualizado: {informe_path}", func_name)
        return True, informe_path
    except Exception as e:
        log_exception(func_name, e, "Error escribiendo informe general")
        return False, str(e)


def generar_md_sesion_coincidencias(videos_procesados_data, clips_generados_en_sesion, estadisticas_escaneo=None, terminos_buscados=None, menciones_tangenciales_data=None):
    """
    Genera un archivo Markdown con el reporte COMPLETO de la sesión:
    - Estadísticas generales (videos encontrados, procesados, etc.)
    - Todas las coincidencias de todos los videos
    - Todos los clips generados
    - Transcripciones de cada video
    - Uso de API Mistral/Voxtral
    
    Se guarda en la carpeta 'videos procesados'.
    """
    func_name = "generar_md_sesion_coincidencias"
    try:
        fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fecha_legible = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        md_filename = f"SESION_COINCIDENCIAS_{fecha_str}.md"
        md_path = os.path.join(CARPETA_PROCESADOS, md_filename)
        
        log_info(f"Generando MD de sesión: {md_path}", func_name)
        
        # Contadores globales
        total_coincidencias = sum(len(v.get('coincidencias_items', [])) for v in videos_procesados_data)
        total_clips = len(clips_generados_en_sesion)
        menciones_tangenciales_data = menciones_tangenciales_data or []
        total_videos_con_coincidencias = len(videos_procesados_data)
        total_tangenciales = len(menciones_tangenciales_data)
        todos_terminos = list(set(
            t for v in videos_procesados_data for t in v.get('terminos_encontrados', [])
        ))
        
        # Uso de Mistral/Voxtral
        mistral_audio_secs = st.session_state.get('mistral_total_audio_seconds', 0)
        mistral_tokens = st.session_state.get('mistral_total_tokens', 0)
        mistral_transcripciones = st.session_state.get('mistral_total_transcripciones', 0)
        mistral_costo = (mistral_audio_secs / 60) * 0.012
        
        md = []
        
        # ============================
        # ENCABEZADO
        # ============================
        md.append(f"# 🎯 Reporte de Sesión - Coincidencias Detectadas")
        md.append(f"")
        md.append(f"> **Radio Analyzer IA v2.0** | Sesión: {fecha_legible}")
        md.append(f"")
        md.append(f"---")
        md.append(f"")
        
        # ============================
        # ESTADÍSTICAS DE LA SESIÓN
        # ============================
        md.append(f"## 📊 Estadísticas de la Sesión")
        md.append(f"")
        md.append(f"| Métrica | Valor |")
        md.append(f"|---------|-------|")
        
        if estadisticas_escaneo:
            md.append(f"| **Total archivos escaneados** | {estadisticas_escaneo.get('total_archivos', 'N/A')} |")
            md.append(f"| **Audios encontrados** | {estadisticas_escaneo.get('total_videos', 'N/A')} |")
            md.append(f"| **Archivos nuevos (a procesar)** | {estadisticas_escaneo.get('archivos_nuevos', 'N/A')} |")
            md.append(f"| **Ya procesados previamente** | {estadisticas_escaneo.get('archivos_procesados', 'N/A')} |")
            md.append(f"| **Omitidos por tamaño** | {estadisticas_escaneo.get('archivos_muy_pequeños', 0)} |")
        
        md.append(f"| **Videos con coincidencias** | {total_videos_con_coincidencias} |")
        md.append(f"| **Total coincidencias detectadas** | {total_coincidencias} |")
        md.append(f"| **Total menciones tangenciales** | {total_tangenciales} |")
        md.append(f"| **Total clips generados** | {total_clips} |")
        md.append(f"| **Términos detectados** | {', '.join(todos_terminos) if todos_terminos else 'Ninguno'} |")
        md.append(f"| **Fecha/hora del reporte** | {fecha_legible} |")
        md.append(f"")
        
        # Términos buscados
        if terminos_buscados:
            terminos_nombres = [t.get('termino', str(t)) if isinstance(t, dict) else str(t) for t in terminos_buscados]
            md.append(f"**Términos buscados en esta sesión:** {', '.join(terminos_nombres)}")
            md.append(f"")
        
        md.append(f"---")
        md.append(f"")
        
        # ============================
        # USO DE API MISTRAL/VOXTRAL
        # ============================
        md.append(f"## 🧠 Uso de API Mistral / Voxtral")
        md.append(f"")
        md.append(f"| Métrica | Valor |")
        md.append(f"|---------|-------|")
        md.append(f"| **Transcripciones realizadas** | {mistral_transcripciones} |")
        audio_min = int(mistral_audio_secs // 60)
        audio_seg = int(mistral_audio_secs % 60)
        md.append(f"| **Audio procesado** | {audio_min}m {audio_seg}s ({mistral_audio_secs:.0f}s) |")
        md.append(f"| **Tokens totales** | {mistral_tokens:,} |")
        md.append(f"| **Prompt tokens** | {st.session_state.get('mistral_total_prompt_tokens', 0):,} |")
        md.append(f"| **Completion tokens** | {st.session_state.get('mistral_total_completion_tokens', 0):,} |")
        md.append(f"| **Costo estimado** | ${mistral_costo:.4f} USD |")
        md.append(f"")
        md.append(f"---")
        md.append(f"")
        
        # ============================
        # DETALLE POR CADA VIDEO
        # ============================
        if videos_procesados_data:
            md.append(f"## 🎬 Detalle por Video ({total_videos_con_coincidencias} con coincidencias)")
            md.append(f"")
            
            for vid_idx, video_data in enumerate(videos_procesados_data, 1):
                nombre = video_data.get('nombre_archivo', '?')
                tipo = video_data.get('tipo_archivo', 'video')
                resumen = video_data.get('resumen_archivo', '')
                coincidencias = video_data.get('coincidencias_items', [])
                clips = video_data.get('clips_info', [])
                transcripcion = video_data.get('transcripcion_completa', '')
                terminos_vid = video_data.get('terminos_encontrados', [])
                
                md.append(f"### 🎵 Audio {vid_idx}: `{nombre}`")
                md.append(f"")
                md.append(f"| Campo | Valor |")
                md.append(f"|-------|-------|")
                md.append(f"| **Archivo** | `{nombre}` |")
                md.append(f"| **Tipo** | {tipo} |")
                md.append(f"| **Coincidencias** | {len(coincidencias)} |")
                md.append(f"| **Clips generados** | {len(clips)} |")
                md.append(f"| **Términos encontrados** | {', '.join(terminos_vid)} |")
                md.append(f"")
                
                # Resumen ejecutivo del video
                if resumen:
                    md.append(f"#### 📝 Resumen Ejecutivo")
                    md.append(f"")
                    md.append(resumen)
                    md.append(f"")
                
                # Coincidencias de este video
                if coincidencias:
                    md.append(f"#### 🔍 Coincidencias ({len(coincidencias)})")
                    md.append(f"")
                    
                    for c_idx, item in enumerate(coincidencias, 1):
                        termino = item.get('termino', '?')
                        contexto = item.get('contexto', '') or item.get('texto', '')
                        timestamp = item.get('timestamp', 0)
                        url_cloud = item.get('url_cloudinary', None)
                        
                        ts_min = int(timestamp // 60) if timestamp else 0
                        ts_seg = int(timestamp % 60) if timestamp else 0
                        ts_display = f"{ts_min}m {ts_seg:02d}s"
                        
                        md.append(f"**{c_idx}. {termino.upper()}** @ {ts_display} ({timestamp:.1f}s)")
                        if url_cloud:
                            md.append(f"   - URL: [{url_cloud}]({url_cloud})")
                        if contexto:
                            md.append(f"   - Contexto: {contexto[:500]}")
                        md.append(f"")
                
                # Clips de este video
                if clips:
                    md.append(f"#### 🎬 Clips Generados ({len(clips)})")
                    md.append(f"")
                    
                    for cl_idx, clip in enumerate(clips, 1):
                        clip_termino = clip.get('termino', '?')
                        clip_tiempo = clip.get('tiempo', '?')
                        clip_path = clip.get('path', '')
                        clip_url = clip.get('url_cloudinary', None)
                        clip_momento = clip.get('momento_exacto', 0)
                        clip_verificado = clip.get('verificado', False)
                        
                        md.append(f"**Clip {cl_idx}:** {clip_termino} @ {clip_tiempo} | Verificado: {'Si' if clip_verificado else 'No'} | Archivo: `{os.path.basename(clip_path)}`")
                        if clip_url:
                            md.append(f"   - URL Cloudinary: [{clip_url}]({clip_url})")
                        md.append(f"")
                
                # Transcripción de este video
                if transcripcion and len(transcripcion.strip()) > 10:
                    md.append(f"#### 📜 Transcripción ({len(transcripcion.split())} palabras)")
                    md.append(f"")
                    md.append(f"<details>")
                    md.append(f"<summary>Ver transcripción completa</summary>")
                    md.append(f"")
                    md.append(f"```")
                    md.append(transcripcion)
                    md.append(f"```")
                    md.append(f"")
                    md.append(f"</details>")
                    md.append(f"")
                
                md.append(f"---")
                md.append(f"")
        else:
            md.append(f"## ℹ️ Sin coincidencias en esta sesión")
            md.append(f"")
            md.append(f"No se encontraron coincidencias en los videos procesados.")
            md.append(f"")
            md.append(f"---")
            md.append(f"")

        if menciones_tangenciales_data:
            _, md_tangenciales, _ = construir_tangenciales_narrativo(menciones_tangenciales_data)
            md.append(f"## ⚠️ Menciones Tangenciales ({total_tangenciales})")
            md.append(f"")
            md.append(md_tangenciales)
            md.append(f"")
            md.append(f"### Detalle Por Ocurrencia")
            md.append(f"")
            md.append(f"| Medio | Término | Motivo | Tiempo en audio | Hora detección |")
            md.append(f"| --- | --- | --- | --- | --- |")
            for tang in menciones_tangenciales_data:
                ts_seg = tang.get('timestamp', 0) or 0
                ta = formato_posicion_en_audio_segundos(ts_seg)
                medio_raw = tang.get('medio') or formatear_nombre_medio_desde_ruta(tang.get('archivo', ''))
                medio = capitalizar_marcas_medios_rd_en_texto(str(medio_raw)).replace('|', '\\|')
                term = capitalizar_marcas_medios_rd_en_texto(str(tang.get('termino', ''))).replace('|', '\\|')
                motivo = capitalizar_marcas_medios_rd_en_texto(str(tang.get('motivo', ''))).replace('|', '\\|')
                hd = _hora_deteccion_formateada(tang).replace('|', '\\|')
                md.append(f"| {medio} | **{term}** | {motivo} | {ta} | {hd} |")
            md.append(f"")
            md.append(f"---")
            md.append(f"")
        
        # ============================
        # PIE
        # ============================
        md.append(f"_Reporte generado automáticamente por Radio Analyzer IA v2.0 - {fecha_legible}_")
        
        # === ESCRIBIR ARCHIVO ===
        contenido_final = "\n".join(md)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(contenido_final)
        ok_informe, msg_informe = escribir_informe_general_radio(contenido_final)
        if not ok_informe:
            log_warning(f"No se pudo actualizar informe general: {msg_informe}", func_name)
        
        log_info(f"✅ MD de sesión guardado: {md_path} ({len(contenido_final)} caracteres, {total_coincidencias} coincidencias)", func_name)
        return True, md_path
        
    except Exception as e:
        log_exception(func_name, e, "Error generando MD de sesión")
        return False, str(e)

# === FUNCIONES DE CORREO BREVO ===

def generar_archivo_coincidencias_md(termino_encontrado, resumen_completo, nombre_video, info_medio="", terminos_detectados=[], video_url=None, video_path=None):
    """
    Genera un archivo Markdown con todas las coincidencias encontradas
    """
    func_name = "generar_archivo_coincidencias_md"
    
    try:
        # Dividir el resumen en transcripción y resumen ejecutivo
        transcripcion = ""
        resumen_ejecutivo = resumen_completo
        
        if "**TRANSCRIPCIÓN DEL CONTENIDO:**" in resumen_completo:
            partes = resumen_completo.split("**RESUMEN EJECUTIVO:**")
            if len(partes) >= 2:
                transcripcion = partes[0].replace("**TRANSCRIPCIÓN DEL CONTENIDO:**", "").strip()
                resumen_ejecutivo = partes[1].strip()
        
        # Nombre del archivo MD
        archivo_md = "coincidencias.md"
        
        # Crear contenido del archivo MD
        contenido_md = f"""
# 🎯 Sistema de Alerta de Medios de EDESUR - Coincidencias

## 📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

---

## 🔍 Coincidencia Detectada: {termino_encontrado}

### 📺 Información del Medio
**{info_medio}**

### 🎧 Audio de la Coincidencia
"""
        
        # Agregar información del audio - OBLIGATORIO incluir URL de Cloudinary
        if video_url:
            contenido_md += f"""
**🎧 Audio de Cloudinary:** [{video_url}]({video_url})

**📁 Archivo:** `{nombre_video}`

**🔗 Enlace directo:** {video_url}

> ✅ **URL de Cloudinary disponible para escucha**
"""
        elif video_path and os.path.exists(video_path):
            contenido_md += f"""
**📁 Archivo Local:** `{video_path}`

**📁 Archivo:** `{nombre_video}`

> ⚠️ **Nota:** Audio local disponible, pero no subido a Cloudinary aún

> ❌ **IMPORTANTE:** Se requiere subir el audio a Cloudinary para obtener URL de consulta
"""
        else:
            contenido_md += f"""
**📁 Archivo:** `{nombre_video}`

> ❌ **ERROR:** Audio no disponible o no procesado

> ❌ **IMPORTANTE:** Se requiere subir el audio a Cloudinary para obtener URL de consulta
"""
        
        # Agregar términos detectados
        if terminos_detectados:
            contenido_md += f"""
### 🔍 Términos Detectados
"""
            for termino in terminos_detectados:
                contenido_md += f"- **{termino}**\n"
        
        # Agregar resumen ejecutivo
        contenido_md += f"""
### 🎯 Resumen Ejecutivo
{resumen_ejecutivo}

### 📝 Transcripción del Contenido
{transcripcion}

---

"""
        
        # Verificar si el archivo ya existe
        if os.path.exists(archivo_md):
            # Leer contenido existente
            with open(archivo_md, 'r', encoding='utf-8') as f:
                contenido_existente = f.read()
            
            # Insertar nueva coincidencia al inicio (después del encabezado)
            lineas = contenido_existente.split('\n')
            indice_insertar = 0
            
            # Encontrar donde insertar (después del encabezado y antes del primer separador)
            for i, linea in enumerate(lineas):
                # Buscar después del encabezado principal y antes del primer contenido
                if linea.startswith('----') and i > 5:  # Después del encabezado
                    indice_insertar = i
                    break
            
            # Insertar nueva coincidencia al principio del contenido
            lineas.insert(indice_insertar, contenido_md)
            contenido_final = '\n'.join(lineas)
        else:
            # Crear archivo nuevo con encabezado
            encabezado = f"""# 🎯 Sistema de Alerta de Medios de EDESUR - Coincidencias

> Archivo generado automáticamente por el sistema de monitoreo de medios
> 
> **Fecha de creación:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
> 
> **Descripción:** Este archivo contiene todas las coincidencias detectadas por el sistema de análisis de audios.

---

"""
            contenido_final = encabezado + contenido_md
        
        contenido_final = capitalizar_marcas_medios_rd_en_texto(contenido_final)

        # Escribir archivo
        with open(archivo_md, 'w', encoding='utf-8') as f:
            f.write(contenido_final)
        
        log_info(f"✅ Archivo MD actualizado: {archivo_md}", func_name)
        return True, f"Archivo MD actualizado: {archivo_md}"
        
    except Exception as e:
        log_exception(func_name, e)
        return False, f"Error generando archivo MD: {str(e)}"

def generar_analisishoy_md(nombre_archivo, termino_encontrado, contexto_termino="", resumen_ejecutivo="", transcripcion_completa="", video_url=None, info_medio=""):
    """
    Genera (o actualiza) el archivo Analisishoy_YYYYMMDD.md con los detalles
    completos de cada coincidencia detectada en la sesión de análisis de hoy.
    El archivo se guarda en el mismo directorio de trabajo del script.
    """
    func_name = "generar_analisishoy_md"
    try:
        fecha_hoy = datetime.now().strftime('%Y%m%d')
        hora_analisis = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        nombre_md = f"Analisishoy_{fecha_hoy}.md"
        ruta_md = os.path.join(CARPETA_PROCESADOS, nombre_md)

        # --- Sección de esta coincidencia ---
        contexto_display = contexto_termino[:800] if contexto_termino else "No disponible"
        transcripcion_display = transcripcion_completa.strip() if transcripcion_completa and len(transcripcion_completa.strip()) > 30 else "No disponible"

        seccion = f"""
---

# 📊 ANÁLISIS COMPLETO: {nombre_archivo}

## 📋 Información General
- **Archivo:** `{nombre_archivo}`
- **Fecha de análisis:** {hora_analisis}
- **Medio:** {info_medio}
- **Término detectado:** **{termino_encontrado}**
- **Audio Cloudinary:** {f'[{video_url}]({video_url})' if video_url else 'No disponible'}

{resumen_ejecutivo}

## 📝 Transcripción Completa

{transcripcion_display}

"""

        seccion = capitalizar_marcas_medios_rd_en_texto(seccion)

        # Si ya existe el archivo de hoy, acumulamos; si no, lo creamos con encabezado
        if os.path.exists(ruta_md):
            with open(ruta_md, 'a', encoding='utf-8') as f:
                f.write(seccion)
        else:
            encabezado = f"""# 📊 Análisis de hoy — {datetime.now().strftime('%d/%m/%Y')}

> Generado automáticamente por el sistema de monitoreo de medios.
> Archivo: `{nombre_md}`

"""
            with open(ruta_md, 'w', encoding='utf-8') as f:
                f.write(encabezado + seccion)

        log_info(f"✅ Analisishoy MD actualizado: {ruta_md}", func_name)
        return True, ruta_md

    except Exception as e:
        log_exception(func_name, e, "Error generando Analisishoy MD")
        return False, str(e)


def append_analisishoy_menciones_tangenciales(menciones_tangenciales_data):
    """
    Añade al Analisishoy_YYYYMMDD.md una sección narrativa agrupada por término (misma lógica
    que correo/UI) más tabla de apoyo. Si el archivo no existe, lo crea con encabezado estándar.
    """
    func_name = "append_analisishoy_menciones_tangenciales"
    if not menciones_tangenciales_data:
        return True, None
    try:
        fecha_hoy = datetime.now().strftime('%Y%m%d')
        nombre_md = f"Analisishoy_{fecha_hoy}.md"
        ruta_md = os.path.join(CARPETA_PROCESADOS, nombre_md)
        hora_lote = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        _, md_narr, _ = construir_tangenciales_narrativo(menciones_tangenciales_data)
        lineas_tabla = [
            "| Medio | Término | Motivo | Tiempo en audio | Hora detección |",
            "| --- | --- | --- | --- | --- |",
        ]
        for tang in menciones_tangenciales_data:
            ts_seg = tang.get('timestamp', 0) or 0
            ta = formato_posicion_en_audio_segundos(ts_seg)
            medio_raw = tang.get('medio') or formatear_nombre_medio_desde_ruta(tang.get('archivo', ''))
            medio = capitalizar_marcas_medios_rd_en_texto(str(medio_raw)).replace('|', '\\|')
            term = capitalizar_marcas_medios_rd_en_texto(str(tang.get('termino', ''))).replace('|', '\\|')
            motivo = capitalizar_marcas_medios_rd_en_texto(str(tang.get('motivo', ''))).replace('|', '\\|')
            hd = _hora_deteccion_formateada(tang).replace('|', '\\|')
            lineas_tabla.append(f"| {medio} | **{term}** | {motivo} | {ta} | {hd} |")

        lineas = [
            "",
            "---",
            "",
            f"*Menciones tangenciales (cierre de ciclo): {hora_lote}*",
            "",
            md_narr,
            "",
            "**Detalle por ocurrencia**",
            "",
            "\n".join(lineas_tabla),
            "",
        ]
        bloque = "\n".join(lineas)

        if os.path.exists(ruta_md):
            with open(ruta_md, 'a', encoding='utf-8') as f:
                f.write(bloque)
        else:
            encabezado = f"""# 📊 Análisis de hoy — {datetime.now().strftime('%d/%m/%Y')}

> Generado automáticamente por el sistema de monitoreo de medios.
> Archivo: `{nombre_md}`

"""
            with open(ruta_md, 'w', encoding='utf-8') as f:
                f.write(encabezado + bloque)

        log_info(f"✅ Analisishoy MD: menciones tangenciales añadidas ({len(menciones_tangenciales_data)}): {ruta_md}", func_name)
        return True, ruta_md
    except Exception as e:
        log_exception(func_name, e, "Error añadiendo menciones tangenciales a Analisishoy MD")
        return False, str(e)


def crear_plantilla_email_html(termino_encontrado, resumen_completo, nombre_video, info_medio="", terminos_detectados=[], video_url=None, transcripcion_segmento=""):
    """Crea una plantilla HTML moderna y elegante para el correo de coincidencia"""
    termino_encontrado = capitalizar_marcas_medios_rd_en_texto(str(termino_encontrado or "")).strip()
    resumen_completo = capitalizar_marcas_medios_rd_en_texto(resumen_completo or "")
    nombre_video = capitalizar_marcas_medios_rd_en_texto(str(nombre_video or ""))
    if info_medio is not None:
        ims = info_medio if isinstance(info_medio, str) else str(info_medio)
        ims = ims.strip()
        info_medio = capitalizar_marcas_medios_rd_en_texto(ims) if ims else ims
    if terminos_detectados:
        terminos_detectados = [capitalizar_marcas_medios_rd_en_texto(str(x)) for x in terminos_detectados]
    if transcripcion_segmento:
        transcripcion_segmento = capitalizar_marcas_medios_rd_en_texto(transcripcion_segmento)

    # Dividir el resumen en transcripción y resumen ejecutivo
    transcripcion = ""
    resumen_ejecutivo = resumen_completo
    
    if "**RESUMEN EJECUTIVO:**" in resumen_completo and "**TRANSCRIPCIÓN DEL CONTENIDO:**" in resumen_completo:
        partes = resumen_completo.split("**RESUMEN EJECUTIVO:**")
        if len(partes) >= 2:
            resto = partes[1]
            if "**TRANSCRIPCIÓN DEL CONTENIDO:**" in resto:
                partes_finales = resto.split("**TRANSCRIPCIÓN DEL CONTENIDO:**")
                resumen_ejecutivo = partes_finales[0].strip()
                transcripcion = partes_finales[1].strip()
    
    # Resaltar el segmento del clip si se proporciona
    if transcripcion and transcripcion_segmento:
        seg = transcripcion_segmento.strip()
        if seg and seg in transcripcion:
            highlight = f'<strong style="background:#fff3cd; padding: 2px 4px; border-radius: 4px; border: 1px solid #ffeeba;">{seg}</strong>'
            transcripcion = transcripcion.replace(seg, highlight)
    
    # Mejorar formato de transcripción con saltos de línea
    if transcripcion:
        # Dividir en párrafos basado en puntos y mayúsculas
        transcripcion = re.sub(r'\. ([A-Z])', r'.\n\n\1', transcripcion)
        # Limpiar espacios extra
        transcripcion = re.sub(r'\n\s*\n\s*\n', '\n\n', transcripcion)
    
    # Color scheme moderno y elegante
    primary_color = "#667eea"
    secondary_color = "#f8f9fa"
    accent_color = "#28a745"
    text_color = "#333"
    
    # Procesar términos encontrados
    terminos_html = ""
    if terminos_detectados:
        terminos_list = []
        for termino in terminos_detectados:
            terminos_list.append(f'<span style="background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%); color: white; padding: 8px 16px; border-radius: 25px; font-size: 14px; font-weight: 500; box-shadow: 0 2px 8px rgba(255, 71, 87, 0.3);">"{termino}"</span>')
        terminos_html = " ".join(terminos_list)
    else:
        terminos_html = f'<span style="background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%); color: white; padding: 8px 16px; border-radius: 25px; font-size: 14px; font-weight: 500; box-shadow: 0 2px 8px rgba(255, 71, 87, 0.3);">"{termino_encontrado}"</span>'
    
    # Información del medio
    medio_section = ""
    if info_medio:
        medio_section = f"""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #2196f3; margin: 20px 0; text-align: center;">
            <strong style="color: #1976d2; font-size: 16px;">📺 {info_medio}</strong>
        </div>
        """
    
    # Botón de reproducir al inicio
    play_intro_section = ""
    if video_url:
        play_intro_section = f"""
        <div style="text-align: center; margin: 30px 0;">
            <a href="{video_url}" target="_blank" style="background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%); border: none; border-radius: 50px; padding: 15px 30px; color: white; font-size: 18px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 12px; box-shadow: 0 4px 15px rgba(255, 71, 87, 0.3); text-decoration: none;">
                ▶️ Reproducir Audio de Coincidencia
            </a>
        </div>
        """
    
    # Sección de audio simplificada (evita embeds de video en clientes de correo)
    video_section = ""
    if video_url:
        video_section = f"""
        <div style="margin: 30px 0; text-align: center;">
            <h3 style="color: {primary_color}; margin: 0 0 20px 0; font-size: 20px; display: flex; align-items: center; justify-content: center;">
                <span style="margin-right: 10px;">🎧</span>
                Audio de la Coincidencia
            </h3>
            
            <div style="max-width: 650px; margin: 0 auto; background: #f8f9fa; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 25px rgba(0,0,0,0.12); padding: 22px;">
                <p style="margin: 0 0 12px 0; color: #495057; font-size: 15px;">
                    Este cliente de correo puede no reproducir audio embebido.
                </p>
                <a href="{video_url}" target="_blank" style="padding: 12px 24px; border-radius: 26px; text-decoration: none; font-weight: 600; display: inline-block; background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); color: white;">
                    🎧 Escuchar audio en nueva pestaña
                </a>
            </div>
            
            <div style="margin-top: 15px; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; border-left: 4px solid {primary_color};">
                <div style="font-weight: 600; color: #495057; margin-bottom: 8px; font-size: 16px;">{nombre_video}</div>
                <div style="color: #6c757d; font-size: 14px; display: flex; align-items: center; justify-content: center; gap: 8px;">
                    🌐 <span>Cloudinary CDN</span>
                </div>
            </div>
        </div>
        """
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema de Alerta de Medios de EDESUR: {termino_encontrado}</title>
    </head>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f8f9fa;">
        <div style="background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden;">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, {primary_color} 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 28px; font-weight: 300;">🎯 Sistema de Alerta de Medios de EDESUR</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;">Término: <strong>{termino_encontrado}</strong></p>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;">📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
            
            <!-- Content -->
            <div style="padding: 30px;">
                
                {play_intro_section}
                
                {video_section}
                
                <!-- Términos Detectados -->
                {f'''
                <div style="margin: 30px 0; padding: 25px; background: #f8f9fa; border-radius: 12px; border-left: 5px solid {accent_color};">
                    <h3 style="margin: 0 0 20px 0; color: {accent_color}; font-size: 20px; display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🔍</span>
                        Términos Detectados
                    </h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
                        {terminos_html}
                    </div>
                </div>
                ''' if terminos_detectados else ''}
                
                {medio_section}
                
                <!-- Resumen Ejecutivo -->
                <div style="background: linear-gradient(135deg, {primary_color} 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin: 25px 0; text-align: center;">
                    <h3 style="margin: 0 0 20px 0; color: white; font-size: 20px;">🎯 Resumen Ejecutivo</h3>
                    <p style="margin: 0; line-height: 1.8;">{resumen_ejecutivo}</p>
                </div>
                
                <!-- Transcripción -->
                {f'''
                <div style="margin: 30px 0; padding: 25px; background: #ffffff; border-radius: 12px; border: 1px solid #dee2e6;">
                    <h3 style="margin: 0 0 20px 0; color: #111111; font-size: 20px; display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">📝</span>
                        Transcripción del Contenido
                    </h3>
                    <div style="background: white; padding: 25px; border-radius: 12px; border: 1px solid #e9ecef; margin: 20px 0; font-family: 'Segoe UI', sans-serif; line-height: 1.8; color: #111111; white-space: pre-line; text-align: justify;">
{transcripcion}
                    </div>
                </div>
                ''' if transcripcion else ''}
                
                <!-- Botones Centrados -->
                <div style="display: flex; gap: 15px; justify-content: center; margin: 30px 0; flex-wrap: wrap;">
                    {f'<a href="{video_url}" target="_blank" style="padding: 15px 30px; border: none; border-radius: 30px; text-decoration: none; font-weight: 600; text-align: center; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 10px; cursor: pointer; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); color: white;">🔗 Escuchar en nueva pestaña</a>' if video_url else ''}
                    {f'<a href="{video_url}" target="_blank" style="padding: 15px 30px; border: none; border-radius: 30px; text-decoration: none; font-weight: 600; text-align: center; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 10px; cursor: pointer; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); color: white;">⬇️ Descargar Audio</a>' if video_url else ''}
                    <a href="#" style="padding: 15px 30px; border: none; border-radius: 30px; text-decoration: none; font-weight: 600; text-align: center; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 10px; cursor: pointer; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); color: white;">📊 Ver Análisis Completo</a>
                </div>
                
            </div>
            
            <!-- Footer -->
            <div style="background: #343a40; color: white; padding: 25px; text-align: center;">
                <p style="margin: 5px 0; opacity: 0.8;"><strong>🤖 Sistema de Monitoreo Automático</strong></p>
                <p style="margin: 5px 0; opacity: 0.8;">Generado automáticamente el {datetime.now().strftime('%d/%m/%Y a las %H:%M:%S')}</p>
                <p style="margin: 5px 0; opacity: 0.8;">Este es un correo automático, no responder.</p>
            </div>
            
        </div>
        
    </body>
    </html>
    """
    
    return html_template


# ============================================================================
# === SISTEMA DE RESUMEN DIARIO POR ENTIDAD (CORREO DIGEST) =================
# ============================================================================

def parsear_coincidencias_md(fecha_filtro=None, hora_inicio=None, hora_fin=None):
    """
    Lee coincidencias.md y extrae todas las coincidencias como lista de dicts.
    Filtra por fecha (formato DD/MM/YYYY) y rango horario opcional.
    
    Returns:
        list[dict]: Lista de coincidencias con campos:
            termino, medio, fecha_hora, video_url, resumen_ejecutivo, transcripcion
    """
    func_name = "parsear_coincidencias_md"
    archivo_md = "coincidencias.md"
    
    if not os.path.exists(archivo_md):
        log_warning(f"Archivo {archivo_md} no encontrado", func_name)
        return []
    
    try:
        with open(archivo_md, 'r', encoding='utf-8') as f:
            contenido = f.read()
    except Exception as e:
        log_warning(f"Error leyendo {archivo_md}: {e}", func_name)
        return []
    
    # Si no se pasa fecha, usar hoy
    if not fecha_filtro:
        fecha_filtro = datetime.now().strftime('%d/%m/%Y')
    
    coincidencias = []
    
    # Dividir por bloques de coincidencia (cada uno empieza con "## 📅")
    bloques = re.split(r'(?=## 📅)', contenido)
    
    for bloque in bloques:
        if not bloque.strip() or '## 📅' not in bloque:
            continue
        
        # Extraer fecha/hora
        match_fecha = re.search(r'## 📅\s*(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})', bloque)
        if not match_fecha:
            continue
        
        fecha_str = match_fecha.group(1)  # DD/MM/YYYY
        hora_str = match_fecha.group(2)   # HH:MM:SS
        
        # Filtrar por fecha
        if fecha_str != fecha_filtro:
            continue
        
        # Filtrar por rango horario si se especifica
        if hora_inicio and hora_fin:
            try:
                hora_actual = datetime.strptime(hora_str, '%H:%M:%S').time()
                h_inicio = datetime.strptime(hora_inicio, '%H:%M').time()
                h_fin = datetime.strptime(hora_fin, '%H:%M').time()
                if not (h_inicio <= hora_actual <= h_fin):
                    continue
            except:
                pass
        
        # Extraer término
        match_termino = re.search(r'Coincidencia Detectada:\s*(.+)', bloque)
        termino = match_termino.group(1).strip() if match_termino else 'Desconocido'
        
        # Extraer medio
        match_medio = re.search(r'Información del Medio\s*\n\*\*(.+?)\*\*', bloque)
        medio = match_medio.group(1).strip() if match_medio else ''
        
        # Extraer URL de Cloudinary
        match_url = re.search(r'Enlace directo:\s*(https://res\.cloudinary\.com/[^\s\n]+)', bloque)
        video_url = match_url.group(1).strip() if match_url else ''
        
        # Extraer nombre del archivo
        match_archivo = re.search(r'Archivo:\*\*\s*`([^`]+)`', bloque)
        nombre_archivo = match_archivo.group(1).strip() if match_archivo else ''
        
        # Extraer resumen ejecutivo
        match_resumen = re.search(r'### 🎯 Resumen Ejecutivo\s*\n(.*?)(?=### 📝|---|\Z)', bloque, re.DOTALL)
        resumen = match_resumen.group(1).strip() if match_resumen else ''
        
        # Extraer transcripción (primeros 500 chars)
        match_trans = re.search(r'### 📝 Transcripción del Contenido\s*\n(.*?)(?=---|$)', bloque, re.DOTALL)
        transcripcion = match_trans.group(1).strip()[:500] if match_trans else ''
        
        coincidencias.append({
            'termino': termino,
            'medio': medio,
            'fecha': fecha_str,
            'hora': hora_str,
            'fecha_hora': f"{fecha_str} {hora_str}",
            'video_url': video_url,
            'nombre_archivo': nombre_archivo,
            'resumen_ejecutivo': resumen,
            'transcripcion': transcripcion
        })
    
    log_info(f"Parseadas {len(coincidencias)} coincidencias del {fecha_filtro}" + 
             (f" ({hora_inicio}-{hora_fin})" if hora_inicio else ""), func_name)
    return coincidencias


def obtener_thumbnail_cloudinary(video_url, segundo=5, ancho=480, alto=270):
    """
    Genera URL de thumbnail/frame de un video en Cloudinary.
    Cloudinary permite extraer frames cambiando extensión y agregando transformaciones.
    
    Ejemplo:
        Input:  https://res.cloudinary.com/CLOUD/video/upload/v123/path/clip.mp4
        Output: https://res.cloudinary.com/CLOUD/video/upload/so_5,w_480,h_270,c_fill/v123/path/clip.jpg
    """
    if not video_url or 'cloudinary.com' not in video_url:
        return ''
    
    try:
        # Cambiar extensión de .mp4 a .jpg
        url_base = re.sub(r'\.(mp4|webm|avi|mov)$', '.jpg', video_url)
        
        # Insertar transformaciones antes de /v (versión)
        url_thumb = re.sub(
            r'(/video/upload/)(v\d+/)',
            rf'\1so_{segundo},w_{ancho},h_{alto},c_fill/\2',
            url_base
        )
        return url_thumb
    except:
        return ''


def agrupar_coincidencias_por_cliente(coincidencias):
    """
    Agrupa coincidencias por cliente usando terminos_guardados.json.
    
    Returns:
        dict: {cliente_id: {'cliente': dict, 'coincidencias': [list]}}
    """
    func_name = "agrupar_coincidencias_por_cliente"
    
    # Cargar mapeo término -> cliente
    try:
        with open('terminos_guardados.json', 'r', encoding='utf-8') as f:
            terminos_data = json.load(f)
        terminos_map = {t['termino'].lower(): t['cliente_id'] for t in terminos_data.get('terminos', [])}
    except:
        terminos_map = {}
    
    # Cargar clientes
    try:
        with open('clientes_config.json', 'r', encoding='utf-8') as f:
            clientes_data = json.load(f)
        clientes_map = {c['id']: c for c in clientes_data.get('clientes', [])}
    except:
        clientes_map = {}
    
    # Agrupar
    grupos = {}
    for coinc in coincidencias:
        termino_lower = coinc['termino'].lower()
        cliente_id = terminos_map.get(termino_lower, 'default')
        
        # Si no se encontró mapeo directo, buscar coincidencia parcial
        if cliente_id == 'default' and termino_lower not in terminos_map:
            for t_key, t_cid in terminos_map.items():
                if t_key in termino_lower or termino_lower in t_key:
                    cliente_id = t_cid
                    break
        
        if cliente_id not in grupos:
            cliente_obj = clientes_map.get(cliente_id, clientes_map.get('default', {}))
            grupos[cliente_id] = {
                'cliente': cliente_obj,
                'coincidencias': []
            }
        
        grupos[cliente_id]['coincidencias'].append(coinc)
    
    log_info(f"Coincidencias agrupadas en {len(grupos)} entidades", func_name)
    return grupos


def generar_html_resumen_diario(coincidencias, cliente_nombre, corte_label, fecha, color_cliente="#1E88E5"):
    """
    Genera HTML moderno para el correo digest de una entidad.
    Incluye frame de video, resumen ejecutivo y link por cada coincidencia.
    """
    total = len(coincidencias)
    terminos_unicos = list(set(c['termino'] for c in coincidencias))
    medios_unicos = list(set(c['medio'] for c in coincidencias if c['medio']))
    
    # Generar cards de coincidencias
    cards_html = ""
    for idx, coinc in enumerate(coincidencias, 1):
        thumb_url = obtener_thumbnail_cloudinary(coinc['video_url'])
        
        # Imagen del frame
        img_section = ""
        if thumb_url:
            img_section = f'''
            <div style="flex-shrink: 0; width: 220px;">
                <a href="{coinc['video_url']}" target="_blank">
                    <img src="{thumb_url}" alt="Frame del video" 
                         style="width: 220px; height: 124px; object-fit: cover; border-radius: 8px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: block;" />
                </a>
            </div>'''
        
        # Botón de video
        btn_video = ""
        if coinc['video_url']:
            btn_video = f'''
            <a href="{coinc['video_url']}" target="_blank" 
               style="display: inline-block; background: linear-gradient(135deg, #ff4757, #ff3742); 
                      color: white; padding: 8px 18px; border-radius: 25px; text-decoration: none; 
                      font-size: 13px; font-weight: 600; margin-top: 8px;">
                ▶️ Ver Video
            </a>'''
        
        # Limpiar resumen (quitar markdown)
        resumen_limpio = capitalizar_marcas_medios_rd_en_texto(
            coinc['resumen_ejecutivo'].replace('**', '').replace('*', '').replace('🤖 Análisis del segmento:', '').strip()
        )
        # Convertir saltos de línea a <br> para HTML
        resumen_html = resumen_limpio.replace('\n\n', '<br><br>').replace('\n', '<br>')
        
        cards_html += f'''
        <div style="background: white; border-radius: 12px; padding: 20px; margin: 15px 0; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08); border-left: 4px solid {color_cliente};">
            <div style="display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap;">
                {img_section}
                <div style="flex: 1; min-width: 250px;">
                    <div style="font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                        Coincidencia #{idx} &middot; {coinc['hora']}
                    </div>
                    <div style="font-size: 18px; font-weight: 700; color: #333; margin-bottom: 6px;">
                        🔍 {capitalizar_marcas_medios_rd_en_texto(str(coinc['termino']))}
                    </div>
                    <div style="font-size: 13px; color: #666; margin-bottom: 10px;">
                        📺 {capitalizar_marcas_medios_rd_en_texto(str(coinc['medio'] if coinc['medio'] else coinc['nombre_archivo']))}
                    </div>
                    <div style="font-size: 13px; color: #444; line-height: 1.6;">
                        {resumen_html}
                    </div>
                    {btn_video}
                </div>
            </div>
        </div>'''
    
    # Términos badges
    terminos_badges = " ".join(
        f'<span style="background: {color_cliente}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">{capitalizar_marcas_medios_rd_en_texto(str(t))}</span>'
        for t in terminos_unicos
    )
    
    # Medios lista
    medios_lista = "<br>".join(
        f"📺 {capitalizar_marcas_medios_rd_en_texto(str(m))}" for m in medios_unicos
    ) if medios_unicos else "Sin datos de medio"
    
    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen {corte_label} - {cliente_nombre}</title>
</head>
<body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f0f2f5;">
    <div style="max-width: 700px; margin: 0 auto;">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, {color_cliente} 0%, {color_cliente}cc 100%); color: white; 
                    padding: 30px; border-radius: 16px 16px 0 0; text-align: center;">
            <h1 style="margin: 0; font-size: 24px; font-weight: 300;">
                📊 Resumen de Coincidencias
            </h1>
            <div style="font-size: 16px; margin-top: 8px; opacity: 0.9;">
                {corte_label} &middot; {fecha}
            </div>
            <div style="font-size: 20px; font-weight: 700; margin-top: 12px;">
                {cliente_nombre}
            </div>
        </div>
        
        <!-- Stats bar -->
        <div style="background: white; padding: 20px 30px; display: flex; justify-content: space-around; 
                    text-align: center; border-bottom: 1px solid #eee; flex-wrap: wrap;">
            <div>
                <div style="font-size: 28px; font-weight: 700; color: {color_cliente};">{total}</div>
                <div style="font-size: 12px; color: #888; text-transform: uppercase;">Coincidencias</div>
            </div>
            <div>
                <div style="font-size: 28px; font-weight: 700; color: {color_cliente};">{len(terminos_unicos)}</div>
                <div style="font-size: 12px; color: #888; text-transform: uppercase;">Términos</div>
            </div>
            <div>
                <div style="font-size: 28px; font-weight: 700; color: {color_cliente};">{len(medios_unicos)}</div>
                <div style="font-size: 12px; color: #888; text-transform: uppercase;">Medios</div>
            </div>
        </div>
        
        <!-- Términos detectados -->
        <div style="background: white; padding: 15px 30px; border-bottom: 1px solid #eee;">
            <div style="font-size: 12px; color: #888; text-transform: uppercase; margin-bottom: 8px;">Términos detectados</div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                {terminos_badges}
            </div>
        </div>
        
        <!-- Coincidencias -->
        <div style="background: #f8f9fa; padding: 20px;">
            {cards_html}
        </div>
        
        <!-- Footer -->
        <div style="background: #343a40; color: white; padding: 20px; border-radius: 0 0 16px 16px; text-align: center;">
            <p style="margin: 4px 0; opacity: 0.8; font-size: 13px;">
                🤖 Resumen generado automáticamente por Radio Analyzer IA v2.0
            </p>
            <p style="margin: 4px 0; opacity: 0.6; font-size: 12px;">
                {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} &middot; {cliente_nombre}
            </p>
        </div>
        
    </div>
</body>
</html>'''
    
    return html


def enviar_resumen_diario_clientes(corte="mañana"):
    """
    Función principal del digest diario.
    
    1. Lee coincidencias.md filtrando por hoy y rango horario del corte
    2. Agrupa por entidad/cliente
    3. Si una entidad NO tiene coincidencias, NO se envía correo
    4. Genera HTML con frame + resumen + link por cada coincidencia
    5. Envía por Brevo SMTP a los destinatarios de cada entidad
    
    Args:
        corte: "mañana" (00:00-10:30), "tarde" (10:31-17:30), "noche" (17:31-23:59)
    
    Returns:
        dict: {cliente_id: (exito, mensaje)}
    """
    func_name = "enviar_resumen_diario_clientes"
    
    # Definir rangos horarios
    rangos = {
        "mañana": ("00:00", "10:30", "Resumen Mañana (hasta 10:30 AM)"),
        "tarde":  ("10:31", "17:30", "Resumen Tarde (10:30 AM - 5:30 PM)"),
        "noche":  ("17:31", "23:59", "Resumen Noche (5:30 PM - 11:59 PM)")
    }
    
    hora_inicio, hora_fin, corte_label = rangos.get(corte, rangos["mañana"])
    fecha_hoy = datetime.now().strftime('%d/%m/%Y')
    
    log_info(f"🔔 Iniciando resumen diario: {corte_label} ({fecha_hoy})", func_name)
    
    # 1. Parsear coincidencias del día en el rango horario
    coincidencias = parsear_coincidencias_md(
        fecha_filtro=fecha_hoy,
        hora_inicio=hora_inicio,
        hora_fin=hora_fin
    )
    
    if not coincidencias:
        log_info(f"Sin coincidencias para el corte {corte} del {fecha_hoy}", func_name)
        return {"_sin_datos": (False, f"Sin coincidencias en el corte {corte_label}")}
    
    log_info(f"📊 {len(coincidencias)} coincidencias encontradas para corte {corte}", func_name)
    
    # 2. Agrupar por entidad
    grupos = agrupar_coincidencias_por_cliente(coincidencias)
    
    resultados = {}
    
    # 3. Para cada entidad con coincidencias, generar y enviar correo
    for cliente_id, grupo in grupos.items():
        cliente = grupo['cliente']
        coincs_cliente = grupo['coincidencias']
        cliente_nombre = nombre_cliente_mostrar_para_ui(cliente, cliente_id)
        color_cliente = cliente.get('color', '#1E88E5')
        
        # Verificar que el cliente tenga Brevo habilitado
        brevo_config = cliente.get('brevo', {})
        if not brevo_config.get('enabled'):
            log_info(f"⏭️ {cliente_nombre}: Brevo no habilitado, saltando", func_name)
            resultados[cliente_id] = (False, "Brevo deshabilitado")
            continue
        
        correos, correos_normalizados = obtener_destinatarios_activos_cliente(cliente)
        if brevo_config.get('correos_destinatarios', []) != correos_normalizados:
            brevo_config['correos_destinatarios'] = correos_normalizados
            cliente['brevo'] = brevo_config
            actualizar_cliente(cliente_id, cliente)
        if not correos:
            log_info(f"⏭️ {cliente_nombre}: Sin destinatarios configurados", func_name)
            resultados[cliente_id] = (False, "Sin destinatarios")
            continue
        
        log_info(f"📧 Generando resumen para {cliente_nombre}: {len(coincs_cliente)} coincidencias", func_name)
        
        # 4. Generar HTML
        html_content = generar_html_resumen_diario(
            coincs_cliente, cliente_nombre, corte_label, fecha_hoy, color_cliente
        )
        
        # 5. Enviar por Brevo SMTP
        try:
            api_key = brevo_config.get('api_key', '')
            sender_email = brevo_config.get('sender_email', '')
            sender_name = brevo_config.get('sender_name', f'Sistema {cliente_nombre}')
            sender_name = capitalizar_marcas_medios_rd_en_texto(sender_name)
            smtp_user = brevo_config.get('smtp_user', sender_email)
            smtp_server = brevo_config.get('smtp_server', 'smtp-relay.brevo.com')
            smtp_port = brevo_config.get('smtp_port', 587)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"{EMAIL_ASUNTO_PREFIJO_RADIO}📊 Resumen {corte_label} - {cliente_nombre} ({len(coincs_cliente)} coincidencias)"
            msg['From'] = f"{sender_name} <{sender_email}>"
            msg['To'] = correos[0]
            if len(correos) > 1:
                msg['Bcc'] = ', '.join(correos[1:])
            
            # Texto plano como alternativa
            text_lines = [f"RESUMEN {corte_label.upper()} - {cliente_nombre}", f"Fecha: {fecha_hoy}", ""]
            for c in coincs_cliente:
                text_lines.append(f"---")
                text_lines.append(
                    "Término: "
                    f"{capitalizar_marcas_medios_rd_en_texto(str(c['termino']))} | {c['hora']}"
                )
                text_lines.append(f"Medio: {capitalizar_marcas_medios_rd_en_texto(str(c['medio']))}")
                text_lines.append(
                    f"Resumen: {capitalizar_marcas_medios_rd_en_texto(c['resumen_ejecutivo'][:300])}"
                )
                if c['video_url']:
                    text_lines.append(f"Clip: {c['video_url']}")
                text_lines.append("")
            
            msg.attach(MIMEText('\n'.join(text_lines), 'plain', 'utf-8'))
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, api_key)
                server.send_message(msg)
            
            log_info(f"✅ Resumen enviado a {cliente_nombre}: {len(correos)} destinatarios", func_name)
            resultados[cliente_id] = (True, f"Enviado a {len(correos)} destinatarios ({len(coincs_cliente)} coincidencias)")
            
        except Exception as e:
            log_warning(f"❌ Error enviando resumen a {cliente_nombre}: {e}", func_name)
            resultados[cliente_id] = (False, f"Error: {str(e)}")
    
    return resultados


# === SCHEDULER DE RESUMEN DIARIO ===

_scheduler_thread = None
_scheduler_activo = False

def _ejecutar_scheduler_resumen():
    """Thread que ejecuta el scheduler de resúmenes diarios."""
    global _scheduler_activo
    import time as _time
    
    func_name = "_ejecutar_scheduler_resumen"
    log_info("🕐 Scheduler de resúmenes diarios iniciado", func_name)
    
    while _scheduler_activo:
        ahora = datetime.now()
        hora_actual = ahora.strftime('%H:%M')
        
        # Verificar cortes
        if hora_actual == '10:30':
            log_info("🔔 Ejecutando resumen MAÑANA (10:30)", func_name)
            try:
                enviar_resumen_diario_clientes("mañana")
            except Exception as e:
                log_warning(f"Error en resumen mañana: {e}", func_name)
        
        elif hora_actual == '17:30':
            log_info("🔔 Ejecutando resumen TARDE (17:30)", func_name)
            try:
                enviar_resumen_diario_clientes("tarde")
            except Exception as e:
                log_warning(f"Error en resumen tarde: {e}", func_name)
        
        elif hora_actual == '23:59':
            log_info("🔔 Ejecutando resumen NOCHE (23:59)", func_name)
            try:
                enviar_resumen_diario_clientes("noche")
            except Exception as e:
                log_warning(f"Error en resumen noche: {e}", func_name)
        
        # Dormir 60 segundos para no revisar más de una vez por minuto
        _time.sleep(60)
    
    log_info("🛑 Scheduler de resúmenes diarios detenido", func_name)


def iniciar_scheduler_resumen():
    """Inicia el scheduler de resúmenes diarios en un thread aparte."""
    global _scheduler_thread, _scheduler_activo
    import threading
    
    if _scheduler_activo:
        return False, "Scheduler ya está activo"
    
    _scheduler_activo = True
    _scheduler_thread = threading.Thread(target=_ejecutar_scheduler_resumen, daemon=True)
    _scheduler_thread.start()
    return True, "Scheduler iniciado"


def detener_scheduler_resumen():
    """Detiene el scheduler de resúmenes diarios."""
    global _scheduler_activo
    _scheduler_activo = False
    return True, "Scheduler detenido"


# ============================================================================
# === FIN SISTEMA DE RESUMEN DIARIO ==========================================
# ============================================================================


def enviar_correo_brevo(termino_encontrado, resumen_completo, nombre_video, video_path=None, info_medio="", terminos_detectados=[], video_url_gdrive=None):
    """Envía correo usando Brevo SMTP con plantilla moderna a múltiples destinatarios"""
    func_name = "enviar_correo_brevo"
    
    try:
        config = cargar_brevo_config()
        
        if not config['enabled']:
            log_info("Correo Brevo deshabilitado", func_name)
            return False, "Correo deshabilitado"
            
        if not all([config['api_key'], config['sender_email']]):
            log_warning("Configuración de Brevo incompleta (API key o sender)", func_name)
            return False, "Configuración incompleta"
        
        # Obtener lista de correos destinatarios
        correos_destinatarios = obtener_correos_activos()
        if not correos_destinatarios:
            log_warning("No hay correos destinatarios configurados", func_name)
            return False, "No hay destinatarios configurados"
        
        # Verificar conectividad
        if not verificar_conectividad():
            log_warning("Sin conectividad - saltando envío de correo", func_name)
            return False, "Sin conectividad"
        
        log_info(f"Enviando correo para término: {termino_encontrado} a {len(correos_destinatarios)} destinatarios", func_name)
        
        termino_asunto = capitalizar_marcas_medios_rd_en_texto(str(termino_encontrado)).strip()

        # Crear mensaje base
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"{EMAIL_ASUNTO_PREFIJO_RADIO}🎯 Coincidencia: {termino_asunto}"
        msg['From'] = f"{config['sender_name']} <{config['sender_email']}>"
        # Primer destinatario en To, resto en Bcc (mejor entrega que solo Bcc en varios proveedores)
        msg['To'] = correos_destinatarios[0]
        if len(correos_destinatarios) > 1:
            msg['Bcc'] = ', '.join(correos_destinatarios[1:])
        
        # Usar URL de Google Drive si está disponible, sino intentar Cloudinary
        video_url = None
        if video_url_gdrive:
            video_url = video_url_gdrive
            log_info(f"✅ Usando URL de Google Drive para player: {video_url}", func_name)
        elif video_path and os.path.exists(video_path):
            log_info(f"Intentando subir clip a Cloudinary: {video_path}", func_name)
            try:
                cloudinary_configurado = configurar_cloudinary()
                log_info(f"Cloudinary configurado: {cloudinary_configurado}", func_name)
                
                if cloudinary_configurado:
                    video_url_result, mensaje_subida = subir_video_cloudinary(video_path, termino_encontrado)
                    if video_url_result:
                        video_url = video_url_result
                        log_info(f"✅ Archivo subido a Cloudinary exitosamente: {video_url}", func_name)
                    else:
                        log_warning(f"❌ Error subiendo video a Cloudinary: {mensaje_subida}", func_name)
                else:
                    log_warning("❌ Cloudinary no está configurado correctamente", func_name)
            except Exception as e:
                log_warning(f"❌ Excepción subiendo video a Cloudinary: {e}", func_name)
        else:
            log_warning(f"❌ No hay URL de Google Drive ni video local: {video_path}", func_name)
        
        # Crear contenido HTML con información completa
        html_content = crear_plantilla_email_html(
            termino_encontrado, 
            resumen_completo, 
            nombre_video, 
            info_medio, 
            terminos_detectados if terminos_detectados else [termino_encontrado], 
            video_url
        )
        
        # Crear versión texto plano completa como respaldo
        terminos_texto = ", ".join([f'"{t}"' for t in terminos_detectados]) if terminos_detectados else f'"{termino_encontrado}"'
        
        text_content = f"""
COINCIDENCIA DETECTADA EN ANÁLISIS DE AUDIOS

TÉRMINOS DETECTADOS: {terminos_texto}

{f"MEDIO: {info_medio}" if info_medio else ""}

RESUMEN COMPLETO DE LA COINCIDENCIA:
{resumen_completo}

INFORMACIÓN TÉCNICA:
- Fecha y Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
- Archivo Analizado: {nombre_video}
- Sistema: Análisis Automático de Audios con IA

Este correo fue generado automáticamente por el Sistema de Análisis de Audios de FGJ Medios.
        """.strip()

        text_content = capitalizar_marcas_medios_rd_en_texto(text_content)
        
        part_text = MIMEText(text_content, 'plain', 'utf-8')
        part_html = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(part_text)
        msg.attach(part_html)
        
        # NO adjuntar audios - solo usar URLs (Cloudinary/Google Drive)
        # Esto evita el error "Max message size exceeded" de Brevo
        if video_url:
            log_info(f"✅ Usando URL para clip en correo: {video_url}", func_name)
            # La URL ya está incluida en el contenido HTML/texto
        else:
            log_warning("⚠️ No hay URL de clip disponible para incluir en el correo", func_name)
        
        # Enviar correo a todos los destinatarios
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            # Login con usuario SMTP pero envío desde email verificado
            smtp_user = config.get('smtp_user', config['sender_email'])
            server.login(smtp_user, config['api_key'])
            server.send_message(msg, to_addrs=correos_destinatarios)
        
        log_info(f"✅ Correo enviado exitosamente a {len(correos_destinatarios)} destinatarios: {', '.join(correos_destinatarios[:3])}{'...' if len(correos_destinatarios) > 3 else ''}", func_name)
        return True, f"Correo enviado a {len(correos_destinatarios)} destinatarios"
        
    except Exception as e:
        error_msg = f"Error enviando correo: {str(e)[:200]}"
        log_exception(func_name, e, error_msg)
        return False, error_msg

def test_brevo_connection():
    """Prueba la conexión con Brevo"""
    func_name = "test_brevo_connection"
    
    try:
        config = cargar_brevo_config()
        
        if not config['enabled']:
            return False, "❌ Brevo está deshabilitado"
            
        if not all([config['api_key'], config['sender_email']]):
            return False, "❌ Configuración incompleta (API key o sender)"
        
        correos_destinatarios = obtener_correos_activos()
        if not correos_destinatarios:
            return False, "❌ No hay destinatarios configurados"
        
        # Probar conexión SMTP
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            # Login con usuario SMTP
            smtp_user = config.get('smtp_user', config['sender_email'])
            server.login(smtp_user, config['api_key'])
        
        # Enviar correo de prueba
        exito, mensaje = enviar_correo_brevo(
            "PRUEBA",
            "**CORREO DE PRUEBA**\\n\\nEste es un correo de prueba del sistema de análisis de audio.\\n\\n✅ Si recibiste este correo, la configuración está funcionando correctamente.\\n\\n🎯 **Funcionalidades probadas:**\\n- Envío a múltiples destinatarios\\n- Plantilla HTML moderna\\n- Adjuntos sin limitación de tamaño\\n- Información completa del medio",
            "test_email.mp4",
            None,  # No hay video_path en prueba
            "Prueba del Sistema de Correos",  # info_medio
            ["PRUEBA", "SISTEMA", "CORREOS"],  # terminos_detectados
            None  # No hay video_url_gdrive en prueba
        )
        
        if exito:
            return True, f"✅ Conexión exitosa y correo de prueba enviado a {len(correos_destinatarios)} destinatarios"
        else:
            return False, f"❌ Error enviando correo de prueba: {mensaje}"
            
    except Exception as e:
        return False, f"❌ Error de conexión: {str(e)[:100]}"

def enviar_mensaje_telegram(mensaje, chat_id=None, bot_token=None, parse_mode='Markdown'):
    """Envía un mensaje de texto a Telegram - VERSIÓN ROBUSTA"""
    func_name = "enviar_mensaje_telegram"
    config = cargar_telegram_config()
    
    bot_token = bot_token or config['bot_token']
    chat_id = chat_id or config['chat_id']
    
    if not bot_token or not chat_id:
        log_info("Token o Chat ID no configurados para Telegram", func_name)
        return False, "Token o Chat ID no configurados"
    
    # Verificar conectividad antes de intentar
    if not verificar_conectividad():
        log_info("Sin conectividad - saltando envío a Telegram", func_name)
        return False, "Sin conectividad a internet"

    if mensaje:
        mensaje = capitalizar_marcas_medios_rd_en_texto(mensaje)

    # Telegram permite 4096 caracteres por mensaje; si es más largo, enviar en varios mensajes (resumen completo)
    MAX_TELEGRAM = 4096
    chunks = []
    resto = mensaje
    while resto:
        if len(resto) <= MAX_TELEGRAM:
            chunks.append(resto)
            break
        pos = resto.rfind('\n', 0, MAX_TELEGRAM)
        if pos <= 0:
            pos = MAX_TELEGRAM
        chunks.append(resto[:pos].strip())
        resto = resto[pos:].lstrip()
    
    log_debug(f"Enviando mensaje a Telegram: {len(chunks)} parte(s), {len(mensaje)} caracteres", func_name)
    
    for idx, chunk in enumerate(chunks):
        # Reintentos con backoff exponencial por cada parte
        max_intentos = 3
        for intento in range(max_intentos):
            try:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = {
                    'chat_id': chat_id,
                    'text': chunk,
                    'disable_web_page_preview': False
                }
                if parse_mode:
                    data['parse_mode'] = parse_mode
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'RadioAnalizer/2.1',
                    'Connection': 'close',
                    'Accept': 'application/json'
                })
                response = session.post(url, json=data, timeout=45, allow_redirects=True)
                session.close()
                if response.status_code == 200:
                    log_info(f"Parte {idx + 1}/{len(chunks)} enviada a Telegram", func_name)
                    break
                else:
                    error_msg = response.text[:150]
                    log_info(f"Telegram respondió HTTP {response.status_code}: {error_msg}", func_name)
                    if intento == max_intentos - 1:
                        return False, f"Error HTTP {response.status_code}: {error_msg}"
            except requests.exceptions.ConnectionError as e:
                error_msg = str(e)[:150]
                if intento == max_intentos - 1:
                    return False, f"Error de conexión: {error_msg}"
            except requests.exceptions.Timeout:
                if intento == max_intentos - 1:
                    return False, "Timeout - Telegram no responde"
            except Exception as e:
                log_exception(func_name, e, f"Parte {idx + 1}")
                if intento == max_intentos - 1:
                    return False, f"Error enviando mensaje: {str(e)[:150]}"
            if intento < max_intentos - 1:
                esperar_con_backoff(intento, max_espera=20)
        # Pequeña pausa entre partes para no saturar la API
        if idx < len(chunks) - 1:
            time.sleep(0.5)
    
    return True, "Mensaje enviado a Telegram" + (f" ({len(chunks)} parte(s))" if len(chunks) > 1 else "")

def enviar_video_telegram(video_path, caption="", chat_id=None, bot_token=None, usar_cloudinary=True):
    """Envía un video a Telegram (directamente o vía Cloudinary) y devuelve la URL del video"""
    config = cargar_telegram_config()
    
    bot_token = bot_token or config['bot_token']
    chat_id = chat_id or config['chat_id']

    if caption:
        caption = capitalizar_marcas_medios_rd_en_texto(caption)

    if not bot_token or not chat_id:
        return False, "Token o Chat ID no configurados"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        
        # Verificar tamaño del archivo
        file_size_mb = os.path.getsize(video_path) / (1024*1024)
        
        video_url_cloudinary = None
        if usar_cloudinary and config.get('use_cloudinary', True):
            # Subir a Cloudinary primero
            video_url_cloudinary, upload_msg = subir_video_cloudinary(video_path)
            
            if video_url_cloudinary:
                # Enviar URL del video
                data = {
                    'chat_id': chat_id,
                    'video': video_url_cloudinary,
                    'caption': caption.replace("🌐 *Vía:* Cloudinary - 5.6MB", f"🌐 *Vía:* Cloudinary - {file_size_mb:.1f}MB"),
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, data=data, timeout=config.get('timeout', 30))
            else:
                return False, f"Error subiendo video: {upload_msg}", None
        
        elif file_size_mb <= config.get('max_file_size_mb', 8):  # Límite configurable
            # Envío directo
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
                data = {
                    'chat_id': chat_id,
                    'caption': caption.replace("🌐 *Vía:* Cloudinary - 5.6MB", f"📹 *Envío directo* - {file_size_mb:.1f}MB")
                }
                
                response = requests.post(url, files=files, data=data, timeout=config.get('timeout', 30))
        else:
            return False, f"Archivo muy grande ({file_size_mb:.1f}MB) y Cloudinary deshabilitado", None
        
        if response.status_code == 200:
            return True, f"Clip enviado a Telegram ({file_size_mb:.1f}MB)", video_url_cloudinary
        else:
            return False, f"Error HTTP {response.status_code}: {response.text[:100]}", None
            
    except Exception as e:
        return False, f"Error enviando clip: {str(e)[:100]}", None

def enviar_video_telegram_directo(video_path, caption, chat_id=None, bot_token=None, parse_mode='Markdown'):
    """
    Envía un video directamente a Telegram usando sendVideo API
    Soporta videos hasta 50MB directamente, o URLs para videos más grandes
    """
    func_name = "enviar_video_telegram_directo"
    config = cargar_telegram_config()
    
    bot_token = bot_token or config['bot_token']
    chat_id = chat_id or config['chat_id']

    if caption:
        caption = capitalizar_marcas_medios_rd_en_texto(caption)

    if not bot_token or not chat_id:
        return False, "Token o Chat ID no configurados", None
    
    try:
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            return False, f"Archivo no encontrado: {video_path}", None
        
        # Obtener información del archivo
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        log_info(f"Enviando video directo a Telegram: {os.path.basename(video_path)} ({file_size_mb:.1f}MB)", func_name)
        
        # Verificar límite de tamaño configurable (por defecto 8MB)
        max_mb = config.get('max_file_size_mb', 8)
        if file_size_mb > max_mb:
            return False, f"Video demasiado grande ({file_size_mb:.1f}MB > {max_mb}MB). Comprimir antes de enviar.", None
        
        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        
        # Preparar datos del video
        with open(video_path, 'rb') as video_file:
            files = {
                'video': (os.path.basename(video_path), video_file, 'video/mp4')
            }
            
            data = {
                'chat_id': chat_id,
                'caption': caption,
                'supports_streaming': True,  # Para videos MP4
                'duration': None,  # Telegram lo detectará automáticamente
                'width': None,    # Telegram lo detectará automáticamente
                'height': None    # Telegram lo detectará automáticamente
            }
            if parse_mode:
                data['parse_mode'] = parse_mode
            
            # Enviar video con timeout más largo para archivos grandes
            timeout = max(60, int(file_size_mb * 2))  # 2 s/MB, mínimo 60s
            response = requests.post(url, data=data, files=files, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                message_id = result['result']['message_id']
                video_info = result['result'].get('video', {})
                
                log_info(f"Clip enviado exitosamente a Telegram: {os.path.basename(video_path)} (ID: {message_id})", func_name)
                return True, f"Clip enviado exitosamente (ID: {message_id})", None
            else:
                error_desc = result.get('description', 'Error desconocido')
                return False, f"Error de Telegram: {error_desc}", None
        else:
            return False, f"Error HTTP: {response.status_code} - {response.text}", None
            
    except requests.exceptions.Timeout:
        return False, f"Timeout: Telegram no respondió en {timeout} segundos", None
    except requests.exceptions.RequestException as e:
        return False, f"Error de conexión: {str(e)}", None
    except Exception as e:
        return False, f"Error inesperado: {str(e)}", None

def enviar_video_telegram_url(video_url, caption, chat_id=None, bot_token=None, parse_mode='Markdown'):
    """
    Envía un clip a Telegram usando una URL (archivos grandes o desde Cloudinary)
    Soporta videos hasta 2GB cuando se usa URL
    """
    func_name = "enviar_video_telegram_url"
    config = cargar_telegram_config()
    
    bot_token = bot_token or config['bot_token']
    chat_id = chat_id or config['chat_id']

    if caption:
        caption = capitalizar_marcas_medios_rd_en_texto(caption)

    if not bot_token or not chat_id:
        return False, "Token o Chat ID no configurados", None
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        
        data = {
            'chat_id': chat_id,
            'video': video_url,
            'caption': caption,
            'parse_mode': parse_mode,
            'supports_streaming': True
        }
        
        response = requests.post(url, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                message_id = result['result']['message_id']
                log_info(f"URL de clip enviada exitosamente a Telegram: {video_url} (ID: {message_id})", func_name)
                return True, f"URL de clip enviada exitosamente (ID: {message_id})", None
            else:
                error_desc = result.get('description', 'Error desconocido')
                return False, f"Error de Telegram: {error_desc}", None
        else:
            return False, f"Error HTTP: {response.status_code} - {response.text}", None
            
    except requests.exceptions.Timeout:
        return False, "Timeout: Telegram no respondió en 30 segundos", None
    except requests.exceptions.RequestException as e:
        return False, f"Error de conexión: {str(e)}", None
    except Exception as e:
        return False, f"Error inesperado: {str(e)}", None

def enviar_video_telegram_inteligente(video_path, caption, chat_id=None, bot_token=None, parse_mode='Markdown', cloudinary_url=None):
    """
    Función inteligente que decide automáticamente el mejor método para enviar video a Telegram:
    1. Si hay URL de Cloudinary -> usar sendVideo con URL (hasta 2GB)
    2. Si video < 50MB -> envío directo
    3. Si video > 50MB -> subir a Cloudinary primero, luego usar URL
    """
    func_name = "enviar_video_telegram_inteligente"
    
    try:
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            return False, f"Archivo no encontrado: {video_path}", None
        
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        log_info(f"Enviando video inteligente a Telegram: {os.path.basename(video_path)} ({file_size_mb:.1f}MB)", func_name)

        # Prefijar método de envío en el caption para identificar API de Telegram
        if caption is None:
            caption = ""
        if not caption.strip().startswith("[AT]"):
            caption = f"[AT] {caption}".strip()
        
        # MÉTODO 1: Si ya tenemos URL de Cloudinary, usarla directamente
        if cloudinary_url:
            log_info(f"Usando URL de Cloudinary existente: {cloudinary_url}", func_name)
            return enviar_video_telegram_url(cloudinary_url, caption, chat_id, bot_token, parse_mode)
        
        # MÉTODO 2: Si video es pequeño (< 50MB), envío directo
        max_mb = config.get('max_file_size_mb', 8)
        if file_size_mb <= max_mb:
            log_info(f"Video pequeño ({file_size_mb:.1f}MB) - envío directo", func_name)
            return enviar_video_telegram_directo(video_path, caption, chat_id, bot_token, parse_mode)
        
        # MÉTODO 3: Video grande (> 50MB) - subir a Cloudinary primero
        log_info(f"Video grande ({file_size_mb:.1f}MB) - subiendo a Cloudinary primero", func_name)
        
        # Subir a Cloudinary
        cloudinary_ok, cloudinary_msg, cloudinary_url = subir_video_cloudinary(video_path, "Clip para Telegram")
        
        if cloudinary_ok and cloudinary_url:
            log_info(f"Archivo subido a Cloudinary exitosamente: {cloudinary_url}", func_name)
            # Ahora enviar usando la URL
            return enviar_video_telegram_url(cloudinary_url, caption, chat_id, bot_token, parse_mode)
        else:
            return False, f"Error subiendo a Cloudinary: {cloudinary_msg}", None
            
    except Exception as e:
        return False, f"Error inesperado: {str(e)}", None

def enviar_clips_a_telegram(clips_generados, resumen, terminos_detectados, video_origen):
    """ENVÍO GARANTIZADO: Resumen → Pausa → Clip de audio → Pausa → Siguiente"""
    config = cargar_telegram_config()
    
    if not config['enabled']:
        return False, "Telegram deshabilitado"
    
    if not config['bot_token'] or not config['chat_id']:
        return False, "Telegram no configurado correctamente"
    
    # ========== CONTROL DE DUPLICADOS ==========
    # Verificar si ya se enviaron estos clips individualmente
    clips_ya_enviados = 0
    clips_pendientes = []
    
    for clip in clips_generados:
        clip_path = clip.get('path', '')
        if clip_path in st.session_state.get('clips_enviados_telegram', []):
            clips_ya_enviados += 1
            st.info(f"⏭️ Clip ya enviado individualmente: {os.path.basename(clip_path)}")
        else:
            clips_pendientes.append(clip)
    
    if clips_ya_enviados == len(clips_generados):
        st.success(f"✅ Todos los clips ya fueron enviados individualmente ({clips_ya_enviados}/{len(clips_generados)})")
        return True, f"✅ Todos los clips ya enviados individualmente"
    
    if clips_pendientes:
        st.info(f"📤 Enviando {len(clips_pendientes)} clips pendientes (de {len(clips_generados)} total)")
        clips_generados = clips_pendientes  # Usar solo los clips pendientes
    
    try:
        # ========== PASO 1: SIEMPRE ENVIAR RESUMEN EJECUTIVO PRIMERO ==========
        mensaje_resumen = f"""🎬 *ANÁLISIS DE VIDEO COMPLETADO*

📹 *Audio:* `{video_origen}`
🔍 *Términos detectados:* {', '.join(terminos_detectados)}
📊 *Total clips generados:* {len(clips_generados)}
⏰ *Fecha:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📋 *RESUMEN EJECUTIVO:*
{resumen}

🌐 *Servidor:* Analizador de Audio IA v2.0

⬇️ *Clips a continuación...*"""
        
        # GARANTIZAR que el resumen se envíe
        intentos_resumen = 0
        resumen_enviado = False
        while intentos_resumen < 3 and not resumen_enviado:
            exito_msg, resultado_msg = enviar_mensaje_telegram(mensaje_resumen)
            if exito_msg:
                st.success(f"📋 ✅ RESUMEN EJECUTIVO ENVIADO: {video_origen}")
                resumen_enviado = True
            else:
                intentos_resumen += 1
                st.warning(f"⚠️ Reintento {intentos_resumen}/3 enviando resumen: {resultado_msg}")
                time.sleep(10)  # Pausa más larga entre reintentos
        
        if not resumen_enviado:
            st.error(f"❌ FALLO CRÍTICO: No se pudo enviar resumen para {video_origen}")
            return False, "❌ Resumen no enviado"
        
        # ========== PASO 2: PAUSA OBLIGATORIA DESPUÉS DEL RESUMEN ==========
        st.info("⏸️ Pausa de 30 segundos después del resumen para evitar congestión...")
        time.sleep(30)
        
        # ========== PASO 3: ENVIAR CADA VIDEO CON SU PAUSA ==========
        if config.get('send_clips', True) and clips_generados:
            clips_enviados = 0
            clips_fallidos = 0
            
            for i, clip in enumerate(clips_generados, 1):
                if not os.path.exists(clip['path']):
                    st.warning(f"⚠️ Archivo no existe: {clip['path']}")
                    continue
                
                # Caption consolidado con toda la información
                caption = f"""🎯 *CLIP {i}/{len(clips_generados)} DE COINCIDENCIA*

📺 *Medio:* {video_origen}
⏰ *Generado:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 *TÉRMINOS DETECTADOS:* {', '.join(terminos_detectados)}

🏷️ *Término específico:* `{clip['termino']}`
⏱️ *Tiempo en audio:* {clip['tiempo']}
📝 *Contexto:* {clip['contexto'][:200]}{'...' if len(clip['contexto']) > 200 else ''}

📋 *RESUMEN DEL VIDEO:*
{resumen}

━━━━━━━━━━━━━━━━━━━━━
🌐 *Vía:* Cloudinary - 5.6MB"""
                
                # GARANTIZAR que cada video se envíe
                intentos_video = 0
                # Forzar API directa sin reintentos ni uso de URL/Cloudinary
                video_enviado = False
                file_size_mb_clip = os.path.getsize(clip['path']) / (1024 * 1024) if os.path.exists(clip['path']) else 0
                if file_size_mb_clip <= 50 and os.path.exists(clip['path']):
                    exito_clip, resultado_clip, _ = enviar_video_telegram_directo(
                        clip['path'],
                        caption,
                        chat_id=config.get('chat_id'),
                        bot_token=config.get('bot_token'),
                        parse_mode=None
                    )
                    if exito_clip:
                        clips_enviados += 1
                        st.success(f"🎬 ✅ VIDEO {i} ENVIADO: {clip['termino']} - {resultado_clip}")
                        video_enviado = True
                    else:
                        st.error(f"❌ FALLO: Video {i} no enviado - {clip['termino']} - {resultado_clip}")
                else:
                    st.warning(f"🚫 Video {i} omitido ({file_size_mb_clip:.1f}MB > 50MB). Solo API directa permitida.")
                
                if not video_enviado:
                    clips_fallidos += 1
                    st.error(f"❌ FALLO: Video {i} no enviado - {clip['termino']}")
                
                # ========== PAUSA OBLIGATORIA ENTRE CADA VIDEO ==========
                if i < len(clips_generados):
                    st.info(f"⏸️ Pausa de 30 segundos antes del próximo clip para evitar congestión...")
                    time.sleep(30)
            
            # ========== PASO 4: MENSAJE FINAL CON PAUSA ==========
            time.sleep(1)
            mensaje_final = f"""✅ *ENVÍO COMPLETADO*

📹 *Audio procesado:* `{video_origen}`
📱 *Clips enviados exitosamente:* {clips_enviados}
❌ *Clips fallidos:* {clips_fallidos}
📊 *Total procesado:* {len(clips_generados)}

━━━━━━━━━━━━━━━━━━━━━"""
            
            enviar_mensaje_telegram(mensaje_final)
            time.sleep(30)  # Pausa final de 30 segundos antes del siguiente video
            
            return True, f"✅ GARANTIZADO: Resumen + {clips_enviados} clips enviados para {video_origen}"
        
        return True, f"✅ GARANTIZADO: Solo resumen enviado para {video_origen}"
        
    except Exception as e:
        st.error(f"❌ ERROR CRÍTICO en envío: {str(e)}")
        return False, f"❌ Error crítico: {str(e)[:100]}"

def test_telegram_connection():
    """Prueba la conexión con Telegram"""
    config = cargar_telegram_config()
    
    if not config['bot_token'] or not config['chat_id']:
        return False, "Token o Chat ID no configurados"
    
    mensaje_test = f"""🧪 *TEST DE CONEXIÓN*

✅ Bot conectado correctamente
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Analizador de Audio IA v2.0

Este es un mensaje de prueba."""
    
    return enviar_mensaje_telegram(mensaje_test)

def cargar_terminos_guardados():
    """Carga términos desde archivo JSON"""
    try:
        if os.path.exists(TERMINOS_CONFIG):
            with open(TERMINOS_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('terminos', [])
    except Exception as e:
        st.warning(f"⚠️ Error cargando términos guardados: {e}")
    return []

def guardar_terminos_archivo(terminos):
    """Guarda términos en archivo JSON"""
    try:
        data = {
            'terminos': terminos,
            'fecha_actualizacion': datetime.now().isoformat(),
            'total_terminos': len(terminos)
        }
        with open(TERMINOS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ Error guardando términos: {e}")
        return False

def cargar_configuracion_completa():
    """Carga toda la configuración guardada"""
    config = {
        'terminos': [],
        'intervalo': 60,
        'duracion_clip': 90,  # 1.5 minutos total (45s antes + 45s después)
        'buffer_anterior': 30,  # 30s antes de la coincidencia
        'mostrar_coincidencias': True
    }
    
    try:
        if os.path.exists(TERMINOS_CONFIG):
            with open(TERMINOS_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                config.update(data)
    except Exception:
        pass
    
    return config

def guardar_configuracion_completa(terminos, intervalo=60, duracion_clip=90, buffer_anterior=30, mostrar_coincidencias=True):
    """Guarda toda la configuración"""
    try:
        data = {
            'terminos': terminos,
            'intervalo': intervalo,
            'duracion_clip': duracion_clip,
            'buffer_anterior': buffer_anterior,
            'mostrar_coincidencias': mostrar_coincidencias,
            'fecha_actualizacion': datetime.now().isoformat(),
            'total_terminos': len(terminos)
        }
        with open(TERMINOS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"❌ Error guardando configuración: {e}")
        return False

# === INICIALIZAR ESTADO DE LA SESIÓN ===
def init_session_state():
    # Cargar configuración guardada
    config_guardada = cargar_configuracion_completa()
    
    defaults = {
        'resumen_global': [],
        'running': False,
        'terminos_continuos': config_guardada['terminos'],  # Cargar términos guardados
        'ultimo_chequeo': datetime.now(),
        'videos_encontrados': 0,
        'videos_procesados': 0,
        'clips_generados': 0,
        'app_restarted': False,
        'intervalo': config_guardada['intervalo'],
        'mostrar_coincidencias': config_guardada['mostrar_coincidencias'],
        'mostrar_solo_actual_relevantes': True,
        'clips_encontrados_sesion': [],
        'duracion_clip': config_guardada.get('duracion_clip', 60),  # Default 1 minuto
        'buffer_anterior': config_guardada.get('buffer_anterior', 30),  # Default 30s
        'coincidencias_enviadas_supabase': set(),  # Control de duplicados para Supabase
        # === Loop continuo ===
        'loop_continuo': True,               # Loop activado por defecto
        'intervalo_loop': 60,                # Segundos entre ciclos cuando hay videos
        'intervalo_loop_vacio': 120,         # Segundos entre ciclos cuando NO hay audios nuevos
        'loop_ciclo_numero': 0,              # Contador de ciclos completados
        # === Contador de uso Mistral/Voxtral ===
        'mistral_total_audio_seconds': 0,       # Segundos de audio procesados
        'mistral_total_prompt_tokens': 0,        # Tokens de prompt consumidos
        'mistral_total_completion_tokens': 0,    # Tokens de completado consumidos
        'mistral_total_tokens': 0,               # Tokens totales consumidos
        'mistral_total_transcripciones': 0,      # Número de transcripciones realizadas
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Inicializar estado después de definir la función
init_session_state()

# === AUTO-INICIAR SCHEDULER DE RESUMEN DIARIO ===
# Se inicia automáticamente al arrancar la app (solo una vez, controlado por variable global)
if not _scheduler_activo:
    try:
        iniciar_scheduler_resumen()
        log_info("✅ Scheduler de resumen diario iniciado automáticamente (cortes: 10:30, 17:30, 23:59)", "STARTUP")
    except Exception as _e:
        log_warning(f"⚠️ Error iniciando scheduler automático: {_e}", "STARTUP")

# === FUNCIÓN MISTRAL CLIENT (DEBE ESTAR ANTES DE verificar_estado_mistral) ===
@st.cache_resource
def cargar_cliente_mistral():
    model = "voxtral-mini-latest"
    client = Mistral(api_key=mistral_api_key)
    return client, model

def verificar_estado_mistral():
    """
    Verifica si Mistral API está disponible de forma simple
    """
    func_name = "verificar_estado_mistral"
    try:
        # Verificación simple sin crear archivos
        client, model = cargar_cliente_mistral()
        
        # Si llegamos aquí, al menos las credenciales están configuradas
        log_debug("Mistral client configurado correctamente", func_name)
        return True, "Configurado (verificación completa requiere audio)"
        
    except Exception as e:
        error_str = str(e).lower()
        if "503" in error_str or "service unavailable" in error_str:
            log_info("Mistral API no disponible (503)", func_name)
            return False, "Service Unavailable (503)"
        elif "500" in error_str:
            log_info("Mistral API error interno (500)", func_name)
            return False, "Internal Server Error (500)"
        elif "api" in error_str and "key" in error_str:
            log_info("Error de API key de Mistral", func_name)
            return False, "Error de API Key"
        else:
            log_info(f"Mistral API error: {e}", func_name)
            return False, f"Error: {str(e)[:50]}"

@st.cache_data
def buscar_todos_los_clips(busqueda_termino="", dias_limite=365):
    clips = []
    ahora = time.time()
    limite_tiempo = ahora - (dias_limite * 24 * 60 * 60) if dias_limite < 9999 else 0
    
    try:
        # Buscar en la carpeta de procesados
        if os.path.exists(CARPETA_PROCESADOS):
            for root, dirs, files in os.walk(CARPETA_PROCESADOS):
                # Verificar si es carpeta procesada con marcador P*
                marcador_procesado = os.path.join(root, "PROCESADO.txt")
                if os.path.exists(marcador_procesado):
                    for file in files:
                        if file.endswith(".mp4") and busqueda_termino.lower() in file.lower():
                            path_ = os.path.join(root, file)
                            if os.path.exists(path_) and os.path.isfile(path_):
                                file_time = os.path.getctime(path_)
                                if file_time >= limite_tiempo:
                                    # Extraer información del archivo
                                    info = extraer_info_clip(file, path_)
                                    clips.append(info)
    except Exception as e:
        st.warning(f"⚠️ Error buscando clips: {e}")
    
    return sorted(clips, key=lambda x: x['timestamp'], reverse=True)

def extraer_info_clip(filename, filepath):
    """Extrae información del nombre del clip"""
    # Formato esperado: YYYYMMDD_HHMMSS_termino_XmYYs.mp4
    try:
        parts = filename.replace('.mp4', '').split('_')
        if len(parts) >= 4:
            fecha_str = parts[0]
            hora_str = parts[1]
            termino = parts[2]
            duracion = parts[3] if len(parts) > 3 else "0m00s"
            
            # Convertir fecha y hora
            fecha_obj = datetime.strptime(f"{fecha_str}_{hora_str}", "%Y%m%d_%H%M%S")
            
            return {
                'filename': filename,
                'filepath': filepath,
                'termino': termino,
                'fecha': fecha_obj.strftime("%Y-%m-%d %H:%M:%S"),
                'fecha_creacion': fecha_obj,
                'timestamp': fecha_obj.timestamp(),
                'duracion': duracion,
                'tiempo_video': duracion,
                'size_mb': round(os.path.getsize(filepath) / (1024*1024), 1) if os.path.exists(filepath) else 0
            }

    except Exception as e:
        # Fallback para archivos con formato no estándar
        try:
            fecha_creacion_obj = datetime.fromtimestamp(os.path.getctime(filepath)) if os.path.exists(filepath) else datetime.now()
            timestamp_val = fecha_creacion_obj.timestamp()
        except:
            fecha_creacion_obj = datetime.now()
            timestamp_val = fecha_creacion_obj.timestamp()
            
        return {
            'filename': filename,
            'filepath': filepath,
            'termino': 'desconocido',
            'fecha': fecha_creacion_obj.strftime("%Y-%m-%d %H:%M:%S"),
            'fecha_creacion': fecha_creacion_obj,
            'timestamp': timestamp_val,
            'duracion': '0m00s',
            'tiempo_video': '0m00s',
            'size_mb': round(os.path.getsize(filepath) / (1024*1024), 1) if os.path.exists(filepath) else 0
        }

def generar_resumen_md(items):
    prompt = (
        "Genera un resumen ejecutivo en Markdown de los análisis de audio realizados:\n\n"
        "DATOS ENCONTRADOS:\n"
    )
    for e in items:
        prompt += f"- **{e['termino']}** en `{e['video']}`: {e['texto'][:100]}...\n"

    prompt += "\n\nGenera un resumen que incluya:\n"
    prompt += "1. Resumen ejecutivo de los hallazgos\n"
    prompt += "2. Términos más frecuentes\n"
    prompt += "3. Audios más relevantes\n"
    prompt += "4. Conclusiones y recomendaciones\n"

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un analista especializado en generar reportes ejecutivos de análisis multimedia."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        md = resp.choices[0].message.content
        
        # Guardar archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resumen_ejecutivo_{timestamp}.md"
        filepath = os.path.join(CARPETA_VIDEOS, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        
        st.success(f"✅ Reporte generado: `{filename}`")
        
        # Mostrar preview del reporte
        with st.expander("👁️ Preview del Reporte"):
            st.markdown(md)
            
    except Exception as e:
        st.error(f"❌ Error generando reporte: {e}")

def borrar_clips_antiguos(dias=7):
    """Borra clips de más de X días"""
    ahora = time.time()
    limite = ahora - (dias * 24 * 60 * 60)
    
    clips_borrados = 0
    carpetas_borradas = 0
    
    try:
        # Buscar en la carpeta de procesados
        if os.path.exists(CARPETA_PROCESADOS):
            for root, dirs, files in os.walk(CARPETA_PROCESADOS):
                # Verificar si es carpeta procesada con marcador P*
                marcador_procesado = os.path.join(root, "PROCESADO.txt")
                if os.path.exists(marcador_procesado):
                    archivos_en_carpeta = 0
                    for file in files:
                        if file.endswith((".mp4", ".txt")):
                            file_path = os.path.join(root, file)
                            if os.path.getctime(file_path) < limite:
                                try:
                                    os.remove(file_path)
                                    clips_borrados += 1
                                except Exception:
                                    pass
                            else:
                                archivos_en_carpeta += 1
                    
                    if archivos_en_carpeta == 0:
                        try:
                            shutil.rmtree(root)
                            carpetas_borradas += 1
                        except Exception:
                            pass
    except Exception:
        pass
    
    return clips_borrados

# Mostrar mensaje si se cargaron términos automáticamente
if st.session_state.terminos_continuos:
    # Extraer nombres de términos (soporta formato dict y string)
    def obtener_nombre_termino(t):
        if isinstance(t, dict):
            return t.get('termino', str(t))
        return str(t)
    
    nombres_terminos = [obtener_nombre_termino(t) for t in st.session_state.terminos_continuos[:3]]
    st.success(f"✅ Se cargaron automáticamente {len(st.session_state.terminos_continuos)} términos desde `terminos_guardados.json`: {', '.join(nombres_terminos)}{'...' if len(st.session_state.terminos_continuos) > 3 else ''}")

# Mostrar estado de servicios preconfigurados
st.info("🚀 **Sistema preconfigurado y listo para usar:**")
col1, col2, col3, col4, col5 = st.columns(5)

# Verificar estado real de cada servicio
webhook_config = cargar_webhook_config()
telegram_config = cargar_telegram_config()
brevo_config = cargar_brevo_config()

with col1:
    if webhook_config.get('enabled', False) and webhook_config.get('url'):
        st.success("🌐 **Webhook** ✅\nMake.com activo")
    else:
        st.warning("🌐 **Webhook** ⚠️\nDeshabilitado")

with col2:
    if telegram_config.get('enabled', False) and telegram_config.get('bot_token') and telegram_config.get('chat_id'):
        st.success("📱 **Telegram** ✅\n@edesuralertas activo")
    else:
        st.warning("📱 **Telegram** ⚠️\nNo configurado")

with col3:
    st.success("☁️ **Google Drive** ✅\nSiempre activo")

with col4:
    cloudinary_config = cargar_cloudinary_config()
    if cloudinary_config.get('cloud_name') and cloudinary_config.get('api_key'):
        st.success("☁️ **Cloudinary** ✅\ndhzxzbkmc activo")
    else:
        st.warning("☁️ **Cloudinary** ⚠️\nNo configurado")

with col5:
    correos_activos_dashboard = obtener_correos_activos()
    if brevo_config.get('enabled', False) and brevo_config.get('api_key') and brevo_config.get('sender_email') and correos_activos_dashboard:
        st.success(f"📧 **Brevo** ✅\n{len(correos_activos_dashboard)} destinatarios")
    else:
        st.warning("📧 **Brevo** ⚠️\nNo configurado")
    
st.title(f"🎙️ RadioAnalizer {RADIO_ANALIZER_VERSION} — Análisis automático de audios")
st.caption(f"Versión **{RADIO_ANALIZER_VERSION}** · tangenciales inmediatas, DeepSeek, Google Sheets (EDESUR / Intrant)")
st.markdown(
    f"📁 **Entrada:** `{CARPETA_VIDEOS}` · 🗂️ **AUDIOCHECKS** (logs, caché, clips, evidencias): `{CARPETA_PROCESADOS}` | "
    f"🌐 Webhook: Make.com | 📱 Telegram | ☁️ Drive | 📧 Brevo"
)
st.info("⏱️ **Configuración de clips:** Por defecto genera clips de 1 minuto (30s antes + 30s después de cada coincidencia)")

_seg_grab_ui = 120
try:
    _seg_grab_ui = int(os.getenv("RADIO_GRABACION_SEGUNDOS", "120"))
except ValueError:
    _seg_grab_ui = 120
try:
    _emisoras_dashboard = escanear_emisoras_entrada(segundos_grabacion=_seg_grab_ui)
except Exception as _ex_em:
    _emisoras_dashboard = []
    log_warning(f"No se pudo listar emisoras: {_ex_em}", "dashboard_emisoras")

with st.expander("📻 Emisoras y estado de grabación", expanded=True):
    st.caption(
        f"Cada subcarpeta dentro de `{CARPETA_VIDEOS}` cuenta como emisora. "
        f"**Grabando** = el último archivo de audio cambió hace menos de **{_seg_grab_ui}s** "
        "(variable `RADIO_GRABACION_SEGUNDOS`). Se ignoran carpetas `c_clip_*` y salidas con `PROCESADO.txt`."
    )
    r1, r2 = st.columns([1, 5])
    with r1:
        if st.button("🔄 Actualizar", key="btn_refresh_emisoras"):
            st.rerun()
    if not _emisoras_dashboard:
        st.warning("No hay emisoras detectadas: crea subcarpetas bajo la carpeta de entrada o coloca audios en la raíz.")
    else:
        _filas = []
        for _e in _emisoras_dashboard:
            _est = "🔴 Grabando" if _e["grabando"] else "⚪ Inactiva"
            _hace = ""
            if _e.get("inactiva_desde_seg") is not None and not _e["grabando"] and _e.get("n_audios", 0) > 0:
                _s = _e["inactiva_desde_seg"]
                if _s < 60:
                    _hace = f" (sin cambios hace {_s:.0f}s)"
                elif _s < 3600:
                    _hace = f" (sin cambios hace {_s/60:.0f} min)"
                else:
                    _hace = f" (sin cambios hace {_s/3600:.1f} h)"
            _filas.append({
                "Emisora": _e["nombre"],
                "Archivos": _e["n_audios"],
                "Estado": _est,
                "Última modificación": _e["ultima_actividad_str"] + _hace,
            })
        st.dataframe(_filas, use_container_width=True, hide_index=True)

# === BOTÓN DE PRUEBA GLOBAL DE TODOS LOS CLIENTES ===
def probar_conexiones_cliente(cliente):
    """Prueba todas las conexiones de un cliente específico"""
    resultados = {}
    cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
    
    # 1. Probar Webhook
    webhook_config = cliente.get('webhook', {})
    if webhook_config.get('enabled') and webhook_config.get('url'):
        try:
            response = requests.post(
                webhook_config['url'],
                json={'test': True, 'cliente': cliente_nombre, 'timestamp': datetime.now().isoformat()},
                timeout=10
            )
            resultados['webhook'] = (response.status_code in [200, 201, 202], f"Status: {response.status_code}")
        except Exception as e:
            resultados['webhook'] = (False, str(e)[:50])
    else:
        resultados['webhook'] = (None, "No configurado")
    
    # 2. Probar Telegram
    telegram_config = cliente.get('telegram', {})
    if telegram_config.get('enabled') and telegram_config.get('bot_token') and telegram_config.get('chat_id'):
        try:
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/getMe"
            response = requests.get(url, timeout=10)
            resultados['telegram'] = (response.status_code == 200, "Bot válido" if response.status_code == 200 else f"Error: {response.status_code}")
        except Exception as e:
            resultados['telegram'] = (False, str(e)[:50])
    else:
        resultados['telegram'] = (None, "No configurado")
    
    # 3. Probar Brevo (usando SMTP, no API REST)
    brevo_config = cliente.get('brevo', {})
    if brevo_config.get('enabled') and brevo_config.get('api_key'):
        try:
            smtp_server = brevo_config.get('smtp_server', 'smtp-relay.brevo.com')
            smtp_port = brevo_config.get('smtp_port', 587)
            smtp_user = brevo_config.get('smtp_user', brevo_config.get('sender_email', ''))
            api_key = brevo_config['api_key']
            
            # Probar conexión SMTP (igual que verificar_conexiones.py)
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_user, api_key)
            
            dest_count = len(brevo_config.get("correos_destinatarios", []))
            resultados['brevo'] = (True, f"SMTP OK ({dest_count} dest.)")
        except Exception as e:
            resultados['brevo'] = (False, str(e)[:50])
    else:
        resultados['brevo'] = (None, "No configurado")
    
    # 4. Probar Google Drive
    gdrive_config = cliente.get('google_drive', {})
    if gdrive_config.get('enabled') and gdrive_config.get('folder_id'):
        try:
            # Verificar usando las credenciales globales o del cliente
            exito, mensaje = test_google_drive_connection()
            resultados['google_drive'] = (exito, mensaje[:50])
        except Exception as e:
            resultados['google_drive'] = (False, str(e)[:50])
    else:
        resultados['google_drive'] = (None, "No configurado")
    
    # 5. Probar Cloudinary (usando configuración y verificación simple)
    cloudinary_config = cliente.get('cloudinary', {})
    if cloudinary_config.get('enabled') and cloudinary_config.get('cloud_name') and cloudinary_config.get('api_key'):
        try:
            # Configurar Cloudinary
            cloudinary.config(
                cloud_name=cloudinary_config['cloud_name'],
                api_key=cloudinary_config['api_key'],
                api_secret=cloudinary_config['api_secret']
            )
            # Verificar con una llamada simple a la API de recursos
            try:
                import cloudinary.api as cloudinary_api
                cloudinary_api.ping()
                resultados['cloudinary'] = (True, "Conectado")
            except AttributeError:
                # Si ping() no existe, intentar resources()
                try:
                    import cloudinary.api as cloudinary_api
                    cloudinary_api.resources(max_results=1)
                    resultados['cloudinary'] = (True, "Conectado")
                except:
                    # Si falla, verificar que la config sea válida
                    if cloudinary_config['cloud_name'] and cloudinary_config['api_key'] and cloudinary_config['api_secret']:
                        resultados['cloudinary'] = (True, "Config válida")
                    else:
                        resultados['cloudinary'] = (False, "Config incompleta")
        except Exception as e:
            resultados['cloudinary'] = (False, str(e)[:50])
    else:
        resultados['cloudinary'] = (None, "No configurado")
    
    # 6. Probar Supabase
    supabase_config = cliente.get('supabase', {})
    if supabase_config.get('enabled') and supabase_config.get('url') and supabase_config.get('anon_key'):
        try:
            test_client = create_client(supabase_config['url'], supabase_config['anon_key'])
            tabla = supabase_config.get('tabla_nombre', 'alertamediosintrant')
            test_client.table(tabla).select('id').limit(1).execute()
            resultados['supabase'] = (True, f"Tabla: {tabla}")
        except Exception as e:
            error_str = str(e)
            if 'does not exist' in error_str.lower():
                resultados['supabase'] = (True, f"Conexión OK (tabla pendiente)")
            else:
                resultados['supabase'] = (False, str(e)[:50])
    else:
        resultados['supabase'] = (None, "No configurado")
    
    return resultados

def probar_todos_los_clientes():
    """Prueba las conexiones de todos los clientes configurados"""
    clientes = obtener_clientes_activos()
    resultados_globales = {}
    
    for cliente in clientes:
        cliente_nombre = nombre_cliente_mostrar_para_ui(cliente)
        resultados_globales[cliente_nombre] = probar_conexiones_cliente(cliente)
    
    return resultados_globales

def guardar_log_pruebas(resultados):
    """Guarda un log detallado de las pruebas de conexiones"""
    log_path = os.path.join(CARPETA_PROCESADOS, "log_pruebas_conexiones.json")
    log_txt_path = os.path.join(CARPETA_PROCESADOS, "log_pruebas_conexiones.txt")
    
    # Crear estructura del log
    log_data = {
        'fecha_prueba': datetime.now().isoformat(),
        'fecha_legible': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'total_clientes': len(resultados),
        'resumen': {
            'clientes_ok': 0,
            'clientes_con_errores': 0,
            'servicios_ok': 0,
            'servicios_fallidos': 0,
            'servicios_no_configurados': 0
        },
        'clientes': {}
    }
    
    # Procesar resultados por cliente
    for cliente_nombre, servicios in resultados.items():
        cliente_data = {
            'servicios': {},
            'exitosos': 0,
            'fallidos': 0,
            'no_configurados': 0
        }
        
        for servicio, (estado, mensaje) in servicios.items():
            cliente_data['servicios'][servicio] = {
                'estado': 'OK' if estado is True else ('ERROR' if estado is False else 'NO_CONFIGURADO'),
                'mensaje': mensaje,
                'timestamp': datetime.now().isoformat()
            }
            
            if estado is True:
                cliente_data['exitosos'] += 1
                log_data['resumen']['servicios_ok'] += 1
            elif estado is False:
                cliente_data['fallidos'] += 1
                log_data['resumen']['servicios_fallidos'] += 1
            else:
                cliente_data['no_configurados'] += 1
                log_data['resumen']['servicios_no_configurados'] += 1
        
        # Determinar estado del cliente
        if cliente_data['fallidos'] > 0:
            log_data['resumen']['clientes_con_errores'] += 1
        else:
            log_data['resumen']['clientes_ok'] += 1
        
        log_data['clientes'][cliente_nombre] = cliente_data
    
    # Guardar JSON
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log_warning(f"Error guardando log JSON: {e}", "guardar_log_pruebas")
    
    # Guardar TXT legible
    try:
        with open(log_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📋 LOG DE PRUEBAS DE CONEXIONES\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"📅 Fecha: {log_data['fecha_legible']}\n")
            f.write(f"👥 Total Clientes: {log_data['total_clientes']}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("📊 RESUMEN GLOBAL\n")
            f.write("-" * 40 + "\n")
            f.write(f"  ✅ Clientes OK: {log_data['resumen']['clientes_ok']}\n")
            f.write(f"  ❌ Clientes con errores: {log_data['resumen']['clientes_con_errores']}\n")
            f.write(f"  🟢 Servicios OK: {log_data['resumen']['servicios_ok']}\n")
            f.write(f"  🔴 Servicios fallidos: {log_data['resumen']['servicios_fallidos']}\n")
            f.write(f"  ⚪ Servicios no configurados: {log_data['resumen']['servicios_no_configurados']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("📋 DETALLE POR CLIENTE\n")
            f.write("=" * 80 + "\n\n")
            
            for cliente_nombre, cliente_data in log_data['clientes'].items():
                estado_cliente = "🟢" if cliente_data['fallidos'] == 0 else "🔴"
                f.write(f"\n{estado_cliente} {cliente_nombre}\n")
                f.write(f"   ✅ Exitosos: {cliente_data['exitosos']} | ❌ Fallidos: {cliente_data['fallidos']} | ⚪ No config: {cliente_data['no_configurados']}\n")
                f.write("-" * 40 + "\n")
                
                for servicio, info in cliente_data['servicios'].items():
                    icono = "✅" if info['estado'] == 'OK' else ("❌" if info['estado'] == 'ERROR' else "⚪")
                    f.write(f"   {icono} {servicio.upper():15} | {info['estado']:15} | {info['mensaje']}\n")
                
                f.write("\n")
            
            # Sección de errores
            f.write("=" * 80 + "\n")
            f.write("🔴 ERRORES DETECTADOS\n")
            f.write("=" * 80 + "\n\n")
            
            hay_errores = False
            for cliente_nombre, cliente_data in log_data['clientes'].items():
                for servicio, info in cliente_data['servicios'].items():
                    if info['estado'] == 'ERROR':
                        hay_errores = True
                        f.write(f"❌ [{cliente_nombre}] {servicio.upper()}: {info['mensaje']}\n")
            
            if not hay_errores:
                f.write("✅ No se detectaron errores en ningún servicio.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"📁 Log guardado en: {log_txt_path}\n")
            f.write(f"📁 Log JSON: {log_path}\n")
            f.write("=" * 80 + "\n")
    
    except Exception as e:
        log_warning(f"Error guardando log TXT: {e}", "guardar_log_pruebas")
    
    return log_path, log_txt_path, log_data

def cargar_ultimo_log_pruebas():
    """Carga el último log de pruebas"""
    log_path = os.path.join(CARPETA_PROCESADOS, "log_pruebas_conexiones.json")
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log_warning(f"Error cargando log: {e}", "cargar_ultimo_log_pruebas")
    return None

# Botón prominente para probar todas las conexiones
st.markdown("---")
col_test1, col_test2, col_test3 = st.columns([2, 1, 1])

with col_test1:
    if st.button("🧪 **PROBAR TODAS LAS CONEXIONES DE CLIENTES**", type="primary", use_container_width=True):
        with st.spinner("🔄 Probando conexiones de todos los clientes..."):
            resultados = probar_todos_los_clientes()
            
            # Guardar log
            log_json_path, log_txt_path, log_data = guardar_log_pruebas(resultados)
            
            st.markdown("### 📊 **RESULTADOS DE PRUEBAS**")
            st.caption(f"📁 Log guardado: `{log_txt_path}`")
            
            for cliente_nombre, servicios in resultados.items():
                # Contar éxitos y fallos
                exitosos = sum(1 for v in servicios.values() if v[0] is True)
                fallidos = sum(1 for v in servicios.values() if v[0] is False)
                no_config = sum(1 for v in servicios.values() if v[0] is None)
                
                # Determinar color del cliente
                if fallidos > 0:
                    emoji = "🔴"
                elif exitosos > 0:
                    emoji = "🟢"
                else:
                    emoji = "⚪"
                
                with st.expander(f"{emoji} **{cliente_nombre}** - ✅ {exitosos} | ❌ {fallidos} | ⚪ {no_config}", expanded=(fallidos > 0)):
                    cols = st.columns(6)
                    
                    servicios_nombres = ['webhook', 'telegram', 'brevo', 'google_drive', 'cloudinary', 'supabase']
                    servicios_iconos = ['🌐', '📱', '📧', '☁️', '🖼️', '🗄️']
                    
                    for i, (servicio, icono) in enumerate(zip(servicios_nombres, servicios_iconos)):
                        with cols[i]:
                            resultado = servicios.get(servicio, (None, "N/A"))
                            if resultado[0] is True:
                                st.success(f"{icono} ✅")
                                st.caption(resultado[1][:20])
                            elif resultado[0] is False:
                                st.error(f"{icono} ❌")
                                st.caption(resultado[1][:20])
                            else:
                                st.info(f"{icono} ⚪")
                                st.caption("No config")
            
            # Resumen global
            total_clientes = len(resultados)
            clientes_ok = sum(1 for servicios in resultados.values() 
                            if all(v[0] is not False for v in servicios.values()))
            
            if clientes_ok == total_clientes:
                st.success(f"✅ **{total_clientes}/{total_clientes} clientes** con todas las conexiones OK")
            else:
                st.warning(f"⚠️ **{clientes_ok}/{total_clientes} clientes** sin errores")
            
            # Mostrar errores específicos
            errores = []
            for cliente_nombre, servicios in resultados.items():
                for servicio, (estado, mensaje) in servicios.items():
                    if estado is False:
                        errores.append(f"❌ **{cliente_nombre}** → {servicio}: {mensaje}")
            
            if errores:
                st.markdown("### 🔴 **ERRORES DETECTADOS:**")
                for error in errores:
                    st.error(error)

with col_test2:
    # Contador de clientes
    clientes_count = len(obtener_clientes_activos())
    st.metric("👥 Clientes", clientes_count)

with col_test3:
    # Contador de términos
    terminos_count = len(st.session_state.get('terminos_continuos', []))
    st.metric("🏷️ Términos", terminos_count)

# Botones para ver logs y errores
col_log1, col_log2, col_log3 = st.columns([1, 1, 2])
with col_log1:
    if st.button("📋 Ver Último Log", use_container_width=True):
        ultimo_log = cargar_ultimo_log_pruebas()
        if ultimo_log:
            st.session_state['mostrar_log'] = True
        else:
            st.warning("No hay logs de pruebas anteriores")

with col_log2:
    # Contar errores de archivos
    errores_count = len(st.session_state.get('errores_archivos', []))
    log_errores_path = os.path.join(CARPETA_PROCESADOS, "log_errores_archivos.json")
    if os.path.exists(log_errores_path):
        try:
            with open(log_errores_path, "r", encoding="utf-8") as f:
                errores_guardados = json.load(f)
                errores_count = len(errores_guardados)
        except:
            pass
    
    btn_label = f"⚠️ Ver Errores ({errores_count})" if errores_count > 0 else "⚠️ Ver Errores"
    if st.button(btn_label, use_container_width=True):
        st.session_state['mostrar_errores_archivos'] = True

with col_log3:
    log_txt_path = os.path.join(CARPETA_PROCESADOS, "log_pruebas_conexiones.txt")
    if os.path.exists(log_txt_path):
        st.caption(f"📁 Último log: {ultimo_log.get('fecha_legible', 'N/A') if (ultimo_log := cargar_ultimo_log_pruebas()) else 'N/A'}")

# Mostrar log si está activo
if st.session_state.get('mostrar_log', False):
    ultimo_log = cargar_ultimo_log_pruebas()
    if ultimo_log:
        with st.expander("📋 **ÚLTIMO LOG DE PRUEBAS**", expanded=True):
            st.markdown(f"**📅 Fecha:** {ultimo_log.get('fecha_legible', 'N/A')}")
            
            resumen = ultimo_log.get('resumen', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Clientes OK", resumen.get('clientes_ok', 0))
            with col2:
                st.metric("❌ Con Errores", resumen.get('clientes_con_errores', 0))
            with col3:
                st.metric("🟢 Servicios OK", resumen.get('servicios_ok', 0))
            with col4:
                st.metric("🔴 Fallidos", resumen.get('servicios_fallidos', 0))
            
            st.markdown("---")
            st.markdown("**📋 Detalle por Cliente:**")
            
            for cliente_nombre, cliente_data in ultimo_log.get('clientes', {}).items():
                estado = "🟢" if cliente_data.get('fallidos', 0) == 0 else "🔴"
                with st.expander(f"{estado} {cliente_nombre}"):
                    for servicio, info in cliente_data.get('servicios', {}).items():
                        icono = "✅" if info['estado'] == 'OK' else ("❌" if info['estado'] == 'ERROR' else "⚪")
                        st.markdown(f"{icono} **{servicio.upper()}**: {info['mensaje']}")
            
            # Botón para descargar log
            log_txt_path = os.path.join(CARPETA_PROCESADOS, "log_pruebas_conexiones.txt")
            if os.path.exists(log_txt_path):
                with open(log_txt_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                st.download_button(
                    label="📥 Descargar Log TXT",
                    data=log_content,
                    file_name=f"log_pruebas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            if st.button("❌ Cerrar Log"):
                st.session_state['mostrar_log'] = False
                st.rerun()

# ========== MOSTRAR ERRORES DE ARCHIVOS FALLIDOS ==========
if st.session_state.get('mostrar_errores_archivos', False):
    log_errores_path = os.path.join(CARPETA_PROCESADOS, "log_errores_archivos.json")
    errores_archivos = []
    
    # Cargar errores del archivo
    if os.path.exists(log_errores_path):
        try:
            with open(log_errores_path, "r", encoding="utf-8") as f:
                errores_archivos = json.load(f)
        except:
            pass
    
    with st.expander("⚠️ **ERRORES DE ARCHIVOS FALLIDOS**", expanded=True):
        if errores_archivos:
            st.warning(f"📋 **{len(errores_archivos)} archivo(s) con errores registrados**")
            st.caption("Estos errores NO se envían a Telegram/Email. Solo se guardan localmente.")
            
            st.markdown("---")
            
            # Mostrar cada error
            for i, error in enumerate(reversed(errores_archivos[-20:]), 1):  # Últimos 20
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i}. 📄 {error.get('archivo', 'N/A')}**")
                        st.caption(f"⏰ {error.get('timestamp', 'N/A')}")
                        st.error(f"❌ {error.get('error', 'Error desconocido')}")
                    with col2:
                        if error.get('ubicacion') and error.get('ubicacion') != 'No movido':
                            st.caption(f"📁 {error.get('ubicacion', '')}")
                    st.markdown("---")
            
            # Botón para limpiar errores
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("🗑️ Limpiar Errores", use_container_width=True):
                    try:
                        with open(log_errores_path, "w", encoding="utf-8") as f:
                            json.dump([], f)
                        st.session_state['errores_archivos'] = []
                        st.success("✅ Errores limpiados")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error limpiando: {e}")
            
            with col_clear2:
                # Descargar log
                log_content = json.dumps(errores_archivos, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Descargar Log JSON",
                    data=log_content,
                    file_name=f"errores_archivos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.success("✅ No hay errores de archivos registrados")
        
        if st.button("❌ Cerrar Errores"):
            st.session_state['mostrar_errores_archivos'] = False
            st.rerun()

st.markdown("---")

# Mostrar flujo de envío completo (colapsado para UI más simple)
with st.expander("FLUJO DE ENVIO MULTICLIENTE", expanded=False):
    st.markdown("""
**Sistema multicliente:** cada termino esta asociado a un cliente con sus credenciales.

**Cuando se detecta una coincidencia:**
1. Deteccion del cliente por termino.
2. Envio a destinos del cliente:
   - Telegram
   - Webhook
   - Brevo
   - Google Drive
   - Supabase

**Resultado:** cada cliente recibe sus coincidencias en sus destinos configurados.
""")

st.markdown("---")

# === SIDEBAR CON ESTADÍSTICAS ===
with st.sidebar:
    st.markdown(f"**RadioAnalizer** `{RADIO_ANALIZER_VERSION}`")
    st.markdown("---")
    st.header("📊 Estadísticas de Sesión")
    st.metric("Audios encontrados", st.session_state.videos_encontrados)
    st.metric("Audios procesados", st.session_state.videos_procesados)
    st.metric("Clips generados", st.session_state.clips_generados)

    st.markdown("---")
    st.header("📻 Emisoras")
    if _emisoras_dashboard:
        _grab = [x["nombre"] for x in _emisoras_dashboard if x.get("grabando")]
        if _grab:
            st.success(f"🔴 Grabando ({len(_grab)}): " + ", ".join(_grab[:6]) + ("…" if len(_grab) > 6 else ""))
        else:
            st.caption("Ninguna con actividad reciente (heurística por mtime).")
        for _em in _emisoras_dashboard[:20]:
            _ic = "🔴" if _em.get("grabando") else "⚪"
            st.caption(f"{_ic} {_em['nombre']} — {_em['n_audios']} arch.")
        if len(_emisoras_dashboard) > 20:
            st.caption(f"… y {len(_emisoras_dashboard) - 20} más (ver tabla arriba)")
    else:
        st.caption("Sin carpetas de emisoras en la entrada.")
    
    # === Contador de uso Mistral/Voxtral ===
    st.markdown("---")
    st.header("🧠 Uso Mistral / Voxtral")
    
    total_audio_secs = st.session_state.get('mistral_total_audio_seconds', 0)
    total_tokens = st.session_state.get('mistral_total_tokens', 0)
    total_transcripciones = st.session_state.get('mistral_total_transcripciones', 0)
    
    # Formatear tiempo de audio
    audio_min = int(total_audio_secs // 60)
    audio_seg = int(total_audio_secs % 60)
    if audio_min > 0:
        audio_display = f"{audio_min}m {audio_seg}s"
    else:
        audio_display = f"{audio_seg}s"
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Transcripciones", total_transcripciones)
        st.metric("Audio procesado", audio_display)
    with col_m2:
        st.metric("Tokens totales", f"{total_tokens:,}")
        # Estimación de costo (Voxtral transcription pricing aprox)
        # Voxtral Mini Transcribe: ~$0.012 por minuto de audio
        costo_estimado = (total_audio_secs / 60) * 0.012
        st.metric("Costo estimado", f"${costo_estimado:.4f}")
    
    if total_transcripciones > 0:
        st.caption(f"Prompt: {st.session_state.get('mistral_total_prompt_tokens', 0):,} | Completion: {st.session_state.get('mistral_total_completion_tokens', 0):,} tokens")
    
    # === Loop Continuo ===
    st.markdown("---")
    st.header("🔄 Loop Continuo")
    
    loop_activo = st.checkbox(
        "Activar loop automático",
        value=st.session_state.get('loop_continuo', True),
        help="Al terminar de procesar, re-escanea automáticamente buscando audios nuevos"
    )
    if loop_activo != st.session_state.get('loop_continuo', True):
        st.session_state.loop_continuo = loop_activo
    
    if loop_activo:
        col_loop1, col_loop2 = st.columns(2)
        with col_loop1:
            nuevo_intervalo_loop = st.number_input(
                "Espera entre ciclos (s)",
                min_value=10, max_value=600, 
                value=st.session_state.get('intervalo_loop', 60),
                step=10,
                help="Segundos de espera entre ciclos cuando hay videos"
            )
            if nuevo_intervalo_loop != st.session_state.get('intervalo_loop', 60):
                st.session_state.intervalo_loop = nuevo_intervalo_loop
        
        with col_loop2:
            nuevo_intervalo_vacio = st.number_input(
                "Espera sin nuevos (s)",
                min_value=30, max_value=1800,
                value=st.session_state.get('intervalo_loop_vacio', 120),
                step=30,
                help="Segundos de espera cuando no hay audios nuevos"
            )
            if nuevo_intervalo_vacio != st.session_state.get('intervalo_loop_vacio', 120):
                st.session_state.intervalo_loop_vacio = nuevo_intervalo_vacio
        
        ciclos = st.session_state.get('loop_ciclo_numero', 0)
        st.caption(f"Ciclos completados en esta sesión: **{ciclos}**")
    else:
        st.caption("El loop está desactivado. El sistema se detendrá al terminar de procesar.")

    st.markdown("---")
    st.header("📤 Envíos por cliente")
    _orden_envios_ids = ("default", "intrant", "minerd")
    _labels_envios = {"default": "EDESUR", "intrant": "Intrant", "minerd": "MINERD"}
    _clist_envios = cargar_clientes()
    for _cid in _orden_envios_ids:
        _c = next((x for x in _clist_envios if x.get("id") == _cid), None)
        if not _c:
            continue
        _label = _labels_envios.get(_cid, _c.get("nombre", _cid))
        _val_actual = _c.get("envios_habilitados", True)
        _toggle = st.toggle(
            f"Envíos {_label}",
            value=_val_actual,
            help="ON: notificar Telegram, correo, webhooks, Drive y Supabase. OFF: pausar todos los envíos a este cliente.",
            key=f"envios_toggle_{_cid}",
        )
        if _toggle != _val_actual:
            actualizar_cliente(_cid, {"envios_habilitados": _toggle})
            st.rerun()

    st.markdown("---")
    st.header("⚙️ Configuración Avanzada")
    
    # Configuración de intervalos con auto-guardado
    nuevo_intervalo = st.selectbox(
        "⏱️ Intervalo de búsqueda:",
        options=[30, 60, 120, 300, 600],
        format_func=lambda x: f"{x} segundos ({x//60}min)" if x >= 60 else f"{x} segundos",
        index=1 if st.session_state.intervalo == 60 else [30, 60, 120, 300, 600].index(st.session_state.intervalo) if st.session_state.intervalo in [30, 60, 120, 300, 600] else 1,
        key="sidebar_intervalo_select"
    )
    
    # Auto-guardar si cambió el intervalo
    if nuevo_intervalo != st.session_state.intervalo:
        st.session_state.intervalo = nuevo_intervalo
        guardar_configuracion_completa(
            st.session_state.terminos_continuos,
            st.session_state.intervalo,
            st.session_state.get('duracion_clip', 90),
            st.session_state.get('buffer_anterior', 30),
            st.session_state.mostrar_coincidencias
        )
    
    # Opciones de visualización con auto-guardado
    nuevo_mostrar = st.checkbox("Mostrar coincidencias en tiempo real", value=st.session_state.mostrar_coincidencias)
    if nuevo_mostrar != st.session_state.mostrar_coincidencias:
        st.session_state.mostrar_coincidencias = nuevo_mostrar
        guardar_configuracion_completa(
            st.session_state.terminos_continuos,
            st.session_state.intervalo,
            st.session_state.get('duracion_clip', 90),
            st.session_state.get('buffer_anterior', 30),
            st.session_state.mostrar_coincidencias
        )
    
    st.session_state.mostrar_solo_actual_relevantes = st.checkbox(
        "Mostrar solo audio actual + relevantes",
        value=st.session_state.get('mostrar_solo_actual_relevantes', True),
        help="Oculta detalle de archivos sin coincidencias y mantiene en pantalla solo el actual, coincidencias y menciones tangenciales."
    )
    
    # Configuración de clips con auto-guardado
    st.markdown("**⏱️ Configuración de Duración de Clips:**")
    st.info("💡 **Cómo funciona:** El clip incluirá [Buffer anterior] + momento de coincidencia + tiempo restante hasta [Duración total]")
    
    nueva_duracion = st.slider("Duración total del clip (segundos)", 30, 180, st.session_state.get('duracion_clip', 90), 
                               help="Duración total del clip generado. El usuario requiere estrictamente 90s para capturar ideas completas.")
    
    # Forzar que el valor mínimo sea 90 si no está configurado
    if nueva_duracion < 90:
        nueva_duracion = 90
        st.warning("⚠️ Duración ajustada a 90 segundos (Mínimo requerido por ti)")
    nuevo_buffer = st.slider("Buffer anterior (segundos)", 10, 90, st.session_state.get('buffer_anterior', 30),
                            help="Tiempo antes de la coincidencia. Por defecto 30s antes")
    
    # Mostrar cálculo del buffer posterior
    buffer_posterior = nueva_duracion - nuevo_buffer
    st.caption(f"📊 **Resultado:** {nuevo_buffer//60}:{nuevo_buffer%60:02d} antes + {buffer_posterior//60}:{buffer_posterior%60:02d} después de la coincidencia")
    
    # Auto-guardar configuración de clips
    if nueva_duracion != st.session_state.get('duracion_clip', 90) or nuevo_buffer != st.session_state.get('buffer_anterior', 30):
        st.session_state.duracion_clip = nueva_duracion
        st.session_state.buffer_anterior = nuevo_buffer
        guardar_configuracion_completa(
            st.session_state.terminos_continuos,
            st.session_state.intervalo,
            st.session_state.duracion_clip,
            st.session_state.buffer_anterior,
            st.session_state.mostrar_coincidencias
        )
    
    st.markdown("---")
    st.header("🌐 Configuración Webhook")
    
    webhook_config = cargar_webhook_config()
    
    # Habilitar/Deshabilitar webhook global
    webhook_enabled = st.checkbox("Activar envío de clips", value=webhook_config['enabled'])
    
    if webhook_enabled:
        st.subheader("📡 Seleccionar Webhooks de Destino")
        
        # Switches individuales para cada webhook
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enviar_makecom = st.checkbox(
                "🔵 Make.com", 
                value=webhook_config.get('enviar_makecom', True),
                help="Enviar a Make.com"
            )
            st.text("hook.us1.make.com")
        
        with col2:
            enviar_n8n = st.checkbox(
                "🟢 N8N", 
                value=webhook_config.get('enviar_n8n', True),
                help="Enviar a N8N principal"
            )
            st.text("webhook/edesurbot")
        
        with col3:
            enviar_n8n_test = st.checkbox(
                "🟡 N8N-Test", 
                value=webhook_config.get('enviar_n8n_test', True),
                help="Enviar a N8N de prueba"
            )
            st.text("webhook-test/edesurbot")
        
        # Mostrar estado de selección
        webhooks_activos = []
        if enviar_makecom:
            webhooks_activos.append("Make.com")
        if enviar_n8n:
            webhooks_activos.append("N8N")
        if enviar_n8n_test:
            webhooks_activos.append("N8N-Test")
            
        if webhooks_activos:
            st.info(f"📤 Enviando a: {', '.join(webhooks_activos)}")
        else:
            st.warning("⚠️ No hay webhooks seleccionados")
        
        # Configuración general
        max_size = st.slider("Tamaño máximo por clip (MB):", 1, 50, min(webhook_config['max_file_size_mb'], 25))
        
        if st.button("💾 Guardar Configuración Webhook"):
            nueva_config = webhook_config.copy()
            nueva_config.update({
                'enabled': webhook_enabled,
                'enviar_makecom': enviar_makecom,
                'enviar_n8n': enviar_n8n,
                'enviar_n8n_test': enviar_n8n_test,
                'max_file_size_mb': max_size
            })
            
            if guardar_webhook_config(nueva_config):
                st.success("✅ Configuración webhook guardada")
            
        # Test de webhooks seleccionados
        if webhooks_activos:
            if st.button("🧪 Probar Webhooks Seleccionados"):
                exito, mensaje = webhook_notification_simple(
                    "test_video.mp4", 
                    "**TÉRMINOS DETECTADOS:** test\n\nEsto es una prueba del webhook", 
                    ["test"]
                )
                if exito:
                    st.success(f"✅ Webhooks OK: {mensaje}")
                else:
                    st.error(f"❌ Error webhooks: {mensaje}")
                    
        st.info("💡 Se enviarán los clips donde se encontraron coincidencias + resumen a los webhooks seleccionados")
    else:
        st.info("Webhook desactivado")
    
    st.markdown("---")
    st.header("📱 Configuración Telegram")
    
    telegram_config = cargar_telegram_config()
    
    # Habilitar/Deshabilitar Telegram
    telegram_enabled = st.checkbox("Activar envío a Telegram", value=telegram_config['enabled'])
    
    if telegram_enabled:
        bot_token = st.text_input("🤖 Bot Token:", value=telegram_config['bot_token'], 
                                 placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz", type="password")
        
        chat_id = st.text_input("💬 Chat ID:", value=telegram_config['chat_id'], 
                               placeholder="-1001234567890")
        
        col1, col2 = st.columns(2)
        with col1:
            send_clips = st.checkbox("Enviar clips", value=telegram_config.get('send_clips', True))
        with col2:
            send_summary = st.checkbox("Enviar resumen", value=telegram_config.get('send_summary', True))
        
        use_cloudinary = st.checkbox("Usar Cloudinary para clips de audio", value=telegram_config.get('use_cloudinary', True))
        
        if st.button("💾 Guardar Telegram"):
            nueva_config = telegram_config.copy()
            nueva_config.update({
                'enabled': telegram_enabled,
                'bot_token': bot_token.strip(),
                'chat_id': chat_id.strip(),
                'send_clips': send_clips,
                'send_summary': send_summary,
                'use_cloudinary': use_cloudinary
            })
            
            if guardar_telegram_config(nueva_config):
                st.success("✅ Configuración Telegram guardada")
        
        # Test de Telegram
        if bot_token.strip() and chat_id.strip():
            if st.button("🧪 Probar Telegram"):
                exito, mensaje = test_telegram_connection()
                if exito:
                    st.success(f"✅ Telegram OK: {mensaje}")
                else:
                    st.error(f"❌ Error Telegram: {mensaje}")
        
        # Configuración de Cloudinary si está habilitada
        if use_cloudinary:
            st.markdown("#### ☁️ Configuración Cloudinary")
            
            cloudinary_config = cargar_cloudinary_config()
            
            cloud_name = st.text_input("Cloud Name:", value=cloudinary_config['cloud_name'], 
                                      placeholder="tu-cloud-name")
            
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input("API Key:", value=cloudinary_config['api_key'], 
                                       placeholder="123456789012345", type="password")
            with col2:
                api_secret = st.text_input("API Secret:", value=cloudinary_config['api_secret'], 
                                          placeholder="abcdefghijklmnopqrstuvwxyz", type="password")
            
            folder_name = st.text_input("Carpeta:", value=cloudinary_config.get('folder', 'video_analyzer_clips'))
            
            if st.button("💾 Guardar Cloudinary"):
                nueva_config = cloudinary_config.copy()
                nueva_config.update({
                    'cloud_name': cloud_name.strip(),
                    'api_key': api_key.strip(),
                    'api_secret': api_secret.strip(),
                    'folder': folder_name.strip()
                })
                
                if guardar_cloudinary_config(nueva_config):
                    st.success("✅ Configuración Cloudinary guardada")
            
            st.info("💡 Cloudinary se usará para subir archivos grandes a Telegram")
        
        st.info("📱 Se enviarán clips y resúmenes a tu canal/chat de Telegram")
    else:
        st.info("Telegram desactivado")
    
    st.markdown("---")
    st.header("☁️ Configuración Google Drive")
    
    # Habilitar/Deshabilitar Google Drive
    gdrive_enabled = st.checkbox("Activar envío a Google Drive", value=True)
    
    if gdrive_enabled:
        st.info(f"📁 **Carpeta destino:** `{GOOGLE_DRIVE_FOLDER_ID}`")
        st.info("🔑 **Credenciales configuradas:** ✅")
        
        # Mostrar información de la carpeta
        col1, col2 = st.columns(2)
        with col1:
            st.caption("📂 ID de carpeta")
            st.code(GOOGLE_DRIVE_FOLDER_ID)
        with col2:
            st.caption("🔐 Cliente ID")
            st.code(GOOGLE_CLIENT_ID[:20] + "...")
        
        # Test de conexión
        if st.button("🧪 Probar Google Drive", help="Verificar conexión con Google Drive"):
            with st.spinner("Probando Google Drive..."):
                exito, mensaje = test_google_drive_connection()
                if exito:
                    st.success(f"✅ Google Drive: {mensaje}")
                else:
                    st.error(f"❌ Google Drive: {mensaje}")
        
        st.info("💡 Se enviarán clips y resúmenes TXT a Google Drive automáticamente")
    else:
        st.info("Google Drive desactivado")
    
    st.markdown("---")
    st.header("📧 Configuración Brevo (Correo)")
    
    brevo_config = cargar_brevo_config()
    
    # Habilitar/Deshabilitar Brevo
    brevo_enabled = st.checkbox("Activar envío de correos", value=brevo_config['enabled'])
    
    if brevo_enabled:
        st.info("💡 **Brevo (ex SendinBlue)** - Servicio profesional de correo electrónico")
        
        # Configuración del remitente
        st.subheader("👤 Configuración del Remitente")
        sender_email = st.text_input("📧 Email del Remitente:", value=brevo_config['sender_email'], 
                                   placeholder="tu-email@dominio.com")
        sender_name = st.text_input("👤 Nombre del Remitente:", value=brevo_config['sender_name'], 
                                  placeholder="Sistema de Análisis de Audio")
        
        # API Key
        api_key = st.text_input("🔑 API Key de Brevo:", value=brevo_config['api_key'], 
                               placeholder="xkeysib-...", type="password",
                               help="Tu API Key de Brevo (SMTP Key)")
        
        # Gestión de múltiples destinatarios
        st.subheader("📨 Lista de Destinatarios")
        
        # Cargar correos guardados
        correos_guardados = cargar_correos_guardados()
        
        # Mostrar correos existentes
        if correos_guardados:
            st.success(f"📧 **{len(correos_guardados)} correos configurados:**")
            
            # Mostrar en columnas
            cols = st.columns(3)
            for i, correo_data in enumerate(correos_guardados):
                with cols[i % 3]:
                    estado = "🟢" if correo_data.get('activo', True) else "🔴"
                    st.write(f"{estado} **{correo_data['nombre']}**")
                    st.caption(f"📧 {correo_data['email']}")
                    
                    # Botón para eliminar
                    if st.button(f"🗑️ Eliminar", key=f"del_{correo_data['email']}"):
                        exito, mensaje = eliminar_correo_de_lista(correo_data['email'])
                        if exito:
                            st.success(mensaje)
                            st.rerun()
                        else:
                            st.error(mensaje)
        else:
            st.info("📭 No hay correos configurados aún")
        
        st.markdown("---")
        
        # Agregar nuevo correo
        st.subheader("➕ Agregar Nuevo Destinatario")
        
        # Inicializar contador para widgets de Brevo
        if 'brevo_widget_counter' not in st.session_state:
            st.session_state.brevo_widget_counter = 0
        
        col1, col2 = st.columns([2, 1])
        with col1:
            nuevo_correo = st.text_input("📧 Email:", placeholder="nuevo@dominio.com", key=f"brevo_nuevo_correo_input_{st.session_state.brevo_widget_counter}")
        with col2:
            nombre_correo = st.text_input("👤 Nombre:", placeholder="Nombre", key=f"brevo_nombre_correo_input_{st.session_state.brevo_widget_counter}")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("➕ Agregar Correo"):
                if nuevo_correo.strip():
                    exito, mensaje = agregar_correo_a_lista(nuevo_correo, nombre_correo)
                    if exito:
                        st.success(mensaje)
                        # Forzar recreación de widgets incrementando contador
                        st.session_state.brevo_widget_counter += 1
                        st.rerun()
                    else:
                        st.error(mensaje)
                else:
                    st.warning("Ingresa un correo válido")
        
        with col2:
            if st.button("📧 Agregar FGJ Medios"):
                exito, mensaje = agregar_correo_a_lista("info@fgjmedios.com", "FGJ Medios")
                if exito:
                    st.success(mensaje)
                    st.rerun()
                else:
                    st.info("El correo ya existe")
        
        with col3:
            if correos_guardados and st.button("🧹 Limpiar Todos los Correos"):
                if guardar_correos_lista([]):
                    st.success("Lista de correos limpiada")
                    st.rerun()
        
        # Configuración avanzada
        with st.expander("⚙️ Configuración Avanzada"):
            smtp_server = st.text_input("🌐 Servidor SMTP:", value=brevo_config['smtp_server'], 
                                       placeholder="smtp-relay.sendinblue.com")
            smtp_port = st.number_input("🔌 Puerto SMTP:", value=brevo_config['smtp_port'], 
                                       min_value=1, max_value=65535, step=1)
        
        # Botón para guardar
        if st.button("💾 Guardar Configuración Brevo"):
            nueva_config = brevo_config.copy()
            nueva_config.update({
                'enabled': brevo_enabled,
                'api_key': api_key.strip(),
                'sender_email': sender_email.strip(),
                'sender_name': sender_name.strip(),
                'recipient_email': '',  # Ya no se usa, se maneja con la lista
                'recipient_name': '',   # Ya no se usa, se maneja con la lista
                'smtp_server': smtp_server.strip(),
                'smtp_port': smtp_port
            })
            
            if guardar_brevo_config(nueva_config):
                st.success("✅ Configuración Brevo guardada exitosamente")
                st.rerun()  # Recargar para mostrar el estado actualizado
        
        # Test de conexión
        if api_key.strip() and sender_email.strip() and correos_guardados:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Probar Conexión Brevo"):
                    with st.spinner("Probando conexión con Brevo..."):
                        exito, mensaje = test_brevo_connection()
                        if exito:
                            st.success(f"✅ Brevo: {mensaje}")
                        else:
                            st.error(f"❌ Brevo: {mensaje}")
            
            with col2:
                if st.button("📧 Enviar Correo de Prueba"):
                    with st.spinner("Enviando correo de prueba..."):
                        exito, mensaje = enviar_correo_brevo(
                            "PRUEBA MANUAL",
                            "**CORREO DE PRUEBA MANUAL**\\n\\nEste es un correo de prueba manual del sistema de análisis de audio de FGJ Medios.\\n\\n✅ Si recibiste este correo, la configuración está funcionando correctamente.\\n\\n🎯 **Características del correo:**\\n- Plantilla HTML moderna y responsive\\n- Envío a múltiples destinatarios\\n- Resumen completo de coincidencias\\n- Archivos adjuntos sin limitación de tamaño\\n- Información detallada del medio\\n- Términos detectados destacados\\n\\n📧 **Sistema configurado para:** Análisis automático de audio con notificaciones inmediatas por correo.",
                            "test_manual.mp4",
                            None,  # No hay video_path en prueba manual
                            "FGJ Medios - Prueba Manual del Sistema",  # info_medio
                            ["PRUEBA MANUAL", "FGJ MEDIOS", "SISTEMA CORREOS"],  # terminos_detectados
                            None  # No hay video_url_gdrive en prueba manual
                        )
                        if exito:
                            st.success(f"✅ {mensaje}")
                        else:
                            st.error(f"❌ Error enviando correo: {mensaje}")
        elif not correos_guardados:
            st.warning("⚠️ Agrega al menos un destinatario para poder probar")
        
        # Información sobre el funcionamiento
        st.info("📧 **Funcionamiento:** Se enviará un correo automáticamente cuando se detecte una coincidencia con:")
        st.markdown("""
        - 🎯 **Asunto:** "Radio — 🎯 Coincidencia: [TÉRMINO]"
        - 📋 **Cuerpo:** Resumen COMPLETO generado por IA en formato HTML moderno
        - 📺 **Medio:** Información detallada del medio donde se detectó
        - 🏷️ **Términos:** Todos los términos detectados destacados
        - 🎬 **Clip de audio:** Incrustado en el correo (si está disponible en Cloudinary)
        - 📎 **Adjunto:** Archivo de video completo (SIN limitación de tamaño)
        - 👥 **Destinatarios:** Envío a TODOS los correos configurados en la lista
        """)
        
        st.success("✨ **Mejoras implementadas:** Resumen completo, múltiples destinatarios, videos sin límite de tamaño")
        
        # Mostrar estado de configuración
        correos_activos = obtener_correos_activos()
        if all([api_key.strip(), sender_email.strip()]) and correos_activos:
            st.success(f"✅ Configuración completa - Lista para enviar correos a {len(correos_activos)} destinatarios")
        elif not correos_activos:
            st.warning("⚠️ Agrega al menos un destinatario para completar la configuración")
        else:
            st.warning("⚠️ Configuración incompleta - Completa API Key y email del remitente")
            
    else:
        st.info("📧 Correo Brevo desactivado")
        st.markdown("""
        **¿Por qué usar Brevo?**
        - ✅ Servicio profesional de correo
        - ✅ Alta deliverability 
        - ✅ Plantillas HTML modernas
        - ✅ Soporte para adjuntos y videos
        - ✅ API confiable y rápida
        """)

# === PANEL DE CONTROL PRINCIPAL ===
st.markdown("## 🎛️ Panel de Control Principal")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.running:
        st.success("🟢 **ACTIVO**")
        st.write(f"⏰ {st.session_state.ultimo_chequeo.strftime('%H:%M:%S')}")
    else:
        st.error("🔴 **INACTIVO**")
        st.write("⏸️ En espera")

with col2:
    if st.session_state.terminos_continuos:
        st.info(f"🔍 **{len(st.session_state.terminos_continuos)} términos configurados**")
        
        # Mostrar estado de guardado
        if os.path.exists(TERMINOS_CONFIG):
            try:
                with open(TERMINOS_CONFIG, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    fecha_guardado = data.get('fecha_actualizacion', '')
                    if fecha_guardado:
                        fecha_dt = datetime.fromisoformat(fecha_guardado.replace('Z', '+00:00').replace('+00:00', ''))
                        st.caption(f"💾 Guardado: {fecha_dt.strftime('%H:%M:%S')}")
            except Exception:
                pass
    else:
        st.warning("⚠️ Sin términos")

with col3:
    # Tiempo de próximo chequeo
    if st.session_state.running:
        proximo = st.session_state.ultimo_chequeo + timedelta(seconds=st.session_state.intervalo)
        tiempo_restante = proximo - datetime.now()
        if tiempo_restante.total_seconds() > 0:
            st.info(f"⏳ Próximo en {int(tiempo_restante.total_seconds())}s")
        else:
            st.info("🔄 Procesando...")
    else:
        st.info("⏸️ Pausado")

with col4:
    # Estado del sistema - Calculado después de definir las funciones
    try:
        total_clips = len(buscar_todos_los_clips())
        st.metric("Total clips", total_clips)
    except Exception:
        st.metric("Total clips", "...")

# === REINICIO Y LIMPIEZA ===
st.markdown("### 🔧 Controles del Sistema")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Reiniciar Aplicación"):
        st.session_state.clear()
        init_session_state()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("✅ Aplicación reiniciada")
        st.rerun()

with col2:
    if st.button("💾 Limpiar Caché"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("🧹 Caché limpiado")
        st.rerun()

with col3:
    if st.button("🗑️ Borrar Clips Antiguos"):
        clips_borrados = borrar_clips_antiguos(dias=7)
        if clips_borrados > 0:
            st.success(f"🗑️ Borrados {clips_borrados} clips antiguos")
            st.rerun()
        else:
            st.info("✅ No hay clips antiguos para borrar")

with col4:
    if st.button("📊 Generar Reporte"):
        if st.session_state.resumen_global:
            generar_resumen_md(st.session_state.resumen_global)
        else:
            st.warning("⚠️ No hay datos para reportar")

# === OPTIMIZACIÓN Y CACHÉ ===
st.markdown("### ⚡ Optimización de Búsqueda")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧹 Limpiar Caché", help="Limpiar caché de archivos escaneados"):
        archivos_limpiados = limpiar_cache_escaneo()
        if archivos_limpiados > 0:
            st.success(f"🧹 Caché limpiado: {archivos_limpiados} archivos eliminados")
        else:
            st.info("✅ Caché ya está limpio")

with col2:
    if st.button("📊 Ver Estadísticas Caché", help="Mostrar estadísticas del caché de escaneo"):
        try:
            cache = cargar_cache_escaneo()
            archivos_cache = cache.get('archivos_escaneados', {})
            ultima_actualizacion = cache.get('ultima_actualizacion', 0)
            
            if archivos_cache:
                st.markdown("**📊 Estadísticas del Caché:**")
                st.metric("Archivos en caché", len(archivos_cache))
                
                if ultima_actualizacion > 0:
                    fecha_actualizacion = datetime.fromtimestamp(ultima_actualizacion)
                    st.metric("Última actualización", fecha_actualizacion.strftime('%Y-%m-%d %H:%M:%S'))
                
                # Contar tipos de archivos
                procesados = sum(1 for info in archivos_cache.values() if info.get('procesado', False))
                muy_pequeños = sum(1 for info in archivos_cache.values() if info.get('muy_pequeño', False))
                nuevos_detectados = sum(1 for info in archivos_cache.values() if info.get('detectado_como_nuevo', False))
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Ya procesados", procesados)
                    st.metric("Muy pequeños", muy_pequeños)
                with col_b:
                    st.metric("Nuevos detectados", nuevos_detectados)
                    st.metric("Pendientes", len(archivos_cache) - procesados - muy_pequeños)
                    
            else:
                st.info("📭 Caché vacío")
        except Exception as e:
            st.error(f"❌ Error leyendo caché: {e}")

with col3:
    if st.button("🔄 Resetear Caché", help="Resetear completamente el caché de escaneo"):
        try:
            if os.path.exists(CACHE_ESCANEO):
                os.remove(CACHE_ESCANEO)
                st.success("🔄 Caché reseteado completamente")
            else:
                st.info("✅ No había caché para resetear")
        except Exception as e:
            st.error(f"❌ Error reseteando caché: {e}")

with col4:
    if st.button("🌐 Diagnóstico Red", help="Verificar conectividad con APIs"):
        with st.spinner("🔍 Verificando conectividad..."):
            resultados = diagnosticar_conectividad()
            
            st.markdown("**🌐 Estado de Conectividad:**")
            
            # Internet general
            if resultados['internet']:
                st.success("✅ Conectividad a internet: OK")
            else:
                st.error("❌ Sin conectividad a internet")
            
            # OpenAI
            if resultados['openai']:
                st.success("✅ OpenAI API: Disponible")
            else:
                st.warning("⚠️ OpenAI API: No disponible")
            
            # Mistral
            if resultados['mistral']:
                st.success("✅ Mistral API: Disponible")
            else:
                st.warning("⚠️ Mistral API: No disponible")
            
            # Resumen
            if not resultados['internet']:
                st.error("🔧 **Solución:** Verificar conexión a internet")
            elif not resultados['openai'] and not resultados['mistral']:
                st.error("🔧 **Solución:** Verificar configuración de APIs y claves")
            else:
                st.info("✅ **Estado:** Al menos una API disponible")

# === ESTADO DE APIS ===
# Verificar estado de Mistral y mostrar alerta si hay problemas
try:
    mistral_disponible, mistral_estado = verificar_estado_mistral()
    if not mistral_disponible:
        st.warning(f"⚠️ **Mistral API no disponible:** {mistral_estado}")
        st.info("🔄 **Sistema de Fallback Activo:** faster-whisper (local) → Voxtral (Mistral) → OpenAI Whisper")
    else:
        st.success(f"✅ **Mistral API:** {mistral_estado}")
except Exception as e:
    st.error(f"❌ Error verificando Mistral: {str(e)[:100]}")

# === PRUEBAS RÁPIDAS DE SERVICIOS ===
st.markdown("### 🧪 Pruebas Rápidas de Servicios")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🌐 Probar Webhook", help="Verificar conexión con Make.com"):
        with st.spinner("Probando webhook..."):
            # Primero probar conectividad básica
            try:
                import requests
                test_response = requests.get("https://httpbin.org/status/200", timeout=5)
                if test_response.status_code == 200:
                    st.info("🌐 Conectividad a internet: ✅")
                else:
                    st.warning("⚠️ Problemas de conectividad")
            except Exception:
                st.error("❌ Sin conexión a internet")
            
            # Luego probar el webhook
            exito, mensaje = webhook_notification_simple(
                "test_video.mp4", 
                "**TÉRMINOS DETECTADOS:** prueba\n\nEste es un test del sistema de análisis de audio.", 
                ["prueba", "test"]
            )
            if exito:
                st.success(f"✅ Webhook funcionando: {mensaje}")
            else:
                st.error(f"❌ Error en webhook: {mensaje}")
                st.info("💡 Tip: El webhook puede tardar unos segundos en responder")

with col2:
    if st.button("📱 Probar Telegram", help="Verificar conexión con el bot de Telegram"):
        with st.spinner("Probando Telegram..."):
            exito, mensaje = test_telegram_connection()
            if exito:
                st.success(f"✅ Telegram funcionando: {mensaje}")
            else:
                st.error(f"❌ Error en Telegram: {mensaje}")

with col3:
    if st.button("☁️ Probar Cloudinary", help="Verificar configuración de Cloudinary"):
        with st.spinner("Probando Cloudinary..."):
            if configurar_cloudinary():
                st.success("✅ Cloudinary configurado correctamente")
                st.info("📹 Listo para subir archivos grandes")
            else:
                st.error("❌ Error en configuración de Cloudinary")

with col4:
    if st.button("☁️ Probar Google Drive", help="Verificar conexión con Google Drive"):
        with st.spinner("Probando Google Drive..."):
            exito, mensaje = test_google_drive_connection()
            if exito:
                st.success(f"✅ Google Drive: {mensaje}")
            else:
                st.error(f"❌ Google Drive: {mensaje}")

with col5:
    if st.button("📤 Test Webhooks", help="Enviar primer resumen ejecutivo guardado a ambos webhooks"):
        with st.spinner("Enviando primer resumen ejecutivo guardado..."):
            # Buscar el primer resumen ejecutivo guardado
            config = cargar_webhook_config()
            
            # Intentar obtener el primer resumen del session_state
            primer_resumen = None
            primer_video = None
            terminos_encontrados = []
            
            if st.session_state.resumen_global and len(st.session_state.resumen_global) > 0:
                # Usar el primer elemento del resumen global
                primer_item = st.session_state.resumen_global[0]
                primer_video = primer_item.get('video', 'video_guardado.mp4')
                primer_resumen = primer_item.get('transcripcion_completa', primer_item.get('texto', 'Resumen ejecutivo guardado'))
                terminos_encontrados = [primer_item.get('termino', 'termino_guardado')]
            else:
                # Fallback si no hay resúmenes guardados
                primer_video = "resumen_ejemplo.mp4"
                primer_resumen = "**TÉRMINOS DETECTADOS:** ejemplo\n\n1. Tema principal: Este es un resumen ejecutivo de ejemplo del sistema\n2. Contexto: Enviado desde el analizador de audio\n3. Puntos clave: Sistema funcionando correctamente\n4. Relevancia: Prueba de conectividad con resumen real"
                terminos_encontrados = ["ejemplo", "prueba"]
            
            # Crear el mensaje con formato de resumen ejecutivo real
            data_prueba = {
                'evento': 'video_analizado',
                'timestamp': datetime.now().isoformat(),
                'video': primer_video,
                'terminos': terminos_encontrados,
                'resumen': primer_resumen[:500] + "..." if len(primer_resumen) > 500 else primer_resumen,
                'servidor': 'analizador_audio_ia_v2'
            }
            
            # Enviar solo a webhooks seleccionados
            exitos = []
            
            # Enviar a webhook principal (Make.com) si está habilitado
            if config.get('enviar_makecom', True):
                exito_principal, mensaje_principal = enviar_a_webhook_individual(
                    config['url'], data_prueba, "test_webhooks", "Make.com"
                )
                exitos.append(exito_principal)
                if exito_principal:
                    st.success(f"✅ Make.com: {mensaje_principal}")
                else:
                    st.error(f"❌ Make.com: {mensaje_principal}")
            else:
                st.info("🔵 Make.com: No seleccionado")
                
            # Enviar a webhook secundario (N8N) si está habilitado
            if config.get('enviar_n8n', True):
                exito_secundario, mensaje_secundario = enviar_a_webhook_individual(
                    config['url_secundario'], data_prueba, "test_webhooks", "N8N"
                )
                exitos.append(exito_secundario)
                if exito_secundario:
                    st.success(f"✅ N8N: {mensaje_secundario}")
                else:
                    st.error(f"❌ N8N: {mensaje_secundario}")
            else:
                st.info("🟢 N8N: No seleccionado")
                
            # Enviar a webhook terciario (N8N-Test) si está habilitado
            if config.get('enviar_n8n_test', True):
                exito_terciario, mensaje_terciario = enviar_a_webhook_individual(
                    config['url_terciario'], data_prueba, "test_webhooks", "N8N-Test"
                )
                exitos.append(exito_terciario)
                if exito_terciario:
                    st.success(f"✅ N8N-Test: {mensaje_terciario}")
                else:
                    st.error(f"❌ N8N-Test: {mensaje_terciario}")
            else:
                st.info("🟡 N8N-Test: No seleccionado")
            
            # Mostrar información del resumen enviado
            st.info(f"📋 **Clip enviado:** {primer_video}")
            st.info(f"🏷️ **Términos:** {', '.join(terminos_encontrados)}")
            with st.expander("📄 Ver resumen enviado"):
                st.text(primer_resumen[:1000] + "..." if len(primer_resumen) > 1000 else primer_resumen)
            
            # Resultado general
            if not exitos:
                st.warning("⚠️ No hay webhooks seleccionados para probar")
            elif all(exitos):
                st.success("🎉 Todos los webhooks seleccionados recibieron el resumen ejecutivo")
            elif any(exitos):
                st.warning("⚠️ Solo algunos webhooks seleccionados funcionaron")
            else:
                st.error("❌ Ningún webhook seleccionado funcionó")

# === CONFIGURACIÓN DE TÉRMINOS ===
st.markdown("## 🔍 Configuración de Búsqueda")

# Función helper para extraer nombres de términos (soporta dict y string)
def extraer_nombres_terminos(terminos_lista):
    nombres = []
    for t in terminos_lista:
        if isinstance(t, dict):
            nombres.append(t.get('termino', ''))
        else:
            nombres.append(str(t))
    return nombres

terminos_input = st.text_input(
    "🏷️ Palabras clave (separadas por coma):",
    value=", ".join(extraer_nombres_terminos(st.session_state.terminos_continuos)),
    help="Ejemplo: EDESUR, apagones, corte, energia, luz",
    key="terminos_input_field"
)

# Botones de acción organizados
col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1])

with col1:
    if st.button("💾 Guardar Términos", help="Guardar la lista de términos"):
        if terminos_input.strip():
            # Obtener términos del input
            terminos_strings = [t.strip().lower() for t in terminos_input.split(",") if t.strip()]
            
            # Preservar asociaciones de cliente existentes
            terminos_existentes = {
                (t.get('termino', t) if isinstance(t, dict) else t): t 
                for t in st.session_state.terminos_continuos
            }
            
            # Crear lista con formato diccionario, preservando cliente_id si existe
            nuevos_terminos = []
            for termino_str in terminos_strings:
                if termino_str in terminos_existentes:
                    # Preservar el formato existente
                    existente = terminos_existentes[termino_str]
                    if isinstance(existente, dict):
                        nuevos_terminos.append(existente)
                    else:
                        nuevos_terminos.append({'termino': termino_str, 'cliente_id': 'default'})
                else:
                    # Nuevo término - asignar a default
                    nuevos_terminos.append({'termino': termino_str, 'cliente_id': 'default'})
            
            st.session_state.terminos_continuos = nuevos_terminos
            if guardar_configuracion_completa(
                st.session_state.terminos_continuos,
                st.session_state.intervalo,
                st.session_state.duracion_clip,
                st.session_state.buffer_anterior,
                st.session_state.mostrar_coincidencias
            ):
                st.success(f"✅ {len(nuevos_terminos)} términos guardados en `terminos_guardados.json`")
            else:
                st.error("❌ Error guardando términos")
        else:
            st.session_state.terminos_continuos = []
            guardar_configuracion_completa([], st.session_state.intervalo, st.session_state.get('duracion_clip', 180), st.session_state.get('buffer_anterior', 90), st.session_state.mostrar_coincidencias)
            st.warning("🗑️ Términos limpiados")

with col2:
    # Botón para verificar APIs
    if st.button("🔍 Verificar APIs", help="Verificar estado de todas las APIs antes del procesamiento"):
        verificar_todas_las_apis()
    
    # Botón para probar envío de video
    if st.button("🎬 Probar clip", help="Probar envío inteligente de clip a Telegram"):
        st.info("🎬 **PRUEBA DE ENVÍO DE VIDEO**")
        st.markdown("---")
        
        # Buscar un video de prueba
        videos_disponibles = []
        if os.path.exists("videos procesados"):
            for archivo in os.listdir("videos procesados"):
                if archivo.endswith('.mp4'):
                    videos_disponibles.append(archivo)
        
        if videos_disponibles:
            video_prueba = videos_disponibles[0]
            ruta_video = os.path.join("videos procesados", video_prueba)
            
            st.info(f"📹 Video de prueba: {video_prueba}")
            
            # Obtener tamaño del video
            file_size = os.path.getsize(ruta_video)
            file_size_mb = file_size / (1024 * 1024)
            
            st.info(f"📏 Tamaño: {file_size_mb:.1f}MB")
            
            # Determinar método
            if file_size_mb <= 50:
                st.success("✅ Método: Envío directo a Telegram")
            else:
                st.info("☁️ Método: Cloudinary + Telegram")
            
            # Probar envío
            caption_prueba = f"🧪 **PRUEBA DE ENVÍO INTELIGENTE**\n\n📹 Archivo: {video_prueba}\n📏 Tamaño: {file_size_mb:.1f}MB\n⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            with st.spinner("🎬 Probando envío de video..."):
                exito, mensaje, url = enviar_video_telegram_inteligente(
                    ruta_video,
                    caption_prueba
                )
                
                if exito:
                    st.success(f"✅ **Clip enviado exitosamente**: {mensaje}")
                    if url:
                        st.info(f"🔗 URL: {url}")
                else:
                    st.error(f"❌ **Error enviando clip**: {mensaje}")
        else:
            st.warning("⚠️ No hay videos disponibles para probar")
    
    if st.button("🚀 Procesar Una Vez", type="primary", help="Procesar audios nuevos una sola vez"):
        if st.session_state.terminos_continuos:
            st.session_state.procesar_una_vez = True
        else:
            st.error("❌ Configura términos primero")

with col3:
    # Estado compacto sin mostrar lista
    if st.session_state.terminos_continuos:
        st.success(f"✅ {len(st.session_state.terminos_continuos)} términos")
    else:
        st.info("📝 Agrega términos")

with col4:
    if st.session_state.terminos_continuos:
        if st.button("🗑️", key="btn_limpiar", help="Limpiar todos"):
            st.session_state.terminos_continuos = []
            guardar_configuracion_completa([], st.session_state.intervalo, st.session_state.get('duracion_clip', 180), st.session_state.get('buffer_anterior', 90), st.session_state.mostrar_coincidencias)
            st.rerun()

# Separador visual
st.markdown("---")

# === GESTIÓN DE CORREOS PARA COINCIDENCIAS ===
st.markdown("## 📧 Correos para coincidencias de audio")

# Cargar correos guardados
correos_guardados = cargar_correos_guardados()

# Mostrar correos actuales
if correos_guardados:
    st.success(f"✅ {len(correos_guardados)} correos configurados")
    
    # Mostrar lista de correos con opción de eliminar
    st.markdown("### 📬 Lista de Destinatarios")
    for i, correo_info in enumerate(correos_guardados):
        col1, col2 = st.columns([4, 1])
        with col1:
            nombre_display = correo_info.get('nombre', 'Sin nombre')
            correo_display = correo_info['correo']
            st.write(f"📧 **{nombre_display}** - {correo_display}")
        with col2:
            if st.button("🗑️", key=f"eliminar_correo_{i}", help=f"Eliminar {correo_display}"):
                eliminar_correo_de_lista(correo_display)
                st.success(f"✅ Correo {correo_display} eliminado")
                st.rerun()
else:
    st.info("📭 No hay correos configurados. Agrega correos para recibir notificaciones de coincidencias.")

# Formulario para agregar nuevo correo
st.markdown("### ➕ Agregar Nuevo Correo")

# Inicializar contador para widgets
if 'correos_widget_counter' not in st.session_state:
    st.session_state.correos_widget_counter = 0

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    nuevo_correo = st.text_input(
        "📧 Correo electrónico:",
        placeholder="ejemplo@correo.com",
        key=f"correos_nuevo_correo_input_{st.session_state.correos_widget_counter}"
    )

with col2:
    nuevo_nombre = st.text_input(
        "👤 Nombre (opcional):",
        placeholder="Nombre del destinatario",
        key=f"correos_nuevo_nombre_input_{st.session_state.correos_widget_counter}"
    )

with col3:
    st.write("")  # Espaciador
    if st.button("➕ Agregar", type="primary", help="Agregar correo a la lista"):
        if nuevo_correo.strip():
            # Validar formato de correo básico
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, nuevo_correo.strip()):
                if agregar_correo_a_lista(nuevo_correo.strip(), nuevo_nombre.strip()):
                    st.success(f"✅ Correo {nuevo_correo.strip()} agregado")
                    # Forzar recreación de widgets incrementando contador
                    if 'correos_widget_counter' not in st.session_state:
                        st.session_state.correos_widget_counter = 0
                    st.session_state.correos_widget_counter += 1
                    st.rerun()
                else:
                    st.warning("⚠️ El correo ya existe en la lista")
            else:
                st.error("❌ Formato de correo inválido")
        else:
            st.error("❌ Ingresa un correo válido")

# Botones de acción rápida
st.markdown("### ⚡ Acciones Rápidas")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏢 Agregar FGJ Medios", help="Agregar info@fgjmedios.com"):
        if agregar_correo_a_lista("info@fgjmedios.com", "FGJ Medios"):
            st.success("✅ FGJ Medios agregado")
            st.rerun()
        else:
            st.info("ℹ️ FGJ Medios ya está en la lista")

with col2:
    if st.button("🧪 Correo de Prueba", help="Agregar autosemana@gmail.com"):
        if agregar_correo_a_lista("autosemana@gmail.com", "Pruebas"):
            st.success("✅ Correo de prueba agregado")
            st.rerun()
        else:
            st.info("ℹ️ Correo de prueba ya está en la lista")

with col3:
    if correos_guardados and st.button("🗑️ Limpiar Todos", help="Eliminar todos los correos"):
        # Confirmar acción
        if 'confirmar_limpiar_correos' not in st.session_state:
            st.session_state.confirmar_limpiar_correos = False
        
        if not st.session_state.confirmar_limpiar_correos:
            st.session_state.confirmar_limpiar_correos = True
            st.warning("⚠️ Presiona nuevamente para confirmar")
        else:
            # Limpiar todos los correos
            guardar_correos_lista([])
            st.success("🗑️ Todos los correos eliminados")
            st.session_state.confirmar_limpiar_correos = False
            st.rerun()

# Información sobre el funcionamiento
with st.expander("ℹ️ Información sobre Notificaciones por Correo"):
    st.markdown("""
    ### 📧 **Cómo Funcionan las Notificaciones por Correo**
    
    **Cuándo se envían:**
    - ✅ Automáticamente cuando se detecta una coincidencia de video
    - ✅ Solo si Brevo está configurado y habilitado
    - ✅ A todos los correos en esta lista
    
    **Contenido del correo:**
    - 📧 **Asunto:** Nombre de la coincidencia detectada
    - 📝 **Cuerpo:** Resumen completo de la coincidencia con información del medio y términos detectados
    - 🎬 **Clip de audio:** Incrustado en el correo con reproductor
    - 🔗 **Enlaces:** Para ver y descargar el video
    
    **Características del Player:**
    - ✅ Player completamente incrustado en el correo
    - ✅ Controles personalizados (play/pause, volumen, progreso)
    - ✅ Funciona directamente en el correo sin controles externos
    - ✅ Diseño profesional y moderno
    
    **Configuración requerida:**
    - 🔧 Brevo debe estar configurado en la sección de configuración
    - 📧 Al menos un correo debe estar en esta lista
    - ✅ El sistema enviará a todos los correos configurados
    
    **Persistencia de datos:**
    - 💾 Los correos se guardan en: `correos_guardados.json` (raíz de la app)
    - 💾 Los términos se guardan en: `terminos_guardados.json` (raíz de la app)
    - 🔒 Los datos persisten entre sesiones y no se pierden al borrar videos
    - 📍 Ubicación segura fuera de la carpeta de videos procesados
    """)

# Separador visual
st.markdown("---")

# ============================================================================
# === UI: RESUMEN DIARIO PROGRAMADO (DIGEST POR ENTIDAD) ====================
# ============================================================================
st.markdown("## 📊 Resumen Diario Programado")
st.caption("Envía un correo resumen con todas las coincidencias del día a cada entidad. Solo se envía si hay coincidencias.")

col_digest1, col_digest2 = st.columns([2, 1])

with col_digest1:
    st.markdown("""
    **Horarios automáticos:**
    - ☀️ **10:30 AM** — Coincidencias de la mañana (00:00 - 10:30)
    - 🌅 **5:30 PM** — Coincidencias de la tarde (10:31 - 17:30)
    - 🌙 **11:59 PM** — Coincidencias de la noche (17:31 - 23:59)
    """)

with col_digest2:
    # Estado del scheduler (se inicia automáticamente con la app)
    if _scheduler_activo:
        st.success("🟢 Scheduler ACTIVO (automático)")
        if st.button("🔴 Detener Scheduler", use_container_width=True):
            detener_scheduler_resumen()
            st.warning("🛑 Scheduler detenido. Se reactivará al reiniciar la app.")
            st.rerun()
    else:
        st.warning("⚪ Scheduler detenido")
        if st.button("🟢 Reactivar Scheduler", type="primary", use_container_width=True):
            iniciar_scheduler_resumen()
            st.success("✅ Scheduler reactivado")
            st.rerun()

# Envío manual de prueba
st.markdown("#### 🧪 Envío Manual de Prueba")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    if st.button("☀️ Enviar Mañana", use_container_width=True, help="Envía resumen del corte mañana (00:00-10:30)"):
        with st.spinner("Generando y enviando resumen mañana..."):
            resultados = enviar_resumen_diario_clientes("mañana")
            for cid, (exito, msg) in resultados.items():
                if exito:
                    st.success(f"✅ {cid}: {msg}")
                else:
                    st.warning(f"⚠️ {cid}: {msg}")

with col_m2:
    if st.button("🌅 Enviar Tarde", use_container_width=True, help="Envía resumen del corte tarde (10:31-17:30)"):
        with st.spinner("Generando y enviando resumen tarde..."):
            resultados = enviar_resumen_diario_clientes("tarde")
            for cid, (exito, msg) in resultados.items():
                if exito:
                    st.success(f"✅ {cid}: {msg}")
                else:
                    st.warning(f"⚠️ {cid}: {msg}")

with col_m3:
    if st.button("🌙 Enviar Noche", use_container_width=True, help="Envía resumen del corte noche (17:31-23:59)"):
        with st.spinner("Generando y enviando resumen noche..."):
            resultados = enviar_resumen_diario_clientes("noche")
            for cid, (exito, msg) in resultados.items():
                if exito:
                    st.success(f"✅ {cid}: {msg}")
                else:
                    st.warning(f"⚠️ {cid}: {msg}")

# Enviar TODO el día de una vez
if st.button("📨 Enviar Resumen Completo del Día", use_container_width=True, help="Envía las 3 franjas del día actual"):
    with st.spinner("Enviando resúmenes de todo el día..."):
        for corte in ["mañana", "tarde", "noche"]:
            st.markdown(f"**Corte: {corte}**")
            resultados = enviar_resumen_diario_clientes(corte)
            for cid, (exito, msg) in resultados.items():
                if exito:
                    st.success(f"✅ {cid}: {msg}")
                elif "_sin_datos" in cid:
                    st.info(f"ℹ️ {msg}")
                else:
                    st.warning(f"⚠️ {cid}: {msg}")

# Preview: mostrar cuántas coincidencias hay hoy
with st.expander("👁️ Preview: Coincidencias de hoy"):
    fecha_hoy = datetime.now().strftime('%d/%m/%Y')
    coincs_hoy = parsear_coincidencias_md(fecha_filtro=fecha_hoy)
    if coincs_hoy:
        st.success(f"📊 **{len(coincs_hoy)} coincidencias** encontradas hoy ({fecha_hoy})")
        grupos_hoy = agrupar_coincidencias_por_cliente(coincs_hoy)
        for cid, grupo in grupos_hoy.items():
            nombre = nombre_cliente_mostrar_para_ui(grupo['cliente'], cid)
            n = len(grupo['coincidencias'])
            terminos = list(set(c['termino'] for c in grupo['coincidencias']))
            st.markdown(f"- **{nombre}**: {n} coincidencias — Términos: {', '.join(terminos)}")
    else:
        st.info(f"Sin coincidencias registradas hoy ({fecha_hoy})")

st.markdown("---")

# === FUNCIONES AUXILIARES ===
def cargar_procesados():
    """
    Carga la lista de archivos ya procesados desde procesados.log Y procesados.txt
    Lee ambos archivos para máxima compatibilidad y respaldo
    """
    procesados = set()
    
    # ========== LEER procesados.log (formato detallado con timestamps) ==========
    if os.path.exists(PROCESADOS_LOG):
        try:
            with open(PROCESADOS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Ignorar líneas vacías, comentarios, timestamps y separadores
                    if line and not line.startswith('#') and not line.startswith('[') and not line.startswith('='):
                        # Extraer solo el nombre del archivo (puede estar en diferentes formatos)
                        if '|' in line and 'VIDEO_PROCESADO:' in line:
                            # Formato: [timestamp] 🎬 VIDEO_PROCESADO: archivo.mp4 | Términos: ...
                            partes = line.split('VIDEO_PROCESADO:')
                            if len(partes) > 1:
                                nombre = partes[1].split('|')[0].strip()
                                procesados.add(nombre)
                        elif not line.startswith('📹') and not 'SUBCLIP' in line:
                            # Formato simple: archivo.mp4 (línea de compatibilidad)
                            procesados.add(line)
            log_debug(f"✅ {len(procesados)} archivos cargados desde procesados.log", "cargar_procesados")
        except Exception as e:
            st.warning(f"⚠️ Error leyendo procesados.log: {e}")
            log_warning(f"Error leyendo procesados.log: {e}", "cargar_procesados")
    
    # ========== LEER procesados.txt (formato simple - respaldo adicional) ==========
    procesados_txt = os.path.join(CARPETA_PROCESADOS, "procesados.txt")
    if os.path.exists(procesados_txt):
        try:
            cantidad_inicial = len(procesados)
            with open(procesados_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Agregar solo líneas válidas (nombres de archivo)
                    if line and not line.startswith('#') and not line.startswith('['):
                        procesados.add(line)
            cantidad_nueva = len(procesados) - cantidad_inicial
            if cantidad_nueva > 0:
                log_debug(f"✅ {cantidad_nueva} archivos adicionales desde procesados.txt", "cargar_procesados")
        except Exception as e:
            log_warning(f"Error leyendo procesados.txt: {e}", "cargar_procesados")
    else:
        # Crear procesados.txt si no existe (para compatibilidad futura)
        try:
            with open(procesados_txt, "w", encoding="utf-8") as f:
                f.write("# Archivo de videos procesados (formato simple)\n")
                f.write("# Un video por línea\n")
                f.write(f"# Creado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_info("✅ Archivo procesados.txt creado", "cargar_procesados")
        except Exception as e:
            log_warning(f"No se pudo crear procesados.txt: {e}", "cargar_procesados")
    
    log_info(f"📊 Total de archivos procesados cargados: {len(procesados)}", "cargar_procesados")
    return procesados

def cargar_cache_escaneo():
    """Carga el caché de archivos escaneados para optimizar búsquedas"""
    cache_default = {
        'archivos_escaneados': {},  # path: {mtime, size, procesado}
        'ultima_actualizacion': 0,
        'version': '1.0'
    }
    
    try:
        if os.path.exists(CACHE_ESCANEO):
            with open(CACHE_ESCANEO, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                # Validar estructura del caché
                if isinstance(cache, dict) and 'archivos_escaneados' in cache:
                    return cache
    except Exception as e:
        log_warning(f"Error cargando caché de escaneo: {e}", "cargar_cache_escaneo")
    
    return cache_default

def guardar_cache_escaneo(cache):
    """Guarda el caché de archivos escaneados"""
    try:
        cache['ultima_actualizacion'] = time.time()
        with open(CACHE_ESCANEO, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log_warning(f"Error guardando caché de escaneo: {e}", "guardar_cache_escaneo")
        return False

def limpiar_cache_escaneo():
    """Limpia archivos que ya no existen del caché"""
    try:
        cache = cargar_cache_escaneo()
        archivos_limpiados = 0
        
        # Crear una copia de las claves para iterar
        paths_to_check = list(cache['archivos_escaneados'].keys())
        
        for path in paths_to_check:
            if not os.path.exists(path):
                del cache['archivos_escaneados'][path]
                archivos_limpiados += 1
        
        if archivos_limpiados > 0:
            guardar_cache_escaneo(cache)
            log_info(f"Caché limpiado: {archivos_limpiados} archivos eliminados", "limpiar_cache_escaneo")
        
        return archivos_limpiados
    except Exception as e:
        log_warning(f"Error limpiando caché: {e}", "limpiar_cache_escaneo")
        return 0

def cargar_archivos_fallidos():
    """Carga la lista de archivos fallidos desde fallidos.txt"""
    try:
        if os.path.exists("fallidos.txt"):
            with open("fallidos.txt", "r", encoding="utf-8") as f:
                archivos_fallidos = [line.strip().split('|')[0] for line in f.readlines() if line.strip()]
            log_info(f"📋 Cargados {len(archivos_fallidos)} archivos fallidos", "cargar_archivos_fallidos")
            return archivos_fallidos
        return []
    except Exception as e:
        log_warning(f"Error cargando archivos fallidos: {e}", "cargar_archivos_fallidos")
        return []

def guardar_archivo_fallido(nombre_archivo, error_mensaje="", archivo_path=None):
    """
    Guarda un archivo fallido:
    1. Muestra mensaje en UI
    2. Envía notificación a plataformas
    3. Mueve archivo a carpeta archivos_fallidos/
    4. Crea archivo .txt con el error
    """
    func_name = "guardar_archivo_fallido"
    
    try:
        archivos_fallidos = cargar_archivos_fallidos()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Evitar duplicados
        if nombre_archivo in archivos_fallidos:
            log_info(f"ℹ️ Archivo ya está en fallidos: {nombre_archivo}", func_name)
            return False
        
        # 1. MOSTRAR MENSAJE EN UI
        st.error(f"❌ **ERROR PROCESANDO ARCHIVO:** `{nombre_archivo}`")
        st.warning(f"⚠️ **Error:** {error_mensaje}")
        st.info("📁 Moviendo archivo a carpeta de fallidos...")
        
        # 2. CREAR CARPETA archivos_fallidos/ si no existe
        carpeta_fallidos = "archivos_fallidos"
        if not os.path.exists(carpeta_fallidos):
            os.makedirs(carpeta_fallidos)
            log_info(f"📁 Carpeta creada: {carpeta_fallidos}", func_name)
        
        # 3. MOVER ARCHIVO a carpeta de fallidos
        archivo_movido = False
        ruta_destino = None
        
        if archivo_path and os.path.exists(archivo_path):
            try:
                nombre_base = os.path.basename(archivo_path)
                ruta_destino = os.path.join(carpeta_fallidos, nombre_base)
                
                # Si ya existe en destino, agregar timestamp
                if os.path.exists(ruta_destino):
                    nombre_sin_ext, ext = os.path.splitext(nombre_base)
                    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nombre_base = f"{nombre_sin_ext}_{timestamp_file}{ext}"
                    ruta_destino = os.path.join(carpeta_fallidos, nombre_base)
                
                shutil.move(archivo_path, ruta_destino)
                archivo_movido = True
                log_info(f"📁 Archivo movido a: {ruta_destino}", func_name)
                st.success(f"✅ Archivo movido a: `{carpeta_fallidos}/{nombre_base}`")
            except Exception as e_move:
                log_warning(f"⚠️ No se pudo mover archivo: {e_move}", func_name)
                st.warning(f"⚠️ No se pudo mover archivo: {e_move}")
        
        # 4. CREAR ARCHIVO .txt CON EL ERROR dentro de la carpeta
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        timestamp_txt = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_txt = f"{nombre_sin_ext}_ERROR_{timestamp_txt}.txt"
        ruta_txt = os.path.join(carpeta_fallidos, nombre_txt)
        
        contenido_error = f"""❌ ERROR AL PROCESAR ARCHIVO
{'='*60}

📄 ARCHIVO: {nombre_archivo}
⏰ FECHA Y HORA: {timestamp}
❌ ERROR: {error_mensaje}

{'='*60}
UBICACIÓN ORIGINAL: {archivo_path if archivo_path else 'Desconocida'}
UBICACIÓN ACTUAL: {ruta_destino if archivo_movido else 'No movido'}
ARCHIVO MOVIDO: {'✅ Sí' if archivo_movido else '❌ No'}

{'='*60}
ACCIONES RECOMENDADAS:
- Verificar que el archivo no esté corrupto
- Verificar formato del archivo
- Revisar logs para más detalles
- Si el problema persiste, contactar soporte

{'='*60}
Generado automáticamente por Radio Analyzer IA
"""
        
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(contenido_error)
        
        log_info(f"📝 Archivo de error creado: {ruta_txt}", func_name)
        st.success(f"📝 Reporte de error creado: `{carpeta_fallidos}/{nombre_txt}`")
        
        # 5. REGISTRAR EN fallidos.txt
        with open("fallidos.txt", "a", encoding="utf-8") as f:
            f.write(f"{nombre_archivo}|{timestamp}|{error_mensaje}\n")
        
        log_warning(f"❌ Archivo agregado a fallidos: {nombre_archivo} - Error: {error_mensaje}", func_name)
        
        # 6. ENVIAR NOTIFICACIONES A PLATAFORMAS
        st.info("📤 Enviando notificaciones...")
        enviar_notificacion_archivo_fallido(nombre_archivo, error_mensaje, ruta_destino if archivo_movido else None)
        
        return True
        
    except Exception as e:
        log_error_critico(func_name, f"Error guardando archivo fallido: {e}")
        st.error(f"❌ Error crítico guardando archivo fallido: {e}")
        return False

def enviar_notificacion_archivo_fallido(nombre_archivo, error_mensaje, ruta_archivo=None):
    """
    Guarda errores de archivos fallidos SOLO localmente (UI + logs).
    NO envía notificaciones a plataformas externas (Telegram, Webhook, Email).
    Los errores se pueden consultar en la UI.
    """
    func_name = "enviar_notificacion_archivo_fallido"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # ========== GUARDAR EN SESSION_STATE PARA CONSULTA EN UI ==========
        if 'errores_archivos' not in st.session_state:
            st.session_state.errores_archivos = []
        
        error_entry = {
            'archivo': nombre_archivo,
            'error': error_mensaje,
            'timestamp': timestamp,
            'ubicacion': ruta_archivo if ruta_archivo else 'No movido'
        }
        
        # Evitar duplicados
        ya_existe = any(e['archivo'] == nombre_archivo and e['error'] == error_mensaje 
                       for e in st.session_state.errores_archivos)
        if not ya_existe:
            st.session_state.errores_archivos.append(error_entry)
        
        # ========== GUARDAR EN ARCHIVO LOG LOCAL ==========
        log_errores_path = os.path.join("videos procesados", "log_errores_archivos.json")
        try:
            if os.path.exists(log_errores_path):
                with open(log_errores_path, "r", encoding="utf-8") as f:
                    errores_guardados = json.load(f)
            else:
                errores_guardados = []
            
            errores_guardados.append(error_entry)
            
            # Mantener solo los últimos 100 errores
            if len(errores_guardados) > 100:
                errores_guardados = errores_guardados[-100:]
            
            with open(log_errores_path, "w", encoding="utf-8") as f:
                json.dump(errores_guardados, f, indent=2, ensure_ascii=False)
            
            log_info(f"📝 Error guardado en log local: {nombre_archivo}", func_name)
            
        except Exception as e:
            log_warning(f"Error guardando log de errores: {e}", func_name)
        
        # ========== MOSTRAR EN UI (ya se muestra en guardar_archivo_fallido) ==========
        log_warning(f"❌ Archivo fallido registrado localmente: {nombre_archivo} - {error_mensaje}", func_name)
        
        # NO SE ENVÍA A TELEGRAM, WEBHOOK NI EMAIL
        # Los errores de archivos dañados solo se guardan localmente para consulta en la UI
        
    except Exception as e:
        log_error_critico(func_name, f"Error guardando error localmente: {e}")

def es_archivo_fallido(nombre_archivo):
    """Verifica si un archivo está en la lista de fallidos"""
    try:
        archivos_fallidos = cargar_archivos_fallidos()
        return nombre_archivo in archivos_fallidos
    except Exception as e:
        log_warning(f"Error verificando archivo fallido: {e}", "es_archivo_fallido")
        return False

def limpiar_archivos_fallidos():
    """Limpia la lista de archivos fallidos"""
    try:
        if os.path.exists("fallidos.txt"):
            os.remove("fallidos.txt")
            log_info("🧹 Lista de archivos fallidos limpiada", "limpiar_archivos_fallidos")
            return True
        return False
    except Exception as e:
        log_warning(f"Error limpiando archivos fallidos: {e}", "limpiar_archivos_fallidos")
        return False

def mostrar_archivos_fallidos():
    """Muestra la lista de archivos fallidos en la interfaz"""
    try:
        archivos_fallidos = cargar_archivos_fallidos()
        if archivos_fallidos:
            st.warning(f"⚠️ **{len(archivos_fallidos)} archivos fallidos** (serán omitidos en el procesamiento)")
            with st.expander("📋 Ver archivos fallidos", expanded=False):
                for i, archivo in enumerate(archivos_fallidos, 1):
                    st.text(f"{i}. {archivo}")
            
            if st.button("🧹 Limpiar lista de fallidos"):
                if limpiar_archivos_fallidos():
                    st.success("✅ Lista de archivos fallidos limpiada")
                    st.rerun()
        else:
            st.success("✅ No hay archivos fallidos")
    except Exception as e:
        st.error(f"❌ Error mostrando archivos fallidos: {e}")

def obtener_duracion(video_path):
    try:
        res = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", video_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(json.loads(res.stdout)["format"]["duration"])
    except Exception as e:
        st.warning(f"⚠️ No se pudo obtener duración: {os.path.basename(video_path)}")
        return 1.0

def generar_intro_audio_elevenlabs(texto_intro, nombre_base, func_name="elevenlabs_intro"):
    """
    Genera audio MP3 de intro con ElevenLabs. Retorna ruta o None.
    """
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        log_warning("ElevenLabs no configurado (API key o voice id faltante)", func_name)
        return None

    try:
        safe_base = re.sub(r"[^\w\-\.]", "_", nombre_base)[:80]
        intro_path = os.path.join(CARPETA_PROCESADOS, f"{safe_base}_intro_tts.mp3")
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": texto_intro,
            "model_id": ELEVENLABS_MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            log_warning(f"ElevenLabs HTTP {resp.status_code}: {resp.text[:200]}", func_name)
            return None

        with open(intro_path, "wb") as f:
            f.write(resp.content)

        if os.path.exists(intro_path) and os.path.getsize(intro_path) > 0:
            log_info(f"Intro TTS generada: {intro_path}", func_name)
            return intro_path

        log_warning("ElevenLabs devolvió audio vacío", func_name)
        return None
    except Exception as e:
        log_warning(f"Error generando intro ElevenLabs: {e}", func_name)
        return None

def concatenar_intro_y_clip_audio(intro_path, clip_path, func_name="concat_intro_audio"):
    """
    Concatena intro + clip y retorna la ruta final o None.
    """
    if not intro_path or not clip_path:
        return None
    if not (os.path.exists(intro_path) and os.path.exists(clip_path)):
        return None

    try:
        base, ext = os.path.splitext(clip_path)
        salida_path = f"{base}_con_intro{ext if ext else '.mp3'}"
        cmd = [
            "ffmpeg", "-y",
            "-i", intro_path,
            "-i", clip_path,
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[aout]",
            "-map", "[aout]",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libmp3lame",
            salida_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log_warning(f"Error ffmpeg concatenando intro: {result.stderr[:220]}", func_name)
            return None
        if os.path.exists(salida_path) and os.path.getsize(salida_path) > 0:
            log_info(f"Audio final con intro generado: {salida_path}", func_name)
            return salida_path
        return None
    except Exception as e:
        log_warning(f"Error concatenando intro y clip: {e}", func_name)
        return None

def preparar_clip_con_intro_elevenlabs(clip_path, nombre_archivo, termino_encontrado, func_name="preparar_intro_clip"):
    """
    Genera intro TTS y la concatena al inicio del clip. Devuelve (ruta_final, ok_intro).
    """
    if not clip_path or not os.path.exists(clip_path):
        return clip_path, False
    try:
        texto_intro = construir_texto_intro_coincidencia(nombre_archivo, termino_encontrado)
        base_intro = os.path.splitext(os.path.basename(clip_path))[0]
        intro_tts_path = generar_intro_audio_elevenlabs(texto_intro, base_intro, func_name)
        if not intro_tts_path:
            log_warning("No se pudo generar intro ElevenLabs; se usa clip original", func_name)
            return clip_path, False
        clip_con_intro = concatenar_intro_y_clip_audio(intro_tts_path, clip_path, func_name)
        if clip_con_intro:
            log_info(f"Intro agregada correctamente al clip: {clip_con_intro}", func_name)
            return clip_con_intro, True
        log_warning("No se pudo concatenar intro; se usa clip original", func_name)
        return clip_path, False
    except Exception as e:
        log_warning(f"Error en pipeline de intro; usando clip original: {e}", func_name)
        return clip_path, False

def registrar_audio_check(
    origen_audio_path,
    nombre_archivo,
    termino_encontrado,
    timestamp_segundos,
    tipo_mencion="real",
    variante_detectada=None,
    func_name="audio_checks"
):
    """
    Copia el audio origen a CARPETA_AUDIOCHECKS_EVIDENCIAS (dentro de AUDIOCHECKS) y registra TXT.
    Registra tanto menciones reales como tangenciales.
    """
    if not origen_audio_path or not os.path.exists(origen_audio_path):
        return None, None

    try:
        checks_dir = CARPETA_AUDIOCHECKS_EVIDENCIAS
        os.makedirs(checks_dir, exist_ok=True)

        nombre_base = os.path.basename(origen_audio_path)
        ruta_copia_audio = os.path.join(checks_dir, nombre_base)
        if not os.path.exists(ruta_copia_audio):
            shutil.copy2(origen_audio_path, ruta_copia_audio)
            log_info(f"Audio origen copiado para revisión: {ruta_copia_audio}", func_name)

        ruta_txt = os.path.join(checks_dir, f"{os.path.splitext(nombre_base)[0]}.txt")
        minutos = int(timestamp_segundos // 60)
        segundos = int(timestamp_segundos % 60)
        marca_tiempo = f"{minutos:02d}:{segundos:02d}"
        tipo_mencion = (tipo_mencion or "real").strip().lower()
        if tipo_mencion not in ("real", "tangencial"):
            tipo_mencion = "real"
        variante = (variante_detectada or termino_encontrado or "").strip()

        linea = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Archivo: {nombre_archivo} | Tipo: {tipo_mencion} | "
            f"Tiempo: {timestamp_segundos:.1f}s ({marca_tiempo}) | "
            f"Palabra: {termino_encontrado} | Variante: {variante}\n"
        )
        with open(ruta_txt, "a", encoding="utf-8") as f_txt:
            f_txt.write(linea)

        return ruta_copia_audio, ruta_txt
    except Exception as e:
        log_warning(f"No se pudo registrar audio check: {e}", func_name)
        return None, None

# === FUNCIONES CACHE Y PRINCIPALES ===
_whisper_device_actual = "cpu"  # "cuda" o "cpu" - se actualiza al cargar el modelo

@st.cache_resource
def cargar_modelo_whisper_timestamps():
    """Intenta CUDA primero, fallback a CPU"""
    global _whisper_device_actual
    try:
        m = WhisperModel("medium", device="cuda", compute_type="float16")
        _whisper_device_actual = "cuda"
        return m
    except Exception as e:
        log_info(f"CUDA no disponible ({e}), usando CPU para faster-whisper", "cargar_modelo_whisper")
        m = WhisperModel("medium", device="cpu", compute_type="int8")
        _whisper_device_actual = "cpu"
        return m

@st.cache_resource
def cargar_modelo_whisper_cpu():
    """Modelo faster-whisper forzado a CPU (para fallback cuando CUDA falla en runtime)"""
    return WhisperModel("medium", device="cpu", compute_type="int8")

# === FUNCIONES PRINCIPALES DE PROCESAMIENTO ===

def _transcribir_faster_whisper_con_segments(audio_path, model, func_name="transcribir_faster"):
    """Transcribe y retorna (texto, segments) en un solo pass"""
    with st.spinner("🧠 Transcribiendo con faster-whisper..."):
        segments_iter, info = model.transcribe(
            audio_path, language="es", beam_size=5, vad_filter=True, word_timestamps=False
        )
        textos = []
        segments_list = []
        for seg in segments_iter:
            textos.append(seg.text.strip())
            segments_list.append({'start': seg.start, 'end': seg.end, 'text': seg.text.strip()})
        result = " ".join(textos)
    # Validación de calidad
    if not result or len(result.strip()) < 10:
        raise Exception(f"Transcripción vacía o muy corta - posible audio sin habla")
    chars_unicos = set(result.strip())
    if len(chars_unicos) <= 3:
        raise Exception(f"Transcripción basura detectada - alucinación de whisper")
    return result, segments_list

def transcribir_con_faster_whisper(audio_path, return_segments=False):
    """
    Transcribe audio con faster-whisper (local, GRATIS).
    Retorna (texto, segments) - un solo pass para evitar procesar 2 veces.
    
    Returns:
        Si return_segments=False: texto (str) - compatibilidad
        Si return_segments=True: (texto, segments)
    """
    func_name = "transcribir_con_faster_whisper"
    try:
        log_info(f"Iniciando transcripción con faster-whisper: {audio_path}", func_name)
        model = cargar_modelo_whisper_timestamps()
        result, segments_list = _transcribir_faster_whisper_con_segments(audio_path, model, func_name)
        log_info(f"Transcripción faster-whisper completada. Longitud: {len(result)} caracteres", func_name)
        return (result, segments_list) if return_segments else result
    except Exception as e:
        log_exception(func_name, e, f"Archivo: {audio_path}")
        raise

def transcribir_con_faster_whisper_cpu(audio_path, return_segments=False):
    """faster-whisper forzado a CPU (fallback cuando CUDA falla en runtime)"""
    func_name = "transcribir_con_faster_whisper_cpu"
    try:
        log_info(f"Iniciando transcripción con faster-whisper CPU: {audio_path}", func_name)
        model = cargar_modelo_whisper_cpu()
        result, segments_list = _transcribir_faster_whisper_con_segments(audio_path, model, func_name)
        log_info(f"Transcripción faster-whisper CPU completada. Longitud: {len(result)} caracteres", func_name)
        return (result, segments_list) if return_segments else result
    except Exception as e:
        log_exception(func_name, e, f"Archivo: {audio_path}")
        raise

def transcribir_audio_mistral(audio_path, con_timestamps=False):
    """
    Transcribe audio usando Voxtral Mini Transcribe (endpoint dedicado de transcripción).
    Soporta archivos de hasta ~30 minutos.
    
    Args:
        audio_path: Ruta al archivo de audio
        con_timestamps: Si True, devuelve (texto, segments) con timestamps por segmento
    
    Returns:
        Si con_timestamps=False: texto transcrito (str)
        Si con_timestamps=True: (texto, segments) donde segments es lista de dicts con start, end, text
    """
    func_name = "transcribir_audio_mistral"
    try:
        log_info(f"Iniciando transcripción con Voxtral: {audio_path}", func_name)
        
        client, _ = cargar_cliente_mistral()
        model = "voxtral-mini-latest"
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        log_debug(f"Tamaño archivo audio: {file_size_mb:.2f}MB", func_name)
        
        file_name = os.path.basename(audio_path)
        
        log_debug("Enviando audio a Voxtral Transcription API", func_name)
        with open(audio_path, "rb") as f:
            transcription_response = client.audio.transcriptions.complete(
                model=model,
                file={
                    "content": f,
                    "file_name": file_name,
                },
                language="es",
                timestamp_granularities=["segment"] if con_timestamps else None,
            )
        
        result = transcription_response.text
        log_info(f"Transcripción Voxtral completada exitosamente. Longitud: {len(result)} caracteres", func_name)
        
        # === Acumular estadísticas de uso de Mistral/Voxtral ===
        try:
            if hasattr(transcription_response, 'usage') and transcription_response.usage:
                usage = transcription_response.usage
                audio_secs = getattr(usage, 'prompt_audio_seconds', 0) or 0
                prompt_tok = getattr(usage, 'prompt_tokens', 0) or 0
                compl_tok = getattr(usage, 'completion_tokens', 0) or 0
                total_tok = getattr(usage, 'total_tokens', 0) or 0
                
                st.session_state.mistral_total_audio_seconds += audio_secs
                st.session_state.mistral_total_prompt_tokens += prompt_tok
                st.session_state.mistral_total_completion_tokens += compl_tok
                st.session_state.mistral_total_tokens += total_tok
                st.session_state.mistral_total_transcripciones += 1
                
                log_info(f"Uso Voxtral: {audio_secs}s audio, {total_tok} tokens. Acumulado sesión: {st.session_state.mistral_total_audio_seconds}s, {st.session_state.mistral_total_tokens} tokens", func_name)
        except Exception as usage_err:
            log_warning(f"Error capturando estadísticas de uso Voxtral: {usage_err}", func_name)
        
        if con_timestamps:
            segments = []
            if hasattr(transcription_response, 'segments') and transcription_response.segments:
                for seg in transcription_response.segments:
                    segments.append({
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text.strip() if hasattr(seg, 'text') else ''
                    })
            log_info(f"Timestamps extraídos: {len(segments)} segmentos", func_name)
            return result, segments
        
        return result
        
    except Exception as e:
        log_exception(func_name, e, f"Archivo: {audio_path}")
        raise

def transcribir_con_openai(audio_path):
    """
    Transcribe audio usando OpenAI Whisper API
    Para archivos > 19MB que no puede procesar Mistral
    """
    func_name = "transcribir_con_openai"
    try:
        log_info(f"Iniciando transcripción con OpenAI Whisper: {audio_path}", func_name)
        
        # Configurar API key desde variable de entorno
        openai.api_key = os.getenv('OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY_BACKUP', '')
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        log_debug(f"Tamaño archivo audio: {file_size_mb:.2f}MB", func_name)
        
        with open(audio_path, "rb") as audio_file:
            log_debug("Enviando audio a OpenAI Whisper API", func_name)
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="es"  # Español
            )
        
        result = transcript.text
        log_info(f"Transcripción OpenAI completada exitosamente. Longitud: {len(result)} caracteres", func_name)
        return result
        
    except Exception as e:
        log_exception(func_name, e, f"Archivo: {audio_path}")
        raise


def _ui_status(level, message):
    """
    Mensajes UI con modo compacto:
    - Si existe un contenedor de estado actual, reutilizarlo para no acumular líneas.
    - Si no, comportamiento normal de Streamlit.
    """
    try:
        status_container = st.session_state.get('ui_status_container')
        if status_container is not None and st.session_state.get('mostrar_solo_actual_relevantes', True):
            with status_container.container():
                getattr(st, level)(message)
        else:
            getattr(st, level)(message)
    except Exception:
        getattr(st, level)(message)


def transcribir_audio_hibrido(audio_path, indice_actual=None, total=None):
    """
    Sistema de transcripción con faster-whisper (local, GRATIS) como motor principal.
    Cadena de fallback: faster-whisper → Voxtral (Mistral) → OpenAI Whisper
    
    - Paso 1: faster-whisper local (sin costo de API)
    - Paso 2 (fallback): Voxtral (Mistral Transcription API)
    - Paso 3 (fallback): OpenAI Whisper API
    - Archivos muy grandes (>100MB): Convierte a MP3 primero
    
    indice_actual, total: opcionales; si se pasan, se muestra progreso tipo "24/56" para mayor control.
    """
    func_name = "transcribir_audio_hibrido"
    try:
        tamaño_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        progreso = f"**{indice_actual}/{total}** " if (indice_actual is not None and total is not None) else ""
        _ui_status("info", f"🔄 {progreso}🎯 **ANÁLISIS DE ARCHIVO:** {os.path.basename(audio_path)} ({tamaño_mb:.1f}MB)")
        
        if tamaño_mb > 100:
            _ui_status("info", f"📊 **DECISIÓN:** Archivo {tamaño_mb:.1f}MB (>100MB) → **Convirtiendo a MP3** antes de transcribir")
            log_info(f"Archivo {tamaño_mb:.1f}MB - Convirtiendo a MP3 (archivo muy grande)", func_name)
            return transcribir_archivo_grande_mp3(audio_path, func_name)
        
        # === PASO 1: faster-whisper GPU (local, GRATIS) ===
        _ui_status("info", f"📊 **DECISIÓN:** Archivo {tamaño_mb:.1f}MB → Intentando **faster-whisper GPU** (local, sin costo)")
        log_info(f"Archivo {tamaño_mb:.1f}MB - Intentando faster-whisper GPU", func_name)
        
        try:
            result, segments = transcribir_con_faster_whisper(audio_path, return_segments=True)
            motor = f"faster-whisper {_whisper_device_actual.upper()}"
            _ui_status("success", f"✅ **{motor}** completó transcripción + timestamps — **SIN COSTO**")
            log_info(f"Transcripción exitosa con {motor}", func_name)
            return result, motor, segments
        except Exception as whisper_error:
            err_str = str(whisper_error)[:80]
            es_cuda = "cublas" in err_str.lower() or "cuda" in err_str.lower() or "dll" in err_str.lower()
            if es_cuda:
                _ui_status("warning", f"⚠️ **faster-whisper GPU falló** → Intentando **faster-whisper CPU**")
                log_info(f"faster-whisper GPU falló ({whisper_error}) - Intentando CPU", func_name)
                try:
                    result, segments = transcribir_con_faster_whisper_cpu(audio_path, return_segments=True)
                    _ui_status("success", f"✅ **faster-whisper CPU** completó transcripción + timestamps — **SIN COSTO**")
                    log_info(f"Transcripción exitosa con faster-whisper CPU", func_name)
                    return result, "faster-whisper CPU", segments
                except Exception as cpu_error:
                    _ui_status("warning", f"⚠️ **faster-whisper CPU falló** → Activando **Voxtral (Mistral)**")
                    log_info(f"faster-whisper CPU falló ({cpu_error}) - Activando Voxtral", func_name)
            else:
                _ui_status("warning", f"⚠️ **faster-whisper falló** ({err_str}...) → Activando **Voxtral (Mistral)**")
                log_info(f"faster-whisper falló ({whisper_error}) - Activando Voxtral", func_name)
        
        # === PASO 2: Voxtral/Mistral API (con timestamps en el mismo call) ===
        _ui_status("info", f"📊 **FALLBACK:** Usando **Voxtral** (Mistral) con timestamps")
        try:
            result, segments = transcribir_audio_mistral(audio_path, con_timestamps=True)
            _ui_status("success", f"✅ **Mistral (Voxtral)** completó transcripción + timestamps")
            log_info(f"Transcripción exitosa con Mistral", func_name)
            return result, "Mistral (Voxtral)", segments
            
        except Exception as mistral_error:
            _ui_status("warning", f"⚠️ **Voxtral falló** → Activando **FALLBACK a OpenAI Whisper**")
            log_info(f"Voxtral falló ({mistral_error}) - Activando FALLBACK a OpenAI", func_name)
            
            if "getaddrinfo failed" in str(mistral_error) or "Connection error" in str(mistral_error):
                _ui_status("warning", "🔍 **DIAGNÓSTICO:** Problema de conectividad detectado")
                diagnosticar_conectividad()
        
        # === PASO 3: OpenAI Whisper API (último fallback - no devuelve timestamps) ===
        _ui_status("info", f"📊 **FALLBACK 2:** Usando **OpenAI Whisper** (último recurso)")
        try:
            result = transcribir_con_openai(audio_path)
            _ui_status("success", f"✅ **OpenAI Whisper** completó la transcripción (sin timestamps - se obtendrán con whisper local)")
            log_info(f"Transcripción exitosa con OpenAI - timestamps se obtendrán aparte", func_name)
            return result, "OpenAI Whisper", None  # segments=None, main loop usará obtener_timestamps_whisper
        except Exception as openai_error:
            _ui_status("error", f"❌ **Las 3 opciones fallaron** — faster-whisper, Voxtral y OpenAI Whisper")
            
            if "getaddrinfo failed" in str(openai_error) or "Connection error" in str(openai_error):
                st.error("🔍 **DIAGNÓSTICO COMPLETO:** Problemas de conectividad en APIs")
                diagnosticar_conectividad()
            
            log_exception(func_name, openai_error, f"Todas las opciones de transcripción fallaron")
            raise Exception(f"Todas las opciones fallaron - faster-whisper: {whisper_error}, Voxtral: {mistral_error}, OpenAI: {openai_error}")
        
    except Exception as e:
        log_exception(func_name, e, f"Archivo: {audio_path}, Tamaño: {tamaño_mb:.1f}MB")
        raise

def transcribir_archivo_grande_mp3(audio_path, func_name):
    """Convierte archivos grandes a MP3 y los transcribe.
    Cadena: faster-whisper (local) → Voxtral (Mistral) → OpenAI Whisper"""
    try:
        audio_mp3 = audio_path.replace('.wav', '_comprimido.mp3')
        
        st.info(f"🔄 **Convirtiendo archivo grande a MP3:** {os.path.basename(audio_path)}")
        log_info(f"Convirtiendo archivo grande a MP3: {audio_path} -> {audio_mp3}", func_name)
        
        try:
            with st.spinner("🎵 Convirtiendo audio a MP3 (64kbps, 16kHz, mono)..."):
                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ac", "1", "-ar", "16000", "-acodec", "mp3", "-b:a", "64k",
                    audio_mp3
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            tamaño_original_mb = os.path.getsize(audio_path) / (1024 * 1024)
            tamaño_mp3_mb = os.path.getsize(audio_mp3) / (1024 * 1024)
            reduccion = ((tamaño_original_mb - tamaño_mp3_mb) / tamaño_original_mb) * 100
            
            st.success(f"✅ **Conversión MP3 exitosa:** {tamaño_original_mb:.1f}MB → {tamaño_mp3_mb:.1f}MB (**{reduccion:.1f}% reducción**)")
            log_info(f"Archivo convertido: {tamaño_original_mb:.1f}MB -> {tamaño_mp3_mb:.1f}MB (reducción: {reduccion:.1f}%)", func_name)
            
        except subprocess.CalledProcessError as e:
            log_exception(func_name, e, f"Error convirtiendo archivo a MP3")
            raise Exception(f"Error convirtiendo a MP3: {e}")
        
        # === PASO 1: faster-whisper GPU/CPU ===
        st.info(f"🧠 **Intentando faster-whisper** (MP3 {tamaño_mp3_mb:.1f}MB)")
        try:
            result, segments = transcribir_con_faster_whisper(audio_mp3, return_segments=True)
            motor = f"faster-whisper {_whisper_device_actual.upper()}"
            st.success(f"✅ **{motor}** completó transcripción del MP3 — **SIN COSTO**")
            if os.path.exists(audio_mp3):
                os.remove(audio_mp3)
                st.info(f"🧹 Archivo temporal MP3 eliminado")
            return result, f"{motor} (MP3 {tamaño_mp3_mb:.1f}MB)", segments
        except Exception as whisper_error:
            try:
                result, segments = transcribir_con_faster_whisper_cpu(audio_mp3, return_segments=True)
                st.success(f"✅ **faster-whisper CPU** completó transcripción del MP3 — **SIN COSTO**")
                if os.path.exists(audio_mp3):
                    os.remove(audio_mp3)
                return result, f"faster-whisper CPU (MP3 {tamaño_mp3_mb:.1f}MB)", segments
            except Exception:
                st.warning(f"⚠️ **faster-whisper falló** con MP3 → Activando **Voxtral**")
        
        # === PASO 2: Voxtral/Mistral con timestamps ===
        st.info(f"🧠 **Intentando Voxtral** (MP3 {tamaño_mp3_mb:.1f}MB)")
        try:
            result, segments = transcribir_audio_mistral(audio_mp3, con_timestamps=True)
            st.success(f"✅ **Voxtral** completó transcripción del MP3 con timestamps")
            if os.path.exists(audio_mp3):
                os.remove(audio_mp3)
            return result, f"Mistral (Voxtral) (MP3 {tamaño_mp3_mb:.1f}MB)", segments
        except Exception as mistral_error:
            st.warning(f"⚠️ **Voxtral falló** → Activando **OpenAI Whisper**")
        
        # === PASO 3: OpenAI Whisper (sin timestamps) ===
        st.info(f"🧠 **Intentando OpenAI Whisper** (MP3 {tamaño_mp3_mb:.1f}MB)")
        try:
            result = transcribir_con_openai(audio_mp3)
            st.success(f"✅ **OpenAI Whisper** completó transcripción del MP3")
            if os.path.exists(audio_mp3):
                os.remove(audio_mp3)
            return result, f"OpenAI Whisper (MP3 {tamaño_mp3_mb:.1f}MB)", None
            
        except Exception as openai_error:
            if os.path.exists(audio_mp3):
                os.remove(audio_mp3)
            
            log_exception(func_name, openai_error, f"Las 3 opciones fallaron con archivo MP3")
            raise Exception(f"Todas fallaron con MP3 - faster-whisper: {whisper_error}, Voxtral: {mistral_error}, OpenAI: {openai_error}")
                
    except Exception as e:
        log_exception(func_name, e, f"Error procesando archivo grande: {audio_path}")
        raise

def obtener_timestamps_whisper(audio_path):
    model = cargar_modelo_whisper_timestamps()
    
    segments, _ = model.transcribe(
        audio_path, language="es", chunk_length=300,
        beam_size=1, vad_filter=True, word_timestamps=False
    )
    
    timestamp_segments = []
    for seg in segments:
        timestamp_segments.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text
        })
    
    return timestamp_segments


def determinar_segmento_inteligente_gemini(transcripcion_con_timestamps, termino_encontrado, timestamp_coincidencia, duracion_maxima=90):
    """
    🌟 USA GEMINI 3 PRO PARA DETERMINAR EL SEGMENTO MÁS LÓGICO Y COHERENTE
    
    Gemini analiza la transcripción completa y determina cuál es el segmento más lógico donde
    el término encontrado es el EJE CENTRAL de la conversación, no una mención tangencial.
    
    Args:
        transcripcion_con_timestamps: Lista de segmentos con 'start', 'end', 'text'
        termino_encontrado: El término que generó la coincidencia
        timestamp_coincidencia: Timestamp donde se encontró el término
        duracion_maxima: Duración máxima del clip en segundos (default: 60)
    
    Returns:
        dict: {'inicio': float, 'fin': float, 'razon': str, 'duracion': float, 'idea_central': str}
        dict rechazo: {'_rechazo_segmento': True, 'razon': str} si la mención es tangencial
    """
    func_name = "determinar_segmento_inteligente_gemini"
    
    # Verificar si Gemini está configurado
    if not gemini_client:
        log_warning("⚠️ Gemini no configurado, usando fallback GPT-4o", func_name)
        return determinar_segmento_inteligente_gpt4(
            transcripcion_con_timestamps, termino_encontrado, 
            timestamp_coincidencia, duracion_maxima
        )
    
    try:
        log_info(f"🌟 Iniciando análisis GEMINI 3 PRO para segmento inteligente del término '{termino_encontrado}'", func_name)
        
        # Construir contexto de transcripción con timestamps
        contexto_transcripcion = []
        for seg in transcripcion_con_timestamps:
            tiempo_inicio = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
            contexto_transcripcion.append(f"[{tiempo_inicio}] {seg['text'].strip()}")
        
        texto_completo_timestamps = "\n".join(contexto_transcripcion)
        
        # Calcular timestamp en formato legible
        minuto_coincidencia = int(timestamp_coincidencia // 60)
        segundo_coincidencia = int(timestamp_coincidencia % 60)
        
        # Prompt optimizado para Gemini - ENFOCADO EN IDEAS CENTRADAS EN LA COINCIDENCIA
        prompt = f"""Eres un experto editor de audio y analista de contenido. Tu tarea es encontrar el SEGMENTO EXACTO donde "{termino_encontrado}" sea el TEMA CENTRAL de la conversación.

🎯 TÉRMINO A ANALIZAR: "{termino_encontrado}"
⏰ TIMESTAMP DE DETECCIÓN: {minuto_coincidencia}:{segundo_coincidencia:02d} ({timestamp_coincidencia:.1f} segundos)

📝 TRANSCRIPCIÓN COMPLETA CON TIMESTAMPS:
{texto_completo_timestamps}

═══════════════════════════════════════════════════════════════════
📋 TU MISIÓN CRÍTICA:
═══════════════════════════════════════════════════════════════════

1️⃣ IDENTIFICAR si "{termino_encontrado}" es el EJE CENTRAL del segmento:
   ✅ APROBADO: La conversación GIRA EN TORNO a "{termino_encontrado}"
   ✅ APROBADO: Hay información CONCRETA y DESARROLLADA sobre "{termino_encontrado}"
   ✅ APROBADO: El término es el PROTAGONISTA, no un actor secundario
   
   ❌ RECHAZAR: Solo se menciona de pasada o en una lista
   ❌ RECHAZAR: El tema principal es OTRO y "{termino_encontrado}" es tangencial
   ❌ RECHAZAR: No hay desarrollo de ideas sobre "{termino_encontrado}"

2️⃣ Si APRUEBAS, determinar el SEGMENTO ÓPTIMO que:
   - Capture la IDEA COMPLETA relacionada con "{termino_encontrado}"
   - Tenga INICIO y FIN naturales (no cortes abruptos)
   - Duración REQUERIDA: EXACTAMENTE {duracion_maxima} segundos (obligatorio para no cortar la idea)
   - Sea COHERENTE y COMPRENSIBLE por sí solo

3️⃣ EXTRAER LA IDEA CENTRAL:
   - ¿Qué se dice ESPECÍFICAMENTE sobre "{termino_encontrado}"?
   - Resume en 1-2 oraciones la información CONCRETA
   - NO resumas toda la transcripción, solo lo relacionado con el término

═══════════════════════════════════════════════════════════════════
📤 FORMATO DE RESPUESTA (JSON estricto, sin markdown):
═══════════════════════════════════════════════════════════════════

Si el término ES el tema central (APROBAR):
{{
  "rechazar": false,
  "inicio_segundos": <número>,
  "fin_segundos": <número>,
  "duracion_segundos": <número>,
  "razon": "<por qué este segmento captura la idea sobre {termino_encontrado}>",
  "idea_central": "<qué se dice CONCRETAMENTE sobre {termino_encontrado} - máximo 2 oraciones>"
}}

Si el término NO es el tema central (RECHAZAR):
{{
  "rechazar": true,
  "razon": "<por qué {termino_encontrado} solo es una mención tangencial>"
}}

⚠️ IMPORTANTE:
- Sé ESTRICTO: Si hay duda, RECHAZA
- La "idea_central" debe responder: "¿Qué dice este segmento SOBRE {termino_encontrado}?"
- NO incluyas información que no esté directamente relacionada con el término

RESPONDE SOLO CON EL JSON:"""

        # Llamar a Gemini 3 Pro
        log_info("📡 Enviando solicitud a GEMINI 3 PRO para análisis de segmento...", func_name)
        
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-preview",  # Gemini 3 Pro (modelo más avanzado)
            contents=prompt,
            config={
                "temperature": 0.2,  # Bajo para respuestas más consistentes
                "max_output_tokens": 600
            }
        )
        
        respuesta_gemini = response.text.strip()
        log_debug(f"Respuesta Gemini: {respuesta_gemini}", func_name)
        
        # Limpiar respuesta (remover markdown si existe)
        respuesta_limpia = respuesta_gemini.replace("```json", "").replace("```", "").strip()
        
        # Parsear JSON
        resultado = json.loads(respuesta_limpia)
        
        # 🚫 VERIFICAR SI GEMINI RECHAZÓ EL SEGMENTO
        if resultado.get('rechazar', False):
            razon_rechazo = resultado.get('razon', 'Mención tangencial sin desarrollo')
            log_warning(f"🚫 GEMINI RECHAZÓ el segmento: {razon_rechazo}", func_name)
            st.warning(f"🚫 **Gemini:** {razon_rechazo}")
            return {'_rechazo_segmento': True, 'razon': razon_rechazo}
        
        # Validar y ajustar resultados
        inicio = float(resultado.get('inicio_segundos', timestamp_coincidencia - 30))
        fin = float(resultado.get('fin_segundos', timestamp_coincidencia + 30))
        razon = resultado.get('razon', 'Segmento determinado por Gemini')
        idea_central = resultado.get('idea_central', '')
        
        # Validaciones de seguridad
        inicio = max(0, inicio)  # No puede ser negativo
        duracion_calculada = fin - inicio
        
        # Si excede duración máxima, ajustar
        if duracion_calculada > duracion_maxima:
            log_warning(f"⚠️ Segmento Gemini ({duracion_calculada:.1f}s) excede máximo ({duracion_maxima}s), ajustando...", func_name)
            fin = inicio + duracion_maxima
            duracion_calculada = duracion_maxima
            razon += " (ajustado a duración máxima)"
        
        # Si es muy corto (< 60s), expandir a 1 minuto mínimo (±30s del centro)
        if duracion_calculada < 60:
            log_warning(f"⚠️ Segmento Gemini corto ({duracion_calculada:.1f}s), expandiendo a 60s...", func_name)
            centro = (inicio + fin) / 2
            inicio = max(0, centro - 30)
            fin = centro + 30
            duracion_calculada = fin - inicio
            razon += " (expandido a 1 minuto estándar)"
        
        resultado_final = {
            'inicio': inicio,
            'fin': fin,
            'razon': razon,
            'duracion': duracion_calculada,
            'idea_central': idea_central  # Nueva: idea centrada en la coincidencia
        }
        
        log_info(f"✅ Gemini determinó segmento: {inicio:.1f}s - {fin:.1f}s ({duracion_calculada:.1f}s)", func_name)
        log_info(f"📝 Razón: {razon}", func_name)
        log_info(f"💡 Idea central: {idea_central[:100]}...", func_name)
        
        # Mostrar en UI
        st.success(f"🌟 **Gemini 3 Pro:** Segmento inteligente: {inicio:.1f}s - {fin:.1f}s ({duracion_calculada:.1f}s)")
        st.info(f"💡 **Razón:** {razon}")
        if idea_central:
            st.info(f"🎯 **Idea central sobre '{termino_encontrado}':** {idea_central}")
        
        return resultado_final
        
    except json.JSONDecodeError as e:
        log_warning(f"⚠️ Error parseando JSON de Gemini: {e}. Intentando fallback GPT-4o", func_name)
        # Fallback a GPT-4o
        return determinar_segmento_inteligente_gpt4(
            transcripcion_con_timestamps, termino_encontrado,
            timestamp_coincidencia, duracion_maxima
        )
    
    except Exception as e:
        log_exception(func_name, e, f"Término: {termino_encontrado}, Timestamp: {timestamp_coincidencia}")
        st.warning(f"⚠️ Error en análisis Gemini, usando fallback GPT-4o: {str(e)}")
        
        # Fallback a GPT-4o
        return determinar_segmento_inteligente_gpt4(
            transcripcion_con_timestamps, termino_encontrado,
            timestamp_coincidencia, duracion_maxima
        )


def determinar_segmento_inteligente_gpt4(transcripcion_con_timestamps, termino_encontrado, timestamp_coincidencia, duracion_maxima=90):
    """
    🤖 USA GPT-4o PARA DETERMINAR EL SEGMENTO MÁS LÓGICO Y COHERENTE
    
    En lugar de recortar mecánicamente X segundos antes/después, GPT-4o analiza
    la transcripción completa y determina cuál es el segmento más lógico que:
    - Contiene la idea completa relacionada con la coincidencia
    - Tiene coherencia narrativa (inicio y fin naturales)
    - Duración recomendada de 90 segundos de duración
    - Captura el contexto relevante sin cortes abruptos
    
    Args:
        transcripcion_con_timestamps: Lista de segmentos con 'start', 'end', 'text'
        termino_encontrado: El término que generó la coincidencia
        timestamp_coincidencia: Timestamp donde se encontró el término
        duracion_maxima: Duración máxima del clip en segundos (default: 60)
    
    Returns:
        dict: {'inicio': float, 'fin': float, 'razon': str, 'duracion': float}
        dict rechazo: {'_rechazo_segmento': True, 'razon': str}
    """
    func_name = "determinar_segmento_inteligente_gpt4"
    
    try:
        log_info(f"🤖 Iniciando análisis GPT-4o para segmento inteligente del término '{termino_encontrado}'", func_name)
        
        # Construir contexto de transcripción con timestamps
        contexto_transcripcion = []
        for seg in transcripcion_con_timestamps:
            tiempo_inicio = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
            contexto_transcripcion.append(f"[{tiempo_inicio}] {seg['text'].strip()}")
        
        texto_completo_timestamps = "\n".join(contexto_transcripcion)
        
        # Calcular timestamp en formato legible
        minuto_coincidencia = int(timestamp_coincidencia // 60)
        segundo_coincidencia = int(timestamp_coincidencia % 60)
        
        # Prompt para GPT-4o
        prompt = f"""Eres un experto editor de audio. Analiza esta transcripción con timestamps y determina el SEGMENTO donde "{termino_encontrado}" sea el TEMA CENTRAL.

🎯 TÉRMINO ENCONTRADO: "{termino_encontrado}"
⏰ TIMESTAMP DE COINCIDENCIA: {minuto_coincidencia}:{segundo_coincidencia:02d} ({timestamp_coincidencia:.1f} segundos)

📝 TRANSCRIPCIÓN COMPLETA CON TIMESTAMPS:
{texto_completo_timestamps}

📋 TU TAREA:
Encuentra el segmento donde "{termino_encontrado}" sea el TEMA PRINCIPAL de la conversación, que cumpla:

🎯 CRITERIO PRINCIPAL (MÁS IMPORTANTE):
- El término "{termino_encontrado}" debe ser el EJE CENTRAL del segmento
- La conversación debe GIRAR ALREDEDOR de "{termino_encontrado}"
- NO aceptes menciones de pasada o tangenciales
- La idea completa debe DESARROLLAR el tema de "{termino_encontrado}"

✅ OTROS REQUISITOS:
1. Tenga un INICIO y FIN NATURAL (no cortes abruptos)
2. Capture el CONTEXTO RELEVANTE que desarrolla el tema
3. Duración REQUERIDA: EXACTAMENTE {duracion_maxima} segundos (obligatorio para capturar la idea completa)
4. Sea COHERENTE y COMPRENSIBLE por sí solo

⚠️ IMPORTANTE:
- Si "{termino_encontrado}" solo se menciona de pasada (sin desarrollar el tema), RESPONDE: {{"rechazar": true, "razon": "Mención tangencial sin desarrollo"}}
- Identifica dónde EMPIEZA el desarrollo del tema (puede ser varios segundos antes del término)
- Identifica dónde TERMINA el desarrollo completo de la idea
- Busca pausas naturales, cambios de tema, o conclusiones de frases
- Si la idea completa excede {duracion_maxima}s, prioriza el núcleo más importante

RESPONDE EN FORMATO JSON (sin markdown, sin comentarios):
{{
  "rechazar": false,
  "inicio_segundos": <timestamp de inicio en segundos como número>,
  "fin_segundos": <timestamp de fin en segundos como número>,
  "razon": "<breve explicación de por qué elegiste este segmento (1-2 líneas)>",
  "duracion_segundos": <duración total del segmento como número>
}}

O si la mención es tangencial/sin desarrollo:
{{
  "rechazar": true,
  "razon": "<explicación de por qué es solo mención tangencial>"
}}"""

        # Llamar a GPT-4o
        log_info("📡 Enviando solicitud a GPT-4o para análisis de segmento...", func_name)
        
        # Usar API key desde variable de entorno
        openai.api_key = os.getenv('OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY_BACKUP', '')
        
        response = openai.chat.completions.create(
            model="gpt-4o",  # Modelo más avanzado de OpenAI
            messages=[
                {"role": "system", "content": "Eres un experto editor de audio que analiza transcripciones para determinar los mejores segmentos de corte."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Bajo para respuestas más consistentes
            max_tokens=1000
        )
        
        respuesta_gpt = response.choices[0].message.content.strip()
        log_debug(f"Respuesta GPT-4o: {respuesta_gpt}", func_name)
        
        # Limpiar respuesta (remover markdown si existe)
        respuesta_limpia = respuesta_gpt.replace("```json", "").replace("```", "").strip()
        
        # Parsear JSON
        resultado = json.loads(respuesta_limpia)
        
        # 🚫 VERIFICAR SI GPT-4o RECHAZÓ EL SEGMENTO
        if resultado.get('rechazar', False):
            razon_rechazo = resultado.get('razon', 'Mención tangencial sin desarrollo')
            log_warning(f"🚫 GPT-4o RECHAZÓ el segmento: {razon_rechazo}", func_name)
            st.warning(f"🚫 **GPT-4o:** {razon_rechazo}")
            return {'_rechazo_segmento': True, 'razon': razon_rechazo}
        
        # Validar y ajustar resultados
        inicio = float(resultado.get('inicio_segundos', timestamp_coincidencia - 30))
        fin = float(resultado.get('fin_segundos', timestamp_coincidencia + 30))
        razon = resultado.get('razon', 'Segmento determinado por GPT-4o')
        
        # Validaciones de seguridad
        inicio = max(0, inicio)  # No puede ser negativo
        duracion_calculada = fin - inicio
        
        # Si excede duración máxima, ajustar
        if duracion_calculada > duracion_maxima:
            log_warning(f"⚠️ Segmento GPT-4o ({duracion_calculada:.1f}s) excede máximo ({duracion_maxima}s), ajustando...", func_name)
            # Mantener el inicio, acortar el fin
            fin = inicio + duracion_maxima
            duracion_calculada = duracion_maxima
            razon += " (ajustado a duración máxima)"
        
        # Si es muy corto (< 60s), expandir a 1 minuto mínimo (±30s del centro)
        if duracion_calculada < 60:
            log_warning(f"⚠️ Segmento GPT-4o corto ({duracion_calculada:.1f}s), expandiendo a 60s...", func_name)
            centro = (inicio + fin) / 2
            inicio = max(0, centro - 30)
            fin = centro + 30
            duracion_calculada = fin - inicio
            razon += " (expandido a 1 minuto estándar)"
        
        resultado_final = {
            'inicio': inicio,
            'fin': fin,
            'razon': razon,
            'duracion': duracion_calculada
        }
        
        log_info(f"✅ GPT-4o determinó segmento: {inicio:.1f}s - {fin:.1f}s ({duracion_calculada:.1f}s)", func_name)
        log_info(f"📝 Razón: {razon}", func_name)
        
        # Mostrar en UI
        st.success(f"🤖 **GPT-4o:** Segmento inteligente determinado: {inicio:.1f}s - {fin:.1f}s ({duracion_calculada:.1f}s)")
        st.info(f"💡 **Razón:** {razon}")
        
        return resultado_final
        
    except json.JSONDecodeError as e:
        log_warning(f"⚠️ Error parseando JSON de GPT-4o: {e}. Usando método tradicional", func_name)
        # Fallback al método tradicional
        return {
            'inicio': max(0, timestamp_coincidencia - 30),
            'fin': timestamp_coincidencia + 30,
            'razon': 'Método tradicional (error en GPT-4o)',
            'duracion': 60
        }
    
    except Exception as e:
        log_exception(func_name, e, f"Término: {termino_encontrado}, Timestamp: {timestamp_coincidencia}")
        st.warning(f"⚠️ Error en análisis GPT-4o, usando método tradicional: {str(e)}")
        
        # Fallback al método tradicional
        return {
            'inicio': max(0, timestamp_coincidencia - 30),
            'fin': timestamp_coincidencia + 30,
            'razon': 'Método tradicional (error en GPT-4o)',
            'duracion': 60
        }

def extraer_idea_general_segmento_gpt4(transcripcion_segmento, termino_encontrado, duracion_segundos):
    """
    🤖 USA GPT-4o PARA EXTRAER LA IDEA GENERAL DE UN SEGMENTO ESPECÍFICO
    
    En lugar de enviar toda la transcripción del audio, GPT-4o extrae solo
    la idea principal y relevante del segmento del clip.
    
    Args:
        transcripcion_segmento: Texto del segmento específico del clip
        termino_encontrado: El término que generó la coincidencia
        duracion_segundos: Duración del segmento en segundos
    
    Returns:
        str: Idea general condensada (máximo 1-2 párrafos)
    """
    func_name = "extraer_idea_general_segmento_gpt4"
    
    try:
        log_info(f"🤖 Extrayendo idea general con GPT-4o para término '{termino_encontrado}'", func_name)
        
        # Prompt para GPT-4o
        prompt = f"""Eres un analista de contenido EXTREMADAMENTE CRÍTICO. Tu trabajo es RECHAZAR menciones superficiales y APROBAR solo cuando el término es el TEMA CENTRAL.

🎯 TÉRMINO CLAVE: "{termino_encontrado}"
⏱️ DURACIÓN DEL SEGMENTO: {duracion_segundos:.1f} segundos

📝 TRANSCRIPCIÓN DEL SEGMENTO:
{transcripcion_segmento}

📋 TU TAREA CRÍTICA:
Analiza si "{termino_encontrado}" es el EJE CENTRAL de este segmento.

🚫 RECHAZA (responde "NO_RELEVANTE") SI:
1. El término solo se menciona de pasada o tangencialmente
2. La conversación NO GIRA ALREDEDOR del término
3. El término aparece en una lista o enumeración sin desarrollo
4. Es solo una referencia sin elaboración del tema
5. La idea principal del segmento es OTRO tema diferente

✅ APRUEBA (resume la idea) SOLO SI:
1. El término es el TEMA PRINCIPAL del segmento
2. La conversación DESARROLLA el tema del término
3. Hay información CONCRETA y SUSTANCIAL sobre el término
4. El término es el EJE que estructura toda la conversación

⚠️ FORMATO DE RESPUESTA:

Si NO es relevante (MAYORÍA de los casos):
"NO_RELEVANTE: El término '{termino_encontrado}' [explicar brevemente por qué no es el tema central]"

Si SÍ es relevante (CASOS EXCEPCIONALES):
Resume la IDEA GENERAL en 1-2 párrafos (máximo 150 palabras):
- Información CONCRETA sobre el término
- Desarrollo sustancial del tema
- Contexto que demuestra que es el eje central

IMPORTANTE:
- Sé EXTREMADAMENTE CRÍTICO
- En caso de duda, RECHAZA
- NO inventes información
- NO resumas la transcripción literal

RESPONDE DIRECTAMENTE:"""

        # Llamar a GPT-4o (API key desde variable de entorno)
        openai.api_key = os.getenv('OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY_BACKUP', '')
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un analista de contenido experto que resume ideas de forma clara y concisa."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        
        idea_general = response.choices[0].message.content.strip()
        
        log_info(f"✅ Idea general extraída: {len(idea_general)} caracteres", func_name)
        log_debug(f"Idea: {idea_general[:100]}...", func_name)
        
        return idea_general
        
    except Exception as e:
        log_exception(func_name, e, f"Término: {termino_encontrado}")
        # Fallback: retornar un resumen simple del segmento
        return f"Segmento relacionado con '{termino_encontrado}': {transcripcion_segmento[:200]}..."


def normalizar_termino_sin_acentos(s):
    if s is None:
        return ""
    buf = unicodedata.normalize("NFD", str(s).strip().lower())
    return "".join(c for c in buf if unicodedata.category(c) != "Mn")


def termino_es_solo_educacion(termino):
    """Coincide únicamente con el término «educación», no «ministro de educación», etc."""
    return normalizar_termino_sin_acentos(termino) == "educacion"


# Reglas cuando el keyword es solo «educación» (MINERD escolar RD vs otros usos del vocablo)
_PROMPT_EDUCACION_ESCOLAR_RD_ANEXO = """
═══════════════════════════════════════════════════════════════════
🏫 REGLA ADICIONAL OBLIGATORIA — TÉRMINO EXACTO: «educación»
═══════════════════════════════════════════════════════════════════
Este monitoreo es para **educación escolar / sistema educativo formal dominicano** (antes de entrar solo a nivel universitario aislado).

**Debes responder con es_relevante: false y relevancia: "baja"** cuando «educación» se refiere a cualquiera de estos casos típicos (no enviar alerta como escolar RD):
• educación vial, financiera, del consumidor, sexual o socioemocional **como tema principal** sin enlace explícito a escuela/sistema RD
• talleres corporativos, «educamos al cliente», clichés deportivos («educación en el día a día») o metáforas
• modelos solo de otros países sin articulación con escuelas, MINERD, estudiantes o política escolar **en República Dominicana**

**Puedes marcar es_relevante: true** (con relevancia adecuada) solo cuando el contenido desarrolla **educación formal en RD**: escuelas, liceos, tecnológicos, estudiantes/docentes, MINERD, matrícula, infraestructura escolar, calendario/presupuesto escolar, desayuno/comedor escolar, pruebas nacionales **en contexto escolar dominicano**, reformas al sistema educativo **del país**, etc.

Ante duda razonable, **rechaza** (es_relevante: false, relevancia: "baja"). Los campos que_se_dice, contexto e idea_general deben ser coherentes con esa decisión.
"""


def extraer_idea_general_segmento_gemini(transcripcion_segmento, termino_encontrado, duracion_segundos, nombre_video=""):
    """
    🌟 USA GEMINI 3 PRO PARA EXTRAER IDEAS CENTRADAS EN LA COINCIDENCIA
    
    Gemini 3 Pro analiza el segmento y extrae ESPECÍFICAMENTE qué se dice
    sobre el término encontrado, NO ideas generales al aire.
    
    Args:
        transcripcion_segmento: Texto del segmento específico del clip
        termino_encontrado: El término que generó la coincidencia
        duracion_segundos: Duración del segmento en segundos
        nombre_video: Nombre del archivo de audio para contexto adicional
    
    Returns:
        dict: {
            'idea_general': str - Resumen contextualizado,
            'relevancia': str - 'alta', 'media', 'baja',
            'tema_principal': str - Tema central identificado,
            'contexto': str - Contexto en que se menciona el término,
            'es_relevante': bool - Si el término es tema central
        }
    """
    func_name = "extraer_idea_general_segmento_gemini"
    
    filtro_educacion_rd = termino_es_solo_educacion(termino_encontrado)
    # Si es solo «educación», por defecto no enviar hasta que la IA apruebe (evita falsos positivos)
    resultado_default = {
        'idea_general': f"Segmento relacionado con '{termino_encontrado}'",
        'relevancia': 'baja' if filtro_educacion_rd else 'media',
        'tema_principal': termino_encontrado,
        'contexto': transcripcion_segmento[:200] if transcripcion_segmento else '',
        'es_relevante': False if filtro_educacion_rd else True,
        'que_se_dice': f"Mención de '{termino_encontrado}' en el segmento"
    }
    
    # Verificar si Gemini está configurado
    if not gemini_client:
        log_warning("⚠️ Gemini no configurado, usando fallback GPT-4o", func_name)
        # Fallback a GPT-4o si Gemini no está disponible
        idea_gpt = extraer_idea_general_segmento_gpt4(transcripcion_segmento, termino_encontrado, duracion_segundos)
        resultado_default['idea_general'] = idea_gpt
        if filtro_educacion_rd:
            log_warning(
                "⏭️ Término «educación»: sin Gemini no se clasifica escolar RD; no se aprueba coincidencia",
                func_name,
            )
        return resultado_default
    
    try:
        log_info(f"🌟 Extrayendo idea CENTRADA EN COINCIDENCIA con GEMINI 3 PRO para término '{termino_encontrado}'", func_name)
        
        # Prompt optimizado para Gemini 3 Pro - ENFOCADO EN IDEAS CENTRADAS EN LA COINCIDENCIA
        prompt = f"""Eres un analista experto de contenido de audio/radio. Tu ÚNICA tarea es extraer QUÉ SE DICE ESPECÍFICAMENTE sobre "{termino_encontrado}".

═══════════════════════════════════════════════════════════════════
📻 AUDIO: {nombre_video if nombre_video else 'Archivo de audio / noticias'}
🎯 TÉRMINO DE INTERÉS: "{termino_encontrado}"
⏱️ DURACIÓN: {duracion_segundos:.1f} segundos
═══════════════════════════════════════════════════════════════════

📝 TRANSCRIPCIÓN DEL SEGMENTO:
---
{transcripcion_segmento}
---

═══════════════════════════════════════════════════════════════════
🎯 TU MISIÓN CRÍTICA:
═══════════════════════════════════════════════════════════════════

Responde ÚNICAMENTE estas preguntas sobre "{termino_encontrado}":

1️⃣ ¿Qué se DICE CONCRETAMENTE sobre "{termino_encontrado}"?
2️⃣ ¿Qué INFORMACIÓN ESPECÍFICA se revela sobre "{termino_encontrado}"?
3️⃣ ¿En qué CONTEXTO se menciona "{termino_encontrado}"?

⚠️ REGLAS ESTRICTAS:
- SOLO incluye información que DIRECTAMENTE mencione o se refiera a "{termino_encontrado}"
- NO resumas el segmento completo
- NO incluyas información sobre OTROS temas
- NO inventes información
- Si no hay información sustancial sobre "{termino_encontrado}", indica relevancia BAJA

📤 RESPONDE EN JSON (sin markdown, sin comentarios):

{{
    "es_relevante": true/false,
    "relevancia": "alta" | "media" | "baja",
    "que_se_dice": "¿Qué dice el segmento ESPECÍFICAMENTE sobre {termino_encontrado}? (1-2 oraciones concretas)",
    "contexto": "¿En qué situación/tema se menciona {termino_encontrado}? (1 oración)",
    "idea_general": "Resumen de lo que se dice SOBRE {termino_encontrado} - NO sobre otros temas (máximo 80 palabras)",
    "tema_principal": "El tema central EN RELACIÓN a {termino_encontrado} (5-10 palabras)"
}}

📌 CRITERIOS:
- ALTA: Se habla DIRECTAMENTE de "{termino_encontrado}" con información sustancial
- MEDIA: Se menciona "{termino_encontrado}" con algo de contexto
- BAJA: Solo se nombra sin desarrollo

RESPONDE SOLO CON EL JSON:"""

        if filtro_educacion_rd:
            prompt += _PROMPT_EDUCACION_ESCOLAR_RD_ANEXO

        # Llamar a Gemini 3 Pro
        log_info("📡 Enviando solicitud a Gemini 3 Pro...", func_name)
        
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-preview",  # Gemini 3 Pro (modelo más avanzado)
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 500
            }
        )
        
        respuesta_texto = response.text.strip()
        log_debug(f"Respuesta Gemini: {respuesta_texto[:200]}...", func_name)
        
        # Limpiar respuesta (remover markdown si existe)
        respuesta_limpia = respuesta_texto.replace("```json", "").replace("```", "").strip()
        
        # Parsear JSON
        try:
            resultado = json.loads(respuesta_limpia)
            
            # Validar campos requeridos
            campos_requeridos = ['es_relevante', 'relevancia', 'tema_principal', 'contexto', 'idea_general', 'que_se_dice']
            for campo in campos_requeridos:
                if campo not in resultado:
                    resultado[campo] = resultado_default.get(campo, '')
            
            # Si hay "que_se_dice", usarlo para enriquecer la idea_general
            if resultado.get('que_se_dice') and resultado.get('idea_general'):
                # Combinar para un resumen más completo centrado en la coincidencia
                log_info(f"📝 Qué se dice sobre '{termino_encontrado}': {resultado['que_se_dice'][:100]}...", func_name)
            
            log_info(f"✅ Análisis Gemini completado - Relevancia: {resultado.get('relevancia', 'N/A')}", func_name)
            
            return resultado
            
        except json.JSONDecodeError as je:
            log_warning(f"⚠️ Error parseando JSON de Gemini: {je}", func_name)
            # Si no es JSON válido, usar el texto como idea general
            resultado_default['idea_general'] = respuesta_limpia[:1000]
            return resultado_default
        
    except Exception as e:
        log_exception(func_name, e, f"Término: {termino_encontrado}")
        
        # Fallback a GPT-4o
        log_info("🔄 Fallback a GPT-4o...", func_name)
        try:
            idea_gpt = extraer_idea_general_segmento_gpt4(transcripcion_segmento, termino_encontrado, duracion_segundos)
            resultado_default['idea_general'] = idea_gpt
        except:
            resultado_default['idea_general'] = f"Segmento sobre '{termino_encontrado}': {transcripcion_segmento[:200]}..."
        
        return resultado_default


def generar_resumen_video(nombre_video, coincidencias, transcripcion_completa):
    terminos_encontrados = list(set([item['termino'] for item in coincidencias]))
    
    prompt = f"""
Analiza el siguiente archivo de audio: "{nombre_video}"

TÉRMINOS ENCONTRADOS: {', '.join(terminos_encontrados)}

TRANSCRIPCIÓN COMPLETA:
{transcripcion_completa[:2000]}...

COINCIDENCIAS ESPECÍFICAS:
"""
    
    for item in coincidencias:
        prompt += f"- **{item['termino']}**: {item['texto']}\n"
    
    prompt += """

Genera un resumen ejecutivo que DEBE empezar exactamente así:
**TÉRMINOS DETECTADOS:** [lista los términos encontrados]

Luego incluir:
1. **Tema principal** del video
2. **Contexto** en que aparecen las palabras clave
3. **Puntos clave** mencionados
4. **Relevancia** de las coincidencias encontradas

Mantén el resumen conciso pero informativo (máximo 200 palabras).
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en análisis de contenido audiovisual que genera resúmenes ejecutivos precisos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        contenido_ia = resp.choices[0].message.content
        
        # Asegurar que empiece con los términos detectados
        if not contenido_ia.startswith("**TÉRMINOS DETECTADOS:**"):
            terminos_texto = f"**TÉRMINOS DETECTADOS:** {', '.join(terminos_encontrados)}\n\n"
            contenido_ia = terminos_texto + contenido_ia
        
        return contenido_ia
        
    except Exception as e:
        # Fallback manual si falla la IA
        terminos_texto = f"**TÉRMINOS DETECTADOS:** {', '.join(terminos_encontrados)}\n\n"
        return terminos_texto + f"Error generando resumen automático: {e}"

def generar_resumen_archivo(nombre_archivo, coincidencias, transcripcion_completa, tipo_archivo):
    """
    Genera resumen ejecutivo para cualquier tipo de archivo (video o audio)
    """
    terminos_encontrados = list(set([item['termino'] for item in coincidencias]))
    
    # Extraer información del medio y hora
    info_medio_hora = extraer_info_medio_hora(nombre_archivo)
    
    prompt = f"""
Analiza el siguiente archivo de {tipo_archivo.lower()}: "{nombre_archivo}"

TÉRMINOS ENCONTRADOS: {', '.join(terminos_encontrados)}

TRANSCRIPCIÓN COMPLETA:
{transcripcion_completa[:2000]}...

COINCIDENCIAS ESPECÍFICAS:
"""
    
    for item in coincidencias:
        prompt += f"- **{item['termino']}**: {item['texto']}\n"
    
    prompt += f"""

Genera un resumen ejecutivo con EXACTAMENTE este formato (sin markdown, sin asteriscos, texto plano):

Tema principal: [describe el tema central del {tipo_archivo.lower()} en 1-2 oraciones]

Contexto: [en qué situación/tema se mencionan las palabras clave, 1-2 oraciones]

Puntos clave: [datos específicos mencionados, nombres, cifras, roles, 2-3 oraciones]

Relevancia: [por qué las coincidencias son importantes, qué información revelan, 1-2 oraciones]

REGLAS:
- NO uses markdown ni asteriscos (**), solo texto plano
- NO incluyas "TÉRMINOS DETECTADOS" (ya se agrega automáticamente)
- NO incluyas "Medio" (ya se agrega automáticamente)
- Mantén el resumen conciso pero informativo (máximo 200 palabras)
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Eres un asistente especializado en análisis de contenido audiovisual que genera resúmenes ejecutivos precisos para archivos de {tipo_archivo.lower()}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        contenido_ia = resp.choices[0].message.content
        
        # Limpiar asteriscos de markdown si la IA los incluye
        contenido_ia = contenido_ia.replace('**', '').replace('*', '')
        
        # Remover líneas de "TÉRMINOS DETECTADOS" y "Medio" si la IA las incluyó
        # (ya se agregan automáticamente en enviar_coincidencia_a_cliente)
        lineas = contenido_ia.split('\n')
        lineas_limpias = []
        for linea in lineas:
            linea_lower = linea.strip().lower()
            if linea_lower.startswith('términos detectados') or linea_lower.startswith('medio:') or linea_lower.startswith('📺'):
                continue
            lineas_limpias.append(linea)
        contenido_ia = '\n'.join(lineas_limpias).strip()
        
        return contenido_ia
        
    except Exception as e:
        # Fallback manual si falla la IA
        return f"Tema principal: Análisis de {tipo_archivo.lower()} con términos: {', '.join(terminos_encontrados)}\n\nContexto: Error generando resumen automático: {e}"

def crear_archivo_consolidado(video_path, nombre_video, coincidencias, transcripcion_completa, resumen, terminos_buscados, clips_info=None, video_url=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_limpio = re.sub(r'[^\w\-_\.]', '_', nombre_video.replace('.mp4', ''))
    archivo_consolidado = os.path.join(os.path.dirname(video_path), f"ANALISIS_{timestamp}_{nombre_limpio}.md")
    
    coincidencias_por_termino = {}
    for item in coincidencias:
        termino = item['termino']
        if termino not in coincidencias_por_termino:
            coincidencias_por_termino[termino] = []
        coincidencias_por_termino[termino].append(item)
    
    # Extraer nombres de términos (soporta dict y string)
    terminos_nombres = [t.get('termino', str(t)) if isinstance(t, dict) else str(t) for t in terminos_buscados]
    
    contenido = f"""# 📊 ANÁLISIS COMPLETO: {nombre_video}

## 📋 Información General
- **Archivo:** `{nombre_video}`
- **Fecha de análisis:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Términos buscados:** {', '.join(terminos_nombres)}
- **Coincidencias encontradas:** {len(coincidencias)} menciones de {len(coincidencias_por_termino)} términos diferentes
- **Video Cloudinary:** {f'[{video_url}]({video_url})' if video_url else 'No disponible'}

## 🎯 Resumen Ejecutivo

{resumen}

## 📈 Estadísticas de Coincidencias

"""
    
    for termino, items in coincidencias_por_termino.items():
        contenido += f"### 🔍 '{termino.upper()}' - {len(items)} mención(es)\n\n"
        for i, item in enumerate(items, 1):
            contexto = item['texto'][:150] + "..." if len(item['texto']) > 150 else item['texto']
            contenido += f"**Mención {i}:**\n> {contexto}\n\n"
    
    contenido += f"""
## 📝 Transcripción Completa

{transcripcion_completa}

---
*Análisis generado automáticamente con IA*
"""
    
    with open(archivo_consolidado, "w", encoding="utf-8") as f:
        f.write(contenido)
    
    return archivo_consolidado

def registrar_archivo_procesado(nombre_archivo, coincidencias, resumen, tipo_archivo, video_url=None):
    """
    Registra un archivo procesado en AMBOS archivos:
    - procesados.log (formato detallado con timestamps y metadatos)
    - procesados.txt (formato simple, una línea por video)
    """
    try:
        # ========== REGISTRAR EN procesados.log (formato detallado) ==========
        with open(PROCESADOS_LOG, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            terminos_encontrados = [item['termino'] for item in coincidencias]
            
            # Línea simple para compatibilidad
            f.write(f"{nombre_archivo}\n")
            
            # Línea de metadatos como comentario
            if coincidencias:
                url_info = f" | Cloudinary: {video_url}" if video_url else ""
                f.write(f"# Procesado: {timestamp} | Tipo: {tipo_archivo} | Términos: {', '.join(terminos_encontrados)} | Coincidencias: {len(coincidencias)}{url_info}\n")
            else:
                f.write(f"# Procesado: {timestamp} | Tipo: {tipo_archivo} | Sin coincidencias\n")
        
        # ========== REGISTRAR EN procesados.txt (formato simple) ==========
        procesados_txt = os.path.join(CARPETA_PROCESADOS, "procesados.txt")
        
        # Verificar si el archivo ya está registrado en procesados.txt
        ya_registrado = False
        if os.path.exists(procesados_txt):
            try:
                with open(procesados_txt, "r", encoding="utf-8") as f:
                    contenido = f.read()
                    ya_registrado = nombre_archivo in contenido
            except Exception:
                pass
        
        # Solo agregar si no está ya registrado (evitar duplicados)
        if not ya_registrado:
            with open(procesados_txt, "a", encoding="utf-8") as f:
                f.write(f"{nombre_archivo}\n")
            log_debug(f"✅ Archivo registrado en procesados.txt: {nombre_archivo}", "registrar_archivo_procesado")
        
        log_debug(f"✅ Archivo registrado en procesados.log: {nombre_archivo}", "registrar_archivo_procesado")
        
    except Exception as e:
        st.warning(f"⚠️ Error registrando archivo procesado: {e}")
        log_exception("registrar_archivo_procesado", e, f"Archivo: {nombre_archivo}")

def buscar_videos_nuevos_optimizado(procesados, func_name):
    """
    Busca audios nuevos de forma optimizada usando caché y timestamps
    """
    nuevos = []
    archivos_escaneados = 0
    carpetas_ignoradas = 0
    cache_hits = 0
    
    try:
        # Cargar caché de escaneo
        cache = cargar_cache_escaneo()
        archivos_cache = cache.get('archivos_escaneados', {})
        
        # Limpiar caché de archivos que ya no existen (cada 10 ejecuciones)
        if len(archivos_cache) > 100 and archivos_escaneados % 10 == 0:
            limpiar_cache_escaneo()
            cache = cargar_cache_escaneo()
            archivos_cache = cache.get('archivos_escaneados', {})
        
        # Obtener timestamp del último procesamiento
        ultimo_procesamiento = 0
        if procesados and os.path.exists(PROCESADOS_LOG):
            try:
                stat_info = os.stat(PROCESADOS_LOG)
                ultimo_procesamiento = stat_info.st_mtime
                log_debug(f"Último procesamiento: {datetime.fromtimestamp(ultimo_procesamiento)}", func_name)
            except Exception:
                pass
        
        # Escanear carpetas de forma optimizada
        for root, dirs, files in os.walk(CARPETA_VIDEOS):
            # OPTIMIZACIÓN 1: Ignorar carpetas con clips generados
            marcador_procesado = os.path.join(root, "PROCESADO.txt")
            if os.path.exists(marcador_procesado):
                carpetas_ignoradas += 1
                log_debug(f"Carpeta ignorada (clips): {os.path.basename(root)}", func_name)
                dirs.clear()  # No procesar subdirectorios
                continue
            
            # OPTIMIZACIÓN 1.5: Ignorar carpetas de subclips generados
            es_carpeta_subclips = False
            for file in files:
                # Patrón de subclips: YYYYMMDD_HHMMSS_termino_XmYYs.(audio)
                if (file.lower().endswith(AUDIO_EXTENSIONS) and 
                    len(file.split('_')) >= 4 and 
                    file.split('_')[0].isdigit() and 
                    len(file.split('_')[0]) == 8):  # YYYYMMDD
                    es_carpeta_subclips = True
                    break
            
            # También verificar archivos .txt de transcripción de clips
            if not es_carpeta_subclips:
                for file in files:
                    if (file.lower().endswith('.txt') and 
                        len(file.split('_')) >= 4 and 
                        file.split('_')[0].isdigit() and 
                        len(file.split('_')[0]) == 8):  # YYYYMMDD
                        es_carpeta_subclips = True
                        break
            
            if es_carpeta_subclips:
                carpetas_ignoradas += 1
                log_debug(f"Carpeta ignorada (subclips): {os.path.basename(root)}", func_name)
                dirs.clear()  # No procesar subdirectorios
                continue
            
            # OPTIMIZACIÓN 2: Verificar fecha de carpeta (SOLO para optimización, no para filtrar)
            try:
                carpeta_mtime = os.path.getmtime(root)
                # NOTA: Comentado temporalmente para evitar filtrar carpetas que pueden tener archivos nuevos
                # if ultimo_procesamiento > 0 and carpeta_mtime < ultimo_procesamiento - 3600:  # 1 hora de margen
                #     log_debug(f"Carpeta sin cambios: {os.path.basename(root)}", func_name)
                #     continue
            except Exception:
                pass
            
            # Procesar archivos de audio solamente
            for file in files:
                if file.lower().endswith(AUDIO_EXTENSIONS):
                    path_full = os.path.join(root, file)
                    archivos_escaneados += 1
                    
                    try:
                        # OPTIMIZACIÓN 3: Usar caché si está disponible
                        if path_full in archivos_cache:
                            cache_info = archivos_cache[path_full]
                            file_stat = os.stat(path_full)
                            
                            # Verificar si el archivo ha cambiado
                            if (file_stat.st_mtime == cache_info.get('mtime', 0) and 
                                file_stat.st_size == cache_info.get('size', 0)):
                                cache_hits += 1
                                
                                # Si ya fue procesado según caché, saltar
                                if cache_info.get('procesado', False):
                                    log_debug(f"Archivo en caché (procesado): {file}", func_name)
                                    continue
                                
                                # Si es muy pequeño según caché, saltar
                                if cache_info.get('size', 0) < TAMANO_MINIMO_BYTES:
                                    log_debug(f"Archivo en caché (muy pequeño): {file}", func_name)
                                    continue
                        
                        # OPTIMIZACIÓN 4: Verificar tamaño mínimo
                        file_size = os.path.getsize(path_full)
                        if file_size < TAMANO_MINIMO_BYTES:
                            # Actualizar caché
                            file_stat = os.stat(path_full)
                            archivos_cache[path_full] = {
                                'mtime': file_stat.st_mtime,
                                'size': file_size,
                                'procesado': False,
                                'muy_pequeño': True
                            }
                            continue
                        
                        # OPTIMIZACIÓN 5: Verificar si ya está procesado
                        rel_path = os.path.relpath(path_full, CARPETA_VIDEOS)
                        nombre_archivo_solo = os.path.basename(rel_path)
                        if rel_path in procesados or nombre_archivo_solo in procesados:
                            # Actualizar caché
                            file_stat = os.stat(path_full)
                            archivos_cache[path_full] = {
                                'mtime': file_stat.st_mtime,
                                'size': file_size,
                                'procesado': True
                            }
                            log_debug(f"Audio ya procesado: {file}", func_name)
                            continue
                        
                        # OPTIMIZACIÓN 6: Verificar fecha del archivo (SOLO para caché, no para filtrar)
                        file_stat = os.stat(path_full)
                        file_mtime = max(file_stat.st_mtime, file_stat.st_ctime)
                        
                        # NOTA: No filtrar por fecha del archivo, solo verificar si ya está procesado
                        # La lógica anterior estaba excluyendo archivos que deberían procesarse
                        
                        # ¡Este archivo es NUEVO y debe procesarse!
                        nuevos.append(path_full)
                        log_info(f"✨ Audio NUEVO detectado: {rel_path}", func_name)
                        
                        # Actualizar caché
                        archivos_cache[path_full] = {
                            'mtime': file_stat.st_mtime,
                            'size': file_size,
                            'procesado': False,
                            'detectado_como_nuevo': True
                        }
                            
                    except Exception as e:
                        log_warning(f"Error verificando {file}: {e}", func_name)
                        continue
        
        # Guardar caché actualizado
        cache['archivos_escaneados'] = archivos_cache
        guardar_cache_escaneo(cache)
        
        # LOG DETALLADO PARA DIAGNÓSTICO
        log_info(f"=== DIAGNÓSTICO DE BÚSQUEDA OPTIMIZADA ===", func_name)
        log_info(f"Archivos escaneados: {archivos_escaneados}", func_name)
        log_info(f"Cache hits: {cache_hits}", func_name)
        log_info(f"Carpetas ignoradas: {carpetas_ignoradas}", func_name)
        log_info(f"Archivos NUEVOS encontrados: {len(nuevos)}", func_name)
        
        # Mostrar algunos ejemplos si hay archivos nuevos
        if nuevos:
            log_info("Ejemplos de archivos nuevos (optimizada):", func_name)
            for i, archivo in enumerate(nuevos[:3]):  # Mostrar máximo 3 ejemplos
                rel_path = os.path.relpath(archivo, CARPETA_VIDEOS)
                try:
                    size_mb = os.path.getsize(archivo) / (1024*1024)
                    log_info(f"  {i+1}. {rel_path} ({size_mb:.1f}MB)", func_name)
                except:
                    log_info(f"  {i+1}. {rel_path} (error obteniendo tamaño)", func_name)
        
        log_info(f"=== FIN DIAGNÓSTICO OPTIMIZADA ===", func_name)
        
    except Exception as e:
        log_exception(func_name, e, "Error en búsqueda optimizada")
        log_info("Activando búsqueda tradicional como fallback", func_name)
        return buscar_videos_tradicional(procesados, func_name)
    
    return nuevos

def buscar_videos_tradicional(procesados, func_name):
    """
    Búsqueda tradicional como fallback si la optimizada falla
    """
    archivos = []
    archivos_muy_pequeños = []
    archivos_en_carpetas_procesadas = []
    archivos_ya_procesados = []
    
    log_info("Ejecutando búsqueda tradicional de audios", func_name)
    
    for root, _, files in os.walk(CARPETA_VIDEOS):
        # IGNORAR carpetas que contienen clips generados
        marcador_procesado = os.path.join(root, "PROCESADO.txt")
        if os.path.exists(marcador_procesado):
            for file in files:
                if file.lower().endswith(AUDIO_EXTENSIONS):
                    archivos_en_carpetas_procesadas.append(os.path.join(root, file))
            continue
        
        # IGNORAR carpetas de subclips generados
        es_carpeta_subclips = False
        for file in files:
            # Patrón de subclips: YYYYMMDD_HHMMSS_termino_XmYYs.mp4/mp3
            if (file.lower().endswith(AUDIO_EXTENSIONS) and 
                len(file.split('_')) >= 4 and 
                file.split('_')[0].isdigit() and 
                len(file.split('_')[0]) == 8):  # YYYYMMDD
                es_carpeta_subclips = True
                break
        
        # También verificar archivos .txt de transcripción de clips
        if not es_carpeta_subclips:
            for file in files:
                if (file.lower().endswith('.txt') and 
                    len(file.split('_')) >= 4 and 
                    file.split('_')[0].isdigit() and 
                    len(file.split('_')[0]) == 8):  # YYYYMMDD
                    es_carpeta_subclips = True
                    break
        
        if es_carpeta_subclips:
            continue
            
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                path_full = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(path_full)
                    if file_size >= TAMANO_MINIMO_BYTES:
                        archivos.append(path_full)
                    else:
                        archivos_muy_pequeños.append(path_full)
                        log_debug(f"Archivo muy pequeño ignorado: {file} ({file_size / (1024*1024):.1f}MB)", func_name)
                except Exception as e:
                    log_warning(f"Error verificando archivo {file}: {e}", func_name)
                    continue

    # Filtrar audios ya procesados
    nuevos = []
    for f in archivos:
        rel_path = os.path.relpath(f, CARPETA_VIDEOS)
        nombre_archivo_solo = os.path.basename(rel_path)
        if rel_path not in procesados and nombre_archivo_solo not in procesados:
            nuevos.append(f)
        else:
            archivos_ya_procesados.append(f)
    
    # LOG DETALLADO PARA DIAGNÓSTICO
    log_info(f"=== DIAGNÓSTICO DE BÚSQUEDA TRADICIONAL ===", func_name)
    log_info(f"Archivos encontrados (>={TAMANO_MINIMO_BYTES/(1024*1024):.0f}MB): {len(archivos)}", func_name)
    log_info(f"Archivos muy pequeños (<{TAMANO_MINIMO_BYTES/(1024*1024):.0f}MB): {len(archivos_muy_pequeños)}", func_name)
    log_info(f"Archivos en carpetas procesadas: {len(archivos_en_carpetas_procesadas)}", func_name)
    log_info(f"Archivos ya procesados: {len(archivos_ya_procesados)}", func_name)
    log_info(f"Archivos NUEVOS para procesar: {len(nuevos)}", func_name)
    
    # Mostrar algunos ejemplos si hay archivos nuevos
    if nuevos:
        log_info("Ejemplos de archivos nuevos encontrados:", func_name)
        for i, archivo in enumerate(nuevos[:3]):  # Mostrar máximo 3 ejemplos
            rel_path = os.path.relpath(archivo, CARPETA_VIDEOS)
            size_mb = os.path.getsize(archivo) / (1024*1024)
            log_info(f"  {i+1}. {rel_path} ({size_mb:.1f}MB)", func_name)
    
    # Mostrar ejemplos de archivos muy pequeños si los hay
    if archivos_muy_pequeños:
        log_info("Ejemplos de archivos muy pequeños ignorados:", func_name)
        for i, archivo in enumerate(archivos_muy_pequeños[:3]):  # Mostrar máximo 3 ejemplos
            rel_path = os.path.relpath(archivo, CARPETA_VIDEOS)
            size_mb = os.path.getsize(archivo) / (1024*1024)
            log_info(f"  {i+1}. {rel_path} ({size_mb:.1f}MB)", func_name)
    
    log_info(f"=== FIN DIAGNÓSTICO ===", func_name)
    
    return nuevos

def escanear_carpeta_completa():
    """
    Escanea toda la carpeta y genera estadísticas completas
    """
    func_name = "escanear_carpeta_completa"
    log_info("Iniciando escaneo completo de la carpeta", func_name)
    
    # Estadísticas
    total_archivos = 0
    audios_encontrados = 0
    archivos_procesados = 0
    archivos_nuevos = 0
    archivos_muy_pequeños = 0
    
    # Cargar lista de archivos ya procesados
    procesados = cargar_procesados()
    
    # Mostrar progreso del escaneo
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔍 **ESCANEO COMPLETO DE CARPETA**")
        progress_bar = st.progress(0, text="Iniciando escaneo...")
    
    try:
        # Escanear toda la carpeta
        for root, dirs, files in os.walk(CARPETA_VIDEOS):
            # Ignorar carpetas con clips generados
            marcador_procesado = os.path.join(root, "PROCESADO.txt")
            if os.path.exists(marcador_procesado):
                continue
            
            # IGNORAR CARPETAS DE SUBCLIPS GENERADOS
            # Verificar si esta carpeta contiene subclips (archivos con timestamp en el nombre)
            es_carpeta_subclips = False
            for file in files:
                # Patrón de subclips: YYYYMMDD_HHMMSS_termino_XmYYs.mp4
                if (file.lower().endswith(AUDIO_EXTENSIONS) and 
                    len(file.split('_')) >= 4 and 
                    file.split('_')[0].isdigit() and 
                    len(file.split('_')[0]) == 8):  # YYYYMMDD
                    es_carpeta_subclips = True
                    break
            
            # También verificar si hay archivos .txt de transcripción de clips
            if not es_carpeta_subclips:
                for file in files:
                    if (file.lower().endswith('.txt') and 
                        len(file.split('_')) >= 4 and 
                        file.split('_')[0].isdigit() and 
                        len(file.split('_')[0]) == 8):  # YYYYMMDD
                        es_carpeta_subclips = True
                        break
            
            if es_carpeta_subclips:
                continue
            
            for file in files:
                if file.lower().endswith(AUDIO_EXTENSIONS):
                    total_archivos += 1
                    path_full = os.path.join(root, file)
                    rel_path = os.path.relpath(path_full, CARPETA_VIDEOS)
                    
                    # Contar audios detectados
                    if file.lower().endswith(AUDIO_EXTENSIONS):
                        audios_encontrados += 1
                    
                    # Verificar tamaño
                    try:
                        file_size = os.path.getsize(path_full)
                        if file_size < TAMANO_MINIMO_BYTES:
                            archivos_muy_pequeños += 1
                            continue
                    except Exception:
                        continue
                    
                    # Verificar si ya fue procesado
                    # También verificar solo el nombre del archivo (sin ruta) por compatibilidad
                    nombre_archivo_solo = os.path.basename(rel_path)
                    if rel_path in procesados or nombre_archivo_solo in procesados:
                        archivos_procesados += 1
                    else:
                        archivos_nuevos += 1
                    
                    # Actualizar progreso cada 10 archivos
                    if total_archivos % 10 == 0:
                        progress_bar.progress(min(0.9, total_archivos / 100), 
                                            text=f"Escaneados {total_archivos} archivos...")
        
        progress_bar.progress(1.0, text="Escaneo completado")
        
        # Mostrar estadísticas detalladas
        st.markdown("---")
        st.markdown("### 📊 **ESTADÍSTICAS DEL ESCANEO**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Total Archivos", total_archivos)
            st.metric("🎧 Audios", audios_encontrados)
        
        with col2:
            st.metric("✅ Ya Procesados", archivos_procesados)
            st.metric("🆕 Nuevos", archivos_nuevos)
            st.metric("📏 Muy Pequeños", archivos_muy_pequeños)
        
        with col3:
            porcentaje_procesados = (archivos_procesados / max(1, total_archivos - archivos_muy_pequeños)) * 100
            st.metric("📈 % Procesados", f"{porcentaje_procesados:.1f}%")
            
            if archivos_nuevos > 0:
                st.success(f"🚀 **{archivos_nuevos} archivos nuevos** listos para procesar")
            else:
                st.info("✅ **Todos los archivos ya fueron procesados**")
        
        with col4:
            # Información de archivos de procesados
            try:
                info_archivos = []
                
                # Contar en procesados.log
                if os.path.exists(PROCESADOS_LOG):
                    with open(PROCESADOS_LOG, 'r', encoding='utf-8') as f:
                        lineas = f.readlines()
                    log_count = len([l for l in lineas if not l.startswith('#') and not l.startswith('[') and not l.startswith('=') and l.strip()])
                    info_archivos.append(f"📄 Log: {log_count}")
                
                # Contar en procesados.txt
                procesados_txt = os.path.join(CARPETA_PROCESADOS, "procesados.txt")
                if os.path.exists(procesados_txt):
                    with open(procesados_txt, 'r', encoding='utf-8') as f:
                        lineas = f.readlines()
                    txt_count = len([l for l in lineas if not l.startswith('#') and l.strip()])
                    info_archivos.append(f"📝 TXT: {txt_count}")
                
                if info_archivos:
                    st.metric("📚 Registros", " | ".join(info_archivos))
                else:
                    st.metric("📚 Registros", "No hay")
            except Exception:
                st.metric("📚 Registros", "Error")
        
        # Mostrar detalles de archivos procesados
        if archivos_procesados > 0:
            with st.expander("📋 Ver archivos ya procesados"):
                archivos_procesados_lista = []
                for root, dirs, files in os.walk(CARPETA_VIDEOS):
                    marcador_procesado = os.path.join(root, "PROCESADO.txt")
                    if os.path.exists(marcador_procesado):
                        continue
                    
                    # Ignorar carpetas de subclips
                    es_carpeta_subclips = False
                    for file in files:
                        if (file.lower().endswith(AUDIO_EXTENSIONS) and 
                            len(file.split('_')) >= 4 and 
                            file.split('_')[0].isdigit() and 
                            len(file.split('_')[0]) == 8):
                            es_carpeta_subclips = True
                            break
                    
                    if not es_carpeta_subclips:
                        for file in files:
                            if (file.lower().endswith('.txt') and 
                                len(file.split('_')) >= 4 and 
                                file.split('_')[0].isdigit() and 
                                len(file.split('_')[0]) == 8):
                                es_carpeta_subclips = True
                                break
                    
                    if es_carpeta_subclips:
                        continue
                    
                    for file in files:
                        if file.lower().endswith(AUDIO_EXTENSIONS):
                            path_full = os.path.join(root, file)
                            rel_path = os.path.relpath(path_full, CARPETA_VIDEOS)
                            if rel_path in procesados:
                                archivos_procesados_lista.append(rel_path)
                
                if archivos_procesados_lista:
                    for archivo in sorted(archivos_procesados_lista)[:20]:  # Mostrar máximo 20
                        st.text(f"✅ {archivo}")
                    if len(archivos_procesados_lista) > 20:
                        st.text(f"... y {len(archivos_procesados_lista) - 20} más")
        
        # Recopilar lista de archivos nuevos (fuera del expander para poder retornarla)
        archivos_nuevos_lista = []
        # Recopilar lista de archivos omitidos por ser menores al tamaño mínimo
        archivos_muy_pequenos_lista = []
        for root, dirs, files in os.walk(CARPETA_VIDEOS):
            marcador_procesado = os.path.join(root, "PROCESADO.txt")
            if os.path.exists(marcador_procesado):
                continue
            
            # Ignorar carpetas de subclips
            es_carpeta_subclips = False
            for file in files:
                if (file.lower().endswith(AUDIO_EXTENSIONS) and 
                    len(file.split('_')) >= 4 and 
                    file.split('_')[0].isdigit() and 
                    len(file.split('_')[0]) == 8):
                    es_carpeta_subclips = True
                    break
            
            if not es_carpeta_subclips:
                for file in files:
                    if (file.lower().endswith('.txt') and 
                        len(file.split('_')) >= 4 and 
                        file.split('_')[0].isdigit() and 
                        len(file.split('_')[0]) == 8):
                        es_carpeta_subclips = True
                        break
            
            if es_carpeta_subclips:
                continue
            
            for file in files:
                if file.lower().endswith(AUDIO_EXTENSIONS):
                    path_full = os.path.join(root, file)
                    rel_path = os.path.relpath(path_full, CARPETA_VIDEOS)
                    try:
                        file_size = os.path.getsize(path_full)
                        if file_size >= TAMANO_MINIMO_BYTES and rel_path not in procesados:
                            archivos_nuevos_lista.append(rel_path)
                        elif file_size < TAMANO_MINIMO_BYTES:
                            archivos_muy_pequenos_lista.append(rel_path)
                    except Exception:
                        continue
        
        # Mostrar detalles de archivos nuevos
        if archivos_nuevos > 0:
            with st.expander("🆕 Ver archivos nuevos"):
                if archivos_nuevos_lista:
                    for archivo in sorted(archivos_nuevos_lista):
                        icono = "🎧"
                        st.text(f"{icono} {archivo}")
        
        # Mostrar lista de omitidos por tamaño mínimo
        if archivos_muy_pequenos_lista:
            umbral_mb = TAMANO_MINIMO_BYTES / (1024 * 1024)
            with st.expander(f"🚫 Ver archivos omitidos (< {umbral_mb:.0f} MB)"):
                for archivo in sorted(archivos_muy_pequenos_lista)[:50]:
                    st.text(f"🚫 {archivo}")
                if len(archivos_muy_pequenos_lista) > 50:
                    st.text(f"... y {len(archivos_muy_pequenos_lista) - 50} más")

            # Registrar omitidos en log diario
            try:
                logs_dir = os.path.join(os.getcwd(), "logs")
                os.makedirs(logs_dir, exist_ok=True)
                date_str = datetime.now().strftime("%Y%m%d")
                omitidos_log = os.path.join(logs_dir, f"omitidos_{date_str}.log")
                with open(omitidos_log, "a", encoding="utf-8") as lf:
                    marca_tiempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    lf.write(f"{marca_tiempo} - Omitidos (< {umbral_mb:.0f} MB): {len(archivos_muy_pequenos_lista)}\n")
                    for rel in archivos_muy_pequenos_lista:
                        lf.write(f"- {rel}\n")
            except Exception as e:
                log_warning(f"No se pudo escribir omitidos en log: {e}", func_name)
        
        log_info(f"Escaneo completado: {total_archivos} total, {archivos_nuevos} nuevos, {archivos_procesados} procesados", func_name)
        
        return {
            'total_archivos': total_archivos,
            'total_videos': audios_encontrados,
            'archivos_procesados': archivos_procesados,
            'archivos_nuevos': archivos_nuevos,
            'archivos_muy_pequeños': archivos_muy_pequeños,
            'archivos_muy_pequeños_lista': archivos_muy_pequenos_lista,
            'procesados': procesados,
            'archivos_nuevos_lista': archivos_nuevos_lista
        }
        
    except Exception as e:
        log_exception(func_name, e, "Error en escaneo completo")
        st.error(f"❌ Error durante el escaneo: {e}")
        return None

def _countdown_seguro(container, segundos, mensaje_template, func_name="countdown"):
    """
    Countdown que no se congela si la sesión de Streamlit se cierra.
    Retorna True si completó normalmente, False si la sesión murió.
    
    mensaje_template debe contener {i} para el número de segundos restantes.
    Ejemplo: "💤 Re-escaneando en {i}s..."
    """
    for i in range(segundos, 0, -1):
        try:
            container.info(mensaje_template.format(i=i))
            # Sleep corto (2 x 0.5s) para poder reaccionar rápido a sesión cerrada
            time.sleep(0.5)
            # Verificar que la sesión sigue viva accediendo a session_state
            _ = st.session_state.running
            time.sleep(0.5)
        except Exception as e:
            log_warning(f"Countdown interrumpido (sesión probablemente cerrada): {e}", func_name)
            try:
                container.empty()
            except Exception:
                pass
            return False  # Sesión muerta
    try:
        container.empty()
    except Exception:
        pass
    return True  # Completó normalmente

def buscar_y_procesar_videos(duracion_clip=90, buffer_anterior=30):
    func_name = "buscar_y_procesar_videos"
    mostrar_solo_relevantes = st.session_state.get('mostrar_solo_actual_relevantes', True)
    log_info(f"Iniciando búsqueda y procesamiento de videos. Duración clip: {duracion_clip}s, Buffer: {buffer_anterior}s", func_name)
    
    # ========== CONTROL DE EJECUCIÓN MÚLTIPLE ==========
    # Verificar si ya hay una ejecución en curso para evitar duplicados
    if hasattr(st.session_state, 'procesamiento_en_curso') and st.session_state.procesamiento_en_curso:
        log_warning("⚠️ PROCESAMIENTO YA EN CURSO - Evitando ejecución duplicada", func_name)
        st.warning("⚠️ Ya hay un procesamiento en curso. Esperando a que termine...")
        return
    
    # Marcar que el procesamiento está en curso
    st.session_state.procesamiento_en_curso = True
    log_info("🔒 Marcando procesamiento como en curso para evitar duplicados", func_name)
    
    # Limpiar control de duplicados de Supabase para nuevo procesamiento
    st.session_state.coincidencias_enviadas_supabase.clear()
    log_info("🧹 Control de duplicados Supabase limpiado para nuevo procesamiento", func_name)
    
    st.session_state.ultimo_chequeo = datetime.now()
    
    terminos = st.session_state.terminos_continuos
    log_debug(f"Términos configurados: {terminos}", func_name)
    
    if not terminos:
        log_info("No hay términos configurados para buscar", func_name)
        st.warning("⚠️ No hay términos configurados para buscar")
        st.session_state.procesamiento_en_curso = False
        return

    # === ESCANEO COMPLETO ANTES DE PROCESAR ===
    st.markdown("---")
    st.markdown("### 🔍 **ESCANEO COMPLETO DE CARPETA**")
    estadisticas = escanear_carpeta_completa()
    
    if not estadisticas:
        st.error("❌ Error en el escaneo, no se puede continuar")
        st.session_state.procesamiento_en_curso = False
        return
    
    # Mostrar resumen detallado del escaneo
    st.markdown("---")
    st.markdown("### 📊 **RESUMEN DEL ESCANEO**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📁 Total Archivos", estadisticas['total_archivos'])
    
    with col2:
        st.metric("🎧 Audios", estadisticas['total_videos'])
    
    with col3:
        st.metric("📊 Procesados", estadisticas['archivos_procesados'])
    
    with col4:
        st.metric("🆕 Nuevos", estadisticas['archivos_nuevos'])
    
    # Mostrar cuántos se omitieron por ser menores al tamaño mínimo
    with col5:
        try:
            umbral_mb = TAMANO_MINIMO_BYTES / (1024 * 1024)
            st.metric(f"🚫 Omitidos < {umbral_mb:.0f} MB", estadisticas.get('archivos_muy_pequeños', 0))
        except Exception:
            st.metric("🚫 Omitidos por tamaño", estadisticas.get('archivos_muy_pequeños', 0))
    
    # Mostrar archivos nuevos
    if estadisticas['archivos_nuevos'] > 0:
        st.success(f"🆕 **{estadisticas['archivos_nuevos']} ARCHIVOS NUEVOS** encontrados para procesar")
        
        # Mostrar lista de audios nuevos solo cuando no estamos en modo "solo relevantes"
        if not mostrar_solo_relevantes:
            with st.expander("📋 Ver audios nuevos a procesar"):
                for archivo in estadisticas['archivos_nuevos_lista'][:10]:  # Mostrar máximo 10
                    st.text(f"🎧 {archivo}")
                if len(estadisticas['archivos_nuevos_lista']) > 10:
                    st.text(f"... y {len(estadisticas['archivos_nuevos_lista']) - 10} más")
    else:
        st.success("✅ **TODOS LOS AUDIOS YA FUERON PROCESADOS**")
        st.info("💡 No hay audios nuevos para procesar. El sistema está al día.")
        st.caption(
            f"📁 **Carpeta de entrada escaneada:** `{CARPETA_VIDEOS}` · "
            "Si tus grabaciones están en otra ruta, define `RADIO_CARPETA_AUDIOS` en el archivo `.env` "
            "(junto a `radioAnalizer.py`) y **reinicia** la app."
        )
        
        # Si el loop está activo, esperar y re-escanear
        if st.session_state.get('loop_continuo', True):
            espera = st.session_state.get('intervalo_loop_vacio', 120)
            st.info(f"🔄 **LOOP ACTIVO** - Re-escaneando en {espera}s buscando audios nuevos...")
            log_info(f"Loop activo, sin audios nuevos. Esperando {espera}s para re-escanear", func_name)
            
            st.session_state.procesamiento_en_curso = False
            
            wait_container = st.empty()
            sesion_viva = _countdown_seguro(
                wait_container, espera,
                "💤 Sin audios nuevos. Re-escaneando en {i}s...",
                func_name
            )
            
            if not sesion_viva:
                log_warning("Sesión cerrada durante espera sin audios nuevos. Saliendo limpiamente.", func_name)
                return
            
            st.info("🔄 Volviendo a escanear...")
            time.sleep(1)
            try:
                st.rerun()
            except Exception as e:
                log_warning(f"st.rerun() falló: {e}", func_name)
                st.warning("⚠️ Si la sesión se cerró, pulsa para continuar.")
                if st.button("🔄 **CONTINUAR ESCANEO**", key="continuar_escaneo_inicial"):
                    st.rerun()
        else:
            # Loop desactivado - mostrar opción manual
            if st.button("🔄 **FORZAR REPROCESAMIENTO**", 
                        help="Reprocesar todos los audios (ignorar audios ya procesados)",
                        key="forzar_reprocesamiento"):
                st.session_state.forzar_escaneo_completo = True
                st.info("🚀 Forzando reprocesamiento de todos los audios...")
                st.rerun()
            
            st.info("💡 Activa el **Loop Continuo** en el sidebar para escanear automáticamente.")
            st.session_state.procesamiento_en_curso = False
            return
    
    # Continuar con el procesamiento de archivos nuevos
    st.markdown("---")
    st.success(f"🚀 **INICIANDO PROCESAMIENTO** - {estadisticas['archivos_nuevos']} audios nuevos encontrados")
    
    # Usar la lista de procesados del escaneo
    procesados = estadisticas['procesados']
    
    # Verificar si se forzó un escaneo completo
    forzar_escaneo = getattr(st.session_state, 'forzar_escaneo_completo', False)
    
    if forzar_escaneo:
        st.info("🚀 Ejecutando escaneo completo forzado (ignorando archivos ya procesados)")
        nuevos = buscar_videos_tradicional(procesados, func_name)
        st.session_state.forzar_escaneo_completo = False  # Resetear flag
        log_info("Escaneo completo forzado ejecutado", func_name)
    else:
        # Buscar SOLO archivos nuevos de forma más eficiente
        nuevos = buscar_videos_nuevos_optimizado(procesados, func_name)
    
    log_info(f"Archivos nuevos encontrados para procesar: {len(nuevos)}", func_name)
    
    if not nuevos:
        st.info("✅ No hay audios nuevos para procesar después del escaneo")
        st.caption(
            f"📁 Escaneo en: `{CARPETA_VIDEOS}`. Si esperabas archivos de otra carpeta, configura `RADIO_CARPETA_AUDIOS` en `.env` y reinicia."
        )
        st.session_state.procesamiento_en_curso = False
        return

    st.success(f"🆕 Encontrados {len(nuevos)} audios nuevos para procesar")
    st.session_state.videos_encontrados += len(nuevos)
    
    clips_generados_en_sesion = []
    videos_procesados_data = []  # Almacenar datos de todos los archivos procesados con coincidencias
    menciones_tangenciales_data = []  # Rechazos tangenciales / relevancia baja para UI, MD y correo

    # Contenedor para el progreso del audio actual (se reemplaza en cada iteración)
    current_video_container = st.empty()
    st.session_state.ui_status_container = current_video_container

    def _ui_transient(level, message):
        """Muestra mensajes temporales por archivo cuando se usa modo solo relevantes."""
        if mostrar_solo_relevantes:
            with current_video_container.container():
                getattr(st, level)(message)
        else:
            getattr(st, level)(message)

    # FASE 1: PROCESAR TODOS LOS AUDIOS PRIMERO (sin enviar webhooks)
    st.info("🎧 FASE 1: Procesando todos los audios (sin envíos)")
    
    for i, archivo_path in enumerate(nuevos):
        rel = os.path.relpath(archivo_path, CARPETA_VIDEOS)
        
        # BLOQUE TRY-EXCEPT GENERAL PARA CADA ARCHIVO
        try:
            # Detectar tipo de archivo (audio)
            es_audio = archivo_path.lower().endswith(AUDIO_EXTENSIONS)
            
            # Icono para audios
            icono = "🎧"
            tipo_archivo = "Audio"
            
            with current_video_container.container():
                st.info(f"**{i+1}/{len(nuevos)}** — {icono} Procesando: `{rel}`")
                st.markdown(f"### {icono} Procesando {tipo_archivo} ({i+1}/{len(nuevos)}): `{rel}`")
                progress_bar = st.progress(0, text=f"Iniciando... ({i+1}/{len(nuevos)})")
            
            # Configurar rutas para audios: convertir temporalmente a WAV mono 16k
            archivo_base, _ = os.path.splitext(archivo_path)
            audio_path = f"{archivo_base}_tmp.wav"
            md_path = f"{archivo_base}_streaming.md"
            
            dur_total = obtener_duracion(archivo_path)

            # === FILTRO: Saltar audios menores a 5 minutos ===
            DURACION_MINIMA_SEGUNDOS = 300  # 5 minutos
            if dur_total and dur_total < DURACION_MINIMA_SEGUNDOS:
                minutos = int(dur_total // 60)
                segundos = int(dur_total % 60)
                if not mostrar_solo_relevantes:
                    st.warning(f"⏭️ **SALTANDO audio corto:** `{rel}` — Duración: {minutos}:{segundos:02d} (mínimo requerido: 5:00 minutos)")
                log_info(f"Audio saltado por duración corta ({dur_total:.0f}s < 300s): {rel}", func_name)
                registrar_archivo_procesado(os.path.basename(archivo_path), [], "Audio saltado: duración menor a 5 minutos", "audio")
                st.session_state.videos_procesados += 1
                continue

            # Extraer/convertir audio según el tipo de archivo
            progress_bar.progress(20, text="🎧 Procesando audio...")
        
            # Función para procesar audio con reintentos
            def procesar_audio_con_reintentos(archivo_path, audio_path, max_reintentos=3):
                for intento in range(max_reintentos):
                    try:
                        # Verificar que el archivo existe y es accesible
                        if not os.path.exists(archivo_path):
                            if intento < max_reintentos - 1:
                                _ui_transient("warning", f"⚠️ Archivo no encontrado (intento {intento + 1}/{max_reintentos}): {archivo_path}")
                                time.sleep(5)  # Esperar 5 segundos antes del siguiente intento
                                continue
                            else:
                                _ui_transient("error", f"❌ Archivo no encontrado después de {max_reintentos} intentos: {archivo_path}")
                                return False
                    
                        # Verificar tamaño del archivo
                        file_size = os.path.getsize(archivo_path)
                        if file_size == 0:
                            if intento < max_reintentos - 1:
                                _ui_transient("warning", f"⚠️ Archivo vacío (intento {intento + 1}/{max_reintentos}): {archivo_path}")
                                time.sleep(5)  # Esperar 5 segundos antes del siguiente intento
                                continue
                            else:
                                _ui_transient("error", f"❌ Archivo vacío después de {max_reintentos} intentos: {archivo_path}")
                                return False
                    
                        # Convertir cualquier audio de entrada a WAV (mono 16k)
                        cmd = [
                            "ffmpeg", "-y", "-i", archivo_path,
                            "-ac", "1", "-ar", "16000", "-f", "wav", audio_path
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                    
                        if result.returncode != 0:
                            if intento < max_reintentos - 1:
                                # Solo mostrar un mensaje breve en el primer intento
                                if intento == 0:
                                    _ui_transient("warning", f"⚠️ Error procesando: {os.path.basename(archivo_path)} - reintentando...")
                                time.sleep(10)  # Esperar 10 segundos antes del siguiente intento
                                continue
                            else:
                                # Solo mostrar un resumen final en una línea
                                _ui_transient("error", f"❌ No se pudo procesar: {os.path.basename(archivo_path)} - Error FFmpeg: {result.returncode}")
                            
                                # Intentar con parámetros más permisivos
                                _ui_transient("info", "🔄 Intentando con parámetros alternativos...")
                                cmd_alt = [
                                    "ffmpeg", "-y", "-i", archivo_path,
                                    "-ac", "1", "-ar", "16000", "-f", "wav", 
                                    "-avoid_negative_ts", "make_zero", audio_path
                                ]
                                result_alt = subprocess.run(cmd_alt, capture_output=True, text=True)
                            
                                if result_alt.returncode != 0:
                                    _ui_transient("error", f"❌ Error persistente incluso con parámetros alternativos")
                                    return False
                                else:
                                    _ui_transient("success", "✅ Audio extraído con parámetros alternativos")
                                    return True
                        else:
                            _ui_transient("success", "✅ Audio extraído exitosamente")
                            return True
                    
                    except Exception as e:
                        if intento < max_reintentos - 1:
                            if intento == 0:
                                _ui_transient("warning", f"⚠️ Error procesando: {os.path.basename(archivo_path)} - reintentando...")
                            time.sleep(5)  # Esperar 5 segundos antes del siguiente intento
                            continue
                        else:
                            _ui_transient("error", f"❌ Error inesperado: {os.path.basename(archivo_path)} - {str(e)[:50]}...")
                            return False
            
                return False
        
            # Ejecutar procesamiento con reintentos
            if not procesar_audio_con_reintentos(archivo_path, audio_path):
                _ui_transient("warning", f"⚠️ Saltando: {os.path.basename(archivo_path)}")
                continue

            # Transcribir (faster GPU → faster CPU → Mistral → OpenAI)
            progress_bar.progress(40, text=f"🧠 Transcribiendo ({i+1}/{len(nuevos)})...")
            start = time.time()
            try:
                transcripcion_mistral, api_usada, segments_timestamps = transcribir_audio_hibrido(
                    audio_path, indice_actual=i+1, total=len(nuevos)
                )
                if not mostrar_solo_relevantes:
                    st.info(f"🎯 Transcripción completada con {api_usada} — **{i+1}/{len(nuevos)}**")
            except Exception as e:
                _ui_transient("error", f"❌ Error en transcripción: {e}")
                continue
            elapsed_mistral = time.time() - start

            # Timestamps: si ya los tenemos (faster/Mistral), usar; si no (OpenAI), obtener con whisper
            if segments_timestamps is None or len(segments_timestamps) == 0:
                progress_bar.progress(55, text="🕐 Obteniendo timestamps (OpenAI no los incluye)...")
                start = time.time()
                try:
                    segments_timestamps = obtener_timestamps_whisper(audio_path)
                except Exception as e:
                    _ui_transient("error", f"❌ Error obteniendo timestamps: {e}")
                    continue
                elapsed_whisper = time.time() - start
            else:
                if not mostrar_solo_relevantes:
                    st.success(f"✅ Timestamps incluidos en la transcripción — sin paso adicional")

            progress_bar.progress(80, text="🔍 Buscando coincidencias...")
        
            coincidencias_md = []
            coincidencias_items = []
            # Crear carpeta principal del archivo (una sola vez)
            archivo_name_clean = os.path.splitext(rel)[0]  # Sin extensión
            archivo_name_safe = "".join(c for c in archivo_name_clean if c.isalnum() or c in (' ', '-', '_')).rstrip()
            archivo_name_safe = archivo_name_safe.replace(' ', '_')[:50]  # Máximo 50 caracteres
            fecha_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        
            # CARPETA PRINCIPAL DEL ARCHIVO (contiene todos los clips) - EN CARPETA PROCESADOS
            archivo_main_dir = os.path.join(CARPETA_PROCESADOS, f"c_{archivo_name_safe}_{fecha_folder}")
            os.makedirs(archivo_main_dir, exist_ok=True)
            log_info(f"Carpeta de coincidencias creada: {archivo_main_dir}", func_name)
        
            # Crear archivo marcador P* en la carpeta principal
            marcador_path = os.path.join(archivo_main_dir, "PROCESADO.txt")
            if not os.path.exists(marcador_path):
                # Extraer nombres de términos (soporta dict y string)
                terminos_nombres = [t.get('termino', str(t)) if isinstance(t, dict) else str(t) for t in terminos]
                with open(marcador_path, "w", encoding="utf-8") as f:
                    f.write(f"🚫 CARPETA PROCESADA - NO REPROCESAR\n")
                    f.write(f"Fecha creación: {datetime.now().isoformat()}\n")
                    f.write(f"Archivo origen: {rel} ({tipo_archivo})\n")
                    f.write(f"Términos encontrados: {', '.join(terminos_nombres)}\n")
                    f.write(f"Generado por: Radio Analyzer IA v2.0\n")
            
            # ========== GUARDAR TRANSCRIPCIÓN COMPLETA DEL VIDEO ==========
            transcripcion_completa_path = os.path.join(archivo_main_dir, "TRANSCRIPCION_COMPLETA.txt")
            if not os.path.exists(transcripcion_completa_path):
                try:
                    # Extraer nombres de términos (soporta dict y string)
                    terminos_nombres_transcripcion = [t.get('termino', str(t)) if isinstance(t, dict) else str(t) for t in terminos]
                    with open(transcripcion_completa_path, "w", encoding="utf-8") as f:
                        f.write(f"📝 TRANSCRIPCIÓN COMPLETA DEL VIDEO\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(f"📹 VIDEO: {rel}\n")
                        f.write(f"📅 FECHA DE ANÁLISIS: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"🧠 MOTOR DE TRANSCRIPCIÓN: {api_usada}\n")
                        f.write(f"⏱️ DURACIÓN TOTAL: {dur_total:.1f} segundos ({int(dur_total//60)}:{int(dur_total%60):02d})\n")
                        f.write(f"🔍 TÉRMINOS BUSCADOS: {', '.join(terminos_nombres_transcripcion)}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"TRANSCRIPCIÓN:\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(transcripcion_mistral)
                        f.write(f"\n\n{'='*80}\n")
                        f.write(f"📊 ESTADÍSTICAS:\n")
                        f.write(f"- Total de palabras: {len(transcripcion_mistral.split())}\n")
                        f.write(f"- Total de caracteres: {len(transcripcion_mistral)}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"✅ Generado automáticamente por Radio Analyzer IA v2.0\n")
                    
                    log_info(f"✅ Transcripción completa guardada: {transcripcion_completa_path}", func_name)
                except Exception as e:
                    log_warning(f"Error guardando transcripción completa: {e}", func_name)

            # Buscar términos - VERSIÓN CORREGIDA Y MEJORADA
            text_lower = transcripcion_mistral.lower()
        
            # ========== ALIASES DE TÉRMINOS (SINÓNIMOS) ==========
            # Diccionario que mapea variantes/errores comunes a sus términos correctos
            # Formato: {'termino_buscado': ['alias1', 'alias2', ...]}
            # Si buscas "intrant", también buscará "intran"
            ALIASES_TERMINOS = {
                'intrant': ['intran', 'in tran', 'in trant', 'intrans'],
                'edesur': ['ede sur', 'ede-sur'],
                'edenorte': ['ede norte', 'ede-norte'],
                'edeeste': ['ede este', 'ede-este'],
                'celso marranzini': ['celso marrancini', 'celsomarranzini', 'celsomarrancini'],
            }
        
            # Función auxiliar para buscar término con variaciones
            def buscar_termino_flexible(termino, texto):
                """Busca un término considerando variaciones comunes y aliases"""
                # Búsqueda exacta primero
                if re.search(rf"\b{re.escape(termino)}\b", texto):
                    return True, termino
            
                # 🆕 BUSCAR ALIASES (sinónimos/variantes conocidas)
                termino_lower = termino.lower()
                aliases_buscar = ALIASES_TERMINOS.get(termino_lower, [])
                for alias in aliases_buscar:
                    if re.search(rf"\b{re.escape(alias)}\b", texto):
                        return True, termino  # Retorna el término original, no el alias
                
                # 🆕 BUSCAR SI EL TÉRMINO ES UN ALIAS DE OTRO
                # Ejemplo: si buscas "intran", verificar si es alias de "intrant"
                for termino_principal, lista_aliases in ALIASES_TERMINOS.items():
                    if termino_lower in lista_aliases:
                        # Buscar el término principal
                        if re.search(rf"\b{re.escape(termino_principal)}\b", texto):
                            return True, termino
            
                # Buscar variaciones comunes (plurales, conjugaciones básicas)
                variaciones = [
                    termino + "s",  # plural
                    termino + "es", # plural alternativo
                    termino + "a",  # género femenino
                    termino + "o",  # género masculino
                ]
            
                for variacion in variaciones:
                    if re.search(rf"\b{re.escape(variacion)}\b", texto):
                        return True, variacion
                
                # 🆕 NORMALIZACIÓN DE ESPACIOS: Buscar sin espacios
                # Ejemplo: "celsomarrancini" encontrará "celso marranzini" o "Celsomarrancini"
                # PERO NO hará match parcial: "edesur" NO encontrará "desur"
                termino_sin_espacios = termino.replace(" ", "").lower()
                
                # Buscar palabras consecutivas que al juntar coincidan EXACTAMENTE
                palabras = texto.split()
                for i in range(len(palabras)):
                    for j in range(i+1, min(i+6, len(palabras)+1)):  # Buscar hasta 5 palabras consecutivas
                        fragmento_sin_espacios = "".join(palabras[i:j]).lower()
                        # Match EXACTO, no substring
                        if termino_sin_espacios == fragmento_sin_espacios:
                            termino_encontrado = " ".join(palabras[i:j])
                            return True, termino_encontrado
            
                return False, None
        
            # ========== TÉRMINOS PRIORITARIOS ==========
            # Estos términos SIEMPRE deben generar clips cuando se mencionen
            # No se aplicarán verificaciones estrictas de relevancia a estos términos
            TERMINOS_PRIORITARIOS = {
                'edesur', 'edenorte', 'edeeste',
                'punta catalina',
                'apagones',
                'egehid', 'ede hid',
                'celso marranzini', 'celso', 'marranzini', 'celsomarrancini',
                'protecom',
                'pegase', 'pégase'
            }
            
            # ========== CONTROL DE DUPLICADOS MEJORADO ==========
            # Lista para rastrear timestamps ya procesados para evitar clips duplicados
            timestamps_procesados = []
            # Lista para rastrear combinaciones de término + timestamp ya procesadas
            coincidencias_procesadas = set()
        
            for termino_item in terminos:
                # Extraer el string del término (soporta dict y string)
                if isinstance(termino_item, dict):
                    termino = termino_item.get('termino', '')
                else:
                    termino = str(termino_item)
                
                if not termino:
                    continue
                    
                # PRIMERA VERIFICACIÓN: ¿El término (o variaciones) existe en la transcripción completa?
                encontrado, termino_encontrado = buscar_termino_flexible(termino, text_lower)
            
                if encontrado:
                    log_info(f"Término '{termino}' encontrado en transcripción completa", func_name)
                    
                    # Verificar si es un término prioritario
                    es_prioritario = termino.lower() in TERMINOS_PRIORITARIOS
                    if es_prioritario:
                        log_info(f"⭐ '{termino}' es un TÉRMINO PRIORITARIO - se aplicarán reglas flexibles", func_name)
                        st.info(f"⭐ Término prioritario detectado: '{termino}'")
                
                    mejor_timestamp = None
                    mejor_texto_contexto = ""
                
                    # SEGUNDA VERIFICACIÓN: ¿El término existe en algún segmento específico con timestamp?
                    for seg in segments_timestamps:
                        seg_encontrado, seg_termino_encontrado = buscar_termino_flexible(termino, seg['text'].lower())
                        if seg_encontrado:
                            mejor_timestamp = seg
                            mejor_texto_contexto = seg['text']
                            log_info(f"Término '{termino}' (variante: '{seg_termino_encontrado}') encontrado en segmento: {seg['text'][:100]}...", func_name)
                            break
                
                    # VERIFICACIÓN CRÍTICA: Solo continuar si encontramos el término en un segmento específico
                    # EXCEPCIÓN: Para términos prioritarios, buscar en todos los segmentos y usar el primero disponible
                    if not mejor_timestamp:
                        if es_prioritario and segments_timestamps:
                            # Para términos prioritarios, usar el segmento central como fallback
                            mejor_timestamp = segments_timestamps[len(segments_timestamps) // 2]
                            mejor_texto_contexto = mejor_timestamp['text']
                            log_info(f"⭐ TÉRMINO PRIORITARIO '{termino}': Usando segmento central como fallback", func_name)
                            st.info(f"⭐ Término prioritario '{termino}': Generando clip desde segmento representativo")
                        else:
                            log_warning(f"⚠️ TÉRMINO '{termino}' ENCONTRADO EN TRANSCRIPCIÓN GENERAL PERO NO EN SEGMENTOS ESPECÍFICOS", func_name)
                            log_warning(f"   - Esto puede indicar error de transcripción o segmentación", func_name)
                            log_warning(f"   - NO se generará clip para evitar falsos positivos", func_name)
                            st.warning(f"⚠️ Término '{termino}' encontrado en transcripción general pero no en momento específico - OMITIDO")
                            continue  # ❌ NO GENERAR CLIP - SALTAR AL SIGUIENTE TÉRMINO
                
                    # ========== CONTROL DE DUPLICADOS MEJORADO ==========
                    timestamp_actual = mejor_timestamp['start']
                    
                    # Crear clave única para esta coincidencia (término + timestamp + archivo)
                    clave_coincidencia = f"{termino}_{timestamp_actual:.1f}_{rel}"
                    
                    # Verificar si ya procesamos esta coincidencia exacta
                    if clave_coincidencia in coincidencias_procesadas:
                        log_info(f"⏭️ DUPLICADO DETECTADO: '{termino}' en {timestamp_actual:.1f}s ya procesado para {rel}", func_name)
                        st.info(f"⏭️ Coincidencia duplicada evitada: '{termino}' en {timestamp_actual:.1f}s")
                        continue  # ❌ NO GENERAR CLIP DUPLICADO
                    
                    # Verificar si ya procesamos un clip para este timestamp (tolerancia de ±60 segundos - 1 minuto)
                    es_duplicado = False
                    for ts_procesado in timestamps_procesados:
                        diferencia = abs(timestamp_actual - ts_procesado)
                        if diferencia <= 60:  # Tolerancia de 60 segundos (1 minuto) para evitar clips repetitivos
                            es_duplicado = True
                            log_info(f"⏭️ Término '{termino}' OMITIDO - Ya existe clip para timestamp similar ({diferencia:.1f}s de diferencia, mínimo requerido: 60s)", func_name)
                            st.info(f"⏭️ Término '{termino}' omitido - Ya existe clip reciente (separación mínima: 1 minuto)")
                            break
                
                    if es_duplicado:
                        continue  # ❌ NO GENERAR CLIP DUPLICADO - SALTAR AL SIGUIENTE TÉRMINO
                
                    # Agregar a las listas de control
                    timestamps_procesados.append(timestamp_actual)
                    coincidencias_procesadas.add(clave_coincidencia)
                    log_info(f"✅ Timestamp {timestamp_actual}s agregado a lista de procesados", func_name)
                    log_info(f"✅ Coincidencia '{clave_coincidencia}' registrada para evitar duplicados", func_name)

                    m, s = divmod(int(mejor_timestamp['start']), 60)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                    # SUBCARPETA PARA ESTE TÉRMINO ESPECÍFICO
                    clip_dir = os.path.join(archivo_main_dir, f"c_clip_{termino}")
                    os.makedirs(clip_dir, exist_ok=True)
                    log_info(f"Subcarpeta de clip creada: {clip_dir}", func_name)

                    # ========== 🌟 USAR GEMINI 3 PRO PARA DETERMINAR SEGMENTO INTELIGENTE ==========
                    momento_termino = mejor_timestamp['start']
                    
                    st.info(f"🌟 Analizando con Gemini 3 Pro el mejor segmento para '{termino}'...")
                    log_info(f"🌟 Solicitando análisis GEMINI 3 PRO para término '{termino}' en timestamp {momento_termino:.1f}s", func_name)
                    
                    # Variable para almacenar la idea central extraída por Gemini
                    idea_central_gemini = ""
                    
                    try:
                        # Llamar a GEMINI 3 PRO para determinar el segmento más lógico
                        segmento_gemini = determinar_segmento_inteligente_gemini(
                            transcripcion_con_timestamps=segments_timestamps,
                            termino_encontrado=termino,
                            timestamp_coincidencia=momento_termino,
                            duracion_maxima=duracion_clip
                        )
                        
                        # 🚫 VERIFICAR SI GEMINI/GPT RECHAZÓ EL SEGMENTO (dict) o respuesta ausente (None)
                        rechazo_seg = (
                            segmento_gemini is None
                            or (isinstance(segmento_gemini, dict) and segmento_gemini.get('_rechazo_segmento'))
                        )
                        if rechazo_seg:
                            if isinstance(segmento_gemini, dict) and segmento_gemini.get('_rechazo_segmento'):
                                razon_t = segmento_gemini.get('razon', 'Mención tangencial sin desarrollo')
                            else:
                                razon_t = 'Mención tangencial sin desarrollo'
                            log_warning(f"🚫 Modelo rechazó el segmento para '{termino}': {razon_t}", func_name)
                            evidencia_tang = extraer_texto_transcripcion_ventana(
                                segments_timestamps, momento_termino, ventana_seg=120, duracion_audio=dur_total
                            )
                            if not evidencia_tang.strip():
                                evidencia_tang = (mejor_texto_contexto or "").strip()
                            excerpt_ui = (evidencia_tang or "").strip()
                            if len(excerpt_ui) > 420:
                                excerpt_ui = excerpt_ui[:420].rsplit(" ", 1)[0] + "…"
                            tang_ui = (
                                f"🚫 **Tangencial (sin clip)** — término `{termino}` en `{rel}`\n\n"
                                f"**Motivo del modelo:** {razon_t}"
                            )
                            if excerpt_ui:
                                tang_ui += f"\n\n**Contexto en audio:** {excerpt_ui}"
                            if not deepseek_tangenciales_activo():
                                tang_ui += (
                                    "\n\n*(Análisis enriquecido DeepSeek desactivado o sin API: "
                                    "solo motivo del clasificador Gemini/GPT.)*"
                                )
                            st.warning(tang_ui)
                            item_tang = crear_item_tangencial(
                                rel, termino, razon_t, momento_termino, texto_evidencia=evidencia_tang
                            )
                            menciones_tangenciales_data.append(item_tang)
                            # Guardar evidencia en audioChecks incluso si la mención es tangencial
                            registrar_audio_check(
                                origen_audio_path=archivo_path,
                                nombre_archivo=rel,
                                termino_encontrado=termino,
                                timestamp_segundos=momento_termino,
                                tipo_mencion="tangencial",
                                variante_detectada=termino,
                                func_name=func_name
                            )
                            if deepseek_tangenciales_activo():
                                enriquecer_motivos_tangenciales_deepseek([item_tang], func_name)
                            enriquecer_tangencial_clip_transcripcion_drive(
                                item_tang,
                                archivo_path,
                                rel,
                                archivo_main_dir,
                                momento_termino,
                                buffer_anterior,
                                duracion_clip,
                                dur_total,
                                clip_path_existente=None,
                                func_name=func_name,
                            )
                            mail_tang = notificar_brevo_tangencial_inmediato_si(
                                obtener_cliente_por_termino(termino), item_tang, func_name
                            )
                            if mail_tang is not None:
                                _ok_m, _msg_m = mail_tang
                                if _ok_m:
                                    st.success(f"📧 **Correo tangencial inmediato enviado:** {_msg_m}")
                                else:
                                    st.warning(
                                        f"📧 **Correo tangencial inmediato no enviado:** {_msg_m}\n\n"
                                        f"_Al terminar el ciclo se reintentará el resumen por correo si hay configuración válida._"
                                    )
                            continue  # ❌ NO GENERAR CLIP - SALTAR AL SIGUIENTE TÉRMINO
                        
                        # Usar los valores determinados por Gemini
                        inicio = segmento_gemini['inicio']
                        fin_clip = segmento_gemini['fin']
                        duracion_clip_real = segmento_gemini['duracion']
                        razon_segmento = segmento_gemini['razon']
                        idea_central_gemini = segmento_gemini.get('idea_central', '')  # Nueva: idea centrada
                        
                        # 📏 EXPANSIÓN AUTOMÁTICA: Si el usuario pide 90s y la IA da menos, expandir
                        # Esto cumple con el requisito de no generar clips de audio más cortos que lo pedido
                        if duracion_clip_real < duracion_clip:
                            log_info(f"📏 Expandiendo clip de {duracion_clip_real:.1f}s a {duracion_clip:.1f}s", func_name)
                            dif = duracion_clip - duracion_clip_real
                            inicio = max(0, inicio - (dif / 2))
                            fin_clip = inicio + duracion_clip
                            duracion_clip_real = duracion_clip
                            razon_segmento += f" (Expandido para cumplir {duracion_clip}s)"
                        
                        log_info(f"✅ Gemini 3 Pro determinó segmento inteligente:", func_name)
                        log_info(f"  - Inicio: {inicio:.2f}s", func_name)
                        log_info(f"  - Fin: {fin_clip:.2f}s", func_name)
                        log_info(f"  - Duración: {duracion_clip_real:.2f}s", func_name)
                        log_info(f"  - Razón: {razon_segmento}", func_name)
                        if idea_central_gemini:
                            log_info(f"  - Idea central: {idea_central_gemini[:100]}...", func_name)
                        
                    except Exception as e:
                        # Fallback al método tradicional si Gemini falla
                        log_warning(f"⚠️ Error en Gemini, usando método tradicional: {e}", func_name)
                        st.warning(f"⚠️ Gemini no disponible, usando método tradicional")
                        
                        inicio = max(0, momento_termino - buffer_anterior)
                        fin_clip = inicio + duracion_clip
                        duracion_clip_real = duracion_clip
                        razon_segmento = "Método tradicional (centrado en coincidencia)"
                    
                    # VERIFICAR LÍMITES DEL AUDIO: Asegurar que no se exceda la duración del archivo
                    try:
                        # Obtener duración del audio original
                        cmd_duracion = [
                            "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                            "-of", "csv=p=0", archivo_path
                        ]
                        resultado = subprocess.run(cmd_duracion, capture_output=True, text=True, check=True)
                        duracion_audio = float(resultado.stdout.strip())
                    
                        log_info(f"📹 Duración del audio original: {duracion_audio:.2f}s", func_name)
                    
                        # Verificar si el clip se excede del archivo de audio
                        if fin_clip > duracion_audio:
                            log_warning(f"⚠️ El clip se excede del audio ({fin_clip:.2f}s > {duracion_audio:.2f}s)", func_name)
                            st.warning(f"⚠️ Ajustando clip a límites del audio")
                            
                            # Ajustar para que quepa dentro del archivo
                            if duracion_clip_real <= duracion_audio:
                                # Mover el inicio hacia atrás
                                inicio = max(0, duracion_audio - duracion_clip_real)
                                fin_clip = duracion_audio
                            else:
                                # Audio más corto que duración deseada
                                inicio = 0
                                fin_clip = duracion_audio
                                duracion_clip_real = duracion_audio
                            
                            log_info(f"  - Segmento ajustado: {inicio:.2f}s - {fin_clip:.2f}s ({duracion_clip_real:.2f}s)", func_name)
                    
                    except Exception as e:
                        log_warning(f"⚠️ No se pudo obtener duración del audio: {e}", func_name)
                        log_info("  - Continuando con el clip asumiendo que hay suficiente duración", func_name)
                
                    # Generar clip de audio MP3
                    clip_name = f"{ts}_{termino}_{m}m{s:02d}s.mp3"
                    clip_path = os.path.join(clip_dir, clip_name)
                
                    # Calcular información del clip
                    buffer_anterior_real = momento_termino - inicio
                    buffer_posterior_real = fin_clip - momento_termino
                
                    st.success(f"🎧 Generando segmento inteligente de {duracion_clip_real:.1f}s para '{termino}'")
                    st.info(f"📊 Segmento: {inicio:.1f}s - {fin_clip:.1f}s | Antes: {buffer_anterior_real:.1f}s | Después: {buffer_posterior_real:.1f}s")
                    st.info(f"💡 {razon_segmento}")
                    
                    # Comando para recortar audio con duración exacta
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(inicio),
                        "-t", str(duracion_clip_real), "-i", archivo_path,
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                        clip_path
                    ]
                
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                        # ========== 🤖 EXTRAER IDEA GENERAL DEL SEGMENTO CON GPT-4o ==========
                        st.info(f"🤖 Extrayendo idea general del segmento con GPT-4o...")
                        
                        # Extraer transcripción solo del segmento del clip
                        transcripcion_segmento = ""
                        for seg in segments_timestamps:
                            # Si el segmento está dentro del rango del clip
                            if seg['start'] >= inicio and seg['start'] <= fin_clip:
                                transcripcion_segmento += seg['text'] + " "
                        
                        # Si no se encontró transcripción del segmento, usar contexto completo
                        if not transcripcion_segmento.strip():
                            transcripcion_segmento = mejor_texto_contexto
                        
                        # Llamar a GEMINI 3.0 para extraer idea general (con fallback a GPT-4o)
                        try:
                            # Usar Gemini 3.0 como modelo principal
                            resultado_gemini = extraer_idea_general_segmento_gemini(
                                transcripcion_segmento=transcripcion_segmento.strip(),
                                termino_encontrado=termino,
                                duracion_segundos=duracion_clip_real,
                                nombre_video=rel
                            )
                            
                            # Extraer la idea general del resultado estructurado
                            idea_general_clip = resultado_gemini.get('idea_general', transcripcion_segmento[:600])
                            relevancia_gemini = resultado_gemini.get('relevancia', 'media')
                            es_relevante_gemini = resultado_gemini.get('es_relevante', True)
                            tema_principal = resultado_gemini.get('tema_principal', termino)
                            contexto_gemini = resultado_gemini.get('contexto', '')
                            
                            log_info(f"✅ Análisis Gemini 3.0 - Relevancia: {relevancia_gemini}, Tema: {tema_principal}", func_name)
                            log_info(f"📝 Idea general: {idea_general_clip[:100]}...", func_name)
                            
                            # ========== VERIFICACIÓN DE RELEVANCIA CON GEMINI ==========
                            # Si Gemini determinó que la mención no es relevante, descartar el clip
                            # EXCEPCIÓN: NO descartar clips de términos prioritarios
                            if not es_relevante_gemini or relevancia_gemini == 'baja':
                                if es_prioritario:
                                    # Para términos prioritarios, mantener el clip aunque Gemini lo considere no relevante
                                    log_info(f"⭐ TÉRMINO PRIORITARIO '{termino}': Clip mantenido a pesar de evaluación de relevancia", func_name)
                                    st.info(f"⭐ Término prioritario '{termino}': Clip generado (cualquier mención es relevante)")
                                    idea_general_clip = f"Mención del término '{termino}': {transcripcion_segmento[:200]}..."
                                else:
                                    st.warning(f"⚠️ Clip descartado (Relevancia: {relevancia_gemini}): {contexto_gemini or idea_general_clip[:100]}")
                                    log_info(f"⏭️ Clip descartado por falta de contexto relevante: {termino}", func_name)
                                    motivo_rel = f"Relevancia baja ({relevancia_gemini})"
                                    if contexto_gemini and str(contexto_gemini).strip():
                                        motivo_rel += f" — {contexto_gemini.strip()}"
                                    evidencia_rel = (transcripcion_segmento or "").strip()
                                    item_tang = crear_item_tangencial(
                                        rel, termino, motivo_rel, momento_termino, texto_evidencia=evidencia_rel
                                    )
                                    menciones_tangenciales_data.append(item_tang)
                                    # Guardar evidencia en audioChecks aunque el clip se descarte por relevancia
                                    registrar_audio_check(
                                        origen_audio_path=archivo_path,
                                        nombre_archivo=rel,
                                        termino_encontrado=termino,
                                        timestamp_segundos=timestamp_actual,
                                        tipo_mencion="tangencial",
                                        variante_detectada=termino,
                                        func_name=func_name
                                    )
                                    if deepseek_tangenciales_activo():
                                        enriquecer_motivos_tangenciales_deepseek([item_tang], func_name)
                                    enriquecer_tangencial_clip_transcripcion_drive(
                                        item_tang,
                                        archivo_path,
                                        rel,
                                        archivo_main_dir,
                                        momento_termino,
                                        buffer_anterior,
                                        duracion_clip,
                                        dur_total,
                                        clip_path_existente=clip_path,
                                        func_name=func_name,
                                    )
                                    mail_tang = notificar_brevo_tangencial_inmediato_si(
                                        obtener_cliente_por_termino(termino), item_tang, func_name
                                    )
                                    if mail_tang is not None:
                                        _ok_m, _msg_m = mail_tang
                                        if _ok_m:
                                            st.success(f"📧 **Correo tangencial inmediato enviado:** {_msg_m}")
                                        else:
                                            st.warning(
                                                f"📧 **Correo tangencial inmediato no enviado:** {_msg_m}"
                                            )
                                    continue  # Saltar al siguiente término
                            
                            st.success(f"✅ Idea extraída: {idea_general_clip[:150]}...")
                        except Exception as e:
                            log_warning(f"⚠️ Error extrayendo idea general: {e}", func_name)
                            idea_general_clip = transcripcion_segmento[:300] + "..."  # Fallback
                    
                        # VERIFICACIÓN POST-GENERACIÓN: Confirmar que el clip contiene el término
                        # Para términos prioritarios, esta verificación es opcional (más tolerante)
                        if not es_prioritario:
                            st.info(f"🔍 Verificando que el clip generado contenga el término '{termino}'...")
                        else:
                            st.info(f"⭐ Término prioritario '{termino}': Verificación opcional")
                    
                        # Extraer audio del clip para verificación
                        clip_audio_path = clip_path.replace(".mp3", "_verify.wav")
                        clip_termino_encontrado = termino
                        verify_cmd = [
                            "ffmpeg", "-y", "-i", clip_path,
                            "-ac", "1", "-ar", "16000", "-f", "wav", clip_audio_path
                        ]
                    
                        try:
                            subprocess.run(verify_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                            # Transcribir el clip para verificar
                            verificacion_transcripcion, _, _ = transcribir_audio_hibrido(clip_audio_path)
                            verificacion_lower = verificacion_transcripcion.lower()
                        
                            # Verificar si el término está presente en el clip
                            clip_encontrado, clip_termino_encontrado = buscar_termino_flexible(termino, verificacion_lower)
                            if clip_encontrado:
                                st.success(f"✅ VERIFICADO: El término '{termino}' (variante: '{clip_termino_encontrado}') está presente en el clip generado")
                                log_info(f"✅ Término '{termino}' verificado en clip: {verificacion_transcripcion[:100]}...", func_name)
                            else:
                                if es_prioritario:
                                    # Para términos prioritarios, mantener el clip aunque no se verifique en la revisión
                                    st.warning(f"⚠️ Término prioritario '{termino}': No verificado en clip pero se mantiene")
                                    log_info(f"⭐ TÉRMINO PRIORITARIO '{termino}': Clip mantenido sin verificación estricta", func_name)
                                    
                                    # Limpiar archivo de verificación pero CONTINUAR con el clip
                                    if os.path.exists(clip_audio_path):
                                        os.remove(clip_audio_path)
                                else:
                                    st.error(f"❌ ERROR: El término '{termino}' NO está presente en el clip generado")
                                    log_warning(f"❌ Término '{termino}' NO verificado en clip. Transcripción: {verificacion_transcripcion[:100]}...", func_name)
                                
                                    # Eliminar el clip defectuoso
                                    if os.path.exists(clip_path):
                                        os.remove(clip_path)
                                        st.warning(f"🗑️ Clip defectuoso eliminado: {clip_name}")
                                        log_warning(f"Clip defectuoso eliminado: {clip_path}", func_name)
                            
                                    # Limpiar archivo de verificación
                                    if os.path.exists(clip_audio_path):
                                        os.remove(clip_audio_path)
                            
                                    continue  # No agregar a la lista ni enviar
                        
                            # Limpiar archivo de verificación
                            if os.path.exists(clip_audio_path):
                                os.remove(clip_audio_path)
                            
                        except Exception as verify_error:
                            st.warning(f"⚠️ No se pudo verificar el clip (continuando): {verify_error}")
                            log_warning(f"Error en verificación de clip: {verify_error}", func_name)

                        # Guardar evidencia para análisis posterior (audio origen + TXT con minuto y palabra)
                        if archivo_path and os.path.exists(archivo_path):
                            _audio_check_copia, _audio_check_txt = registrar_audio_check(
                                origen_audio_path=archivo_path,
                                nombre_archivo=rel,
                                termino_encontrado=termino,
                                timestamp_segundos=timestamp_actual,
                                tipo_mencion="real",
                                variante_detectada=clip_termino_encontrado if 'clip_termino_encontrado' in locals() else termino,
                                func_name=func_name
                            )
                            if _audio_check_copia and _audio_check_txt:
                                st.info(f"🗂️ Audio check guardado: {os.path.basename(_audio_check_copia)}")
                    
                        # === 90s sobre el segmento (sin intro); luego intro TTS; luego Cloudinary ===
                        MIN_DURACION_CLIP_ENVIO = 90
                        if clip_path and os.path.exists(clip_path):
                            try:
                                _cmd_dur = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                                            "-of", "csv=p=0", clip_path]
                                _res_dur = subprocess.run(_cmd_dur, capture_output=True, text=True)
                                _dur_clip = float(_res_dur.stdout.strip())
                                if _dur_clip < MIN_DURACION_CLIP_ENVIO:
                                    log_warning(
                                        f"⚠️ Clip descartado por requerimiento de usuario: duración {_dur_clip:.1f}s < {MIN_DURACION_CLIP_ENVIO}s",
                                        func_name)
                                    st.warning(
                                        f"⚠️ Video descartado por ser menor a {MIN_DURACION_CLIP_ENVIO}s (según tu requerimiento).")
                                    clip_path = None
                            except Exception as _e_dur:
                                log_warning(f"⚠️ No se pudo verificar duración del clip: {_e_dur}", func_name)

                        if clip_path and os.path.exists(clip_path):
                            with st.spinner("🎙️ Generando intro de voz y uniendo al clip..."):
                                clip_path, _intro_ok = preparar_clip_con_intro_elevenlabs(
                                    clip_path, rel, termino, func_name)
                            if _intro_ok:
                                st.info("🎙️ Intro agregada al inicio del audio (se sube y envía este archivo).")

                        # SUBIR CLIP A CLOUDINARY INMEDIATAMENTE (ya con intro si ElevenLabs funcionó)
                        url_cloudinary_clip = None
                        if clip_path and os.path.exists(clip_path):
                            with st.spinner("☁️ Subiendo clip a Cloudinary..."):
                                try:
                                    cloudinary_configurado = configurar_cloudinary()
                                    if cloudinary_configurado:
                                        video_url_cloudinary, mensaje_cloudinary = subir_video_cloudinary(clip_path, termino)
                                        if video_url_cloudinary:
                                            url_cloudinary_clip = video_url_cloudinary
                                            st.success(f"☁️ ✅ **CLIP subido a Cloudinary**: {video_url_cloudinary}")
                                            log_info(f"Clip subido a Cloudinary: {video_url_cloudinary}", func_name)
                                        else:
                                            st.warning(f"⚠️ Error subiendo clip a Cloudinary: {mensaje_cloudinary}")
                                            log_warning(f"Error subiendo clip a Cloudinary: {mensaje_cloudinary}", func_name)
                                    else:
                                        st.warning("⚠️ Cloudinary no está configurado")
                                        log_warning("Cloudinary no está configurado para subir clip", func_name)
                                except Exception as e:
                                    st.warning(f"⚠️ Error subiendo clip a Cloudinary: {str(e)}")
                                    log_warning(f"Error subiendo clip a Cloudinary: {str(e)}", func_name)

                            clips_generados_en_sesion.append({
                                'path': clip_path,
                                'termino': termino,
                                'tiempo': f"{m}m{s:02d}s",
                                'contexto': mejor_texto_contexto,
                                'archivo_origen': rel,
                                'momento_exacto': momento_termino,
                                'verificado': True,
                                'url_cloudinary': url_cloudinary_clip
                            })
                            st.session_state.clips_generados += 1
                    
                        # 🚀 ENVÍO INMEDIATO DE COINCIDENCIA
                        st.info(f"🚀 Enviando coincidencia inmediata para '{termino}'...")
                        
                        # Log de la coincidencia detectada para evitar duplicados
                        try:
                            coincidencias_logger.coincidencias_logger.info(
                                f"🎯 COINCIDENCIA DETECTADA | Audio: {rel} | Término: {termino} | Timestamp: {timestamp_actual:.1f}s | Duración: 0s | Confianza: N/A"
                            )
                        except Exception as e:
                            log_warning(f"Error registrando coincidencia en log: {e}", func_name)
                        
                        # 📋 CONSTRUIR RESUMEN EJECUTIVO ESTRUCTURADO COMPLETO
                        # Usar todos los campos de Gemini para el formato completo
                        try:
                            _tiene_gemini = 'resultado_gemini' in locals() and isinstance(resultado_gemini, dict)
                            _tema = resultado_gemini.get('tema_principal', termino) if _tiene_gemini else termino
                            _contexto_g = resultado_gemini.get('contexto', '') if _tiene_gemini else ''
                            _que_dice = resultado_gemini.get('que_se_dice', '') if _tiene_gemini else ''
                            _relevancia_g = resultado_gemini.get('relevancia', 'media') if _tiene_gemini else 'media'
                        except:
                            _tema = termino
                            _contexto_g = ''
                            _que_dice = ''
                            _relevancia_g = 'media'
                        
                        resumen_estructurado = f"Tema principal: {_tema}\n\n"
                        if _contexto_g:
                            resumen_estructurado += f"Contexto: {_contexto_g}\n\n"
                        if _que_dice:
                            resumen_estructurado += f"Puntos clave: {_que_dice}\n\n"
                        # Relevancia: solo texto legible; nunca JSON crudo (evitar "esrelevante: true, relevancia:")
                        _relevancia_texto = (_relevancia_g or "media").capitalize()
                        if idea_general_clip and idea_general_clip != _que_dice:
                            _es_json = any(x in (idea_general_clip or "").lower() for x in ("es_relevante", "esrelevante", '"relevancia"', "relevancia:", "true,", "false,"))
                            if not _es_json and len((idea_general_clip or "").strip()) > 0:
                                resumen_estructurado += f"Relevancia: {_relevancia_texto} — {idea_general_clip.strip()}"
                            else:
                                resumen_estructurado += f"Relevancia: {_relevancia_texto}"
                        else:
                            resumen_estructurado += f"Relevancia: {_relevancia_texto}"

                        exito_envio, mensaje_envio, url_cloudinary_clip = enviar_coincidencia_inmediata(
                            rel,  # nombre del archivo
                            termino,  # término encontrado
                            mejor_texto_contexto,  # contexto del término
                            tipo_archivo,  # tipo de archivo
                            clip_path,  # ruta del clip
                            transcripcion_mistral,  # transcripción completa
                            timestamp_actual,  # timestamp para control de duplicados
                            resumen_estructurado,  # 📋 RESUMEN EJECUTIVO ESTRUCTURADO COMPLETO
                            video_url=url_cloudinary_clip, # URL de Cloudinary
                            transcripcion_segmento=transcripcion_segmento # Segmento resaltado
                        )
                    
                        if exito_envio:
                            log_info(f"✅ Coincidencia enviada exitosamente: {termino} en {rel}", func_name)
                            if mostrar_solo_relevantes:
                                _ui_transient("success", f"✅ Coincidencia enviada: **{termino}**")
                            else:
                                st.success(f"✅ Coincidencia enviada inmediatamente: {mensaje_envio}")
                        else:
                            log_warning(f"❌ Error enviando coincidencia: {mensaje_envio}", func_name)
                            if mostrar_solo_relevantes:
                                _ui_transient("warning", f"⚠️ Error envío: {termino} — {mensaje_envio}")
                            else:
                                st.warning(f"⚠️ Error enviando coincidencia inmediata: {mensaje_envio}")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Error generando clip para {termino}: {e}")
                        continue

                    # Guardar transcripción en archivo TXT (solo si hay clip válido en disco)
                    if not clip_path or not os.path.exists(clip_path):
                        continue
                    txt_path = clip_path.replace(".mp3", ".txt")

                    buffer_posterior = duracion_clip - buffer_anterior
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(f"""TRANSCRIPCIÓN COMPLETA DEL {tipo_archivo.upper()}
    ===============================================

    {tipo_archivo.upper()} ORIGEN: {rel}
    TÉRMINO ENCONTRADO: {termino}
    TIEMPO EN {tipo_archivo.upper()}: {m}m{s:02d}s
    FECHA ANÁLISIS: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ===============================================
    CONFIGURACIÓN DEL CLIP:
    ===============================================

    - Archivo de {tipo_archivo.lower()}: {clip_name}
    - Duración total del clip: {duracion_clip} segundos ({duracion_clip//60}:{duracion_clip%60:02d})
    - Tiempo antes de coincidencia: {buffer_anterior} segundos ({buffer_anterior//60}:{buffer_anterior%60:02d})
    - Tiempo después de coincidencia: {buffer_posterior} segundos ({buffer_posterior//60}:{buffer_posterior%60:02d})
    - Tiempo de inicio del clip: {inicio:.2f}s
    - Momento de coincidencia: {mejor_timestamp['start']:.2f}s
    - API de transcripción utilizada: Mistral/OpenAI Whisper

    ===============================================
    TRANSCRIPCIÓN COMPLETA (Mistral):
    ===============================================

    {transcripcion_mistral}

    ===============================================
    CONTEXTO DEL TIMESTAMP:
    ===============================================

    {mejor_texto_contexto}
    """)
                
                    log_info(f"💾 Clip y transcripción guardados: {clip_name}", func_name)

                    coincidencias_items.append({
                        "termino": termino, 
                        "archivo": rel,
                        "tipo_archivo": tipo_archivo.lower(),
                        "texto": idea_general_clip,  # 🤖 USAR IDEA GENERAL EN LUGAR DE CONTEXTO
                        "contexto": idea_general_clip,  # 🤖 IDEA GENERAL DEL SEGMENTO
                        "timestamp": timestamp_actual,  # Agregar timestamp para control de duplicados
                        "transcripcion_completa": idea_general_clip,  # 🤖 ENVIAR SOLO IDEA GENERAL
                        "url_cloudinary": url_cloudinary_clip  # URL del clip en Cloudinary
                    })

            progress_bar.progress(90, text="📝 Generando resumen...")
        
            # Si hubo coincidencias, generar resumen
            # Inicializar resumen_archivo
            resumen_archivo = ""
        
            if coincidencias_items:
                try:
                    # Copiar audio original con coincidencias a carpeta de salida RadioAnalizer
                    try:
                        destino_coincidencias = CARPETA_PROCESADOS
                        os.makedirs(destino_coincidencias, exist_ok=True)
                        nombre_base_audio = os.path.basename(archivo_path)
                        nombre_destino_audio = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{nombre_base_audio}"
                        ruta_destino_audio = os.path.join(destino_coincidencias, nombre_destino_audio)
                        shutil.copy2(archivo_path, ruta_destino_audio)
                        log_info(f"✅ Audio con coincidencias copiado: {ruta_destino_audio}", func_name)
                    except Exception as e_copia:
                        log_warning(f"⚠️ No se pudo copiar audio con coincidencias: {e_copia}", func_name)

                    resumen_archivo = generar_resumen_archivo(rel, coincidencias_items, transcripcion_mistral, tipo_archivo)
                
                    # Crear archivo consolidado con información de clips
                    clips_info = []
                    for clip in clips_generados_en_sesion:
                        if clip['archivo_origen'] == rel:  # Solo clips de este archivo
                            clips_info.append(clip)
                
                    # Llamada con argumentos posicionales para evitar errores
                    try:
                        # Obtener el URL de Cloudinary si está disponible
                        video_url_consolidado = None
                        if coincidencias_items:
                            # Intentar obtener el URL del primer clip si no hay uno general
                            video_url_consolidado = coincidencias_items[0].get('url_cloudinary')

                        archivo_completo = crear_archivo_consolidado(
                            archivo_path, rel, coincidencias_items, transcripcion_mistral, resumen_archivo, terminos, clips_info, video_url_consolidado
                        )
                    except TypeError:
                        # Fallback sin clips_generados si hay error
                        archivo_completo = crear_archivo_consolidado(
                            archivo_path, rel, coincidencias_items, transcripcion_mistral, resumen_archivo, terminos
                        )
                
                    if st.session_state.mostrar_coincidencias:
                        st.markdown("### 📋 Resumen de coincidencias encontradas:")
                        st.markdown("---")
                    
                        # Mostrar resumen completo en un expander expandido por defecto
                        with st.expander("📄 **RESUMEN EJECUTIVO COMPLETO**", expanded=True):
                            st.markdown(resumen_archivo)
                    
                        # Mostrar detalles adicionales de las coincidencias
                        if coincidencias_items:
                            st.markdown("### 🔍 **DETALLES DE COINCIDENCIAS**")
                            for i, item in enumerate(coincidencias_items, 1):
                                with st.expander(f"🎯 Coincidencia {i}: **{item['termino'].upper()}**", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**🏷️ Término encontrado:** {item['termino']}")
                                        st.markdown(f"**📄 Tipo de archivo:** {item['tipo_archivo']}")
                                        st.markdown(f"**📁 Archivo:** {item['archivo']}")
                                    with col2:
                                        st.markdown(f"**📝 Contexto:**")
                                        st.text_area("", value=item['texto'], height=100, disabled=True, key=f"contexto_{i}_{item['termino']}")
                    
                        st.markdown("---")
                
                    st.session_state.resumen_global.extend(coincidencias_items)
                    st.session_state.videos_procesados += 1
                
                    progress_bar.progress(100, text="✅ Completado")
                    log_info(
                        f"✅ Procesado: {len(coincidencias_items)} coincidencias — {os.path.basename(archivo_completo)}",
                        func_name,
                    )
                    if not mostrar_solo_relevantes:
                        st.success(
                            f"✅ Procesado: {len(coincidencias_items)} coincidencias - Archivo: `{os.path.basename(archivo_completo)}`"
                        )
                
                    # Enviar coincidencias a Supabase
                    try:
                        with st.spinner("🗄️ Enviando coincidencias a Supabase..."):
                            supabase_success, supabase_msg = enviar_coincidencias_a_supabase(
                                coincidencias_items, rel, tipo_archivo, resumen_archivo, transcripcion_mistral, None, None
                            )
                            if supabase_success:
                                log_info(f"🗄️ Supabase: {supabase_msg}", func_name)
                                if not mostrar_solo_relevantes:
                                    st.success(f"🗄️ {supabase_msg}")
                            else:
                                log_warning(f"Supabase: {supabase_msg}", func_name)
                                if not mostrar_solo_relevantes:
                                    st.warning(f"⚠️ {supabase_msg}")
                    except Exception as e:
                        log_warning(f"Error Supabase: {e}", func_name)
                        if not mostrar_solo_relevantes:
                            st.warning(f"⚠️ Error enviando a Supabase: {str(e)}")
                
                    # Almacenar datos del archivo procesado para envío posterior
                    terminos_encontrados = list(set([item['termino'] for item in coincidencias_items]))
                    videos_procesados_data.append({
                        'nombre_archivo': rel,
                        'tipo_archivo': tipo_archivo,
                        'resumen_archivo': resumen_archivo,
                        'terminos_encontrados': terminos_encontrados,
                        'clips_info': clips_info,
                        'coincidencias_items': coincidencias_items,
                        'transcripcion_completa': transcripcion_mistral  # Agregar transcripción completa
                    })
                
                    log_info(f"📦 {tipo_archivo} almacenado para envío posterior: {rel}", func_name)
                    
                    # ========== REGISTRAR VIDEO PROCESADO (CON COINCIDENCIAS) ==========
                    registrar_archivo_procesado(rel, coincidencias_items, resumen_archivo, tipo_archivo)
                    log_info(f"✅ Video registrado en procesados.log: {rel} ({len(coincidencias_items)} coincidencias)", func_name)
                
                except Exception as e:
                    st.warning(f"⚠️ Error en resumen: {e}")
            else:
                progress_bar.progress(100, text="✅ Sin coincidencias")
                tangs_rel = [x for x in menciones_tangenciales_data if x.get('archivo') == rel]
                if tangs_rel:
                    tnames = ", ".join(sorted({(t.get('termino') or '').strip() for t in tangs_rel if t.get('termino')}))
                    st.warning(
                        f"⚠️ **En este archivo no hubo clips**, pero sí **{len(tangs_rel)} mención(es) tangencial(es)** "
                        f"({tnames or 'véase detalle arriba'}). Resumen de tangenciales al **cierre del ciclo** "
                        f"y correo Brevo si está configurado."
                    )
                elif not mostrar_solo_relevantes:
                    st.info(f"🔍 Sin coincidencias en `{rel}`")

                # Limpieza
                if os.path.exists(audio_path) and audio_path != archivo_path: 
                    os.remove(audio_path)
            
                # ========== REGISTRAR VIDEO PROCESADO (SIN COINCIDENCIAS) ==========
                registrar_archivo_procesado(rel, coincidencias_items, resumen_archivo, tipo_archivo)
                log_info(f"✅ Video registrado en procesados.log: {rel} (sin coincidencias)", func_name)
        
        except Exception as e:
            # MANEJO DE ERRORES: Archivo falló
            error_mensaje = f"{type(e).__name__}: {str(e)}"
            log_error_critico(func_name, f"Error procesando archivo {rel}: {error_mensaje}")
            
            # Guardar archivo fallido (mueve, crea txt, envía notificaciones)
            guardar_archivo_fallido(
                nombre_archivo=rel,
                error_mensaje=error_mensaje,
                archivo_path=archivo_path
            )
            
            # Continuar con el siguiente archivo
            continue

    # FASE 2: ENVIAR TODOS LOS ARCHIVOS PROCESADOS AL WEBHOOK
    if videos_procesados_data:
        st.success(f"✅ FASE 1 COMPLETADA: {len(videos_procesados_data)} archivos procesados exitosamente")
        st.info("🌐 FASE 2: Enviando todos los archivos al webhook con pausas de 60s")
        envio_actual_container = st.empty()
        
        webhook_config = cargar_webhook_config()
        telegram_config = cargar_telegram_config()
        
        for i, archivo_data in enumerate(videos_procesados_data, 1):
            icono_archivo = "🎞️" if archivo_data.get('tipo_archivo', '').lower() == 'video' else "🎵"
            with envio_actual_container.container():
                st.markdown(f"---\n### 📤 Enviando {archivo_data.get('tipo_archivo', 'Archivo')} {i}/{len(videos_procesados_data)}: {icono_archivo} `{archivo_data['nombre_archivo']}`")
            
            # PASO 1: Enviar SOLO el resumen ejecutivo al webhook
            if webhook_config['enabled'] and webhook_config['url']:
                try:
                    with envio_actual_container.container(), st.spinner("🌐 Enviando resumen ejecutivo al webhook..."):
                        exito, mensaje = webhook_notification_simple(
                            archivo_data['nombre_archivo'], 
                            archivo_data['resumen_archivo'], 
                            archivo_data['terminos_encontrados']
                        )
                        
                        if exito:
                            st.success(f"🌐 Webhook - Resumen ejecutivo enviado: {mensaje}")
                            
                            # PASO 2: Pausa obligatoria de 60 segundos después del resumen
                            log_info(f"PASO 1 COMPLETADO - Esperando 60 segundos después del resumen ejecutivo del video {i}", "buscar_y_procesar_videos")
                            with envio_actual_container.container(), st.spinner("⏳ PASO 1: Esperando 60s después del resumen ejecutivo..."):
                                time.sleep(60)
                            with envio_actual_container.container():
                                st.info("✅ PASO 1 completado - Procediendo a enviar clips")
                            
                            # PASO 3: Ahora enviar los clips individualmente con pausas de 60s
                            if archivo_data['clips_info']:
                                with envio_actual_container.container():
                                    st.info("🌐 Enviando clips individuales al webhook con pausas de 60s...")
                                exito_clips, mensaje_clips = enviar_clips_individuales_webhook(
                                    archivo_data['clips_info'],
                                    archivo_data['resumen_archivo'], 
                                    archivo_data['terminos_encontrados'],
                                    archivo_data['nombre_archivo']
                                )
                                
                                with envio_actual_container.container():
                                    if exito_clips:
                                        st.success(f"🌐 Webhook - Clips enviados: {mensaje_clips}")
                                    else:
                                        st.warning(f"⚠️ Webhook - Error en clips: {mensaje_clips}")
                            else:
                                with envio_actual_container.container():
                                    st.info("📭 No hay clips para enviar en este video")
                        else:
                            with envio_actual_container.container():
                                st.warning(f"⚠️ Webhook - Error en resumen: {mensaje}")
                except Exception as e:
                    with envio_actual_container.container():
                        st.warning(f"⚠️ Error enviando webhook: {e}")
            
            # PASO 4: Enviar a Telegram (DESACTIVADO PARA EVITAR DUPLICADOS)
            # COMENTADO: El envío masivo a Telegram está desactivado porque ya se envían
            # los clips individualmente cuando se detecta cada término en enviar_coincidencia_inmediata()
            # if telegram_config['enabled'] and telegram_config['bot_token'] and telegram_config['chat_id']:
            #     try:
            #         with st.spinner("📱 Enviando a Telegram..."):
            #             exito_telegram, mensaje_telegram = enviar_clips_a_telegram(
            #                 archivo_data['clips_info'],
            #                 archivo_data['resumen_archivo'], 
            #                 archivo_data['terminos_encontrados'],
            #                 archivo_data['nombre_archivo']
            #             )
            #             
            #             if exito_telegram:
            #                 st.success(f"📱 Telegram: {mensaje_telegram}")
            #             else:
            #                 st.warning(f"⚠️ Telegram: {mensaje_telegram}")
            #     except Exception as e:
            #         st.warning(f"⚠️ Error enviando a Telegram: {e}")
            
            # PASO 4.5: Enviar a Google Drive (DESACTIVADO PARA EVITAR DUPLICADOS)
            # COMENTADO: El envío masivo a Google Drive está desactivado porque ya se suben
            # los clips individualmente cuando se detecta cada término en enviar_coincidencia_inmediata()
            # try:
            #     with st.spinner("☁️ Enviando a Google Drive..."):
            #         exito_gdrive, mensaje_gdrive = enviar_clips_a_google_drive(
            #             archivo_data['clips_info'],
            #             archivo_data['resumen_archivo'], 
            #             archivo_data['terminos_encontrados'],
            #             archivo_data['nombre_archivo'],
            #             archivo_data.get('transcripcion_completa', '')  # Agregar transcripción completa
            #         )
            #         
            #         if exito_gdrive:
            #             st.success(f"☁️ Google Drive: {mensaje_gdrive}")
            #         else:
            #             st.warning(f"⚠️ Google Drive: {mensaje_gdrive}")
            # except Exception as e:
            #     st.warning(f"⚠️ Error enviando a Google Drive: {e}")
            
            # INFORMACIÓN: Los clips ya fueron enviados individualmente durante el procesamiento
            with envio_actual_container.container():
                st.info("ℹ️ Los clips ya fueron enviados individualmente durante el procesamiento (Telegram + Google Drive + Correo)")
            
            # PASO 5: Pausa final de 60 segundos antes del siguiente archivo (excepto el último)
            if i < len(videos_procesados_data):
                log_info(f"PROCESAMIENTO COMPLETADO - Esperando 60 segundos antes del siguiente archivo ({i+1}/{len(videos_procesados_data)})", "buscar_y_procesar_videos")
                with envio_actual_container.container(), st.spinner(f"⏳ FINAL: Esperando 60s antes del siguiente archivo ({i+1}/{len(videos_procesados_data)})..."):
                    time.sleep(60)
                with envio_actual_container.container():
                    st.info(f"✅ Listo para procesar archivo {i+1}")
        
        st.success(f"🎉 FASE 2 COMPLETADA: Todos los {len(videos_procesados_data)} archivos enviados exitosamente")
    else:
        st.info("📭 No se procesaron archivos con coincidencias en esta sesión")

    if menciones_tangenciales_data and deepseek_tangenciales_activo():
        with st.spinner("Profundizando motivos tangenciales (DeepSeek)..."):
            enriquecer_motivos_tangenciales_deepseek(menciones_tangenciales_data, func_name)

    if menciones_tangenciales_data:
        try:
            res_sheets_tang = enviar_tangenciales_a_google_sheets(menciones_tangenciales_data)
            exitos_sheets_tang = sum(1 for _, ok, _ in res_sheets_tang if ok)
            if res_sheets_tang:
                st.info(f"📊 Tangenciales enviadas a Google Sheets: {exitos_sheets_tang}/{len(res_sheets_tang)}")
        except Exception as e_sheets_tang:
            st.warning(f"⚠️ Google Sheets (tangenciales): {e_sheets_tang}")
            log_warning(f"Error Google Sheets tangenciales: {e_sheets_tang}", func_name)

    # Mostrar solo resultados relevantes de la sesión: coincidencias y tangenciales
    if videos_procesados_data or menciones_tangenciales_data:
        st.markdown("---")
        st.markdown("## 📋 **RESULTADOS RELEVANTES DE LA SESIÓN**")

        if videos_procesados_data:
            st.markdown("### 🎯 Coincidencias detectadas")
            for i, video_data in enumerate(videos_procesados_data, 1):
                with st.expander(f"📄 **RESUMEN {i}: {video_data['nombre_archivo']}**", expanded=True):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### 📊 **Resumen Ejecutivo:**")
                        st.markdown(video_data['resumen_archivo'])

                    with col2:
                        st.markdown("### ℹ️ **Información:**")
                        st.markdown(f"**📁 Archivo:** {video_data['nombre_archivo']}")
                        st.markdown(f"**🎞️ Tipo:** {video_data.get('tipo_archivo', 'Audio')}")
                        st.markdown(f"**🔍 Términos:** {', '.join(video_data['terminos_encontrados'])}")
                        st.markdown(f"**🎬 Clips generados:** {len(video_data.get('clips_info', []))}")

                        if video_data.get('clips_info'):
                            st.markdown("**📹 Clips:**")
                            for clip in video_data['clips_info']:
                                st.markdown(f"• {clip['termino']} ({clip['tiempo']})")

        if menciones_tangenciales_data:
            _, md_tang, _ = construir_tangenciales_narrativo(menciones_tangenciales_data)
            st.markdown("### ⚠️ Menciones tangenciales")
            st.markdown(md_tang)
            with st.expander("Detalle por ocurrencia (tabla)", expanded=False):
                for i, tang in enumerate(menciones_tangenciales_data, 1):
                    ts_seg = tang.get('timestamp', 0) or 0
                    ta = formato_posicion_en_audio_segundos(ts_seg)
                    medio = formato_linea_emision_legible(tang.get('archivo', ''))
                    hd = _hora_deteccion_formateada(tang)
                    st.markdown(
                        f"{i}. **{tang.get('termino', '')}** — {medio} — {ta} en audio — detección {hd} — "
                        f"{tang.get('motivo', '')}"
                    )
    
    # Mostrar clips generados en esta sesión
    if clips_generados_en_sesion:
        st.session_state.clips_encontrados_sesion.extend(clips_generados_en_sesion)
        mostrar_player_clips(clips_generados_en_sesion, titulo="🎬 Clips Generados en Esta Sesión")
    
    # === GENERAR MD DE SESIÓN CON TODAS LAS COINCIDENCIAS ===
    try:
        md_sesion_ok, md_sesion_resultado = generar_md_sesion_coincidencias(
            videos_procesados_data=videos_procesados_data,
            clips_generados_en_sesion=clips_generados_en_sesion,
            estadisticas_escaneo=estadisticas if 'estadisticas' in locals() else None,
            terminos_buscados=terminos,
            menciones_tangenciales_data=menciones_tangenciales_data,
        )
        if md_sesion_ok:
            st.success(f"📄 **Reporte de sesión guardado:** `{os.path.basename(md_sesion_resultado)}`")
            log_info(f"✅ MD de sesión generado: {md_sesion_resultado}", func_name)
        else:
            st.warning(f"⚠️ Error generando reporte de sesión: {md_sesion_resultado}")
    except Exception as e:
        st.warning(f"⚠️ Error generando reporte de sesión: {str(e)}")
        log_warning(f"Error generando MD de sesión: {e}", func_name)
    
    # === ANALISISHOY MD: menciones tangenciales (mismo formato y orden que la UI) ===
    try:
        ok_tang, ruta_o_err = append_analisishoy_menciones_tangenciales(menciones_tangenciales_data)
        if ok_tang:
            if ruta_o_err and menciones_tangenciales_data:
                st.success(f"📄 **AnalisisHoy:** menciones tangenciales añadidas a `{os.path.basename(ruta_o_err)}`")
        else:
            st.warning(f"⚠️ AnalisisHoy MD (tangenciales): {ruta_o_err}")
            log_warning(f"No se pudo añadir tangenciales a Analisishoy: {ruta_o_err}", func_name)
    except Exception as e_tang:
        st.warning(f"⚠️ AnalisisHoy MD (tangenciales): {e_tang}")
        log_warning(f"Error Analisishoy tangenciales: {e_tang}", func_name)
    
    # === CORREO BREVO: menciones tangenciales por entidad (fin de ciclo) ===
    try:
        res_correos_tang = enviar_correos_tangenciales_fin_ciclo(menciones_tangenciales_data)
        for nombre_c, ok_c, msg_c in res_correos_tang:
            if ok_c:
                st.success(f"📧 **Brevo (tangenciales)** — {nombre_c}: {msg_c}")
            else:
                st.info(f"📧 **Brevo (tangenciales)** — {nombre_c}: {msg_c}")
    except Exception as e_mail_tang:
        st.warning(f"⚠️ Correo tangenciales: {e_mail_tang}")
        log_warning(f"Error correo tangenciales fin ciclo: {e_mail_tang}", func_name)
    
    # Limpiar referencia de contenedor transitorio para evitar arrastre entre ciclos
    st.session_state.ui_status_container = None

    # === FINALIZACIÓN - LOOP AUTOMÁTICO ===
    st.markdown("---")
    st.success(f"✅ **CICLO COMPLETADO** - {len(videos_procesados_data)} archivos procesados en este ciclo")
    
    # Contar ciclo
    if 'loop_ciclo_numero' not in st.session_state:
        st.session_state.loop_ciclo_numero = 0
    st.session_state.loop_ciclo_numero += 1
    
    log_info(f"Ciclo #{st.session_state.loop_ciclo_numero} completado: {len(videos_procesados_data)} archivos procesados", func_name)
    
    # Verificar si el loop está activo
    loop_activo = st.session_state.get('loop_continuo', True)
    intervalo_loop = st.session_state.get('intervalo_loop', 60)  # segundos entre ciclos
    
    if loop_activo:
        st.info(f"🔄 **LOOP ACTIVO** (Ciclo #{st.session_state.loop_ciclo_numero}) - Re-escaneando en {intervalo_loop}s para buscar audios nuevos...")
        log_info(f"Loop activo - Esperando {intervalo_loop}s antes de re-escanear", func_name)
        
        # Liberar flag ANTES de la espera
        st.session_state.procesamiento_en_curso = False
        
        # Countdown visual seguro (no se congela si la sesión se cierra)
        countdown_container = st.empty()
        ciclo_num = st.session_state.loop_ciclo_numero
        sesion_viva = _countdown_seguro(
            countdown_container, intervalo_loop,
            f"⏳ Re-escaneando en {{i}}s... (Ciclo #{ciclo_num} completado)",
            func_name
        )
        
        if not sesion_viva:
            log_warning("Sesión cerrada durante countdown post-ciclo. Saliendo limpiamente.", func_name)
            return
        
        # Re-escanear: cargar procesados actualizados
        st.info("🔍 **RE-ESCANEANDO** - Buscando audios nuevos...")
        log_info("Re-escaneando carpeta para buscar nuevos videos...", func_name)
        
        try:
            procesados_actualizados = cargar_procesados()
            nuevos_encontrados = buscar_videos_nuevos_optimizado(procesados_actualizados, func_name)
        except Exception as e:
            log_exception(func_name, e, "Error al re-escanear")
            st.error(f"❌ Error al re-escanear: {e}")
            st.session_state.procesamiento_en_curso = False
            return
        
        if nuevos_encontrados:
            st.success(f"🆕 **{len(nuevos_encontrados)} VIDEOS NUEVOS** - Iniciando nuevo ciclo...")
            log_info(f"Loop: {len(nuevos_encontrados)} audios nuevos, iniciando nuevo ciclo", func_name)
            time.sleep(2)
            try:
                st.rerun()  # Solo refresca UI, mantiene session_state
            except Exception as e:
                log_warning(f"st.rerun() falló (sesión puede haber cerrado): {e}", func_name)
                st.warning("⚠️ Si la sesión se cerró, pulsa el botón para continuar.")
                if st.button("🔄 **CONTINUAR LOOP** (re-escanear ahora)", key="continuar_loop_despues_nuevos"):
                    st.rerun()
        else:
            st.success("✅ **NO HAY VIDEOS NUEVOS** - Esperando...")
            log_info("Loop: No hay audios nuevos, entrando en espera", func_name)
            
            wait_container = st.empty()
            espera_sin_nuevos = st.session_state.get('intervalo_loop_vacio', 120)
            sesion_viva = _countdown_seguro(
                wait_container, espera_sin_nuevos,
                "💤 Sin audios nuevos. Re-escaneando en {i}s... (Desactiva el loop en sidebar)",
                func_name
            )
            
            if not sesion_viva:
                log_warning("Sesión cerrada durante espera sin audios nuevos (post-ciclo). Saliendo limpiamente.", func_name)
                return
            
            st.info("🔄 Volviendo a escanear...")
            time.sleep(1)
            try:
                st.rerun()
            except Exception as e:
                log_warning(f"st.rerun() falló: {e}", func_name)
                st.warning("⚠️ Si la sesión se cerró, pulsa para continuar.")
                if st.button("🔄 **CONTINUAR LOOP** (re-escanear ahora)", key="continuar_loop_sin_nuevos"):
                    st.rerun()
    else:
        # Loop desactivado - mostrar resumen final
        st.markdown("### 📊 **RESUMEN FINAL**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📁 Archivos Procesados", len(videos_procesados_data))
        
        with col2:
            total_archivos_procesados = st.session_state.get('loop_ciclo_numero', 1)
            st.metric("🔄 Ciclos Completados", total_archivos_procesados)
        
        with col3:
            total_terminos = sum(len(archivo.get('terminos_encontrados', [])) for archivo in videos_procesados_data)
            st.metric("🔍 Términos Encontrados", total_terminos)
        
        # Botón para procesar manualmente
        if st.button("🔄 **PROCESAR NUEVOS ARCHIVOS**", 
                    type="primary", 
                    help="Buscar y procesar archivos nuevos agregados a la carpeta",
                    key="procesar_nuevos"):
            st.info("🔍 Iniciando búsqueda de archivos nuevos...")
            st.session_state.nueva_verificacion_solicitada = True
            st.rerun()
        
        st.info("💡 **Tip:** Activa el Loop desde el sidebar para que el sistema escanee continuamente.")
    
    # ========== LIBERAR FLAG DE PROCESAMIENTO ==========
    st.session_state.procesamiento_en_curso = False
    log_info("🔓 Flag de procesamiento liberado", func_name)

# === BUCLE CONTINUO (REMOVIDO - AHORA SE USA LÓGICA SÍNCRONA) ===
# La lógica continua ahora se maneja de forma síncrona en el flujo principal

# === PLAYER DE CLIPS AVANZADO ===
def mostrar_player_clips(clips_list, titulo="🎧 Player de Segmentos"):
    """Player avanzado para mostrar clips con controles"""
    
    st.markdown(f"## {titulo}")
    
    if not clips_list:
        st.info("📭 No hay clips para mostrar")
        return
    
    # Estadísticas de clips
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clips", len(clips_list))
    with col2:
        terminos_unicos = len(set([clip['termino'] for clip in clips_list]))
        st.metric("Términos Únicos", terminos_unicos)
    with col3:
        videos_origen = len(set([clip.get('video_origen', 'desconocido') for clip in clips_list]))
        st.metric("Audios Origen", videos_origen)
    with col4:
        tamano_total = sum([os.path.getsize(clip['path']) for clip in clips_list if os.path.exists(clip['path'])])
        st.metric("Tamaño Total", f"{tamano_total / (1024*1024):.1f} MB")
    
    # Filtros avanzados
    st.markdown("### 🔧 Filtros y Controles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtro por término
        terminos_disponibles = ['Todos'] + sorted(list(set([clip['termino'] for clip in clips_list])))
        termino_filtro = st.selectbox("🏷️ Filtrar por término:", terminos_disponibles, key=f"filtro_termino_{titulo.replace(' ', '_')}")
    
    with col2:
        # Filtro por video origen
        videos_disponibles = ['Todos'] + sorted(list(set([clip.get('video_origen', 'desconocido') for clip in clips_list])))
        video_filtro = st.selectbox("🎧 Filtrar por audio origen:", videos_disponibles, key=f"filtro_video_{titulo.replace(' ', '_')}")
    
    with col3:
        # Modo de visualización
        modo_vista = st.selectbox("👁️ Modo de vista:", 
                                 ["Lista completa", "Solo reproductores", "Compacto"], 
                                 key=f"modo_vista_{titulo.replace(' ', '_')}")
    
    # Aplicar filtros
    clips_filtrados = clips_list
    if termino_filtro != 'Todos':
        clips_filtrados = [clip for clip in clips_filtrados if clip['termino'] == termino_filtro]
    if video_filtro != 'Todos':
        clips_filtrados = [clip for clip in clips_filtrados if clip.get('video_origen', 'desconocido') == video_filtro]
    
    # Búsqueda por texto
    busqueda_texto = st.text_input("🔍 Buscar en contexto:", placeholder="Buscar texto en las transcripciones...", key=f"busqueda_texto_{titulo.replace(' ', '_')}")
    if busqueda_texto:
        clips_filtrados = [clip for clip in clips_filtrados 
                          if busqueda_texto.lower() in clip.get('contexto', '').lower()]
    
    # Ordenamiento
    orden = st.selectbox("📊 Ordenar por:", 
                        ["Más recientes", "Más antiguos", "Por término", "Por duración"],
                        key=f"orden_{titulo.replace(' ', '_')}")
    
    if orden == "Más recientes":
        clips_filtrados.sort(key=lambda x: os.path.getctime(x['path']) if os.path.exists(x['path']) else 0, reverse=True)
    elif orden == "Más antiguos":
        clips_filtrados.sort(key=lambda x: os.path.getctime(x['path']) if os.path.exists(x['path']) else 0)
    elif orden == "Por término":
        clips_filtrados.sort(key=lambda x: x['termino'])
    
    st.info(f"📊 Mostrando {len(clips_filtrados)} de {len(clips_list)} clips")
    
    # Control de reproducción automática
    col1, col2 = st.columns(2)
    with col1:
        autoplay = st.checkbox("▶️ Reproducción automática", value=False, key=f"autoplay_{titulo.replace(' ', '_')}")
    with col2:
        clips_por_pagina = st.slider("Clips por página:", 5, 50, 10, key=f"clips_pagina_{titulo.replace(' ', '_')}")
    
    # Paginación
    if clips_filtrados:
        total_paginas = (len(clips_filtrados) - 1) // clips_por_pagina + 1
        if total_paginas > 1:
            pagina_actual = st.selectbox(f"📄 Página (de {total_paginas}):", 
                                        range(1, total_paginas + 1), index=0,
                                        key=f"pagina_{titulo.replace(' ', '_')}")
        else:
            pagina_actual = 1
        
        inicio = (pagina_actual - 1) * clips_por_pagina
        fin = inicio + clips_por_pagina
        clips_pagina = clips_filtrados[inicio:fin]
        
        # Debug info
        st.caption(f"Mostrando clips {inicio + 1}-{min(fin, len(clips_filtrados))} de {len(clips_filtrados)} total")
    else:
        clips_pagina = []
        st.info("📭 No hay clips que coincidan con los filtros aplicados")
    
    # Mostrar clips según el modo
    titulo_limpio = titulo.replace(' ', '_').replace('🎬', '').replace('🆕', '').strip()
    
    if modo_vista == "Lista completa":
        mostrar_clips_completos(clips_pagina, titulo_limpio)
    elif modo_vista == "Solo reproductores":
        mostrar_solo_reproductores(clips_pagina, autoplay, titulo_limpio)
    else:  # Compacto
        mostrar_clips_compactos(clips_pagina, titulo_limpio)
    
    # Botones de acción masiva
    if clips_filtrados:
        st.markdown("### 🔧 Acciones Masivas")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Exportar Lista", key=f"exportar_{titulo.replace(' ', '_')}"):
                exportar_lista_clips(clips_filtrados)
        
        with col2:
            if st.button("📊 Generar Estadísticas", key=f"stats_{titulo.replace(' ', '_')}"):
                generar_estadisticas_clips(clips_filtrados)
        
        with col3:
            if st.button("🗑️ Limpiar Filtrados", key=f"limpiar_{titulo.replace(' ', '_')}"):
                eliminar_clips_filtrados(clips_filtrados)
        
        with col4:
            if st.button("📁 Crear Playlist", key=f"playlist_{titulo.replace(' ', '_')}"):
                crear_playlist_clips(clips_filtrados)

def mostrar_clips_completos(clips, titulo_seccion="clips"):
    """Muestra clips con información completa"""
    for i, clip in enumerate(clips):
        if not os.path.exists(clip['path']):
            continue
            
        with st.expander(f"🎧 {clip['termino'].upper()} - {clip['tiempo']} | {os.path.basename(clip['path'])}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.audio(clip['path'])
            
            with col2:
                st.markdown(f"**🏷️ Término:** {clip['termino']}")
                st.markdown(f"**⏱️ Tiempo:** {clip['tiempo']}")
                st.markdown(f"**🎧 Audio origen:** {clip.get('video_origen', 'Desconocido')}")
                
                # Información del archivo
                file_info = os.stat(clip['path'])
                st.markdown(f"**📊 Tamaño:** {file_info.st_size / (1024*1024):.2f} MB")
                st.markdown(f"**📅 Creado:** {datetime.fromtimestamp(file_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Botones de acción con keys únicos
                if st.button(f"🗑️ Eliminar", key=f"del_clip_{i}_{titulo_seccion}_{hash(clip['path'])}"):
                    try:
                        os.remove(clip['path'])
                        st.success("Clip eliminado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                if st.button(f"📁 Abrir carpeta", key=f"folder_clip_{i}_{titulo_seccion}_{hash(clip['path'])}"):
                    folder_path = os.path.dirname(clip['path'])
                    os.startfile(folder_path)
            
            # Contexto
            if clip.get('contexto'):
                st.markdown("**📝 Contexto:**")
                st.markdown(f"> {clip['contexto']}")

def mostrar_solo_reproductores(clips, autoplay=False, titulo_seccion="reproductores"):
    """Muestra solo los reproductores de audio"""
    for i, clip in enumerate(clips):
        if os.path.exists(clip['path']):
            st.markdown(f"**🎧 {clip['termino'].upper()} - {clip['tiempo']}**")
            st.audio(clip['path'], autoplay=autoplay and i==0)
            if i < len(clips) - 1:
                st.markdown("---")

def mostrar_clips_compactos(clips, titulo_seccion="compactos"):
    """Muestra clips en formato compacto"""
    for i, clip in enumerate(clips):
        if os.path.exists(clip['path']):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"**{clip['termino']}** - {clip['tiempo']}")
            with col2:
                st.markdown(f"_{clip.get('video_origen', 'Desconocido')}_")
            with col3:
                if st.button("▶️", key=f"play_compact_{titulo_seccion}_{i}_{hash(clip['path'])}"):
                    st.audio(clip['path'], autoplay=True)

def exportar_lista_clips(clips):
    """Exporta lista de clips a CSV"""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Termino', 'Tiempo', 'Video_Origen', 'Archivo', 'Tamaño_MB', 'Fecha_Creacion'])
    
    for clip in clips:
        if os.path.exists(clip['path']):
            file_info = os.stat(clip['path'])
            writer.writerow([
                clip['termino'],
                clip['tiempo'],
                clip.get('video_origen', 'Desconocido'),
                os.path.basename(clip['path']),
                f"{file_info.st_size / (1024*1024):.2f}",
                datetime.fromtimestamp(file_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    csv_data = output.getvalue()
    st.download_button(
        label="📥 Descargar CSV",
        data=csv_data,
        file_name=f"clips_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def generar_estadisticas_clips(clips):
    """Genera estadísticas detalladas de los clips"""
    st.markdown("### 📊 Estadísticas Detalladas")
    
    # Estadísticas por término
    terminos_count = {}
    for clip in clips:
        termino = clip['termino']
        terminos_count[termino] = terminos_count.get(termino, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Clips por término:**")
        for termino, count in sorted(terminos_count.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"• {termino}: {count} clips")
    
    with col2:
        # Estadísticas de tiempo
        fechas = []
        for clip in clips:
            if os.path.exists(clip['path']):
                fechas.append(datetime.fromtimestamp(os.path.getctime(clip['path'])))
        
        if fechas:
            fechas.sort()
            st.markdown("**Distribución temporal:**")
            st.markdown(f"• Primer clip: {fechas[0].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"• Último clip: {fechas[-1].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"• Período: {(fechas[-1] - fechas[0]).days} días")

def eliminar_clips_filtrados(clips):
    """Elimina clips filtrados (con confirmación)"""
    st.warning(f"⚠️ Esto eliminará {len(clips)} clips permanentemente")
    if st.button("🗑️ CONFIRMAR ELIMINACIÓN", type="primary", key=f"confirm_delete_{len(clips)}_{hash(str(clips))}"):
        eliminados = 0
        for clip in clips:
            try:
                if os.path.exists(clip['path']):
                    os.remove(clip['path'])
                    eliminados += 1
            except Exception:
                pass
        st.success(f"✅ Eliminados {eliminados} clips")
        st.rerun()

def crear_playlist_clips(clips):
    """Crea un archivo playlist con los clips"""
    playlist_content = "#EXTM3U\n"
    for clip in clips:
        if os.path.exists(clip['path']):
            playlist_content += f"#EXTINF:-1,{clip['termino']} - {clip['tiempo']}\n"
            playlist_content += f"{clip['path']}\n"
    
    st.download_button(
        label="📁 Descargar Playlist",
        data=playlist_content,
        file_name=f"playlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.m3u",
        mime="audio/x-mpegurl"
    )

# === PROCESAR ACCIÓN PENDIENTE ===
# Ejecutar procesamiento si se solicitó
if hasattr(st.session_state, 'procesar_una_vez') and st.session_state.get('procesar_una_vez', False):
    # Obtener configuración del session_state
    duracion_clip = getattr(st.session_state, 'duracion_clip', 90)  # 1.5 minutos total por defecto
    buffer_anterior = getattr(st.session_state, 'buffer_anterior', 30)  # 30s antes por defecto
    
    buscar_y_procesar_videos(duracion_clip, buffer_anterior)
    st.session_state.procesar_una_vez = False

# === PROCESAR NUEVA VERIFICACIÓN SOLICITADA ===
# Ejecutar nueva verificación solo si se solicitó manualmente
if hasattr(st.session_state, 'nueva_verificacion_solicitada') and st.session_state.get('nueva_verificacion_solicitada', False):
    # Obtener configuración del session_state
    duracion_clip = getattr(st.session_state, 'duracion_clip', 90)  # 1.5 minutos total por defecto
    buffer_anterior = getattr(st.session_state, 'buffer_anterior', 30)  # 30s antes por defecto
    
    st.info("🔍 **NUEVA VERIFICACIÓN INICIADA** - Buscando archivos nuevos agregados después del último procesamiento")
    buscar_y_procesar_videos(duracion_clip, buffer_anterior)
    st.session_state.nueva_verificacion_solicitada = False

# === PROCESAR BÚSQUEDA CONTINUA ===
# Ejecutar procesamiento continuo si está activo
if st.session_state.running and st.session_state.terminos_continuos:
    # Inicializar timestamp del último procesamiento si no existe
    if 'ultimo_procesamiento_continuo' not in st.session_state:
        st.session_state.ultimo_procesamiento_continuo = 0
    
    # Verificar si es hora de procesar
    tiempo_actual = time.time()
    tiempo_desde_ultimo = tiempo_actual - st.session_state.ultimo_procesamiento_continuo
    
    if tiempo_desde_ultimo >= st.session_state.intervalo:
        # Obtener configuración del session_state  
        duracion_clip = getattr(st.session_state, 'duracion_clip', 90)  # 1.5 minutos total por defecto
        buffer_anterior = getattr(st.session_state, 'buffer_anterior', 30)  # 30s antes por defecto
        
        # Mostrar que está en modo continuo
        st.info(f"🔄 **MODO CONTINUO ACTIVO** - Ejecutando ciclo de procesamiento")
        
        # Ejecutar procesamiento (igual que "Procesar Una Vez")
        buscar_y_procesar_videos(duracion_clip, buffer_anterior)
        
        # Actualizar timestamp
        st.session_state.ultimo_procesamiento_continuo = time.time()
        
        # Mostrar mensaje de espera
        st.success(f"✅ Ciclo completado. Próximo procesamiento en {st.session_state.intervalo} segundos...")
        
        # Recargar para continuar el ciclo inmediatamente
        st.rerun()
    else:
        # Mostrar countdown hasta el próximo procesamiento
        tiempo_restante = st.session_state.intervalo - int(tiempo_desde_ultimo)
        st.info(f"⏳ **MODO CONTINUO ACTIVO** - Próximo procesamiento en {tiempo_restante} segundos")
        
        # Usar st.rerun() para actualizar el countdown cada 5 segundos
        # Agregar un pequeño delay para evitar refresh demasiado frecuente
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = 0
        
        if time.time() - st.session_state.last_refresh >= 5:
            st.session_state.last_refresh = time.time()
            st.rerun()
        else:
            # Mostrar mensaje estático si no es momento de refresh
            pass

# === CONTROLES PRINCIPALES ===
st.markdown("## ⚡ Control de Búsqueda Continua")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ INICIAR BÚSQUEDA CONTINUA", 
                disabled=st.session_state.running,
                type="primary",
                help="Iniciar monitoreo automático continuo"):
        if not st.session_state.terminos_continuos:
            st.error("❌ Primero configura los términos de búsqueda")
        else:
            st.session_state.running = True
            st.success(f"🚀 Búsqueda continua iniciada (cada {st.session_state.intervalo}s)")
            st.rerun()

with col2:
    if st.button("⏹️ DETENER BÚSQUEDA", 
                disabled=not st.session_state.running,
                type="secondary"):
        st.session_state.running = False
        st.warning("🛑 Búsqueda continua detenida")
        st.rerun()

# === PLAYER PRINCIPAL DE CLIPS ===
st.markdown("---")

# Tabs para diferentes vistas
tab1, tab2, tab3, tab4 = st.tabs(["🎬 Todos los Clips", "🆕 Clips de Sesión", "📊 Análisis", "🏢 Entidades y Rutas"])

with tab1:
    # Mostrar todos los clips disponibles
    todos_los_clips = buscar_todos_los_clips()
    if todos_los_clips:
        # Convertir a formato compatible
        clips_convertidos = []
        for clip_info in todos_los_clips:
            clips_convertidos.append({
                'path': clip_info['filepath'],
                'termino': clip_info['termino'],
                'tiempo': clip_info['tiempo_video'],
                'contexto': f"Generado el {clip_info['fecha_creacion'].strftime('%Y-%m-%d %H:%M:%S')}",
                'video_origen': 'Análisis previo'
            })
        mostrar_player_clips(clips_convertidos, "🎬 Biblioteca Completa de Clips")
    else:
        st.info("📭 No hay clips disponibles. Ejecuta un análisis para generar clips.")

with tab2:
    # Mostrar clips de la sesión actual
    if st.session_state.clips_encontrados_sesion:
        mostrar_player_clips(st.session_state.clips_encontrados_sesion, "🆕 Clips de Esta Sesión")
    else:
        st.info("📭 No se han generado clips en esta sesión. Ejecuta un análisis para ver clips aquí.")

with tab3:
    # Análisis y estadísticas
    st.markdown("### 📊 Análisis de la Biblioteca de Clips")
    
    todos_los_clips = buscar_todos_los_clips()
    if todos_los_clips:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Clips", len(todos_los_clips))
        
        with col2:
            terminos_unicos = len(set([clip['termino'] for clip in todos_los_clips]))
            st.metric("Términos Únicos", terminos_unicos)
        
        with col3:
            # Clips de hoy
            hoy = datetime.now().date()
            clips_hoy = sum(1 for clip in todos_los_clips 
                           if clip['fecha_creacion'].date() == hoy)
            st.metric("Clips de Hoy", clips_hoy)
        
        with col4:
            # Tamaño total
            tamano_total = sum([clip['size_mb'] for clip in todos_los_clips])
            st.metric("Tamaño Total", f"{tamano_total:.1f} MB")
        
        # Gráfico de distribución por términos
        if len(todos_los_clips) > 0:
            st.markdown("#### 📈 Distribución por Términos")
            terminos_count = {}
            for clip in todos_los_clips:
                termino = clip['termino']
                terminos_count[termino] = terminos_count.get(termino, 0) + 1
            
            # Crear DataFrame para el gráfico
            df_terminos = pd.DataFrame(list(terminos_count.items()), 
                                     columns=['Término', 'Cantidad'])
            df_terminos = df_terminos.sort_values('Cantidad', ascending=True)
            
            st.bar_chart(df_terminos.set_index('Término'))
            
            # Timeline de clips
            st.markdown("#### 📅 Timeline de Generación de Clips")
            df_timeline = pd.DataFrame(todos_los_clips)
            df_timeline['fecha'] = pd.to_datetime(df_timeline['fecha_creacion']).dt.date
            clips_por_dia = df_timeline.groupby('fecha').size().reset_index(name='clips')
            
            st.line_chart(clips_por_dia.set_index('fecha'))
    else:
        st.info("📭 No hay datos para analizar. Genera algunos clips primero.")

with tab4:
    # === ENTIDADES Y RUTAS - VISTA LIMPIA ===
    _clientes = obtener_clientes_activos()
    _config_t = cargar_configuracion_completa()
    _terminos = _config_t.get('terminos', [])
    
    # --- Encabezado ---
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("### 🏢 Entidades y Rutas de Envío")
    with col_header2:
        if st.button("➕ Nueva Entidad", type="primary", key="btn_abrir_nueva"):
            st.session_state['mostrar_form_nueva'] = not st.session_state.get('mostrar_form_nueva', False)

    # Vista rápida simplificada por cliente (una sola ventana con tabs)
    if _clientes:
        st.markdown("#### Vista rápida por cliente")
        _tabs_rapidos = st.tabs(
            [f"🏢 {nombre_cliente_mostrar_para_ui(c, c.get('id', ''))}" for c in _clientes]
        )
        for _cli, _tab_rapido in zip(_clientes, _tabs_rapidos):
            with _tab_rapido:
                _cid = _cli.get('id', '')
                _brevo = _cli.get('brevo', {})
                _active_emails, _emails_all = obtener_destinatarios_activos_cliente(_cli)
                _terms_count = len([t for t in _terminos if isinstance(t, dict) and t.get('cliente_id') == _cid])
                _c1, _c2, _c3 = st.columns(3)
                with _c1:
                    st.metric("Términos", _terms_count)
                with _c2:
                    st.metric("Correos activos", len(_active_emails))
                with _c3:
                    st.metric("Correos total", len(_emails_all))

                st.caption("Destinatarios con control independiente ON/OFF")
                _emails_changed = False
                for _idx, _item in enumerate(_emails_all):
                    _email = _item.get('email', '')
                    _nombre = _item.get('nombre', _email)
                    _ec1, _ec2 = st.columns([5, 1])
                    with _ec1:
                        st.write(f"{_nombre} (`{_email}`)")
                    with _ec2:
                        _on = st.toggle("Activo", value=_item.get('activo', True), key=f"quick_email_{_cid}_{_idx}_{_email}")
                        if _on != _item.get('activo', True):
                            _emails_all[_idx]['activo'] = _on
                            _emails_changed = True

                if _emails_changed and st.button("💾 Guardar destinatarios", key=f"quick_save_emails_{_cid}"):
                    _brevo['correos_destinatarios'] = _emails_all
                    _brevo['enabled'] = _brevo.get('enabled', True)
                    _cli['brevo'] = _brevo
                    actualizar_cliente(_cid, _cli)
                    st.success("Destinatarios guardados")
                    st.rerun()
        st.markdown("---")
    
    # ================================================================
    # FORMULARIO NUEVA ENTIDAD (toggle)
    # ================================================================
    if st.session_state.get('mostrar_form_nueva', False):
        st.markdown("---")
        st.markdown("#### 🆕 Crear Nueva Entidad")
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            _new_nombre = st.text_input("Nombre:", placeholder="Ej: EDENOR", key="ent_new_nombre")
        with c2:
            _new_color = st.color_picker("Color:", value="#FF5722", key="ent_new_color")
        with c3:
            _new_id = st.text_input("ID:", placeholder="edenor", key="ent_new_id")
        _new_terms = st.text_input("Términos (separados por coma):", placeholder="edenor, apagones norte", key="ent_new_terms")
        c_tg, c_email, c_drive = st.columns(3)
        with c_tg:
            st.markdown("**📱 Telegram**")
            _ntg = st.checkbox("Activar", key="ent_ntg")
            _ntg_token = st.text_input("Token:", type="password", key="ent_ntg_token") if _ntg else ""
            _ntg_chat = st.text_input("Chat ID:", key="ent_ntg_chat") if _ntg else ""
        with c_email:
            st.markdown("**📧 Email**")
            _nem = st.checkbox("Activar", key="ent_nem")
            _nem_correos = st.text_input("Correos (coma):", key="ent_nem_correos") if _nem else ""
        with c_drive:
            st.markdown("**☁️ Drive**")
            _ngd = st.checkbox("Activar", key="ent_ngd")
            _ngd_folder = st.text_input("Folder ID:", key="ent_ngd_folder") if _ngd else ""
        c_wh, c_sb, c_cl = st.columns(3)
        with c_wh:
            st.markdown("**🌐 Webhook**")
            _nwh = st.checkbox("Activar", key="ent_nwh")
            _nwh_url = st.text_input("URL:", key="ent_nwh_url") if _nwh else ""
        with c_sb:
            st.markdown("**🗄️ Supabase**")
            _nsb = st.checkbox("Activar", key="ent_nsb")
            _nsb_tabla = st.text_input("Tabla:", key="ent_nsb_tabla") if _nsb else ""
        with c_cl:
            st.markdown("**🖼️ Cloudinary**")
            _ncl = st.checkbox("Activar", value=True, key="ent_ncl")
        if st.button("✅ Crear Entidad", type="primary", key="btn_crear_ent"):
            if _new_nombre.strip():
                nc = crear_cliente_nuevo(_new_nombre.strip(), _new_color)
                if _new_id.strip():
                    nc['id'] = _new_id.strip().lower().replace(' ', '_')
                if _ntg and _ntg_token and _ntg_chat:
                    nc['telegram'] = {'enabled': True, 'bot_token': _ntg_token.strip(), 'chat_id': _ntg_chat.strip(), 'send_clips': True, 'send_summary': True, 'use_cloudinary': True}
                if _nem and _nem_correos:
                    correos = [c.strip() for c in _nem_correos.split(',') if '@' in c.strip()]
                    _brevo_key = (os.getenv("BREVO_API_KEY") or os.getenv("BREVO_SMTP_KEY") or "").strip()
                    _brevo_user = (os.getenv("BREVO_SMTP_USER") or "").strip()
                    _brevo_sender = (os.getenv("BREVO_SENDER_EMAIL") or "").strip()
                    nc['brevo'] = {
                        'enabled': True,
                        'api_key': _brevo_key,
                        'smtp_user': _brevo_user,
                        'smtp_server': os.getenv('BREVO_SMTP_SERVER', 'smtp-relay.brevo.com'),
                        'smtp_port': int(os.getenv('BREVO_SMTP_PORT', '587') or 587),
                        'sender_email': _brevo_sender,
                        'sender_name': f'FGJ Medios - {_new_nombre.strip()}',
                        'correos_destinatarios': correos,
                    }
                if _ngd and _ngd_folder:
                    nc['google_drive'] = {'enabled': True, 'folder_id': _ngd_folder.strip()}
                if _nwh and _nwh_url:
                    nc['webhook'] = {'enabled': True, 'url': _nwh_url.strip(), 'url_secundario': '', 'url_terciario': ''}
                if _nsb and _nsb_tabla:
                    nc['supabase'] = {
                        'enabled': True,
                        'url': SUPABASE_URL or '',
                        'anon_key': SUPABASE_ANON_KEY or '',
                        'tabla_nombre': _nsb_tabla.strip(),
                    }
                if _ncl:
                    nc['cloudinary'] = {
                        'enabled': True,
                        'cloud_name': os.getenv('CLOUDINARY_CLOUD_NAME', ''),
                        'api_key': os.getenv('CLOUDINARY_API_KEY', ''),
                        'api_secret': os.getenv('CLOUDINARY_API_SECRET', ''),
                        'folder': os.getenv('CLOUDINARY_FOLDER', 'video_analyzer_clips'),
                    }
                ok, msg = agregar_cliente(nc)
                if ok:
                    if _new_terms.strip():
                        terms = [t.strip().lower() for t in _new_terms.split(',') if t.strip()]
                        ct = cargar_configuracion_completa()
                        for tn in terms:
                            if not any((isinstance(x, dict) and x.get('termino', '').lower() == tn) for x in ct.get('terminos', [])):
                                ct['terminos'].append({'termino': tn, 'cliente_id': nc['id']})
                        ct['total_terminos'] = len(ct['terminos'])
                        ct['fecha_actualizacion'] = datetime.now().isoformat()
                        with open(TERMINOS_CONFIG, 'w', encoding='utf-8') as f:
                            json.dump(ct, f, indent=2, ensure_ascii=False)
                    st.session_state['mostrar_form_nueva'] = False
                    st.success(f"Entidad **{_new_nombre}** creada con ID **{nc['id']}**")
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Ingresa un nombre")
    
    st.markdown("---")
    
    # ================================================================
    # TARJETAS DE ENTIDADES
    # ================================================================
    for _ent in _clientes:
        _eid = _ent.get('id')
        _enombre = _ent.get('nombre', '?')
        _ecolor = _ent.get('color', '#4CAF50')
        _eterms = [t.get('termino', '') for t in _terminos if isinstance(t, dict) and t.get('cliente_id') == _eid]
        _destinos = []
        if _ent.get('telegram', {}).get('enabled'): _destinos.append(f"📱 {_ent['telegram'].get('chat_id', '')}")
        if _ent.get('brevo', {}).get('enabled'):
            _activos_tmp, _norm_tmp = obtener_destinatarios_activos_cliente(_ent)
            _destinos.append(f"📧 {len(_activos_tmp)} activos / {len(_norm_tmp)} total")
        if _ent.get('google_drive', {}).get('enabled'): _destinos.append("☁️ Drive")
        if _ent.get('webhook', {}).get('enabled'): _destinos.append("🌐 Webhook")
        if _ent.get('cloudinary', {}).get('enabled'): _destinos.append("🖼️ Cloudinary")
        if _ent.get('supabase', {}).get('enabled'): _destinos.append(f"🗄️ {_ent['supabase'].get('tabla_nombre', '')}")
        _terms_str = ", ".join(_eterms) if _eterms else "sin términos"
        _dest_str = " | ".join(_destinos) if _destinos else "sin destinos"
        
        with st.expander(f"**{_enombre}** — {len(_eterms)} términos — {len(_destinos)}/6 destinos", expanded=False):
            st.markdown(f"**{_enombre}** — {len(_eterms)} términos — {len(_destinos)}/6 destinos")
            st.markdown(f"**Términos:** {_terms_str}")
            st.markdown(f"**Destinos:** {_dest_str}")
            st.markdown("---")
            # Términos
            st.markdown("##### 🔍 Términos")
            for _it, _term in enumerate(_eterms):
                _tc1, _tc2 = st.columns([5, 1])
                with _tc1:
                    st.text(f"  {_term}")
                with _tc2:
                    if st.button("✕", key=f"x_term_{_eid}_{_it}"):
                        _cfg = cargar_configuracion_completa()
                        _cfg['terminos'] = [t for t in _cfg.get('terminos', []) if not (isinstance(t, dict) and t.get('termino', '').lower() == _term.lower() and t.get('cliente_id') == _eid)]
                        _cfg['total_terminos'] = len(_cfg['terminos'])
                        with open(TERMINOS_CONFIG, 'w', encoding='utf-8') as f:
                            json.dump(_cfg, f, indent=2, ensure_ascii=False)
                        st.rerun()
            _at1, _at2 = st.columns([4, 1])
            with _at1:
                _nuevo_t = st.text_input("Nuevo término:", key=f"add_t_{_eid}", label_visibility="collapsed", placeholder="agregar término...")
            with _at2:
                if st.button("➕", key=f"btn_add_t_{_eid}"):
                    if _nuevo_t.strip():
                        _cfg2 = cargar_configuracion_completa()
                        _cfg2['terminos'].append({'termino': _nuevo_t.strip().lower(), 'cliente_id': _eid})
                        _cfg2['total_terminos'] = len(_cfg2['terminos'])
                        _cfg2['fecha_actualizacion'] = datetime.now().isoformat()
                        with open(TERMINOS_CONFIG, 'w', encoding='utf-8') as f:
                            json.dump(_cfg2, f, indent=2, ensure_ascii=False)
                        st.rerun()
            st.markdown("---")
            # Destinos en grid
            st.markdown("##### 📤 Destinos")
            _d1, _d2, _d3 = st.columns(3)
            with _d1:
                _tg = _ent.get('telegram', {})
                _tg_on = st.checkbox("📱 Telegram", value=_tg.get('enabled', False), key=f"tg_{_eid}")
                if _tg_on:
                    _tg_tok = st.text_input("Token:", value=_tg.get('bot_token', ''), type="password", key=f"tg_tok_{_eid}")
                    _tg_ch = st.text_input("Chat ID:", value=_tg.get('chat_id', ''), key=f"tg_ch_{_eid}")
                    if st.button("💾", key=f"sv_tg_{_eid}"):
                        _ent['telegram'] = {'enabled': True, 'bot_token': _tg_tok.strip(), 'chat_id': _tg_ch.strip(), 'send_clips': True, 'send_summary': True, 'use_cloudinary': True}
                        actualizar_cliente(_eid, _ent); st.success("OK")
            with _d2:
                _br = _ent.get('brevo', {})
                _br_on = st.checkbox("📧 Email", value=_br.get('enabled', False), key=f"br_{_eid}")
                if _br_on:
                    _correos_activos, _correos = obtener_destinatarios_activos_cliente(_ent)
                    st.caption(f"{len(_correos_activos)} activos de {len(_correos)} destinatarios")
                    if _correos:
                        st.markdown("**Destinatarios (ON/OFF independiente):**")
                        _cambios_estado = False
                        for _idx, _correo_item in enumerate(_correos):
                            _email = _correo_item.get('email', '')
                            _nombre = _correo_item.get('nombre', _email)
                            _col_c1, _col_c2, _col_c3 = st.columns([4, 2, 1])
                            with _col_c1:
                                st.caption(f"{_nombre} — `{_email}`")
                            with _col_c2:
                                _activo = st.checkbox(
                                    "Activo",
                                    value=_correo_item.get('activo', True),
                                    key=f"br_active_{_eid}_{_idx}_{_email}"
                                )
                                if _activo != _correo_item.get('activo', True):
                                    _correos[_idx]['activo'] = _activo
                                    _cambios_estado = True
                            with _col_c3:
                                if st.button("🗑️", key=f"br_del_{_eid}_{_idx}_{_email}"):
                                    _correos = [c for c in _correos if (c.get('email') or '') != _email]
                                    _ent['brevo']['correos_destinatarios'] = _correos
                                    _ent['brevo']['enabled'] = True
                                    actualizar_cliente(_eid, _ent)
                                    st.rerun()
                        if _cambios_estado and st.button("💾 Guardar ON/OFF", key=f"br_save_toggle_{_eid}"):
                            _ent['brevo']['correos_destinatarios'] = _correos
                            _ent['brevo']['enabled'] = True
                            actualizar_cliente(_eid, _ent)
                            st.success("Destinatarios actualizados")
                            st.rerun()
                    _new_c = st.text_input("Agregar:", key=f"br_add_{_eid}", placeholder="email@...")
                    if st.button("➕", key=f"br_btn_{_eid}") and _new_c.strip() and '@' in _new_c:
                        _correos.append({
                            'email': _new_c.strip(),
                            'correo': _new_c.strip(),
                            'nombre': _new_c.strip().split('@')[0],
                            'activo': True
                        })
                        _ent['brevo']['correos_destinatarios'] = _correos; _ent['brevo']['enabled'] = True
                        if not _ent['brevo'].get('api_key'):
                            _bk = (os.getenv("BREVO_API_KEY") or os.getenv("BREVO_SMTP_KEY") or "").strip()
                            _bu = (os.getenv("BREVO_SMTP_USER") or "").strip()
                            _bs = (os.getenv("BREVO_SENDER_EMAIL") or "").strip()
                            _ent['brevo'].update({
                                'api_key': _bk,
                                'smtp_user': _bu,
                                'smtp_server': os.getenv('BREVO_SMTP_SERVER', 'smtp-relay.brevo.com'),
                                'smtp_port': int(os.getenv('BREVO_SMTP_PORT', '587') or 587),
                                'sender_email': _bs,
                                'sender_name': f'FGJ Medios - {_enombre}',
                            })
                        actualizar_cliente(_eid, _ent); st.rerun()
            with _d3:
                _gd = _ent.get('google_drive', {})
                _gd_on = st.checkbox("☁️ Drive", value=_gd.get('enabled', False), key=f"gd_{_eid}")
                if _gd_on:
                    _gd_f = st.text_input("Folder ID:", value=_gd.get('folder_id', ''), key=f"gd_f_{_eid}")
                    if st.button("💾", key=f"sv_gd_{_eid}"):
                        _ent['google_drive'] = {'enabled': True, 'folder_id': _gd_f.strip()}
                        actualizar_cliente(_eid, _ent); st.success("OK")
            _d4, _d5, _d6 = st.columns(3)
            with _d4:
                _wh = _ent.get('webhook', {})
                _wh_on = st.checkbox("🌐 Webhook", value=_wh.get('enabled', False), key=f"wh_{_eid}")
                if _wh_on:
                    _wh_u = st.text_input("URL:", value=_wh.get('url', ''), key=f"wh_u_{_eid}")
                    if st.button("💾", key=f"sv_wh_{_eid}"):
                        _ent['webhook'] = {'enabled': True, 'url': _wh_u.strip(), 'url_secundario': '', 'url_terciario': ''}
                        actualizar_cliente(_eid, _ent); st.success("OK")
            with _d5:
                _cl = _ent.get('cloudinary', {})
                _cl_on = st.checkbox("🖼️ Cloudinary", value=_cl.get('enabled', False), key=f"cl_{_eid}")
                if _cl_on and not _cl.get('cloud_name'):
                    if st.button("Activar", key=f"sv_cl_{_eid}"):
                        _ent['cloudinary'] = {
                            'enabled': True,
                            'cloud_name': os.getenv('CLOUDINARY_CLOUD_NAME', ''),
                            'api_key': os.getenv('CLOUDINARY_API_KEY', ''),
                            'api_secret': os.getenv('CLOUDINARY_API_SECRET', ''),
                            'folder': os.getenv('CLOUDINARY_FOLDER', 'video_analyzer_clips'),
                        }
                        actualizar_cliente(_eid, _ent); st.success("OK")
                elif _cl_on:
                    st.caption(f"Cloud: {_cl.get('cloud_name', '')}")
            with _d6:
                _sb = _ent.get('supabase', {})
                _sb_on = st.checkbox("🗄️ Supabase", value=_sb.get('enabled', False), key=f"sb_{_eid}")
                if _sb_on:
                    _sb_t = st.text_input("Tabla:", value=_sb.get('tabla_nombre', ''), key=f"sb_t_{_eid}")
                    if st.button("💾", key=f"sv_sb_{_eid}"):
                        _ent['supabase'] = {'enabled': True, 'url': _sb.get('url', 'https://sfvbcprhfmwglqpyyfxz.supabase.co'), 'anon_key': _sb.get('anon_key', ''), 'tabla_nombre': _sb_t.strip()}
                        actualizar_cliente(_eid, _ent); st.success("OK")
            if _eid != 'default':
                st.markdown("---")
                if st.button(f"🗑️ Eliminar {_enombre}", key=f"del_{_eid}"):
                    _cdel = cargar_configuracion_completa()
                    _cdel['terminos'] = [t for t in _cdel.get('terminos', []) if not (isinstance(t, dict) and t.get('cliente_id') == _eid)]
                    _cdel['total_terminos'] = len(_cdel['terminos'])
                    with open(TERMINOS_CONFIG, 'w', encoding='utf-8') as f:
                        json.dump(_cdel, f, indent=2, ensure_ascii=False)
                    eliminar_cliente(_eid); st.rerun()

# === FUNCIÓN PARA SINCRONIZAR CON SUPABASE DESDE coincidencias.md ===
def sincronizar_coincidencias_md_a_supabase():
    """
    Lee el archivo coincidencias.md y sincroniza con Supabase:
    1. Elimina duplicados
    2. Inserta coincidencias faltantes
    """
    func_name = "sincronizar_coincidencias_md_a_supabase"
    
    if not supabase:
        return False, "❌ Cliente de Supabase no inicializado"
    
    # Verificar si existe el archivo
    if not os.path.exists("coincidencias.md"):
        return False, "❌ Archivo coincidencias.md no encontrado"
    
    try:
        # Leer el archivo
        with open("coincidencias.md", "r", encoding="utf-8") as f:
            contenido = f.read()
        
        # PASO 1: Limpiar duplicados por URL de Cloudinary
        st.info("🧹 Paso 1: Limpiando registros duplicados por URL de Cloudinary...")
        result = supabase.table('alertas_medios').select('*').order('fecha_detencion', desc=False).execute()
        
        # Agrupar por URL de Cloudinary (url_video o enlace_directo)
        registros_por_url = {}
        duplicados_ids = []
        
        for reg in result.data:
            url = reg.get('url_video', '') or reg.get('enlace_directo', '')
            
            # Solo procesar si hay URL de Cloudinary
            if url and 'cloudinary' in url.lower():
                if url in registros_por_url:
                    # Ya existe uno con esta URL
                    # Comparar fechas para mantener el más antiguo
                    registro_existente = registros_por_url[url]
                    fecha_existente = registro_existente.get('fecha_detencion', '')
                    fecha_actual = reg.get('fecha_detencion', '')
                    
                    # Si el actual es más reciente, lo marcamos para eliminar
                    if fecha_actual > fecha_existente:
                        duplicados_ids.append(reg['id'])
                        log_info(f"Duplicado reciente encontrado: ID {reg['id']} (fecha: {fecha_actual})", func_name)
                    else:
                        # El existente es más reciente, eliminamos el existente y guardamos el actual
                        duplicados_ids.append(registro_existente['id'])
                        registros_por_url[url] = reg
                        log_info(f"Duplicado reciente encontrado: ID {registro_existente['id']} (fecha: {fecha_existente})", func_name)
                else:
                    # Primera vez que vemos esta URL
                    registros_por_url[url] = reg
        
        if duplicados_ids:
            eliminados = 0
            for dup_id in duplicados_ids:
                try:
                    supabase.table('alertas_medios').delete().eq('id', dup_id).execute()
                    eliminados += 1
                except Exception as e:
                    log_warning(f"Error eliminando duplicado ID {dup_id}: {e}", func_name)
            
            st.success(f"✅ Eliminados {eliminados} registros duplicados (manteniendo los más antiguos)")
            log_info(f"Duplicados eliminados: {eliminados} de {len(duplicados_ids)} intentos", func_name)
        else:
            st.success("✅ No se encontraron duplicados por URL de Cloudinary")
        
        # PASO 2: Extraer coincidencias del MD
        st.info("📖 Paso 2: Extrayendo coincidencias del archivo MD...")
        
        # Extraer URLs de Cloudinary del contenido
        import re
        urls = re.findall(r'https://res\.cloudinary\.com/[^\s\)]+', contenido)
        
        # Extraer información de cada coincidencia
        bloques = re.split(r'\d+\.\s+', contenido)[1:]  # Dividir por numeración
        
        coincidencias_nuevas = []
        for i, bloque in enumerate(bloques):
            try:
                # Extraer datos del bloque
                termino_match = re.search(r'Menciones de (\w+)', bloque, re.IGNORECASE)
                medio_match = re.search(r'Medio:\s*([^\n]+)', bloque)
                hora_match = re.search(r'Hora:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{1,2}:\d{2})', bloque)
                contexto_match = re.search(r'Contexto:\s*([^\n]+)', bloque)
                resumen_match = re.search(r'Resumen:\s*([^\n]+)', bloque)
                archivo_match = re.search(r'URL Video:\s*([^\n]+)', bloque)
                
                if i < len(urls):
                    url_video = urls[i]
                    
                    coincidencia = {
                        'fecha_detencion': datetime.now().isoformat(),
                        'termino_detectado': termino_match.group(1) if termino_match else 'desconocido',
                        'nombre_medio': medio_match.group(1).strip() if medio_match else 'Medio de Comunicacion',
                        'contexto': contexto_match.group(1).strip() if contexto_match else 'Sin contexto',
                        'resumen_ejecutivo': resumen_match.group(1).strip() if resumen_match else 'Sin resumen',
                        'url_video': url_video,
                        'nombre_archivo': archivo_match.group(1).strip() if archivo_match else 'desconocido',
                        'enlace_directo': url_video,
                        'transcripcion': bloque[:500],
                        'relevancia': 'Alta'
                    }
                    
                    # Parsear fecha y hora
                    if hora_match:
                        fecha_str = hora_match.group(1)  # DD/MM/YYYY
                        hora_str = hora_match.group(2)   # HH:MM
                        
                        try:
                            from datetime import datetime as dt, date, time
                            fecha_parts = fecha_str.split('/')
                            hora_parts = hora_str.split(':')
                            
                            coincidencia['fecha_programa'] = date(
                                int(fecha_parts[2]), 
                                int(fecha_parts[1]), 
                                int(fecha_parts[0])
                            ).isoformat()
                            
                            coincidencia['hora_programa'] = time(
                                int(hora_parts[0]), 
                                int(hora_parts[1])
                            ).isoformat()
                        except:
                            from datetime import date, time
                            coincidencia['fecha_programa'] = date.today().isoformat()
                            coincidencia['hora_programa'] = time(12, 0).isoformat()
                    
                    coincidencias_nuevas.append(coincidencia)
            except Exception as e:
                log_warning(f"Error procesando bloque {i+1}: {e}", func_name)
        
        # PASO 3: Insertar coincidencias
        st.info(f"📥 Paso 3: Insertando {len(coincidencias_nuevas)} coincidencias...")
        
        insertadas = 0
        ya_existen = 0
        
        for coincidencia in coincidencias_nuevas:
            try:
                # Verificar si ya existe
                existing = supabase.table('alertas_medios').select('id').eq('url_video', coincidencia['url_video']).execute()
                
                if existing.data and len(existing.data) > 0:
                    ya_existen += 1
                else:
                    result = supabase.table('alertas_medios').insert(coincidencia).execute()
                    if result.data:
                        insertadas += 1
            except Exception as e:
                log_warning(f"Error insertando coincidencia: {e}", func_name)
        
        mensaje = f"✅ Sincronización completada:\n- Coincidencias insertadas: {insertadas}\n- Ya existían: {ya_existen}\n- Duplicados eliminados: {len(duplicados_ids)}"
        return True, mensaje
        
    except Exception as e:
        error_msg = f"Error en sincronización: {str(e)}"
        log_error_critico(func_name, error_msg)
        return False, f"❌ {error_msg}"

# === SECCIÓN DE SINCRONIZACIÓN CON SUPABASE ===
st.markdown("---")
st.markdown("## 🗄️ Sincronización con Supabase")

col1, col2 = st.columns([3, 1])
with col1:
    st.write("Sincroniza el archivo `coincidencias.md` con Supabase: limpia duplicados e inserta coincidencias faltantes")
with col2:
    if st.button("🔄 Sincronizar desde MD", help="Lee coincidencias.md y sincroniza con Supabase"):
        with st.spinner("Sincronizando..."):
            exito, mensaje = sincronizar_coincidencias_md_a_supabase()
            if exito:
                st.success(mensaje)
            else:
                st.error(mensaje)

# === SECCIÓN DE LOGS ===
st.markdown("---")
st.markdown("## 📋 Sistema de Logs")

def mostrar_logs():
    """
    Interfaz para ver los logs del sistema
    """
    log_dir = Path("logs")
    
    if not log_dir.exists():
        st.warning("📁 No se encontró el directorio de logs")
        return
    
    # Selector de tipo de log
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 Ver Errores", help="Mostrar solo errores críticos"):
            st.session_state.log_type = "errors"
    
    with col2:
        if st.button("ℹ️ Ver Info General", help="Mostrar información general de la aplicación"):
            st.session_state.log_type = "info"
    
    with col3:
        if st.button("🔍 Ver Debug", help="Mostrar información detallada de debug"):
            st.session_state.log_type = "debug"
    
    # Inicializar tipo de log si no existe
    if 'log_type' not in st.session_state:
        st.session_state.log_type = "errors"
    
    # Obtener archivos de log del día actual
    today = datetime.now().strftime("%Y%m%d")
    log_files = {
        "errors": log_dir / f"errors_{today}.log",
        "info": log_dir / f"app_{today}.log", 
        "debug": log_dir / f"debug_{today}.log"
    }
    
    selected_file = log_files[st.session_state.log_type]
    
    if selected_file.exists():
        try:
            # Leer las últimas 100 líneas del archivo
            with open(selected_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Mostrar las últimas líneas
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            
            st.markdown(f"### 📄 {st.session_state.log_type.title()} - Últimas {len(recent_lines)} entradas")
            
            # Mostrar en un contenedor con scroll
            log_content = "".join(recent_lines)
            st.text_area(
                f"Logs de {st.session_state.log_type}:",
                value=log_content,
                height=400,
                help=f"Archivo: {selected_file.name}"
            )
            
            # Información adicional
            file_size = selected_file.stat().st_size / 1024  # KB
            st.caption(f"📊 Tamaño del archivo: {file_size:.1f} KB | Total líneas: {len(lines)}")
            
            # Botón para limpiar logs
            if st.button("🗑️ Limpiar Logs", help="Eliminar logs antiguos"):
                try:
                    selected_file.unlink()
                    st.success(f"✅ Logs de {st.session_state.log_type} eliminados")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error eliminando logs: {e}")
                    
        except Exception as e:
            st.error(f"❌ Error leyendo archivo de log: {e}")
            log_exception("mostrar_logs", e, f"Archivo: {selected_file}")
    else:
        st.info(f"📁 No hay logs de {st.session_state.log_type} para hoy")
        st.caption(f"Archivo esperado: {selected_file.name}")

# Mostrar la interfaz de logs
with st.expander("📋 Ver Logs del Sistema", expanded=False):
    mostrar_logs()

# === INFORMACIÓN DE ESTADO FINAL ===
if st.session_state.running:
    st.markdown("---")
    st.success("🔄 **Búsqueda continua activa** - El procesamiento se ejecutará automáticamente con progreso visual completo")
    
    # Mostrar información del próximo procesamiento
    if 'ultimo_procesamiento_continuo' in st.session_state and st.session_state.ultimo_procesamiento_continuo > 0:
        tiempo_actual = time.time()
        tiempo_desde_ultimo = tiempo_actual - st.session_state.ultimo_procesamiento_continuo
        tiempo_restante = st.session_state.intervalo - int(tiempo_desde_ultimo)
        
        if tiempo_restante > 0:
            st.info(f"⏳ Próximo procesamiento automático en {tiempo_restante} segundos")
        else:
            st.info("🔄 Procesamiento automático iniciando...")

# === AUTO-REFRESH PARA BÚSQUEDA CONTINUA ===
# El auto-refresh ahora se maneja en la lógica principal de procesamiento continuo

# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":
    # This ensures the Streamlit app only runs when executed directly
    # and not when imported as a module
    # Streamlit app runs automatically when script is executed directly
    pass
