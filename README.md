# Radio Analyzer

**Versión 4.5**

Aplicación Streamlit para monitorear carpetas de **audio**, transcribir con IA (Whisper / Mistral / OpenAI), detectar **términos por cliente**, generar **clips** y notificar a **Telegram**, **correo (Brevo)**, **webhooks**, **Google Drive**, **Cloudinary**, **Supabase** y **Google Sheets**.

## Requisitos

- Python 3.10+ (recomendado 3.11–3.12)
- **FFmpeg** y **ffprobe** en el `PATH`
- Cuentas y claves en `.env` (ver `env.radio.example` o documentación interna del proyecto)

## Instalación rápida

```text
python -m venv venv
venv\Scripts\activate
pip install -r requirements-radio.txt
```

- API keys solo en `.env` (p. ej. `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_*`, `CLOUDINARY_*`, OAuth Google). Para el asistente de **nueva entidad** en la UI: `BREVO_API_KEY` o `BREVO_SMTP_KEY`, `BREVO_SMTP_USER`, `BREVO_SENDER_EMAIL`, `BREVO_SMTP_SERVER` (opcional).

## Ejecución

Desde la raíz del repositorio:

```text
streamlit run radioAnalizer.py
```

## Configuración principal

| Recurso | Uso |
|--------|-----|
| `clientes_config.json` | Clientes, términos, Telegram, Brevo, Drive, Sheets |
| `terminos_guardados.json` | Términos globales / sesión |
| Variables `CARPETA_VIDEOS` / `CARPETA_PROCESADOS` | Entrada de audios y carpeta AUDIOCHECKS (clips, logs, snapshot) |

No subas `.env` ni JSON con secretos a Git.

## Novedades en 4.5

- **Envíos de coincidencias reales:** el flujo principal (`enviar_coincidencia_inmediata` → `enviar_coincidencia_a_cliente`) **ya no se omite** por el flag `envios_habilitados` del cliente (comportamiento alineado con notificaciones críticas / tangenciales).
- **Google Sheets:** la fila de coincidencia se escribe cuando hay hoja configurada, sin depender de `envios_habilitados`.
- **Reenvío:** se guarda un snapshot en `ultima_coincidencia_reenvio.json` dentro de `CARPETA_PROCESADOS` tras un envío correcto, para poder repetir el envío con herramientas del proyecto.

## Estructura mínima del módulo principal

- `radioAnalizer.py` — UI Streamlit, bucle de escaneo, transcripción, detección, clips, integraciones multi-cliente.

Otros archivos del árbol local (p. ej. `intro_coincidencia_tts.py`, `emisoras_catalogo.py`, `coincidencias_logger.py`) son **dependencias** del mismo proyecto; para un clon mínimo necesitas el conjunto de imports que declare el propio `radioAnalizer.py`.

## Licencia y autor

Uso interno / según convención del repositorio `videonalizer`.
