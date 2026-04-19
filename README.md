# 🧠 Analizador de Videos Pro - Sistema Inteligente de Detección con GPT-4o

## 📋 DESCRIPCIÓN GENERAL

`transmistral2.py` es un analizador inteligente de videos/audios potenciado por **GPT-4o** que automáticamente:
1. 🔍 Escanea carpetas buscando archivos multimedia
2. 🎤 Los transcribe usando IA (Whisper, Mistral, OpenAI)
3. 🔎 Busca términos específicos en las transcripciones
4. 🤖 **GPT-4o determina el segmento más lógico y coherente** (no recorte mecánico)
5. ✂️ Genera clips inteligentes cuando encuentra coincidencias
6. 🤖 **GPT-4o evalúa críticamente la relevancia** (descarta menciones superficiales)
7. 🤖 **GPT-4o extrae la idea general del segmento** (no toda la transcripción)
8. 📤 Envía solo información relevante a múltiples destinos automáticamente

## 🆕 NUEVA FUNCIONALIDAD GPT-4o (v2.6)

### 🎯 **1. SEGMENTACIÓN INTELIGENTE DE CLIPS**

**ANTES** (Método mecánico):
```
Coincidencia encontrada → Recortar 30s antes + 30s después
❌ Cortes abruptos en medio de frases
❌ Sin considerar contexto narrativo
❌ Duración fija sin sentido lógico
```

**AHORA** (GPT-4o inteligente):
```
Coincidencia encontrada → GPT-4o analiza transcripción completa
✅ Identifica inicio NATURAL de la idea
✅ Identifica fin NATURAL de la idea
✅ Captura contexto relevante completo
✅ Duración variable según contenido (10-60s)
✅ Transiciones suaves y coherentes
```

**Función implementada:** `determinar_segmento_inteligente_gpt4()` (Líneas 5659-5814)

**Características:**
- 📝 Analiza transcripción con timestamps
- 🎯 Identifica segmento más lógico para cada término
- ⏱️ Respeta límite de 60 segundos máximo
- 🛡️ Validaciones automáticas (min 10s, max 60s)
- 🔄 Fallback al método tradicional si GPT-4o falla
- 💡 Proporciona **razón** de cada decisión

### 🤖 **2. EXTRACCIÓN INTELIGENTE DE IDEAS**

**ANTES** (Envío completo):
```
📤 Envío a Telegram/Supabase/Correo:
- Transcripción completa del video (varios minutos)
- Contexto genérico de 300 caracteres
- Información redundante
❌ Mensajes largos difíciles de leer
❌ Mucha información irrelevante
```

**AHORA** (GPT-4o condensado):
```
📤 Envío a Telegram/Supabase/Correo:
- 🤖 Idea general del segmento (1-2 párrafos)
- Extraída por GPT-4o del clip específico
- Solo lo relevante al término encontrado
✅ Mensajes concisos y claros
✅ Solo información relevante (150 palabras max)
✅ Fácil de entender y procesar
```

**Función implementada:** `extraer_idea_general_segmento_gpt4()` (Líneas 5816-5881)

**Características:**
- 📝 Recibe solo transcripción del segmento del clip
- 🤖 GPT-4o extrae idea principal en 1-2 párrafos
- 🎯 Enfocado exclusivamente en el término encontrado
- ⚡ Respuesta directa sin introducción innecesaria
- 🔄 Fallback a resumen simple si GPT-4o falla

### 🔍 **3. ANÁLISIS CRÍTICO DE RELEVANCIA** ⭐ NUEVO (v2.6)

**PROBLEMA ANTERIOR:**
```
❌ Se generaban clips para TODA mención del término
❌ "apagón" mencionado de pasada → CLIP generado
❌ "Hoy hablaremos de inflación, apagones, salud..." → CLIP generado
❌ Información sin contexto sustancial → CLIP generado
❌ Múltiples clips del mismo momento (cada 5 segundos)
```

**SOLUCIÓN AHORA:**
```
✅ GPT-4o evalúa CRÍTICAMENTE la relevancia del contexto
✅ Solo clips con información CONCRETA y SUSTANCIAL
✅ Descarta menciones superficiales automáticamente
✅ Mínimo 1 MINUTO entre clips del mismo video
```

**Características Implementadas:**

#### 🎯 **A) Evaluación Crítica de Contexto**

GPT-4o ahora evalúa cada segmento con estos **criterios estrictos**:

1. ¿El término se discute con **detalle** o solo se menciona de pasada?
2. ¿Hay información **concreta** (causas, consecuencias, ubicación, tiempo, detalles)?
3. ¿Se desarrolla una **idea completa** o solo es una mención superficial?
4. ¿El segmento aporta **información valiosa** sobre el término?

**Respuestas de GPT-4o:**

```python
# Si NO es relevante:
"NO_RELEVANTE: El término 'apagón' se menciona sin contexto sustancial."
→ El clip se ELIMINA automáticamente

# Si ES relevante:
"El apagón afectó a 3 estados durante 8 horas debido a una falla en 
la central eléctrica principal. Las autoridades estiman que la 
normalización del servicio tomará hasta 12 horas..."
→ El clip se PROCESA y ENVÍA
```

**Ejemplos de Menciones:**

| Transcripción | ¿Se Procesa? | Razón |
|---------------|--------------|-------|
| "Hoy hablaremos de inflación, apagones y economía..." | ❌ NO | Mención sin contexto |
| "Entre los temas: educación, apagón, salud..." | ❌ NO | Lista superficial |
| "Se menciona un posible apagón..." | ❌ NO | Vago, sin detalles |
| "El apagón afectó 3 estados por 8 horas debido a falla en central..." | ✅ SÍ | Información concreta |
| "Los apagones programados serán de 2 horas cada tarde en zona norte..." | ✅ SÍ | Detalles específicos |
| "Reportan apagón masivo que dejó sin luz a 500,000 familias..." | ✅ SÍ | Datos sustanciales |

#### 🚫 **B) Control de Duplicados Mejorado**

**ANTES (v2.5):**
```
Tolerancia: ±5 segundos entre clips
Resultado: Múltiples clips del mismo momento
Problema: Información repetitiva
```

**AHORA (v2.6):**
```
Tolerancia: ±60 segundos (1 minuto) entre clips
Resultado: Solo 1 clip por minuto del video
Beneficio: Elimina redundancia de información
```

**Implementación:**
```python
# Línea 7047 en transmistral2.py
if diferencia <= 60:  # Tolerancia de 60 segundos (1 minuto)
    es_duplicado = True
    log_info(f"⏭️ Término '{termino}' OMITIDO - Ya existe clip...")
    st.info(f"⏭️ Término '{termino}' omitido - Ya existe clip reciente")
```

#### ⚙️ **C) Flujo Completo con Verificación**

```
1. 🔍 Detecta término "apagón" en timestamp 2:30

2. 🤖 GPT-4o ANALIZA SEGMENTO
   └─> Determina inicio/fin inteligente
   └─> Genera clip de 25 segundos

3. 🤖 GPT-4o EVALÚA RELEVANCIA
   ├─> ¿Hay contexto sustancial? NO
   └─> Responde: "NO_RELEVANTE: El término se menciona sin contexto..."

4. 🗑️ ELIMINA CLIP AUTOMÁTICAMENTE
   └─> No envía a Telegram, Webhook, Supabase, Email
   └─> Registra en logs el descarte
   └─> Continúa con siguiente término

5. ✅ O SI ES RELEVANTE:
   └─> Procesa clip normalmente
   └─> Envía a todos los destinos
   └─> Registra timestamp procesado (control duplicados)

6. ⏱️ CONTROL DE DUPLICADOS
   └─> Si detecta mismo término a 2:45 (15s después)
   └─> Diferencia: 15s < 60s → OMITIR
   └─> Si detecta mismo término a 3:45 (75s después)
   └─> Diferencia: 75s > 60s → PROCESAR
```

**Funciones Actualizadas:**
- `extraer_idea_general_segmento_gpt4()` - Prompt crítico mejorado (Líneas 5849-5881)
- `buscar_y_procesar_videos()` - Control de duplicados 60s (Línea 7047)
- `buscar_y_procesar_videos()` - Verificación de relevancia (Líneas 7200-7211)

**Beneficios:**

| Aspecto | Impacto |
|---------|---------|
| **Calidad de clips** | Solo información sustancial y valiosa |
| **Reducción de ruido** | Elimina 60-80% de menciones superficiales |
| **Eficiencia** | Menos clips = menos storage, menos envíos |
| **Experiencia de usuario** | Notificaciones más relevantes y accionables |
| **Costos** | Reduce uso de GPT-4o en clips no relevantes |
| **Precisión** | Mayor confianza en alertas recibidas |

---

## 📤 DESTINOS DE ENVÍO

Cuando se detecta una coincidencia, el sistema envía datos a **6 destinos diferentes**:

### 1. 📱 TELEGRAM
- Resumen ejecutivo en texto
- Clip de video (subido a Cloudinary)
- Información del medio y hora
- Contexto de la coincidencia

**Funciones:** `enviar_mensaje_telegram()`, `enviar_video_telegram_inteligente()`, `enviar_clips_a_telegram()`

### 2. 🌐 WEBHOOK
Envía JSON con:
- Tipo de evento
- Archivo origen
- Término detectado
- Contexto
- Resumen ejecutivo
- Timestamp

**Funciones:** `enviar_clips_a_webhook()`, `enviar_a_webhook_individual()`

### 3. ☁️ GOOGLE DRIVE
- Clips de video generados (.mp4)
- Transcripciones completas (.txt)
- Archivos de coincidencias con resumen (.txt)

**Funciones:** `subir_archivo_google_drive()`, `enviar_clips_a_google_drive()`

### 4. 🗄️ SUPABASE (Base de datos)
Tabla: `alertas_medios`

Datos enviados:
- termino_detectado
- nombre_medio
- hora_programa / fecha_programa
- **url_video** (URL de Cloudinary del CLIP)
- nombre_archivo
- contexto
- resumen_ejecutivo
- transcripcion
- relevancia

⚠️ **IMPORTANTE:** Se envía la URL del CLIP, NO del video principal.

**Funciones:** `enviar_coincidencias_a_supabase()`

### 5. 📧 BREVO (Email)
- Email HTML con resumen ejecutivo
- Enlace al clip en Google Drive
- URL de Cloudinary del video
- Información del medio y términos detectados

**API Key:** configurar en **`.env`** como `BREVO_API_KEY` (no incluir claves reales en documentación ni en el repositorio).

**Funciones:** `enviar_correo_brevo()`, `crear_plantilla_email_html()`

### 6. ☁️ CLOUDINARY (Hosting de videos)
- Sube SOLO los clips de coincidencias (no el video principal)
- Genera URLs públicas para compartir
- Estas URLs se envían a Supabase, Telegram, Email

**Funciones:** `subir_video_cloudinary()`, `configurar_cloudinary()`

---

## 🔄 FLUJO DE TRABAJO CON GPT-4o

```
1. 🔍 ESCANEO
   └─> Busca: .mp4, .mp3, .wav, .mkv, .avi
   └─> Verifica si ya fueron procesados
   └─> Verifica si están en fallidos.txt ⭐

2. 🎤 TRANSCRIPCIÓN
   └─> Extrae audio con ffmpeg
   └─> Transcribe con Mistral/OpenAI/Faster-Whisper
   └─> Obtiene timestamps precisos (Whisper)

3. 🔍 BÚSQUEDA
   └─> Busca términos en transcripción
   └─> Identifica timestamps exactos
   └─> Encuentra mejor coincidencia

4. 🤖 ANÁLISIS GPT-4o DEL SEGMENTO ⭐
   └─> Envía transcripción completa con timestamps
   └─> GPT-4o determina inicio NATURAL de la idea
   └─> GPT-4o determina fin NATURAL de la idea
   └─> Retorna: inicio, fin, duración, razón
   └─> Valida límites (10-60 segundos)

5. ✂️ GENERACIÓN DE CLIPS INTELIGENTES
   └─> Corta clip usando tiempos de GPT-4o
   └─> Duración variable (no siempre 60s)
   └─> Verifica que el clip contenga el término
   └─> Si no lo contiene, lo descarta

6. 🤖 EXTRACCIÓN DE IDEA GENERAL ⭐
   └─> Extrae transcripción SOLO del segmento del clip
   └─> GPT-4o analiza y condensa en 1-2 párrafos
   └─> Genera resumen ejecutivo (150 palabras max)
   └─> Enfocado solo en el término detectado

6.5. 🔍 ANÁLISIS CRÍTICO DE RELEVANCIA ⭐ NUEVO (v2.6)
   └─> GPT-4o evalúa si la mención es sustancial
   └─> Verifica criterios: detalle, contexto concreto, desarrollo
   ├─> SI NO ES RELEVANTE:
   │   └─> Elimina el clip automáticamente
   │   └─> Registra descarte en logs
   │   └─> NO envía a ningún destino
   └─> SI ES RELEVANTE:
       └─> Continúa con envío normal

6.6. 🚫 CONTROL DE DUPLICADOS ⭐ NUEVO (v2.6)
   └─> Verifica si hay clip reciente (< 60 segundos)
   ├─> SI HAY CLIP RECIENTE:
   │   └─> Omite este clip (evita redundancia)
   │   └─> Registra omisión en logs
   └─> SI NO HAY CLIP RECIENTE:
       └─> Continúa con procesamiento

7. ☁️ SUBIDA A CLOUDINARY
   └─> Sube SOLO los clips (no video principal)
   └─> Obtiene URL pública del clip

8. 📤 ENVÍO INMEDIATO (en paralelo) - CON IDEA GENERAL:
   ├─> 📱 Telegram: 🤖 Idea general + clip
   ├─> 🌐 Webhook: JSON con 🤖 idea general
   ├─> ☁️ Google Drive: Clips + 🤖 idea general
   ├─> 🗄️ Supabase: 🤖 Idea general (no transcripción completa)
   └─> 📧 Brevo: Email con 🤖 idea general

9. 📊 RESUMEN FINAL
   └─> Muestra estadísticas de la sesión
   └─> Lista archivos procesados
   └─> Muestra archivos fallidos ⭐
```

### 📊 EJEMPLO COMPLETO DEL FLUJO CON GPT-4o (v2.6)

#### Ejemplo 1: Mención RELEVANTE (se procesa)
```
1. 🎬 Detecta "presidente" en timestamp 2:30

2. 🤖 GPT-4o ANALIZA SEGMENTO:
   Prompt: "Analiza esta transcripción y determina el segmento más lógico..."
   Respuesta: {
     "inicio_segundos": 145.0,
     "fin_segundos": 168.5,
     "razon": "Captura el anuncio completo desde introducción hasta conclusión",
     "duracion_segundos": 23.5
   }

3. ✂️ CORTA CLIP: 145.0s - 168.5s (23.5 segundos)

4. 🤖 GPT-4o EXTRAE Y EVALÚA IDEA:
   Prompt: "Analiza si el término se menciona con contexto relevante..."
   Respuesta: "El presidente anuncia medidas económicas para controlar 
               inflación. Incluyen ajustes en tasas de interés y apoyo 
               a familias de bajos ingresos. Entran en vigor próxima semana."
   
   ✅ RELEVANTE: Información concreta y desarrollada

5. 🚫 VERIFICA DUPLICADOS: 
   └─> No hay clips recientes (último a 1:15, diferencia: 75s > 60s)
   └─> ✅ PROCESAR

6. 📤 ENVÍA SOLO LA IDEA (no toda la transcripción del video)
```

#### Ejemplo 2: Mención NO RELEVANTE (se descarta)
```
1. 🎬 Detecta "apagón" en timestamp 5:20

2. 🤖 GPT-4o ANALIZA SEGMENTO:
   Respuesta: {
     "inicio_segundos": 310.0,
     "fin_segundos": 330.0,
     "duracion_segundos": 20.0
   }

3. ✂️ CORTA CLIP: 310.0s - 330.0s (20 segundos)

4. 🤖 GPT-4o EXTRAE Y EVALÚA IDEA:
   Transcripción: "Hoy hablaremos de inflación, apagones, inseguridad..."
   Respuesta: "NO_RELEVANTE: El término 'apagón' se menciona sin contexto sustancial."
   
   ❌ NO RELEVANTE: Solo mención en lista, sin desarrollo

5. 🗑️ CLIP DESCARTADO AUTOMÁTICAMENTE
   └─> Se elimina el archivo .mp4
   └─> NO se envía a ningún destino
   └─> Se registra en logs

6. ⏭️ CONTINÚA CON SIGUIENTE TÉRMINO
```

#### Ejemplo 3: Mención DUPLICADA (se omite)
```
1. 🎬 Detecta "apagón" en timestamp 3:45

2. 🚫 VERIFICA DUPLICADOS:
   └─> Hay clip reciente a 3:10 (diferencia: 35s < 60s)
   └─> ❌ DUPLICADO DETECTADO

3. ⏭️ CLIP OMITIDO (sin generar)
   └─> No se corta video
   └─> No se procesa con GPT-4o
   └─> Se registra omisión en logs

4. ⏭️ CONTINÚA CON SIGUIENTE TÉRMINO
```

---

## 🛠️ FUNCIONES PRINCIPALES (99 funciones)

### Categorías de Funciones:

**🤖 GPT-4o (NUEVAS):**
- `determinar_segmento_inteligente_gpt4()` ⭐ - Determina segmento lógico del clip
- `extraer_idea_general_segmento_gpt4()` ⭐ - Extrae idea general del segmento

**📹 Procesamiento de Video:**
- `buscar_y_procesar_videos()` - Función principal
- `obtener_duracion()` - Duración del video
- `extraer_info_medio_hora()` - Extrae info del nombre

**🎤 Transcripción:**
- `transcribir_audio_mistral()` - Transcribe con Mistral
- `transcribir_con_openai()` - Transcribe con OpenAI
- `transcribir_audio_hibrido()` - Múltiples servicios
- `obtener_timestamps_whisper()` - Timestamps precisos

**🤖 Resúmenes IA:**
- `generar_resumen_video()` - Resumen del video
- `generar_resumen_archivo()` - Resumen ejecutivo

**📤 Envíos:**
- `enviar_coincidencia_inmediata()` - FUNCIÓN CENTRAL de envío (actualizada con GPT-4o)

**🌐 Webhooks:**
- `enviar_clips_a_webhook()`
- `enviar_a_webhook_individual()`

**📱 Telegram:**
- `enviar_mensaje_telegram()`
- `enviar_video_telegram_inteligente()`
- `enviar_clips_a_telegram()`

**☁️ Cloudinary:**
- `subir_video_cloudinary()`
- `configurar_cloudinary()`

**☁️ Google Drive:**
- `subir_archivo_google_drive()`
- `enviar_clips_a_google_drive()`

**🗄️ Supabase:**
- `enviar_coincidencias_a_supabase()`

**📧 Brevo:**
- `enviar_correo_brevo()`
- `crear_plantilla_email_html()`

**❌ Archivos Fallidos ⭐:**
- `cargar_archivos_fallidos()`
- `guardar_archivo_fallido()`
- `es_archivo_fallido()`
- `limpiar_archivos_fallidos()`
- `mostrar_archivos_fallidos()`

**🔍 Búsqueda y Escaneo:**
- `buscar_videos_nuevos_optimizado()`
- `escanear_carpeta_completa()`
- `cargar_cache_escaneo()`

**📊 Gestión de Clips:**
- `buscar_todos_los_clips()`
- `mostrar_player_clips()`
- `exportar_lista_clips()`
- `borrar_clips_antiguos()`

**🔧 Configuración:**
- `cargar_terminos_guardados()`
- `cargar_configuracion_completa()`
- `init_session_state()`

**🔍 Verificación:**
- `verificar_todas_las_apis()`
- `test_google_drive_connection()`
- `test_telegram_connection()`
- `test_brevo_connection()`

**📝 Logging:**
- `configurar_logging()`
- `log_info()`, `log_debug()`, `log_warning()`, `log_exception()`

---

## ⚠️ CARACTERÍSTICAS IMPORTANTES

### 🤖 Segmentación Inteligente con GPT-4o ⭐
- GPT-4o analiza contexto completo
- Determina inicio y fin NATURAL de ideas
- Duración variable (10-60s) según contenido
- Transiciones suaves y coherentes
- Fallback automático si GPT-4o falla

### 🤖 Extracción de Ideas con GPT-4o ⭐
- Condensa información del segmento
- Solo 1-2 párrafos relevantes (150 palabras max)
- Enfocado en el término detectado
- Respuesta directa sin fluff
- Envíos limpios y profesionales

### 🔍 Análisis Crítico de Relevancia ⭐ NUEVO (v2.6)
- **GPT-4o evalúa críticamente** cada mención detectada
- **Descarta automáticamente** menciones superficiales
- **Criterios estrictos:** detalle, contexto concreto, desarrollo completo
- **Reduce ruido 60-80%** en alertas
- **Solo clips valiosos** se procesan y envían
- **Logs detallados** de clips descartados

### 🚫 Control de Duplicados Mejorado ⭐ NUEVO (v2.6)
- **Separación mínima de 60 segundos** (1 minuto) entre clips
- **Elimina redundancia** de información
- **Evita saturación** de notificaciones
- **Un solo clip por minuto** del video
- **Optimiza recursos** (storage, procesamiento, envíos)

### ✅ Control de Duplicados en Supabase ⭐
- Session-level tracking de coincidencias
- Evita envíos duplicados a Supabase
- Clave única: `termino_timestamp_archivo`
- Limpieza automática entre procesamiento
- Logs detallados de duplicados evitados

### ✅ Sistema de Archivos Fallidos
- Si un archivo da error, se guarda en `fallidos.txt`
- En futuras rondas, se omite automáticamente
- Se puede limpiar desde la interfaz
- Formato: `archivo|timestamp|mensaje_error`

### ✅ Verificación de Clips
- Verifica que el clip contenga el término
- Si no lo contiene, lo descarta
- Solo envía clips verificados

### ✅ Envío Solo de Clips
- NO se sube el video principal a Cloudinary
- SOLO los clips se suben
- La URL del clip se envía a Supabase

### ✅ Transcripción Completa Local
- Guarda transcripción completa en carpeta del video
- Archivo: `TRANSCRIPCION_COMPLETA.txt`
- NO se envía a servicios externos
- Solo para referencia local

### ✅ AudioChecks para Análisis Posterior
- Crea automáticamente la carpeta `audioChecks` si no existe
- Copia el audio origen para cada mención detectada (relevante o tangencial) a `audioChecks`
- Crea/actualiza un TXT por audio (mismo nombre base del archivo)
- Registra una línea por coincidencia con:
  - Fecha y hora del registro
  - Archivo origen
  - Tipo de mención (`real` o `tangencial`)
  - Tiempo exacto en segundos y formato minuto (`399.7s` y `06:39`)
  - Palabra objetivo
  - Variante detectada en transcripción (ej.: `Intran`/`intrant`)
- Formato de línea:
  - `[YYYY-MM-DD HH:MM:SS] Archivo: NOMBRE_AUDIO | Tipo: tangencial | Tiempo: 399.7s (06:39) | Palabra: intrant | Variante: Intran`

### ✅ Caché de Escaneo
- Optimiza escaneo con caché
- Evita re-escanear archivos
- Se limpia automáticamente

### ✅ Configuración Persistente
- Se guarda en archivos JSON
- Se carga automáticamente

### ✅ Envío Inmediato
- Notificaciones en tiempo real
- No espera a terminar todo

### ✅ Resúmenes con IA Mejorados
- GPT-4o para análisis inteligente
- Resúmenes profesionales y concisos
- Contexto y relevancia optimizados

---

## 📦 DEPENDENCIAS

```bash
streamlit
openai
mistralai
faster-whisper
pandas
supabase
cloudinary
google-auth-oauthlib
google-auth-httplib2
google-api-python-client
requests
```

**Software externo:**
- FFmpeg
- FFprobe

---

## 🚀 INSTALACIÓN

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar FFmpeg (agregar al PATH)

# 3. Configurar APIs (crear archivos JSON)

# 4. Ejecutar
streamlit run transmistral2.py
```

---

## 📂 ESTRUCTURA DE ARCHIVOS

```
grabaciones/
├── transmistral2.py              # Aplicación principal
├── coincidencias_logger.py       # Sistema de logging
├── requirements.txt               # Dependencias
├── README.md                      # Este archivo
├── fallidos.txt                  # Archivos fallidos ⭐
├── archivos_procesados.json      # Archivos procesados
├── cache_escaneo.json            # Caché
├── webhook_config.json           # Config webhook
├── telegram_config.json          # Config Telegram
├── cloudinary_config.json        # Config Cloudinary
├── brevo_config.json             # Config Brevo
├── correos_guardados.json        # Lista correos
├── terminos_guardados.json       # Términos búsqueda
├── configuracion.json            # Config general
├── credentials.json              # Google Drive creds
├── token.json                   # OAuth token
├── logs/                        # Logs
│   ├── app_YYYYMMDD.log
│   ├── errors_YYYYMMDD.log
│   └── debug_YYYYMMDD.log
└── clips/                       # Clips generados
    └── TERMINO_*.mp4
```

En **Video Analyzer v3** (`VIDEOAnalizer3.py`), la configuración multi-cliente usa además:

- `clientes_config.json` — una entrada por **entidad** (Telegram, Brevo, Drive, Supabase, etc.).
- `terminos_guardados.json` — cada término puede llevar `cliente_id` para enrutar alertas a esa entidad.

**Quitar un cliente del archivo `clientes_config.json`** no borra la configuración de los demás. Si dejas términos en `terminos_guardados.json` con un `cliente_id` que ya no existe, el analizador puede resolver el cliente como **default** (EDESUR) para esos términos; conviene borrar o reasignar esos términos. No conviene eliminar el cliente `default` sin saberlo. Quitar un cliente del JSON **no** borra datos en Supabase ni archivos en Drive.

**Análisis por entidad (sidebar en `VIDEOAnalizer3.py`):** cada cliente puede tener `incluir_en_analisis` (por defecto `true`). Si está en `false`, **no** se buscan en el análisis los términos asociados a ese `cliente_id`; el cambio se guarda en `clientes_config.json`.

---

## 🎯 CASOS DE USO

- **Monitoreo de Medios:** Detectar menciones en TV/radio
- **Análisis de Contenido:** Analizar volúmenes grandes
- **Alertas en Tiempo Real:** Notificaciones inmediatas
- **Archivo:** Documentación con clips y transcripciones

---

## 🎓 ARQUITECTURA HÍBRIDA

**Supabase:**
- Metadatos estructurados
- Info de usuarios
- Registros de actividad

**Pinecone:**
- Vectores de embeddings
- Búsqueda semántica
- Ranking de relevancia

**Flujo RAG:**
1. Usuario pregunta → 2. Embedding → 3. Pinecone busca → 4. Supabase metadatos → 5. LLM responde → 6. Guardar historial

---

## 🏆 MEJORES PRÁCTICAS

1. ✅ Aprovechar GPT-4o para segmentación inteligente
2. ✅ Dejar que GPT-4o determine duraciones variables
3. ✅ **Confiar en el análisis crítico de GPT-4o** (descarta menciones irrelevantes)
4. ✅ **Respetar separación de 60s entre clips** (evita redundancia)
5. ✅ Enviar solo ideas generales (no transcripciones completas)
6. ✅ No subir vectores directamente a Supabase
7. ✅ Validar vectores (no NaN) antes de Pinecone
8. ✅ Usar sistema de fallidos para archivos problemáticos
9. ✅ Verificar clips antes de enviar
10. ✅ Mantener logs actualizados
11. ✅ Confiar en el fallback automático si GPT-4o falla
12. ✅ **Revisar logs para ver clips descartados** (ajustar términos si es necesario)

---

## 📺 EJEMPLOS DE MENSAJES ENVIADOS

### ANTES (Sin GPT-4o):
```
📺 Medio: ANTENA 7 - 21:38:59

TÉRMINOS DETECTADOS: presidente

Tema principal: Se detectó una mención del término "presidente" en el contenido.

Contexto: Bienvenidos al noticiero de las nueve, hoy tenemos importantes 
noticias sobre la situación económica del país. El presidente ha realizado 
un anuncio que cambiará el panorama financiero en los próximos meses...

[2000 caracteres más de transcripción...]
```

### AHORA (Con GPT-4o):
```
📺 Medio: ANTENA 7 - 21:38:59

TÉRMINOS DETECTADOS: presidente

🤖 Análisis del segmento:

El presidente anuncia un paquete de medidas económicas destinadas a controlar 
la inflación y estimular el crecimiento. Las principales acciones incluyen 
ajustes en las tasas de interés, incentivos fiscales para pequeñas empresas, 
y un programa de apoyo directo a familias de bajos ingresos. Estas medidas 
entrarán en vigor la próxima semana y se espera que tengan un impacto positivo 
en la economía en los próximos meses.

Término detectado: "presidente"

🎬 Clip: 20251008_presidente_2m30s.mp4 (23.5 segundos)
☁️ URL: https://res.cloudinary.com/xxx/video/upload/xxx.mp4
```

---

## 🎯 CASOS DE USO MEJORADOS

### 1. **Monitoreo de Medios** 
- ✅ Clips inteligentes con contexto completo
- ✅ Ideas condensadas fáciles de revisar
- ✅ Alertas instantáneas con información relevante

### 2. **Análisis de Contenido**
- ✅ GPT-4o identifica segmentos más importantes
- ✅ Resúmenes ejecutivos automáticos
- ✅ Procesamiento eficiente de grandes volúmenes

### 3. **Alertas en Tiempo Real**
- ✅ Notificaciones instantáneas con contexto
- ✅ Mensajes concisos en Telegram
- ✅ Información accionable inmediatamente

### 4. **Archivo y Documentación**
- ✅ Clips con segmentos lógicos completos
- ✅ Transcripciones completas guardadas localmente
- ✅ Ideas generales para búsqueda rápida

---

## 🆚 COMPARATIVA: ANTES vs AHORA

| Aspecto | Antes (v2.0) | v2.5 (GPT-4o) | v2.6 Actual (+ Análisis Crítico) |
|---------|--------------|---------------|----------------------------------|
| **Segmentación de clips** | Mecánica (30s antes/después) | 🤖 Inteligente (GPT-4o analiza) | 🤖 Inteligente (GPT-4o analiza) |
| **Duración de clips** | Fija (60s) | Variable (10-60s según contenido) | Variable (10-60s según contenido) |
| **Cortes** | Abruptos | Naturales y coherentes | Naturales y coherentes |
| **Evaluación de relevancia** | ❌ No disponible | ❌ No disponible | ✅ GPT-4o evalúa críticamente |
| **Filtrado de menciones** | Ninguno (todo se procesa) | Ninguno (todo se procesa) | ✅ Descarta menciones superficiales |
| **Control de duplicados** | ±5 segundos | ±5 segundos | ✅ ±60 segundos (1 minuto) |
| **Envío de datos** | Transcripción completa (2000+ caracteres) | 🤖 Idea general (150 palabras) | 🤖 Idea general (150 palabras) |
| **Legibilidad** | Difícil (mucho texto) | Excelente (conciso) | Excelente (conciso) |
| **Calidad de clips** | Baja (muchos irrelevantes) | Media | ✅ Alta (solo relevantes) |
| **Reducción de ruido** | 0% | 0% | ✅ 60-80% |
| **Procesamiento** | Rápido | Ligeramente más lento | Ligeramente más lento |
| **Contexto** | Genérico | Específico y relevante | ✅ Crítico y sustancial |
| **Fallback** | No disponible | ✅ Automático si GPT-4o falla | ✅ Automático si GPT-4o falla |

---

## 🔧 CONFIGURACIÓN DE GPT-4o

El sistema usa GPT-4o con estas configuraciones:

**Para segmentación de clips:**
```python
model="gpt-4o"
temperature=0.3  # Respuestas consistentes
max_tokens=500
```

**Para extracción de ideas:**
```python
model="gpt-4o"
temperature=0.4  # Balance entre creatividad y precisión
max_tokens=300   # Máximo 150 palabras
```

**API Key:** Configurada en el código (variable `openai.api_key`)

---

**Desarrollado con ❤️ por el equipo de Video Analyzer IA**

**Versión:** 2.6 (con GPT-4o + Análisis Crítico de Relevancia)

**Última actualización:** 10 de Octubre de 2025

---

## Repositorio público (GitHub) vs desarrollo local

La rama publicada en **GitHub** (`videonalizer`) **no incluye claves API ni tokens en el código** (p. ej. Brevo/Sendinblue, Supabase anon, Cloudinary). Así se cumple la protección de secretos al hacer push y se evita filtrar credenciales.

### En GitHub (código clonado)

- Configura un archivo **`.env`** en la raíz del proyecto (o variables de entorno del sistema) con al menos:
  - **`BREVO_API_KEY`**, **`BREVO_SMTP_USER`**, **`BREVO_SMTP_SERVER`** (por defecto suele ser `smtp-relay.brevo.com`), **`BREVO_SMTP_PORT`** (p. ej. `587`), **`BREVO_SENDER_EMAIL`** — para correo transaccional vía Brevo.
  - **`SUPABASE_URL`** y **`SUPABASE_ANON_KEY`** — si usas Supabase.
- **Cloudinary:** el script puede tomar credenciales desde la configuración local (`cargar_cloudinary_config()` / archivo de configuración bajo la carpeta de procesados), no desde el repositorio.
- Ejecutar la app: `streamlit run VIDEOAnalizer3.py` (tras instalar dependencias y `ffmpeg` según tu entorno).

### En tu máquina (desarrollo privado)

Puedes seguir usando **`clientes_config.json`**, **`terminos_guardados.json`** y la UI de entidades con valores **sin** publicarlos. No hace falta que tu copia local coincida con el árbol de Git si prefieres no commitear secretos.

### Versión etiquetada

El tag **`v4.0.0`** en este repositorio corresponde a la línea de código **VIDEOAnalizer3.py** alineada con esta política (sin secretos embebidos en las rutas de creación de entidad que GitHub escanea).
