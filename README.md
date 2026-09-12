# Simple Video Autocut

Utilidad ligera en **Python + FFmpeg** para analizar y cortar vídeos automáticamente sin edición creativa, filtros ni transiciones. Siempre que es posible, los clips y montajes se generan mediante **stream copy (`-c copy`)**, por lo que no se vuelve a codificar el vídeo.

La versión **0.6.0** incluye cinco modos:

- **`black`**: elimina tramos negros o casi negros, por ejemplo cuando el móvil está apoyado, la lente está tapada o apenas entra luz.
- **`person`**: conserva los planos completos en los que aparece una persona concreta usando una o varias imágenes de referencia.
- **`scenes`**: divide una compilación en escenas independientes, útil para recopilaciones de saltos de skate, memes, highlights, clips unidos, etc.
- **`trim`**: elimina automáticamente una cantidad fija de segundos del principio, final y/o centro de uno o muchos vídeos.
- **`captures`**: selecciona y exporta automáticamente los mejores fotogramas de cada vídeo por calidad general, personas, paisajes o situaciones/acción.

Los modos `black`, `person` y `trim` permiten exportar **clips**, **un montaje final** o **ambas cosas**. El modo `scenes` exporta cada escena como un archivo independiente.

---

## Requisitos

- Python 3.11 o superior.
- FFmpeg y FFprobe disponibles en `PATH`.
- OpenCV y NumPy, instalados automáticamente con pip.
- Para `person` y para `captures --kind person --reference ...`: modelos ONNX de YuNet y SFace.

Comprobar FFmpeg:

```powershell
ffmpeg -version
ffprobe -version
```

---

## Instalación

Desde la carpeta del repositorio:

```powershell
python -m pip install -e .
```

Comprobar la versión:

```powershell
python -m video_autocut.cli --version
```

Debe mostrar:

```text
video-autocut 0.6.0
```

Si la carpeta `Scripts` de Python está incluida en `PATH`, también puedes usar directamente:

```powershell
video-autocut --version
```

Para el modo `person`, descarga una vez los modelos oficiales de OpenCV:

```powershell
python scripts\download_models.py
```

---

# Estructura de salida

Si el vídeo original es:

```text
video_001.mp4
```

y especificas:

```powershell
--output "D:\Autocut\exports"
```

los clips se almacenan así:

```text
D:\Autocut\exports\
└── video_001\
    ├── c001_video_001.mp4
    ├── c002_video_001.mp4
    ├── c003_video_001.mp4
    └── manifest.json
```

Si exportas un montaje:

```text
D:\Autocut\exports\
└── video_001\
    ├── video_001.mp4
    └── manifest.json
```

Con:

```text
--export both
```

se generan clips y montaje dentro de la misma carpeta.

Si la entrada es una carpeta, cada vídeo obtiene su propia carpeta de salida.

---

# Modo `black`

Detecta y elimina tramos negros o casi negros manteniendo las escenas oscuras que todavía contienen información visible.

## Exportar clips

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --export clips --hwaccel auto
```

## Exportar solo el montaje final

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --export montage --hwaccel auto
```

El resultado será un único vídeo con el mismo nombre que el original, pero sin los tramos descartados.

## Exportar clips y montaje

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --export both --hwaccel auto
```

## Procesar una carpeta completa y elegir la salida

```powershell
python -m video_autocut.cli black "C:\Videos\Phone" --export both --output "D:\Autocut\Black" --overwrite --hwaccel auto
```

## Analizar sin generar vídeos

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --dry-run --hwaccel auto
```

## Máxima velocidad sin refinado de bordes

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --dry-run --hwaccel auto --no-refine
```

Esto puede reducir la precisión de los timestamps aproximadamente hasta el intervalo de muestreo utilizado en la primera pasada.

## Forzar CUDA / NVDEC

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --dry-run --hwaccel cuda
```

Opciones disponibles:

```text
--hwaccel auto
--hwaccel cpu
--hwaccel cuda
--hwaccel d3d11va
```

## Generar miniaturas de los cortes y reporte HTML

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --export both --preview --hwaccel auto
```

Añade archivos similares a:

```text
preview_cuts_001.jpg
preview_cuts_002.jpg
report.html
```

Cada corte muestra una imagen antes, durante y después del intervalo eliminado.

## Generar una hoja de miniaturas cada 5 segundos

```powershell
python -m video_autocut.cli black "C:\Videos\video_001.mp4" --dry-run --contact-sheet 5 --hwaccel auto
```

También puedes usar, por ejemplo:

```powershell
--contact-sheet 2
```

para obtener una miniatura cada 2 segundos.

---

# Modo `person`

`person` utiliza una o varias imágenes de referencia para localizar a una persona concreta dentro de los vídeos.

Puedes guardar la foto directamente en la misma carpeta que los vídeos.

Ejemplo:

```text
C:\Videos\Podcast\
├── himar.jpg
├── episodio_001.mp4
├── episodio_002.mp4
└── episodio_003.mp4
```

Ejecuta:

```powershell
python -m video_autocut.cli person "C:\Videos\Podcast" --reference "himar.jpg" --export clips
```

`himar.jpg` se busca automáticamente dentro de la carpeta de entrada. Esa misma referencia se utiliza para todos los vídeos encontrados en ella.

Las imágenes no se procesan como vídeos, por lo que es seguro guardarlas junto a los `.mp4`.

## Varias imágenes de la misma persona

Para mejorar la detección de diferentes ángulos puedes usar varias referencias:

```text
C:\Videos\Podcast\refs\
├── frontal.jpg
├── perfil_izq.jpg
├── perfil_der.jpg
└── tres_cuartos.jpg
```

Y pasar la carpeta completa:

```powershell
python -m video_autocut.cli person "C:\Videos\Podcast" --reference "refs" --export both
```

## Solo montaje final

```powershell
python -m video_autocut.cli person "C:\Videos\Podcast" --reference "himar.jpg" --export montage
```

## Clips + montaje + miniaturas + salida personalizada

```powershell
python -m video_autocut.cli person "C:\Videos\Podcast" --reference "himar.jpg" --export both --preview --output "D:\Autocut\Person" --overwrite
```

El funcionamiento interno es:

1. Detectar cambios de plano.
2. Muestrear varios fotogramas de cada plano.
3. Detectar todas las caras visibles.
4. Compararlas con la referencia.
5. Conservar el plano completo si aparece la persona objetivo.

Por tanto, un plano con varias personas también se conserva si la persona de referencia aparece en él.

---

# Modo `scenes`

Divide una compilación en vídeos independientes detectando cambios visuales fuertes entre escenas.

Casos típicos:

- recopilaciones de saltos de skate;
- compilaciones de memes;
- vídeos de highlights;
- múltiples clips unidos en un único archivo;
- recopilaciones con cortes claros entre contenidos.

## Dividir una compilación

```powershell
python -m video_autocut.cli scenes "C:\Videos\skate_compilation.mp4"
```

Resultado:

```text
exports\
└── skate_compilation\
    ├── c001_skate_compilation.mp4
    ├── c002_skate_compilation.mp4
    ├── c003_skate_compilation.mp4
    └── manifest.json
```

También se acepta `scene` como alias:

```powershell
python -m video_autocut.cli scene "C:\Videos\memes.mp4"
```

## Dividir todas las compilaciones de una carpeta

```powershell
python -m video_autocut.cli scenes "C:\Videos\Compilations" --output "D:\Autocut\Scenes" --overwrite
```

## Generar miniaturas de las escenas

```powershell
python -m video_autocut.cli scenes "C:\Videos\memes.mp4" --preview
```

Añade:

```text
preview_scenes_001.jpg
report.html
```

## Ajustar sensibilidad

El umbral por defecto es:

```text
0.30
```

Un valor **más bajo** detecta cambios menos fuertes y genera más clips:

```powershell
python -m video_autocut.cli scenes "C:\Videos\memes.mp4" --threshold 0.20
```

Un valor **más alto** requiere un cambio visual mayor y genera menos clips:

```powershell
python -m video_autocut.cli scenes "C:\Videos\memes.mp4" --threshold 0.45
```

Para evitar que flashes, transiciones o planos muy cortos se conviertan en clips independientes:

```powershell
python -m video_autocut.cli scenes "C:\Videos\memes.mp4" --threshold 0.25 --min-scene 0.50
```

> Este modo funciona especialmente bien con cortes duros. Las transiciones largas, fundidos o contenidos visualmente muy similares pueden requerir ajustar `--threshold`.

---

# Modo `trim`

Permite eliminar automáticamente una cantidad fija de segundos de uno o todos los vídeos de una carpeta.

El modo de exportación predeterminado de `trim` es:

```text
montage
```

porque normalmente se quiere obtener un único vídeo final por archivo original.

## Eliminar los últimos 3 segundos

Un vídeo:

```powershell
python -m video_autocut.cli trim "C:\Videos\video_001.mp4" --remove-end 3
```

Carpeta completa:

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-end 3 --output "D:\Autocut\Trimmed" --overwrite
```

Para:

```text
C:\Videos\Phone\
├── video_001.mp4
├── video_002.mp4
└── video_003.mp4
```

se genera:

```text
D:\Autocut\Trimmed\
├── video_001\video_001.mp4
├── video_002\video_002.mp4
└── video_003\video_003.mp4
```

## Eliminar los primeros 3 segundos

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-start 3 --output "D:\Autocut\Trimmed" --overwrite
```

## Eliminar 3 segundos exactamente del centro

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-middle 3 --output "D:\Autocut\Trimmed" --overwrite
```

En un vídeo de 10 segundos, `--remove-middle 3` elimina:

```text
00:03.500 -> 00:06.500
```

y une automáticamente los dos tramos restantes.

## Combinar operaciones

Eliminar 2 segundos del principio y 3 del final:

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-start 2 --remove-end 3 --output "D:\Autocut\Trimmed" --overwrite
```

Eliminar principio, centro y final en una sola operación:

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-start 2 --remove-middle 4 --remove-end 3 --output "D:\Autocut\Trimmed" --overwrite
```

## Exportar los tramos restantes como clips independientes

Especialmente útil con `--remove-middle`:

```powershell
python -m video_autocut.cli trim "C:\Videos\video_001.mp4" --remove-middle 3 --export clips
```

Resultado:

```text
c001_video_001.mp4
c002_video_001.mp4
```

## Generar clips y montaje

```powershell
python -m video_autocut.cli trim "C:\Videos\video_001.mp4" --remove-middle 3 --export both
```

## Miniaturas del tramo eliminado

```powershell
python -m video_autocut.cli trim "C:\Videos\video_001.mp4" --remove-end 3 --preview
```

La preview muestra fotogramas antes, durante y después del intervalo eliminado y genera `report.html`.

---

# Modo `captures` / `screenshots`

Selecciona automáticamente los mejores fotogramas de uno o varios vídeos y los exporta como imágenes independientes.

El selector no se limita a tomar una captura cada N segundos. Trabaja en dos pasos:

1. recorre el vídeo a baja frecuencia y puntúa candidatos;
2. distribuye las selecciones para evitar capturas casi idénticas;
3. alrededor de cada candidato vuelve a revisar varios fotogramas;
4. escoge el fotograma más estable, nítido, bien expuesto y con menos clipping;
5. exporta la imagen a resolución original.

Los criterios de calidad incluyen nitidez, exposición, luces/sombras quemadas, contraste, entropía visual y estabilidad aproximada entre fotogramas.

## Cinco mejores capturas generales

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 5 --kind quality
```

## Diez mejores capturas de cada vídeo de una carpeta

```powershell
python -m video_autocut.cli captures "C:\Videos\Viaje" --count 10 --kind quality --output "D:\Autocut\Capturas" --overwrite
```

La estructura de salida es:

```text
D:\Autocut\Capturas\
├── captures001\
│   ├── s001_viaje.jpg
│   ├── s002_viaje.jpg
│   ├── s003_viaje.jpg
│   ├── ...
│   └── s010_viaje.jpg
├── captures002\
│   └── ...
└── manifest.json
```

Cada ejecución crea la siguiente carpeta correlativa (`captures001`, `captures002`, ...), sin incluir el nombre o identificador temporal del vídeo en la ruta. El único `manifest.json` de la raíz se conserva y acumula los lotes de capturas.

## Capturas con personas

Para buscar buenos fotogramas donde aparezca cualquier persona:

```powershell
python -m video_autocut.cli captures "C:\Videos\evento.mp4" --count 10 --kind person
```

En este modo sin referencia se utiliza detección facial ligera como señal de presencia de personas. Prioriza fotogramas donde haya rostros visibles y de tamaño suficiente.

## Capturas de una persona concreta

Puedes reutilizar exactamente el mismo concepto que en el modo `person`. Por ejemplo:

```text
C:\Videos\Podcast\
├── himar.jpg
├── episodio_001.mp4
├── episodio_002.mp4
└── episodio_003.mp4
```

Y ejecutar:

```powershell
python -m video_autocut.cli captures "C:\Videos\Podcast" --count 10 --kind person --reference "himar.jpg"
```

La misma referencia se aplica a todos los vídeos de la carpeta. También puedes pasar un directorio con varias fotos de la misma persona:

```powershell
python -m video_autocut.cli captures "C:\Videos\Podcast" --count 10 --kind person --reference "refs"
```

En este caso solo entran en la selección los fotogramas donde SFace confirma a la persona de referencia.

## Capturas de paisajes

```powershell
python -m video_autocut.cli captures "C:\Videos\Cuenca" --count 10 --kind landscape
```

`landscape` favorece fotogramas nítidos, detallados y bien expuestos, penalizando encuadres dominados por caras. Es un selector visual heurístico; no intenta describir semánticamente el lugar.

## Capturas de situaciones o acción

```powershell
python -m video_autocut.cli captures "C:\Videos\skate.mp4" --count 10 --kind situation
```

`situation` aumenta el peso de movimiento, detalle y actividad visual, pero sigue penalizando fotogramas borrosos. Es útil para saltos, acción, gestos y momentos de actividad.

## Generar una hoja de miniaturas de las capturas elegidas

```powershell
python -m video_autocut.cli captures "C:\Videos\skate.mp4" --count 10 --kind situation --preview
```

Además de las imágenes individuales genera:

```text
captures_preview_001.jpg
```

con timestamp y puntuación de cada selección.

## PNG en lugar de JPEG

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 5 --kind landscape --format png
```

JPEG es el formato predeterminado y se exporta con alta calidad.

## Ajustar separación entre capturas

Por defecto la separación mínima se calcula automáticamente según la duración del vídeo y el número solicitado. Puedes fijarla manualmente:

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 10 --kind landscape --min-gap 8
```

Esto exige aproximadamente 8 segundos entre selecciones siempre que haya suficientes candidatos. Si no puede completar el número solicitado, el selector relaja progresivamente la separación antes de devolver menos imágenes.

## Ajustar velocidad y precisión

Escanear un candidato cada 2 segundos:

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 10 --scan-every 2
```

Más precisión en la búsqueda final alrededor de cada candidato:

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 10 --refine-window 0.75 --refine-fps 12
```

El valor predeterminado busca en `±0.50 s` a `8 FPS`.

## Aceleración de decodificación

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 10 --hwaccel auto
```

También están disponibles:

```text
--hwaccel cpu
--hwaccel cuda
--hwaccel d3d11va
```

Con CUDA y un FFmpeg compatible, el escaneo inicial puede reducir el fotograma en GPU antes de enviarlo a Python. El refinado local y la puntuación visual siguen ejecutándose principalmente en CPU/OpenCV.

## Solo analizar

```powershell
python -m video_autocut.cli captures "C:\Videos\viaje.mp4" --count 10 --kind landscape --dry-run
```

Actualiza `manifest.json` con timestamps y puntuaciones, pero no exporta las imágenes.

Los alias siguientes son equivalentes:

```text
captures
capture
screenshots
screenshot
```

---


# Modos de exportación

Disponibles en `black`, `person` y `trim`:

```text
--export clips
--export montage
--export both
```

- `clips`: genera `c001_...`, `c002_...`, etc.
- `montage`: genera un único vídeo final con el mismo nombre que el original.
- `both`: genera ambos formatos.

`black` y `person` usan `clips` por defecto.

`trim` usa `montage` por defecto.

`scenes` siempre genera clips independientes, ya que volver a unir todas las escenas reconstruiría prácticamente el vídeo original.

---

# Carpeta de salida personalizada

Todos los modos permiten:

```text
--output RUTA
```

Ejemplo:

```powershell
python -m video_autocut.cli scenes "C:\Videos\memes.mp4" --output "D:\Resultados"
```

La estructura siempre mantiene una carpeta por vídeo:

```text
D:\Resultados\
└── memes\
    ├── c001_memes.mp4
    ├── c002_memes.mp4
    └── manifest.json
```

---

# Semántica de `stream copy`

Los clips y montajes utilizan FFmpeg con:

```text
-c copy
```

siempre que es posible.

Esto significa que el programa no añade:

- filtros;
- cambios de resolución;
- corrección de color;
- transiciones;
- modificación del audio;
- recodificación de vídeo.

Sin embargo, códecs como H.264 y H.265 utilizan GOPs y keyframes. Por ello, un corte mediante stream copy no puede garantizar siempre precisión exacta de un fotograma y puede incluir unas décimas adicionales alrededor del punto de corte según la posición de los keyframes del archivo original.

---

# Configuración

La configuración predeterminada está en:

```text
config.toml
```

Para escenas:

```toml
[scenes]
threshold = 0.30
min_scene_duration_s = 0.25
```

Para trim:

```toml
[trim]
min_keep_duration_s = 0.01
```

También permanecen disponibles las secciones `[black]`, `[person]` y `[captures]`.

---

# `manifest.json`, miniaturas y reportes

Salvo que se desactive en configuración, cada vídeo procesado genera un `manifest.json`.

Dependiendo del modo puede contener:

- vídeo de origen;
- intervalos conservados;
- intervalos eliminados;
- timestamps de escenas;
- configuración del detector;
- modo de exportación;
- nombres de archivos generados;
- referencias faciales utilizadas;
- decisiones del detector de persona;
- estadísticas de rendimiento de `black`;
- nombres de previews y reportes.

Las previews son opcionales mediante `--preview` y los reportes HTML mediante `--report`. El modo `captures` usa un único `manifest.json` en la raíz de salida, que acumula los timestamps, métricas y puntuaciones de cada lote.

---

# Comandos rápidos

### Quitar negros y obtener clips + montaje

```powershell
python -m video_autocut.cli black "C:\Videos\video.mp4" --export both --preview --hwaccel auto
```

### Sacar todos los planos de una persona usando una foto guardada junto a los vídeos

```powershell
python -m video_autocut.cli person "C:\Videos\Podcast" --reference "persona.jpg" --export both --preview
```

### Dividir una compilación de memes

```powershell
python -m video_autocut.cli scenes "C:\Videos\memes.mp4" --preview
```

### Dividir todas las compilaciones de una carpeta

```powershell
python -m video_autocut.cli scenes "C:\Videos\Compilaciones" --output "D:\Autocut\Scenes" --overwrite
```

### Quitar los últimos 3 segundos de todos los vídeos de una carpeta

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-end 3 --output "D:\Autocut\Trimmed" --overwrite
```

### Quitar los primeros 3 segundos

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-start 3 --output "D:\Autocut\Trimmed" --overwrite
```

### Quitar 3 segundos del centro

```powershell
python -m video_autocut.cli trim "C:\Videos\Phone" --remove-middle 3 --output "D:\Autocut\Trimmed" --overwrite
```

---

# Tests

Instalar dependencias de desarrollo:

```powershell
python -m pip install -e ".[dev]"
```

Ejecutar:

```powershell
python -m pytest
```

La suite de la versión 0.5.0 cubre:

- detección de negro;
- muestreo y referencias del modo persona;
- previews;
- exportación de clips y montajes mediante stream copy;
- separación por escenas;
- planificación de cortes al principio, final y centro;
- opciones del CLI.

---

# Estructura del proyecto

```text
simple-video-autocut/
├── config.toml
├── CHANGELOG.md
├── README.md
├── models/
├── scripts/
│   └── download_models.py
├── src/video_autocut/
│   ├── black.py
│   ├── cli.py
│   ├── config.py
│   ├── exporter.py
│   ├── faces.py
│   ├── ffmpeg_tools.py
│   ├── manifest.py
│   ├── media.py
│   ├── person.py
│   ├── preview.py
│   ├── scenes.py
│   ├── segments.py
│   └── trim.py
└── tests/
```

---

## Licencia

MIT.

### Sacar las 10 mejores capturas de un vídeo

```powershell
python -m video_autocut.cli captures "C:\Videos\video.mp4" --count 10 --kind quality --preview
```

### Sacar 10 paisajes de cada vídeo de una carpeta

```powershell
python -m video_autocut.cli captures "C:\Videos\Viaje" --count 10 --kind landscape --output "D:\Autocut\Capturas" --overwrite
```

### Sacar 5 capturas de una persona concreta

```powershell
python -m video_autocut.cli captures "C:\Videos\Podcast" --count 5 --kind person --reference "persona.jpg" --preview
```
