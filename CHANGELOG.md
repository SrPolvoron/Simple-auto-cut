# Changelog

## 0.6.0

- Nuevo modo `captures` con alias `capture`, `screenshots` y `screenshot`.
- Selección configurable de 5, 10 o cualquier número de fotogramas por vídeo con `--count`.
- Cuatro perfiles de selección: `quality`, `person`, `landscape` y `situation`.
- `captures --kind person` puede buscar cualquier persona mediante detección facial ligera.
- `captures --kind person --reference ...` reutiliza YuNet + SFace para seleccionar una persona concreta.
- Las referencias relativas se buscan junto a la carpeta de vídeos, igual que en el modo `person`.
- Puntuación de calidad basada en nitidez, exposición, clipping, contraste, entropía y estabilidad aproximada.
- Selección temporal diversa para evitar devolver varias imágenes prácticamente iguales.
- Segunda pasada local alrededor de cada candidato (`--refine-window`, `--refine-fps`) para escoger un fotograma más nítido y estable.
- Soporte de decodificación `auto`, `cpu`, `cuda` y `d3d11va` durante el escaneo inicial.
- Exportación JPEG de alta calidad o PNG a `exports/<video>/captures/`.
- Nuevo `captures.json` con timestamps, métricas y puntuaciones.
- `--preview` genera hojas `captures_preview_XXX.jpg` con timestamp y score.
- Nuevos controles `--scan-every`, `--min-gap`, `--format` y `--no-progress`.
- README actualizado íntegramente en español con ejemplos de todos los modos.
- 38 tests pasando.

## 0.5.0

- Nuevo modo `scenes` para dividir compilaciones en clips independientes.
- Nuevo modo `trim` para eliminar segundos del inicio, final o centro en lote.
- README en español.
