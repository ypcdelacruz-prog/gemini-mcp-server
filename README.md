# gemini-mcp-server

Servidor MCP que conecta Claude con la API de Google Gemini (incluyendo Veo para video), para que Claude pueda:

- **gemini_ask** — hacerle una pregunta de texto a Gemini (segunda opinión, cross-check).
- **gemini_analyze_image** — mandarle una imagen (foto de obra, maquinaria, plano, factura, etc.) junto con una pregunta.
- **gemini_analyze_video** — mandarle un video corto (≲20MB) junto con una pregunta (avance de obra, maquinaria en funcionamiento, etc.).
- **gemini_generate_video** — iniciar la generación de un video de 8s con Veo a partir de un prompt de texto (y opcionalmente una imagen de referencia). Devuelve un `operation_name`.
- **gemini_check_video_status** — revisar el estado de una generación iniciada con `gemini_generate_video` y, cuando termina, descargar el video.

## Variables de entorno

- `GEMINI_API_KEY` (requerida) — API key de Google AI Studio (https://aistudio.google.com/apikey).
- `TRANSPORT=http` — para correr como servidor remoto (Streamable HTTP) en vez de stdio.
- `PORT` — puerto HTTP (default 3000).
- `ALLOWED_HOST` — dominio público del servicio (por ejemplo `gemini-mcp-server.onrender.com`), para protección contra DNS rebinding.

## Despliegue en Render

1. `npm install && npm run build`
2. Start command: `npm start`
3. Variables de entorno: `GEMINI_API_KEY`, `TRANSPORT=http`, `ALLOWED_HOST=<tu-dominio-de-render>`

## Notas sobre modelos y Veo

Los nombres de modelo de Gemini/Veo cambian con frecuencia. Los parámetros `model` de cada herramienta aceptan
cualquier id de modelo actual (no están fijos en el código), así que si Google lanza un modelo nuevo no hace
falta volver a desplegar — basta con pasar el nuevo id al llamar la herramienta. Ver
https://ai.google.dev/gemini-api/docs/models y https://ai.google.dev/gemini-api/docs/veo para los ids vigentes.

Veo genera videos de 8 segundos; la generación toma entre ~1 y 6 minutos, por eso el flujo está dividido en
"iniciar" (`gemini_generate_video`) y "revisar estado" (`gemini_check_video_status`) en vez de una sola llamada
bloqueante.

## Desarrollo local

```bash
npm install
npm run build
GEMINI_API_KEY=... npm run smoke
```
