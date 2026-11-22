#!/usr/bin/env node
/**
 * MCP server for the Google Gemini API.
 * Lets Claude: ask Gemini a question, analyze an image or video, and
 * generate new videos with Veo (start + check status, since generation
 * takes 1-6+ minutes and a single tool call should not block that long).
 * Serves stdio by default, Streamable HTTP when TRANSPORT=http.
 */
import { createMcpExpressApp } from "@modelcontextprotocol/express";
import { toNodeHandler } from "@modelcontextprotocol/node";
import { createMcpHandler, McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import * as z from "zod/v4";

const API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const API_KEY = process.env.GEMINI_API_KEY;
const CHARACTER_LIMIT = 25_000;
const DEFAULT_TEXT_MODEL = "gemini-2.5-pro";
const DEFAULT_VEO_MODEL = "veo-3.1-generate-preview";

if (!API_KEY) {
  console.error("GEMINI_API_KEY environment variable is required");
  process.exit(1);
}

const ModelField = z
  .string()
  .min(1)
  .describe(
    "Gemini model id, e.g. 'gemini-2.5-pro' (default, best quality) or 'gemini-2.5-flash' (faster/cheaper). " +
      "A newer model id can be used here without redeploying the server; see https://ai.google.dev/gemini-api/docs/models"
  );

// ---- Upstream API helpers: module scope, shared by every server instance ----

function httpErrorMessage(status: number, bodyText: string): string {
  switch (status) {
    case 400:
      return `Gemini rechazó la solicitud (entrada inválida). Detalle: ${bodyText.slice(0, 500)}`;
    case 401:
    case 403:
      return "Gemini rechazó la API key. Verifica que GEMINI_API_KEY sea válida y tenga habilitada la Generative Language API.";
    case 404:
      return "No encontrado. Revisa el argumento 'model' u 'operation_name'.";
    case 429:
      return "Se alcanzó el límite de uso (rate limit / quota) de Gemini. Espera un momento antes de volver a intentar.";
    case 500:
    case 503:
      return "Los servidores de Gemini tuvieron un error interno. Intenta de nuevo en unos minutos.";
    default:
      return `La API de Gemini devolvió HTTP ${status}. Detalle: ${bodyText.slice(0, 500)}`;
  }
}

class ApiError extends Error {
  constructor(readonly status: number, bodyText: string) {
    super(httpErrorMessage(status, bodyText));
  }
}

async function geminiFetch(path: string, init: RequestInit, signal: AbortSignal, timeoutMs = 60_000): Promise<Response> {
  const url = path.startsWith("http") ? path : `${API_BASE_URL}/${path}`;
  const sep = url.includes("?") ? "&" : "?";
  const fullUrl = path.startsWith("http") ? url : `${url}${sep}key=${API_KEY}`;
  let response: Response;
  try {
    response = await fetch(fullUrl, {
      ...init,
      headers: { ...init.headers, "x-goog-api-key": API_KEY as string },
      signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)])
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new Error(`Gemini no respondió dentro de ${Math.round(timeoutMs / 1000)}s. Intenta de nuevo o reduce el tamaño del contenido.`);
    }
    throw error;
  }
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new ApiError(response.status, text);
  }
  return response;
}

// ---- Text / image / video understanding (generateContent) ----

interface GenerateContentPart {
  text?: string;
  inline_data?: { mime_type: string; data: string };
}

interface GenerateContentRequest {
  contents: { role: "user"; parts: GenerateContentPart[] }[];
  system_instruction?: { parts: { text: string }[] };
}

interface GenerateContentResponse {
  candidates?: { content?: { parts?: { text?: string }[] }; finishReason?: string }[];
  promptFeedback?: { blockReason?: string };
}

function extractText(response: GenerateContentResponse): string {
  if (response.promptFeedback?.blockReason) {
    return `Gemini bloqueó esta solicitud (motivo: ${response.promptFeedback.blockReason}). Reformula el pedido e intenta de nuevo.`;
  }
  const candidate = response.candidates?.[0];
  const text = candidate?.content?.parts?.map(part => part.text ?? "").join("") ?? "";
  if (!text) {
    return `Gemini no devolvió texto (finishReason: ${candidate?.finishReason ?? "desconocido"}).`;
  }
  return text;
}

function truncate(text: string): string {
  if (text.length <= CHARACTER_LIMIT) return text;
  return text.slice(0, CHARACTER_LIMIT) + "\n\n[Respuesta truncada por longitud.]";
}

async function generateContent(model: string, body: GenerateContentRequest, signal: AbortSignal, timeoutMs?: number): Promise<string> {
  const response = await geminiFetch(
    `models/${encodeURIComponent(model)}:generateContent`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    signal,
    timeoutMs
  );
  const data = (await response.json()) as GenerateContentResponse;
  return truncate(extractText(data));
}

// ---- Veo video generation (predictLongRunning) ----

interface VeoOperation {
  name: string;
  done?: boolean;
  error?: { message?: string };
  response?: {
    generateVideoResponse?: { generatedSamples?: { video?: { uri?: string; bytesBase64Encoded?: string } }[] };
    generatedVideos?: { video?: { uri?: string; bytesBase64Encoded?: string } }[];
  };
}

function extractVideo(op: VeoOperation): { uri?: string; bytesBase64Encoded?: string } | undefined {
  return op.response?.generateVideoResponse?.generatedSamples?.[0]?.video ?? op.response?.generatedVideos?.[0]?.video;
}

// ---- Server factory: serveStdio calls it per connection, createMcpHandler per HTTP request ----

function createServer(): McpServer {
  const server = new McpServer({ name: "gemini-mcp-server", version: "1.0.0" });

  server.registerTool(
    "gemini_ask",
    {
      title: "Ask Gemini",
      description: `Send a text question or request to Google Gemini and return its answer.

Use this for a second opinion from Gemini, a cross-check on an analysis, or any text-only request.
Use gemini_analyze_image / gemini_analyze_video instead when the question is about an image or video.
Use gemini_generate_video instead to create a new video.`,
      inputSchema: z.strictObject({
        prompt: z.string().min(1).max(100_000).describe("The question or request to send to Gemini"),
        system_instruction: z
          .string()
          .max(5_000)
          .optional()
          .describe("Optional system instruction / role for Gemini, e.g. 'Eres un analista financiero'"),
        model: ModelField.default(DEFAULT_TEXT_MODEL)
      }),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: false, openWorldHint: true }
    },
    async ({ prompt, system_instruction, model }, ctx) => {
      const body: GenerateContentRequest = { contents: [{ role: "user", parts: [{ text: prompt }] }] };
      if (system_instruction) body.system_instruction = { parts: [{ text: system_instruction }] };
      const text = await generateContent(model, body, ctx.mcpReq.signal);
      return { content: [{ type: "text", text }] };
    }
  );

  server.registerTool(
    "gemini_analyze_image",
    {
      title: "Analyze Image with Gemini",
      description: `Send an image (base64-encoded) plus a question to Gemini for visual analysis.

Useful for photos of obras (avance de construcción), maquinaria, planos, facturas, capturas de pantalla, etc.
The image must already be base64-encoded (no "data:" prefix).`,
      inputSchema: z.strictObject({
        prompt: z.string().min(1).max(20_000).describe("What to ask about the image, e.g. '¿Qué avance de obra muestra esta foto?'"),
        image_base64: z.string().min(1).describe("Base64-encoded image data, without the 'data:image/...;base64,' prefix"),
        mime_type: z
          .enum(["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"])
          .default("image/jpeg"),
        model: ModelField.default(DEFAULT_TEXT_MODEL)
      }),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: false, openWorldHint: true }
    },
    async ({ prompt, image_base64, mime_type, model }, ctx) => {
      const body: GenerateContentRequest = {
        contents: [{ role: "user", parts: [{ inline_data: { mime_type, data: image_base64 } }, { text: prompt }] }]
      };
      const text = await generateContent(model, body, ctx.mcpReq.signal);
      return { content: [{ type: "text", text }] };
    }
  );

  server.registerTool(
    "gemini_analyze_video",
    {
      title: "Analyze Video with Gemini",
      description: `Send a short video (base64-encoded) plus a question to Gemini for analysis.

Useful for videos of avance de obra, maquinaria en funcionamiento, visitas a proyectos, etc.
Limit: total request under ~20MB (roughly a video of one minute or less at normal quality). For longer/larger
videos this tool will fail — in that case, trim the clip or ask for the File API upload flow to be added.
The video must already be base64-encoded (no "data:" prefix).`,
      inputSchema: z.strictObject({
        prompt: z.string().min(1).max(20_000).describe("What to ask about the video, e.g. '¿Qué avance muestra este video de la obra?'"),
        video_base64: z.string().min(1).describe("Base64-encoded video data, without the 'data:video/...;base64,' prefix"),
        mime_type: z
          .enum(["video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"])
          .default("video/mp4"),
        model: ModelField.default(DEFAULT_TEXT_MODEL)
      }),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: false, openWorldHint: true }
    },
    async ({ prompt, video_base64, mime_type, model }, ctx) => {
      const body: GenerateContentRequest = {
        contents: [{ role: "user", parts: [{ inline_data: { mime_type, data: video_base64 } }, { text: prompt }] }]
      };
      const text = await generateContent(model, body, ctx.mcpReq.signal, 120_000);
      return { content: [{ type: "text", text }] };
    }
  );

  server.registerTool(
    "gemini_generate_video",
    {
      title: "Generate Video with Veo (start)",
      description: `Start generating a new 8-second video with Google Veo from a text prompt (and optionally a reference image).

This only STARTS the generation, which normally takes 1-6 minutes. It returns an operation_name.
Call gemini_check_video_status with that operation_name afterwards (wait ~30-60s before the first check) to
retrieve the finished video.`,
      inputSchema: z.strictObject({
        prompt: z.string().min(1).max(5_000).describe("Description of the video to generate, in detail (subject, action, style, camera, audio/dialogue if any)"),
        image_base64: z
          .string()
          .optional()
          .describe("Optional base64-encoded reference image (no 'data:' prefix) to guide the video (image-to-video)"),
        image_mime_type: z.enum(["image/jpeg", "image/png", "image/webp"]).default("image/jpeg"),
        aspect_ratio: z.enum(["16:9", "9:16"]).default("16:9").describe("16:9 landscape or 9:16 portrait"),
        negative_prompt: z.string().max(2_000).optional().describe("What to avoid in the generated video"),
        model: z
          .string()
          .min(1)
          .default(DEFAULT_VEO_MODEL)
          .describe("Veo model id, e.g. 'veo-3.1-generate-preview' (default) or 'veo-3.1-fast-generate-preview' for a faster/cheaper draft")
      }),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true }
    },
    async ({ prompt, image_base64, image_mime_type, aspect_ratio, negative_prompt, model }, ctx) => {
      const instance: Record<string, unknown> = { prompt };
      if (image_base64) instance.image = { bytesBase64Encoded: image_base64, mimeType: image_mime_type };
      const parameters: Record<string, unknown> = { aspectRatio: aspect_ratio };
      if (negative_prompt) parameters.negativePrompt = negative_prompt;

      const response = await geminiFetch(
        `models/${encodeURIComponent(model)}:predictLongRunning`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ instances: [instance], parameters }) },
        ctx.mcpReq.signal
      );
      const op = (await response.json()) as VeoOperation;
      return {
        content: [
          {
            type: "text",
            text: `Generación de video iniciada.\noperation_name: ${op.name}\n\nEspera ~30-60 segundos y llama a gemini_check_video_status con este operation_name para obtener el resultado (puede tardar varios minutos en total).`
          }
        ]
      };
    }
  );

  server.registerTool(
    "gemini_check_video_status",
    {
      title: "Check Veo Video Status",
      description: `Check the status of a video generation started with gemini_generate_video.

Pass the operation_name it returned. If still running, says so (call again after ~20-30s).
If finished, downloads the video and returns it as a base64-encoded video/mp4 resource.`,
      inputSchema: z.strictObject({
        operation_name: z.string().min(1).describe("The operation_name returned by gemini_generate_video, e.g. 'models/veo-3.1-generate-preview/operations/abc123'")
      }),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true }
    },
    async ({ operation_name }, ctx) => {
      const response = await geminiFetch(operation_name, { method: "GET" }, ctx.mcpReq.signal, 30_000);
      const op = (await response.json()) as VeoOperation;

      if (!op.done) {
        return { content: [{ type: "text", text: "Todavía generando el video. Vuelve a llamar a gemini_check_video_status en 20-30 segundos." }] };
      }
      if (op.error) {
        return { isError: true, content: [{ type: "text", text: `La generación del video falló: ${op.error.message ?? "error desconocido"}` }] };
      }
      const video = extractVideo(op);
      if (!video) {
        return { isError: true, content: [{ type: "text", text: `El video terminó pero no se encontró en la respuesta: ${JSON.stringify(op).slice(0, 1000)}` }] };
      }

      let base64Data = video.bytesBase64Encoded;
      if (!base64Data && video.uri) {
        const videoResponse = await geminiFetch(video.uri, { method: "GET" }, ctx.mcpReq.signal, 120_000);
        const buffer = Buffer.from(await videoResponse.arrayBuffer());
        base64Data = buffer.toString("base64");
      }
      if (!base64Data) {
        return { isError: true, content: [{ type: "text", text: "El video terminó pero no se pudo descargar." }] };
      }

      return {
        content: [
          { type: "text", text: "Video generado correctamente." },
          { type: "resource", resource: { uri: `gemini-video://${operation_name}`, mimeType: "video/mp4", blob: base64Data } }
        ]
      };
    }
  );

  return server;
}

// ---- Transport selection ----

if (process.env.TRANSPORT === "http") {
  const handler = createMcpHandler(createServer);
  const app = createMcpExpressApp({
    host: "0.0.0.0",
    ...(process.env.ALLOWED_HOST ? { allowedHosts: [process.env.ALLOWED_HOST] } : {})
  });
  const nodeHandler = toNodeHandler(handler);
  app.all("/mcp", (req, res) => void nodeHandler(req, res, req.body));
  app.get("/", (_req, res) => res.send("gemini-mcp-server is running. MCP endpoint: /mcp"));

  const port = Number(process.env.PORT ?? 3000);
  app.listen(port, "0.0.0.0", () => {
    console.error(`gemini-mcp-server listening on http://0.0.0.0:${port}/mcp`);
  });
} else {
  serveStdio(createServer);
  console.error("gemini-mcp-server running on stdio");
}
