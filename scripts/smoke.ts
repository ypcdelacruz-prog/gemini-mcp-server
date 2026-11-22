/**
 * Self-check: starts the built server over stdio, calls its tools, and exits non-zero on any failure.
 * Run with: npm run build && npm run smoke
 * Requires GEMINI_API_KEY to be set in the environment.
 */
import { strict as assert } from "node:assert";
import { Client } from "@modelcontextprotocol/client";
import { getDefaultEnvironment, StdioClientTransport } from "@modelcontextprotocol/client/stdio";

const client = new Client({ name: "smoke-test", version: "1.0.0" });
await client.connect(
  new StdioClientTransport({
    command: "node",
    args: ["dist/index.js"],
    env: { ...getDefaultEnvironment(), GEMINI_API_KEY: process.env.GEMINI_API_KEY ?? "" }
  })
);

const { tools } = await client.listTools();
assert.deepEqual(tools.map(tool => tool.name).sort(), ["gemini_analyze_image", "gemini_ask"]);

const ask = await client.callTool({
  name: "gemini_ask",
  arguments: { prompt: "Responde solo con la palabra: ok" }
});
assert.ok(!ask.isError, `gemini_ask failed: ${JSON.stringify(ask.content)}`);

// Invalid arguments must come back as a tool error the model can read
const invalid = await client.callTool({ name: "gemini_ask", arguments: { prompt: "" } });
assert.equal(invalid.isError, true, "an empty prompt should be rejected");

await client.close();
console.error("Smoke test passed");
