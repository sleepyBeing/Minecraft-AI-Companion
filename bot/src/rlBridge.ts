import type { Bot } from "mineflayer";
import { WebSocket, WebSocketServer } from "ws";
import { StageOneArena } from "./rl/stageOneArena.js";
import { StageTwoArena } from "./rl/stageTwoArena.js";
import { StageThreeArena } from "./rl/stageThreeArena.js";
import { StageFourArena } from "./rl/stageFourArena.js";
import type { BridgeRequest } from "./rl/shared.js";

export interface RlBridgeOptions {
  onTrainingStart?: () => void;
  onTrainingEnd?: () => void;
}

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8765;

export function startRlBridge(bot: Bot, options: RlBridgeOptions = {}): () => void {
  const host = process.env.RL_BRIDGE_HOST ?? DEFAULT_HOST;
  const port = parsePort(process.env.RL_BRIDGE_PORT);
  const server = new WebSocketServer({ host, port, maxPayload: 64 * 1024 });
  const stageOne = new StageOneArena(bot);
  const stageTwo = new StageTwoArena(bot);
  const stageThree = new StageThreeArena(bot);
  const stageFour = new StageFourArena(bot);
  let client: WebSocket | null = null;
  let trainingActive = false;

  server.on("listening", () => {
    console.log(`RL WebSocket bridge listening on ws://${host}:${port}.`);
  });

  server.on("error", (error) => {
    console.error("RL WebSocket bridge error:", error);
  });

  server.on("connection", (socket) => {
    if (client && client.readyState === WebSocket.OPEN) {
      socket.close(1013, "Another RL client is already connected");
      return;
    }

    client = socket;
    let requestQueue = Promise.resolve();

    socket.on("message", (data) => {
      requestQueue = requestQueue
        .then(() => handleRequest(socket, data.toString()))
        .catch((error) => {
          console.error("RL bridge request failed:", error);
          send(socket, {
            id: null,
            ok: false,
            error: error instanceof Error ? error.message : String(error)
          });
        });
    });

    socket.on("close", () => {
      if (client !== socket) return;
      client = null;
      bot.clearControlStates();
      void stageFour.restoreWorldSettings().catch((error) => {
        console.error("Could not restore the Stage-four world settings:", error);
      });
      if (trainingActive) {
        trainingActive = false;
        options.onTrainingEnd?.();
      }
    });
  });

  async function handleRequest(socket: WebSocket, rawMessage: string): Promise<void> {
    let request: BridgeRequest;
    try {
      request = JSON.parse(rawMessage) as BridgeRequest;
    } catch {
      send(socket, { id: null, ok: false, error: "Request must be valid JSON" });
      return;
    }

    if (!Number.isSafeInteger(request.id) || typeof request.type !== "string") {
      send(socket, { id: request.id ?? null, ok: false, error: "Invalid request envelope" });
      return;
    }

    try {
      switch (request.type) {
        case "ping":
          send(socket, { id: request.id, ok: true, type: "pong" });
          return;

        case "stage1.reset": {
          if (!trainingActive) {
            trainingActive = true;
            options.onTrainingStart?.();
          }
          const state = await stageOne.reset(request);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage1.step": {
          if (!trainingActive) throw new Error("Call stage1.reset before stage1.step");
          if (!Number.isInteger(request.action) || request.action! < 0 || request.action! > 6)
            throw new Error("Stage-one action must be an integer from 0 to 6");
          const state = await stageOne.step(request.action!);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage1.observe": {
          if (!trainingActive) throw new Error("Call stage1.reset before observing");
          send(socket, { id: request.id, ok: true, state: stageOne.observe() });
          return;
        }

        case "stage2.reset": {
          if (!trainingActive) {
            trainingActive = true;
            options.onTrainingStart?.();
          }
          const state = await stageTwo.reset(request);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage2.step": {
          if (!trainingActive) throw new Error("Call stage2.reset before stage2.step");
          if (!Number.isInteger(request.action) || request.action! < 0 || request.action! > 7)
            throw new Error("Stage-two action must be an integer from 0 to 7");
          const state = await stageTwo.step(request.action!);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage2.observe": {
          if (!trainingActive) throw new Error("Call stage2.reset before observing");
          send(socket, { id: request.id, ok: true, state: stageTwo.observe() });
          return;
        }

        case "stage3.reset": {
          if (!trainingActive) {
            trainingActive = true;
            options.onTrainingStart?.();
          }
          const state = await stageThree.reset(request);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage3.step": {
          if (!trainingActive) throw new Error("Call stage3.reset before stage3.step");
          if (!Number.isInteger(request.action) || request.action! < 0 || request.action! > 7)
            throw new Error("Stage-three action must be an integer from 0 to 7");
          const state = await stageThree.step(request.action!);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage3.observe": {
          if (!trainingActive) throw new Error("Call stage3.reset before observing");
          send(socket, { id: request.id, ok: true, state: stageThree.observe() });
          return;
        }

        case "stage4.reset": {
          if (!trainingActive) {
            trainingActive = true;
            options.onTrainingStart?.();
          }
          const state = await stageFour.reset(request);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage4.step": {
          if (!trainingActive) throw new Error("Call stage4.reset before stage4.step");
          if (!Number.isInteger(request.action) || request.action! < 0 || request.action! > 9)
            throw new Error("Stage-four action must be an integer from 0 to 9");
          const state = await stageFour.step(request.action!);
          send(socket, { id: request.id, ok: true, state });
          return;
        }

        case "stage4.observe": {
          if (!trainingActive) throw new Error("Call stage4.reset before observing");
          send(socket, { id: request.id, ok: true, state: stageFour.observe() });
          return;
        }

        default:
          send(socket, { id: request.id, ok: false, error: `Unknown request type: ${request.type}` });
      }
    } catch (error) {
      send(socket, {
        id: request.id,
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  return () => {
    bot.clearControlStates();
    void stageFour.restoreWorldSettings().catch((error) => {
      console.error("Could not restore the Stage-four world settings:", error);
    });
    client?.close(1001, "Bot shutting down");
    server.close();
    if (trainingActive) {
      trainingActive = false;
      options.onTrainingEnd?.();
    }
  };
}

function send(socket: WebSocket, response: object): void {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(response));
}

function parsePort(value: string | undefined): number {
  if (!value) return DEFAULT_PORT;
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65_535)
    throw new Error(`Invalid RL_BRIDGE_PORT: ${value}`);
  return port;
}
