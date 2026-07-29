import type { Bot } from "mineflayer";
import type { Entity } from "prismarine-entity";
import { WebSocket, WebSocketServer } from "ws";

export interface RlBridgeOptions {
  onTrainingStart?: () => void;
  onTrainingEnd?: () => void;
}

interface BridgeRequest {
  id: number;
  type:
    | "ping"
    | "stage1.reset"
    | "stage1.step"
    | "stage1.observe"
    | "stage2.reset"
    | "stage2.step"
    | "stage2.observe";
  action?: number;
  botPosition?: [number, number];
  targetPosition?: [number, number];
  botYaw?: number;
  rebuildArena?: boolean;
}

interface ArenaOrigin {
  x: number;
  y: number;
  z: number;
}

const ROOM_SIZE = 15;
const ACTION_DURATION_MS = 100;
const TURN_RADIANS = Math.PI * 0.1;
// Leave a margin below Minecraft's nominal three-block survival reach.  A
// horizontal centre-to-centre distance of exactly three blocks can still put
// the target hitbox outside server-validated reach.
const ATTACK_RANGE = 2.5;
const IRON_SWORD_COOLDOWN_SECONDS = 0.625;
const ATTACK_AIM_SETTLE_MS = 75;
const MOVEMENT_SETTLE_TIMEOUT_MS = 450;
const SETTLED_HORIZONTAL_SPEED = 0.025;
const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8765;

/**
 * Starts a localhost-only WebSocket server used by the Python Gymnasium
 * environment 
 * Only one training client can control the bot at a time
 */
export function startRlBridge(bot: Bot, options: RlBridgeOptions = {}): () => void {
  const host = process.env.RL_BRIDGE_HOST ?? DEFAULT_HOST;
  const port = parsePort(process.env.RL_BRIDGE_PORT);
  const server = new WebSocketServer({ host, port, maxPayload: 64 * 1024 });
  const stageOne = new StageOneArena(bot);
  const stageTwo = new StageTwoArena(bot);
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
    client?.close(1001, "Bot shutting down");
    server.close();
    if (trainingActive) {
      trainingActive = false;
      options.onTrainingEnd?.();
    }
  };
}

class StageOneArena {
  private origin: ArenaOrigin | null = null;
  private arenaBuilt = false;
  private targetPosition: [number, number] = [12.5, 12.5];
  private elapsedSeconds = 0;

  constructor(private readonly bot: Bot) {}

  async reset(request: BridgeRequest) {
    const botPosition = validatePosition(request.botPosition ?? [2.5, 2.5], "botPosition");
    const targetPosition = validatePosition(
      request.targetPosition ?? [12.5, 12.5],
      "targetPosition"
    );
    const yaw = Number.isFinite(request.botYaw) ? request.botYaw! : 0;

    if (distanceBetween(botPosition, targetPosition) <= 3)
      throw new Error("Initial bot and target positions must be outside attack range");

    if (!this.origin) this.origin = this.createOrigin();
    if (!this.arenaBuilt || request.rebuildArena) await this.buildArena();
    else await this.sanitizeArena();

    this.bot.clearControlStates();
    this.bot.pathfinder.setGoal(null);
    this.targetPosition = targetPosition;
    this.elapsedSeconds = 0;

    await this.command("clear @s");
    await this.command("effect clear @s");
    await this.command("attribute @s minecraft:max_health base set 20");
    await this.command("effect give @s minecraft:instant_health 1 255 true");
    await this.command("effect give @s minecraft:saturation 999999 0 true");
    await this.command("kill @e[type=minecraft:armor_stand,tag=rl_stage1_target]");
    await this.clearArenaEntities();

    const target = this.worldPosition(targetPosition);
    await this.command(
      `summon minecraft:armor_stand ${target.x} ${this.origin.y + 1} ${target.z} ` +
      `{Tags:["rl_stage1_target"],NoGravity:1b,Invulnerable:1b,Silent:1b}`
    );

    const spawn = this.worldPosition(botPosition);
    const yawDegrees = yaw * 180 / Math.PI;
    await this.command(`tp @s ${spawn.x} ${this.origin.y + 1} ${spawn.z} ${yawDegrees} 0`);
    await sleep(250);

    const resetDistance = Math.hypot(
      this.bot.entity.position.x - spawn.x,
      this.bot.entity.position.z - spawn.z
    );
    if (resetDistance > 1.5) {
      throw new Error(
        "The arena reset could not teleport the bot. Ensure the bot is a server operator."
      );
    }

    const targetSpawned = Object.values(this.bot.entities).some((entity) =>
      entity.name === "armor_stand" &&
      Math.hypot(entity.position.x - target.x, entity.position.z - target.z) <= 2
    );
    if (!targetSpawned) {
      throw new Error(
        "The stationary target did not spawn. Ensure the bot is a server operator."
      );
    }

    return this.observe();
  }

  async step(action: number) {
    this.bot.clearControlStates();
    try {
      switch (action) {
        case 0:
          break;
        case 1:
          this.bot.setControlState("forward", true);
          break;
        case 2:
          this.bot.setControlState("back", true);
          break;
        case 3:
          this.bot.setControlState("left", true);
          break;
        case 4:
          this.bot.setControlState("right", true);
          break;
        case 5:
          await this.bot.look(this.bot.entity.yaw - TURN_RADIANS, 0, true);
          break;
        case 6:
          await this.bot.look(this.bot.entity.yaw + TURN_RADIANS, 0, true);
          break;
      }
      await sleep(ACTION_DURATION_MS);
    } finally {
      this.bot.clearControlStates();
    }

    this.elapsedSeconds += ACTION_DURATION_MS / 1_000;
    return this.observe();
  }

  observe() {
    if (!this.origin) throw new Error("Stage-one arena has not been initialized");
    const botX = this.bot.entity.position.x - this.origin.x;
    const botZ = this.bot.entity.position.z - this.origin.z;
    const dx = this.targetPosition[0] - botX;
    const dz = this.targetPosition[1] - botZ;

    return {
      botPosition: [botX, botZ],
      targetPosition: [...this.targetPosition],
      yaw: this.bot.entity.yaw,
      health: this.bot.health,
      hasEquipment: this.bot.inventory.items().length > 0,
      distance: Math.hypot(dx, dz),
      elapsedSeconds: this.elapsedSeconds,
      roomSize: ROOM_SIZE
    };
  }

  private createOrigin(): ArenaOrigin {
    const configuredX = parseCoordinate(process.env.RL_ARENA_X);
    const configuredY = parseCoordinate(process.env.RL_ARENA_Y);
    const configuredZ = parseCoordinate(process.env.RL_ARENA_Z);

    return {
      x: configuredX ?? Math.floor(this.bot.entity.position.x) - 7,
      y: configuredY ?? Math.floor(this.bot.entity.position.y) - 1,
      z: configuredZ ?? Math.floor(this.bot.entity.position.z) - 7
    };
  }

  private async buildArena(): Promise<void> {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    const { x, y, z } = this.origin;

    // The floor is 15x15. Walls sit just outside it, preserving all 15 blocks
    // of walkable width while preventing the bot from leaving the arena.
    await this.command(`fill ${x - 1} ${y} ${z - 1} ${x + 15} ${y + 4} ${z + 15} air`);
    await this.command(`fill ${x} ${y} ${z} ${x + 14} ${y} ${z + 14} smooth_stone`);
    await this.command(`fill ${x - 1} ${y} ${z - 1} ${x - 1} ${y + 3} ${z + 15} glass`);
    await this.command(`fill ${x + 15} ${y} ${z - 1} ${x + 15} ${y + 3} ${z + 15} glass`);
    await this.command(`fill ${x} ${y} ${z - 1} ${x + 14} ${y + 3} ${z - 1} glass`);
    await this.command(`fill ${x} ${y} ${z + 15} ${x + 14} ${y + 3} ${z + 15} glass`);
    // A solid illuminated ceiling prevents hostile spawns and flying mobs from
    // entering while leaving three full blocks of interior height.
    await this.command(`fill ${x - 1} ${y + 4} ${z - 1} ${x + 15} ${y + 4} ${z + 15} sea_lantern`);
    this.arenaBuilt = true;
  }

  private async sanitizeArena(): Promise<void> {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    const { x, y, z } = this.origin;

    // Remove any blocks, liquids, fire, or other hazards introduced during the
    // previous episode, then restore the safe floor and illuminated ceiling.
    await this.command(`fill ${x} ${y + 1} ${z} ${x + 14} ${y + 3} ${z + 14} air`);
    await this.command(`fill ${x} ${y} ${z} ${x + 14} ${y} ${z + 14} smooth_stone`);
    await this.command(`fill ${x - 1} ${y + 4} ${z - 1} ${x + 15} ${y + 4} ${z + 15} sea_lantern`);
  }

  private async clearArenaEntities(): Promise<void> {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    const { x, y, z } = this.origin;
    // Players are never affected. The old target is removed separately by its
    // tag, while this clears mobs, projectiles, dropped items, and vehicles.
    await this.command(
      `kill @e[x=${x},y=${y + 1},z=${z},dx=14,dy=3,dz=14,` +
      "type=!minecraft:player,type=!minecraft:armor_stand]"
    );
  }

  private worldPosition(local: [number, number]) {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    return {
      x: this.origin.x + local[0],
      z: this.origin.z + local[1]
    };
  }

  private async command(command: string): Promise<void> {
    this.bot.chat(`/${command}`);
    await sleep(75);
  }
}

class StageTwoArena {
  private origin: ArenaOrigin | null = null;
  private arenaBuilt = false;
  private targetPosition: [number, number] = [10.5, 7.5];
  private targetEntityId: number | null = null;
  private lastKnownTargetHealth = 20;
  private targetConfirmedDead = false;
  private elapsedSeconds = 0;
  private nextAttackTime = 0;
  private pendingAttackUntil = 0;
  private healthObjectiveReady = false;
  private lastActionResult = createEmptyAttackResult();

  constructor(private readonly bot: Bot) {
    bot.on("entityHurt", (entity) => {
      if (!entity || entity.id !== this.targetEntityId) return;
      if (entity.health !== undefined && entity.health !== null)
        this.lastKnownTargetHealth = Math.min(
          this.lastKnownTargetHealth,
          Math.max(0, entity.health)
        );
      if (Date.now() <= this.pendingAttackUntil)
        this.lastActionResult.attackLanded = true;
    });
    bot.on("entityDead", (entity) => {
      if (!entity || entity.id !== this.targetEntityId) return;
      this.targetConfirmedDead = true;
      this.lastKnownTargetHealth = 0;
      this.lastActionResult.confirmedKill = true;
      if (Date.now() <= this.pendingAttackUntil)
        this.lastActionResult.attackLanded = true;
    });
  }

  async reset(request: BridgeRequest) {
    const botPosition = validatePosition(request.botPosition ?? [4.5, 7.5], "botPosition");
    const targetPosition = validatePosition(
      request.targetPosition ?? [10.5, 7.5],
      "targetPosition"
    );
    const startingDistance = distanceBetween(botPosition, targetPosition);
    if (startingDistance < 5 || startingDistance > 8)
      throw new Error("Stage-two positions must start between 5 and 8 blocks apart");

    const yaw = Number.isFinite(request.botYaw) ? request.botYaw! : 0;
    if (!this.origin) this.origin = this.createOrigin();
    if (!this.arenaBuilt || request.rebuildArena) await this.buildArena();
    else await this.sanitizeArena();

    this.bot.clearControlStates();
    this.bot.pathfinder.setGoal(null);
    this.targetPosition = targetPosition;
    this.targetEntityId = null;
    this.lastKnownTargetHealth = 20;
    this.targetConfirmedDead = false;
    this.elapsedSeconds = 0;
    this.nextAttackTime = 0;
    this.pendingAttackUntil = 0;
    this.lastActionResult = createEmptyAttackResult();

    await this.command("clear @s");
    await this.command("effect clear @s");
    await this.command("attribute @s minecraft:max_health base set 20");
    await this.command("effect give @s minecraft:instant_health 1 255 true");
    await this.command("effect give @s minecraft:saturation 999999 0 true");
    // Native 1.21.11 currently has a player hitbox/physics edge case affecting
    // Mineflayer at the default scale.  This imperceptible scale adjustment
    // keeps the server and client collision/reach calculations aligned.
    await this.command("attribute @s minecraft:scale base set 0.9999999");
    await this.command("kill @e[type=minecraft:zombie,tag=rl_stage2_target]");
    await this.clearArenaEntities();
    await this.command("give @s minecraft:iron_sword 1");

    const target = this.worldPosition(targetPosition);
    await this.command(
      `summon minecraft:zombie ${target.x} ${this.origin.y + 1} ${target.z} ` +
      `{Tags:["rl_stage2_target"],NoAI:1b,PersistenceRequired:1b,Silent:1b,` +
      `CanPickUpLoot:0b,IsBaby:0b,Health:20.0f}`
    );
    await this.command(
      "attribute @e[type=minecraft:zombie,tag=rl_stage2_target,limit=1] " +
      "minecraft:knockback_resistance base set 1"
    );

    const spawn = this.worldPosition(botPosition);
    const yawDegrees = yaw * 180 / Math.PI;
    await this.command(`tp @s ${spawn.x} ${this.origin.y + 1} ${spawn.z} ${yawDegrees} 0`);
    await sleep(300);

    const ironSword = this.bot.inventory.items().find((item) => item.name === "iron_sword");
    if (!ironSword)
      throw new Error("The bot did not receive its iron sword. Ensure it is a server operator.");
    await this.bot.equip(ironSword, "hand");

    const targetEntity = Object.values(this.bot.entities).find((entity) =>
      entity.name === "zombie" &&
      Math.hypot(entity.position.x - target.x, entity.position.z - target.z) <= 2
    );
    if (!targetEntity)
      throw new Error("The stage-two zombie did not spawn. Ensure difficulty is not peaceful.");
    this.targetEntityId = targetEntity.id;
    this.lastKnownTargetHealth = targetEntity.health ?? 20;
    await this.ensureHealthObjective();
    const serverHealth = await this.queryTargetHealthFromServer();
    if (serverHealth !== null) this.lastKnownTargetHealth = serverHealth;

    const resetDistance = Math.hypot(
      this.bot.entity.position.x - spawn.x,
      this.bot.entity.position.z - spawn.z
    );
    if (resetDistance > 1.5)
      throw new Error("The arena reset could not teleport the bot.");

    return this.observe();
  }

  async step(action: number) {
    this.bot.clearControlStates();
    this.lastActionResult = createEmptyAttackResult();
    try {
      switch (action) {
        case 0:
          break;
        case 1:
          this.bot.setControlState("forward", true);
          break;
        case 2:
          this.bot.setControlState("back", true);
          break;
        case 3:
          this.bot.setControlState("left", true);
          break;
        case 4:
          this.bot.setControlState("right", true);
          break;
        case 5:
          await this.bot.look(this.bot.entity.yaw - TURN_RADIANS, 0, true);
          break;
        case 6:
          await this.bot.look(this.bot.entity.yaw + TURN_RADIANS, 0, true);
          break;
        case 7: {
          this.lastActionResult.attackSelected = true;
          let target = this.getTargetEntity();
          let distance = target
            ? this.attackDistanceTo(target)
            : this.horizontalDistanceToWorldTarget();
          this.lastActionResult.attackDistance = distance;
          this.lastActionResult.outOfRange =
            !target || distance > ATTACK_RANGE;
          if (this.lastActionResult.outOfRange) break;

          const movementSettled = await this.waitForMovementToSettle();
          this.lastActionResult.movementSettled = movementSettled;
          target = this.getTargetEntity();
          distance = target
            ? this.attackDistanceTo(target)
            : this.horizontalDistanceToWorldTarget();
          this.lastActionResult.attackDistance = distance;
          this.lastActionResult.outOfRange =
            !movementSettled || !target || distance > ATTACK_RANGE;
          this.lastActionResult.cooldownBlocked =
            !this.lastActionResult.outOfRange &&
            this.elapsedSeconds < this.nextAttackTime;
          if (target && !this.lastActionResult.outOfRange && !this.lastActionResult.cooldownBlocked) {
            const healthBeforeAttack = this.lastKnownTargetHealth;
            await this.bot.lookAt(
              target.position.offset(0, target.height * 0.65, 0),
              true
            );
            // Give the server one tick to process the forced rotation before
            // sending the interact-entity attack packet.
            await sleep(ATTACK_AIM_SETTLE_MS);

            // Looking takes time and the entity reference can change. Re-check
            // actual reach immediately before the attack packet is sent.
            target = this.getTargetEntity();
            distance = target
              ? this.attackDistanceTo(target)
              : this.horizontalDistanceToWorldTarget();
            this.lastActionResult.attackDistance = distance;
            this.lastActionResult.outOfRange =
              !target || distance > ATTACK_RANGE;

            if (target && !this.lastActionResult.outOfRange) {
              this.lastActionResult.validAttackAttempt = true;
              this.lastActionResult.attackPacketSent = true;
              this.pendingAttackUntil = Date.now() + 1_000;
              this.bot.attack(target);
              this.nextAttackTime =
                this.elapsedSeconds + IRON_SWORD_COOLDOWN_SECONDS;
              await this.waitForAttackResult(healthBeforeAttack);
              const serverHealth = await this.queryTargetHealthFromServer();
              if (serverHealth !== null) {
                this.lastActionResult.serverHealthVerified = true;
                this.lastKnownTargetHealth = serverHealth;
                if (serverHealth < healthBeforeAttack)
                  this.lastActionResult.attackLanded = true;
                if (serverHealth <= 0) {
                  this.targetConfirmedDead = true;
                  this.lastActionResult.confirmedKill = true;
                }
              }
            }
          }
          break;
        }
      }
      await sleep(ACTION_DURATION_MS);
    } finally {
      this.bot.clearControlStates();
    }

    this.elapsedSeconds += ACTION_DURATION_MS / 1_000;
    return this.observe();
  }

  observe() {
    if (!this.origin) throw new Error("Stage-two arena has not been initialized");
    const botX = this.bot.entity.position.x - this.origin.x;
    const botZ = this.bot.entity.position.z - this.origin.z;
    const target = this.getTargetEntity();
    if (target?.health !== undefined && target.health !== null)
      this.lastKnownTargetHealth = Math.min(
        this.lastKnownTargetHealth,
        Math.max(0, target.health)
      );

    const targetAlive = !this.targetConfirmedDead;
    return {
      botPosition: [botX, botZ],
      targetPosition: [...this.targetPosition],
      yaw: this.bot.entity.yaw,
      health: this.bot.health,
      hasIronSword: this.bot.heldItem?.name === "iron_sword",
      targetHealth: this.lastKnownTargetHealth,
      targetAlive,
      targetVisible: target !== null,
      distance: this.horizontalDistanceToWorldTarget(),
      attackReady: this.elapsedSeconds >= this.nextAttackTime,
      attackResult: { ...this.lastActionResult },
      elapsedSeconds: this.elapsedSeconds,
      roomSize: ROOM_SIZE
    };
  }

  private getTargetEntity() {
    if (this.targetConfirmedDead || !this.origin) return null;
    if (this.targetEntityId !== null) {
      const entity = this.bot.entities[this.targetEntityId];
      if (entity?.name === "zombie") return entity;
    }

    const expected = this.worldPosition(this.targetPosition);
    const reacquired = Object.values(this.bot.entities).find((entity) =>
      entity.name === "zombie" &&
      Math.hypot(entity.position.x - expected.x, entity.position.z - expected.z) <= 2
    );
    if (reacquired) {
      this.targetEntityId = reacquired.id;
      return reacquired;
    }
    return null;
  }

  private async waitForAttackResult(healthBeforeAttack: number): Promise<void> {
    const deadline = Date.now() + 500;
    while (Date.now() < deadline) {
      const target = this.getTargetEntity();
      if (
        this.targetConfirmedDead ||
        (target?.health !== undefined &&
          target.health !== null &&
          target.health < healthBeforeAttack)
      ) {
        if (target?.health !== undefined && target.health !== null)
          this.lastKnownTargetHealth = Math.max(0, target.health);
        return;
      }
      await sleep(25);
    }
  }

  private attackDistanceTo(target: Entity): number {
    const eye = this.bot.entity.position.offset(0, 1.62, 0);
    const targetCenter = target.position.offset(0, target.height * 0.5, 0);
    return eye.distanceTo(targetCenter);
  }

  private async waitForMovementToSettle(): Promise<boolean> {
    this.bot.clearControlStates();
    const deadline = Date.now() + MOVEMENT_SETTLE_TIMEOUT_MS;
    let settledSamples = 0;

    while (Date.now() < deadline) {
      const horizontalSpeed = Math.hypot(
        this.bot.entity.velocity.x,
        this.bot.entity.velocity.z
      );
      if (horizontalSpeed <= SETTLED_HORIZONTAL_SPEED) {
        settledSamples += 1;
        if (settledSamples >= 2) return true;
      } else {
        settledSamples = 0;
      }
      await sleep(25);
    }
    return false;
  }

  private async ensureHealthObjective(): Promise<void> {
    if (this.healthObjectiveReady) return;
    // Objectives persist with the world, so "already exists" is harmless.
    await this.command("scoreboard objectives add rl_stage2_health dummy");
    this.healthObjectiveReady = true;
  }

  /**
   * Mineflayer's living-entity health metadata is not consistently refreshed
   * on every supported protocol.  Read the zombie's actual NBT health through
   * a scoreboard so rewards are based on server state rather than stale client
   * metadata.
   */
  private async queryTargetHealthFromServer(): Promise<number | null> {
    if (this.targetConfirmedDead) return 0;
    await this.ensureHealthObjective();
    // A missing entity must not reuse the previous zombie's score.
    await this.command("scoreboard players set #target rl_stage2_health -1");
    await this.command(
      "execute store result score #target rl_stage2_health run data get entity " +
      "@e[type=minecraft:zombie,tag=rl_stage2_target,limit=1] Health 100"
    );

    return new Promise<number | null>((resolve) => {
      let settled = false;
      const finish = (health: number | null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        this.bot.off("messagestr", onMessage);
        resolve(health);
      };
      const onMessage = (message: string) => {
        if (!message.includes("rl_stage2_health")) return;
        const match = message.match(/#target has (-?\d+)/);
        if (match) {
          const storedHealth = Number(match[1]);
          finish(storedHealth < 0 ? null : storedHealth / 100);
        }
      };
      const timeout = setTimeout(() => finish(null), 750);
      this.bot.on("messagestr", onMessage);
      this.bot.chat("/scoreboard players get #target rl_stage2_health");
    });
  }

  private horizontalDistanceTo(x: number, z: number): number {
    return Math.hypot(this.bot.entity.position.x - x, this.bot.entity.position.z - z);
  }

  private horizontalDistanceToWorldTarget(): number {
    const target = this.worldPosition(this.targetPosition);
    return this.horizontalDistanceTo(target.x, target.z);
  }

  private createOrigin(): ArenaOrigin {
    const configuredX = parseCoordinate(process.env.RL_ARENA_X);
    const configuredY = parseCoordinate(process.env.RL_ARENA_Y);
    const configuredZ = parseCoordinate(process.env.RL_ARENA_Z);
    return {
      x: configuredX ?? Math.floor(this.bot.entity.position.x) - 7,
      y: configuredY ?? Math.floor(this.bot.entity.position.y) - 1,
      z: configuredZ ?? Math.floor(this.bot.entity.position.z) - 7
    };
  }

  private async buildArena(): Promise<void> {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    const { x, y, z } = this.origin;
    await this.command(`fill ${x - 1} ${y} ${z - 1} ${x + 15} ${y + 4} ${z + 15} air`);
    await this.command(`fill ${x} ${y} ${z} ${x + 14} ${y} ${z + 14} smooth_stone`);
    await this.command(`fill ${x - 1} ${y} ${z - 1} ${x - 1} ${y + 3} ${z + 15} glass`);
    await this.command(`fill ${x + 15} ${y} ${z - 1} ${x + 15} ${y + 3} ${z + 15} glass`);
    await this.command(`fill ${x} ${y} ${z - 1} ${x + 14} ${y + 3} ${z - 1} glass`);
    await this.command(`fill ${x} ${y} ${z + 15} ${x + 14} ${y + 3} ${z + 15} glass`);
    await this.command(
      `fill ${x - 1} ${y + 4} ${z - 1} ${x + 15} ${y + 4} ${z + 15} sea_lantern`
    );
    this.arenaBuilt = true;
  }

  private async sanitizeArena(): Promise<void> {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    const { x, y, z } = this.origin;
    await this.command(`fill ${x} ${y + 1} ${z} ${x + 14} ${y + 3} ${z + 14} air`);
    await this.command(`fill ${x} ${y} ${z} ${x + 14} ${y} ${z + 14} smooth_stone`);
    await this.command(
      `fill ${x - 1} ${y + 4} ${z - 1} ${x + 15} ${y + 4} ${z + 15} sea_lantern`
    );
  }

  private async clearArenaEntities(): Promise<void> {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    const { x, y, z } = this.origin;
    await this.command(
      `kill @e[x=${x},y=${y + 1},z=${z},dx=14,dy=3,dz=14,type=!minecraft:player]`
    );
  }

  private worldPosition(local: [number, number]) {
    if (!this.origin) throw new Error("Arena origin is unavailable");
    return { x: this.origin.x + local[0], z: this.origin.z + local[1] };
  }

  private async command(command: string): Promise<void> {
    this.bot.chat(`/${command}`);
    await sleep(75);
  }
}

function validatePosition(position: [number, number], name: string): [number, number] {
  if (
    !Array.isArray(position) ||
    position.length !== 2 ||
    !position.every(Number.isFinite)
  ) {
    throw new Error(`${name} must contain two finite coordinates`);
  }
  if (position.some((coordinate) => coordinate < 0.5 || coordinate > ROOM_SIZE - 0.5))
    throw new Error(`${name} must be within the 15x15 arena`);
  return [position[0], position[1]];
}

function distanceBetween(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function createEmptyAttackResult() {
  return {
    attackSelected: false,
    validAttackAttempt: false,
    outOfRange: false,
    cooldownBlocked: false,
    attackLanded: false,
    confirmedKill: false,
    attackDistance: null as number | null,
    serverHealthVerified: false,
    movementSettled: false,
    attackPacketSent: false
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

function parseCoordinate(value: string | undefined): number | null {
  if (value === undefined) return null;
  const coordinate = Number(value);
  if (!Number.isFinite(coordinate)) throw new Error(`Invalid RL arena coordinate: ${value}`);
  return Math.floor(coordinate);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
