import type { Bot } from "mineflayer";
import {
  ROOM_SIZE,
  parseCoordinate,
  sleep,
  type ArenaOrigin
} from "./shared.js";

type ArenaCommand = (command: string) => Promise<void>;

export function validateStartingHealth(value: number): number {
  if (![5, 10, 20].includes(value))
    throw new Error("botHealth must be one of 5, 10, or 20");
  return value;
}

export function createEmptyAttackResult() {
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
    movementNotSettled: false,
    targetMissing: false,
    attackPacketSent: false
  };
}

export async function waitForBotReady(
  bot: Bot,
  stageName: string
): Promise<void> {
  const deadline = Date.now() + 5_000;
  while (bot.health <= 0 && Date.now() < deadline) await sleep(50);
  if (bot.health <= 0)
    throw new Error(`${stageName} reset timed out waiting for the bot to respawn.`);
}

export async function waitForInventoryItem(
  bot: Bot,
  name: string,
  timeoutMs: number
) {
  const deadline = Date.now() + timeoutMs;
  do {
    const item = bot.inventory.items().find((entry) => entry.name === name);
    if (item) return item;
    await sleep(50);
  } while (Date.now() < deadline);
  return null;
}

export function createArenaOrigin(bot: Bot): ArenaOrigin {
  const configuredX = parseCoordinate(process.env.RL_ARENA_X);
  const configuredY = parseCoordinate(process.env.RL_ARENA_Y);
  const configuredZ = parseCoordinate(process.env.RL_ARENA_Z);
  return {
    x: configuredX ?? Math.floor(bot.entity.position.x) - 7,
    y: configuredY ?? Math.floor(bot.entity.position.y) - 1,
    z: configuredZ ?? Math.floor(bot.entity.position.z) - 7
  };
}

export async function buildCombatArena(
  command: ArenaCommand,
  origin: ArenaOrigin
): Promise<void> {
  const { x, y, z } = origin;
  await command(`fill ${x - 1} ${y} ${z - 1} ${x + 15} ${y + 4} ${z + 15} air`);
  await command(`fill ${x} ${y} ${z} ${x + 14} ${y} ${z + 14} smooth_stone`);
  await command(`fill ${x - 1} ${y} ${z - 1} ${x - 1} ${y + 3} ${z + 15} glass`);
  await command(`fill ${x + 15} ${y} ${z - 1} ${x + 15} ${y + 3} ${z + 15} glass`);
  await command(`fill ${x} ${y} ${z - 1} ${x + 14} ${y + 3} ${z - 1} glass`);
  await command(`fill ${x} ${y} ${z + 15} ${x + 14} ${y + 3} ${z + 15} glass`);
  await command(
    `fill ${x - 1} ${y + 4} ${z - 1} ${x + 15} ${y + 4} ${z + 15} sea_lantern`
  );
}

export async function sanitizeCombatArena(
  command: ArenaCommand,
  origin: ArenaOrigin
): Promise<void> {
  const { x, y, z } = origin;
  await command(`fill ${x} ${y + 1} ${z} ${x + 14} ${y + 3} ${z + 14} air`);
  await command(`fill ${x} ${y} ${z} ${x + 14} ${y} ${z + 14} smooth_stone`);
  await command(
    `fill ${x - 1} ${y + 4} ${z - 1} ${x + 15} ${y + 4} ${z + 15} sea_lantern`
  );
}

export async function clearCombatArenaEntities(
  command: ArenaCommand,
  origin: ArenaOrigin
): Promise<void> {
  const { x, y, z } = origin;
  await command(
    `kill @e[x=${x},y=${y + 1},z=${z},dx=14,dy=3,dz=14,type=!minecraft:player]`
  );
}

export class TargetHealthReader {
  private objectiveReady = false;

  constructor(
    private readonly bot: Bot,
    private readonly objective: string,
    private readonly targetTag: string,
    private readonly command: ArenaCommand
  ) {}

  async ensureObjective(): Promise<void> {
    if (this.objectiveReady) return;
    await this.command(`scoreboard objectives add ${this.objective} dummy`);
    this.objectiveReady = true;
  }

  async query(targetConfirmedDead: boolean): Promise<number | null> {
    if (targetConfirmedDead) return 0;
    await this.ensureObjective();
    await this.command(`scoreboard players set #target ${this.objective} -1`);
    await this.command(
      `execute store result score #target ${this.objective} ` +
      "run data get entity " +
      `@e[type=minecraft:zombie,tag=${this.targetTag},limit=1] Health 100`
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
        if (!message.includes(this.objective)) return;
        const match = message.match(/#target has (-?\d+)/);
        if (match) {
          const storedHealth = Number(match[1]);
          finish(storedHealth < 0 ? null : storedHealth / 100);
        }
      };
      const timeout = setTimeout(() => finish(null), 750);
      this.bot.on("messagestr", onMessage);
      this.bot.chat(`/scoreboard players get #target ${this.objective}`);
    });
  }
}
