import type { Bot } from "mineflayer";
import {
  ACTION_DURATION_MS,
  ROOM_SIZE,
  TURN_RADIANS,
  distanceBetween,
  parseCoordinate,
  sleep,
  validatePosition
} from "./shared.js";
import type { ArenaOrigin, BridgeRequest } from "./shared.js";

export class StageOneArena {
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
    // tag, while this clears mobs, projectiles, and dropped items.
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
