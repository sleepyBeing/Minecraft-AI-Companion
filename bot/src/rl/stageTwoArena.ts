import type { Bot } from "mineflayer";
import type { Entity } from "prismarine-entity";
import {
  ACTION_DURATION_MS,
  ATTACK_AIM_SETTLE_MS,
  ATTACK_RANGE,
  CLOSE_ATTACK_RANGE,
  IRON_SWORD_COOLDOWN_SECONDS,
  MOVEMENT_SETTLE_TIMEOUT_MS,
  ROOM_SIZE,
  SETTLED_HORIZONTAL_SPEED,
  TARGET_REACQUIRE_TIMEOUT_MS,
  TURN_RADIANS,
  distanceBetween,
  parseCoordinate,
  sleep,
  validatePosition
} from "./shared.js";
import type { ArenaOrigin, BridgeRequest } from "./shared.js";

export interface CombatArenaOptions {
  stageName: string;
  targetTag: string;
  healthObjective: string;
  stationaryTarget: boolean;
}

const STAGE_TWO_OPTIONS: CombatArenaOptions = {
  stageName: "Stage-two",
  targetTag: "rl_stage2_target",
  healthObjective: "rl_stage2_health",
  stationaryTarget: true
};

export class StageTwoArena {
  private origin: ArenaOrigin | null = null;
  private arenaBuilt = false;
  private targetPosition: [number, number] = [10.5, 7.5];
  private targetEntityId: number | null = null;
  private lastKnownTargetHealth = 20;
  private targetConfirmedDead = false;
  private botConfirmedDead = false;
  private elapsedSeconds = 0;
  private nextAttackTime = 0;
  private pendingAttackUntil = 0;
  private healthObjectiveReady = false;
  private lastActionResult = createEmptyAttackResult();

  constructor(
    private readonly bot: Bot,
    private readonly options: CombatArenaOptions = STAGE_TWO_OPTIONS
  ) {
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
    // Health can already be restored by the time the next WebSocket state is
    // sampled. Latch the actual Mineflayer death event until reset so the
    // Gymnasium episode cannot continue across a respawn.
    bot.on("death", () => {
      this.botConfirmedDead = true;
      this.bot.clearControlStates();
    });
  }

  async reset(request: BridgeRequest) {
    await this.waitForBotReady();
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
    this.botConfirmedDead = false;
    this.elapsedSeconds = 0;
    this.nextAttackTime = 0;
    this.pendingAttackUntil = 0;
    this.lastActionResult = createEmptyAttackResult();

    await this.command("clear @s");
    await this.command("effect clear @s");
    await this.command("attribute @s minecraft:max_health base set 20");
    await this.command("effect give @s minecraft:instant_health 1 255 true");
    await this.command("effect give @s minecraft:saturation 999999 0 true");
    await this.command("attribute @s minecraft:scale base set 0.9999999");
    await this.command(
      `kill @e[type=minecraft:zombie,tag=${this.options.targetTag}]`
    );
    await this.clearArenaEntities();
    await this.command("give @s minecraft:iron_sword 1");

    const target = this.worldPosition(targetPosition);
    const stationaryNbt = this.options.stationaryTarget ? "NoAI:1b," : "";
    await this.command(
      `summon minecraft:zombie ${target.x} ${this.origin.y + 1} ${target.z} ` +
      `{Tags:["${this.options.targetTag}"],${stationaryNbt}` +
      `PersistenceRequired:1b,Silent:1b,` +
      `CanPickUpLoot:0b,IsBaby:0b,Health:20.0f}`
    );
    if (this.options.stationaryTarget) {
      await this.command(
        `attribute @e[type=minecraft:zombie,tag=${this.options.targetTag},limit=1] ` +
        "minecraft:knockback_resistance base set 1"
      );
    }

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
          let target = await this.waitForTargetEntity();
          if (!target) {
            this.lastActionResult.targetMissing = true;
            break;
          }

          let distance = this.attackDistanceTo(target);
          this.lastActionResult.attackDistance = distance;
          this.lastActionResult.outOfRange = distance > ATTACK_RANGE;
          if (this.lastActionResult.outOfRange) break;

          this.bot.clearControlStates();
          const movementSettled =
            distance <= CLOSE_ATTACK_RANGE
              ? true
              : await this.waitForMovementToSettle();
          this.lastActionResult.movementSettled = movementSettled;
          target = await this.waitForTargetEntity();
          if (!target) {
            this.lastActionResult.targetMissing = true;
            break;
          }

          distance = this.attackDistanceTo(target);
          this.lastActionResult.attackDistance = distance;
          this.lastActionResult.outOfRange = distance > ATTACK_RANGE;
          this.lastActionResult.movementNotSettled =
            distance > CLOSE_ATTACK_RANGE && !movementSettled;
          this.lastActionResult.cooldownBlocked =
            !this.lastActionResult.outOfRange &&
            !this.lastActionResult.movementNotSettled &&
            this.elapsedSeconds < this.nextAttackTime;
          if (
            !this.lastActionResult.outOfRange &&
            !this.lastActionResult.movementNotSettled &&
            !this.lastActionResult.cooldownBlocked
          ) {
            const healthBeforeAttack = this.lastKnownTargetHealth;
            await this.bot.lookAt(
              target.position.offset(0, target.height * 0.65, 0),
              true
            );
            await sleep(ATTACK_AIM_SETTLE_MS);
            target = await this.waitForTargetEntity();
            if (!target) {
              this.lastActionResult.targetMissing = true;
              break;
            }

            distance = this.attackDistanceTo(target);
            this.lastActionResult.attackDistance = distance;
            this.lastActionResult.outOfRange = distance > ATTACK_RANGE;

            if (!this.lastActionResult.outOfRange) {
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

      // Tracking can disappear during any action, not only ATTACK. Give
      // Mineflayer the same short reacquisition window so the Python
      // environment can truncate immediately instead of collecting an
      // unusable episode until the policy happens to attack again.
      if (
        !this.targetConfirmedDead &&
        !this.lastActionResult.targetMissing &&
        !this.getTargetEntity() &&
        !(await this.waitForTargetEntity())
      ) {
        this.lastActionResult.targetMissing = true;
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
    const observedTargetPosition: [number, number] = target
      ? [
          target.position.x - this.origin.x,
          target.position.z - this.origin.z
        ]
      : [...this.targetPosition];
    const distance = target
      ? this.horizontalDistanceTo(target.position.x, target.position.z)
      : this.horizontalDistanceToWorldTarget();
    return {
      botPosition: [botX, botZ],
      targetPosition: observedTargetPosition,
      yaw: this.bot.entity.yaw,
      health: this.bot.health,
      botDefeated: this.botConfirmedDead,
      hasIronSword: this.bot.heldItem?.name === "iron_sword",
      targetHealth: this.lastKnownTargetHealth,
      targetAlive,
      targetVisible: target !== null,
      distance,
      attackReady: this.elapsedSeconds >= this.nextAttackTime,
      attackResult: { ...this.lastActionResult },
      elapsedSeconds: this.elapsedSeconds,
      roomSize: ROOM_SIZE
    };
  }

  private getTargetEntity(): Entity | null {
    if (this.targetConfirmedDead || !this.origin) return null;
    if (this.targetEntityId !== null) {
      const entity = this.bot.entities[this.targetEntityId];
      if (entity?.name === "zombie") return entity;
    }

    const expected = this.worldPosition(this.targetPosition);
    const reacquired = Object.values(this.bot.entities)
      .filter(
        (entity): entity is Entity =>
          entity.name === "zombie" && this.isInsideArena(entity)
      )
      .sort(
        (a, b) =>
          Math.hypot(a.position.x - expected.x, a.position.z - expected.z) -
          Math.hypot(b.position.x - expected.x, b.position.z - expected.z)
      )[0];
    if (reacquired) {
      this.targetEntityId = reacquired.id;
      return reacquired;
    }
    return null;
  }

  private async waitForTargetEntity(): Promise<Entity | null> {
    const deadline = Date.now() + TARGET_REACQUIRE_TIMEOUT_MS;
    do {
      const target = this.getTargetEntity();
      if (target) return target;
      await sleep(25);
    } while (Date.now() < deadline);
    return null;
  }

  private isInsideArena(entity: Entity): boolean {
    if (!this.origin) return false;
    const { x, y, z } = this.origin;
    return (
      entity.position.x >= x &&
      entity.position.x <= x + ROOM_SIZE &&
      entity.position.y >= y &&
      entity.position.y <= y + 4 &&
      entity.position.z >= z &&
      entity.position.z <= z + ROOM_SIZE
    );
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
    await this.command(
      `scoreboard objectives add ${this.options.healthObjective} dummy`
    );
    this.healthObjectiveReady = true;
  }

  /**
   * Read the zombie's actual NBT health through
   * a scoreboard so rewards are based on server state rather than client
   * metadata.
   */
  private async queryTargetHealthFromServer(): Promise<number | null> {
    if (this.targetConfirmedDead) return 0;
    await this.ensureHealthObjective();
    // A missing entity must not reuse the previous zombie's score.
    await this.command(
      `scoreboard players set #target ${this.options.healthObjective} -1`
    );
    await this.command(
      `execute store result score #target ${this.options.healthObjective} ` +
      "run data get entity " +
      `@e[type=minecraft:zombie,tag=${this.options.targetTag},limit=1] Health 100`
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
        if (!message.includes(this.options.healthObjective)) return;
        const match = message.match(/#target has (-?\d+)/);
        if (match) {
          const storedHealth = Number(match[1]);
          finish(storedHealth < 0 ? null : storedHealth / 100);
        }
      };
      const timeout = setTimeout(() => finish(null), 750);
      this.bot.on("messagestr", onMessage);
      this.bot.chat(
        `/scoreboard players get #target ${this.options.healthObjective}`
      );
    });
  }

  private async waitForBotReady(): Promise<void> {
    const deadline = Date.now() + 5_000;
    while (this.bot.health <= 0 && Date.now() < deadline) await sleep(50);
    if (this.bot.health <= 0) {
      throw new Error(
        `${this.options.stageName} reset timed out waiting for the bot to respawn.`
      );
    }
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
    movementNotSettled: false,
    targetMissing: false,
    attackPacketSent: false
  };
}
