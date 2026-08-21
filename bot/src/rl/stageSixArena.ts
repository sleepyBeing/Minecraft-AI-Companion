import type { Bot } from "mineflayer";
import type { Entity } from "prismarine-entity";
import { StageFiveArena } from "./stageFiveArena.js";
import { TargetHealthReader } from "./stageTwoSupport.js";
import {
  ROOM_SIZE,
  sleep,
  validatePosition,
  type BridgeRequest
} from "./shared.js";

const NPC_TAG = "rl_stage6_npc";
const NPC_HEALTH_OBJECTIVE = "rl_stage6_npc_hp";
const NPC_MAX_HEALTH = 20;
const NPC_ZOMBIE_MIN_DISTANCE = 2.5;
const NPC_ZOMBIE_MAX_DISTANCE = 5;
const NPC_THREAT_DISTANCE = 2.25;

/** Stage Five arena extended with a stationary villager protection target. */
export class StageSixArena extends StageFiveArena {
  protected npcPosition: [number, number] = [7.5, 7.5];
  protected npcEntityId: number | null = null;
  protected npcHealth = NPC_MAX_HEALTH;
  protected npcConfirmedDead = false;
  protected npcHealthDirty = false;
  protected readonly npcHealthReader: TargetHealthReader;

  constructor(bot: Bot) {
    super(bot);
    this.npcHealthReader = new TargetHealthReader(
      bot,
      NPC_HEALTH_OBJECTIVE,
      NPC_TAG,
      (command) => this.command(command),
      "minecraft:villager"
    );
    bot.on("entityHurt", (entity) => {
      if (entity?.id !== this.npcEntityId) return;
      this.npcHealthDirty = true;
      if (entity.health !== undefined && entity.health !== null)
        this.npcHealth = Math.min(this.npcHealth, Math.max(0, entity.health));
    });
    bot.on("entityDead", (entity) => {
      if (entity?.id !== this.npcEntityId) return;
      this.npcConfirmedDead = true;
      this.npcHealth = 0;
    });
  }

  override async reset(request: BridgeRequest) {
    const npcPosition = validatePosition(
      request.npcPosition ?? [7.5, 7.5],
      "npcPosition"
    );
    const targetPosition = validatePosition(
      request.targetPosition ?? [10.5, 7.5],
      "targetPosition"
    );
    const npcZombieDistance = distance(npcPosition, targetPosition);
    if (
      npcZombieDistance < NPC_ZOMBIE_MIN_DISTANCE ||
      npcZombieDistance > NPC_ZOMBIE_MAX_DISTANCE
    ) {
      throw new Error("Stage-six zombie must start 2.5 to 5 blocks from the NPC");
    }

    this.npcPosition = npcPosition;
    this.npcEntityId = null;
    this.npcHealth = NPC_MAX_HEALTH;
    this.npcConfirmedDead = false;
    this.npcHealthDirty = false;
    await super.reset({ ...request, freezeTarget: true });
    if (!this.origin) throw new Error("Stage-six arena origin is unavailable");

    const npc = this.worldPosition(npcPosition);
    await this.command(
      `summon minecraft:villager ${npc.x} ${this.origin.y + 1} ${npc.z} ` +
      `{Tags:["${NPC_TAG}"],NoAI:1b,PersistenceRequired:1b,Silent:1b,` +
      `Invulnerable:1b,Health:${NPC_MAX_HEALTH}.0f}`
    );
    await this.command(
      `attribute @e[type=minecraft:villager,tag=${NPC_TAG},limit=1] ` +
      "minecraft:knockback_resistance base set 1"
    );
    await sleep(200);
    const entity = this.findNpcEntity();
    if (!entity)
      throw new Error("Stage-six NPC did not spawn; ensure the bot is operator");
    this.npcEntityId = entity.id;
    await this.npcHealthReader.ensureObjective();
    const serverHealth = await this.npcHealthReader.query(false);
    if (serverHealth !== null) this.npcHealth = serverHealth;

    await this.command("effect give @s minecraft:resistance 5 4 true");
    try {
      await this.healBotToFullHealth();
      await this.command("effect give @s minecraft:invisibility 999999 0 true");
    } finally {
      await this.command("effect clear @s minecraft:resistance");
    }
    await this.setTargetFrozen(request.freezeTarget === true);
    await this.command(
      `data merge entity @e[type=minecraft:villager,tag=${NPC_TAG},limit=1] ` +
      "{Invulnerable:0b,Health:20.0f}"
    );
    this.npcHealth = NPC_MAX_HEALTH;
    this.npcHealthDirty = false;
    return this.observe();
  }

  override async step(action: number) {
    await super.step(action);
    if (this.npcHealthDirty) {
      const serverHealth = await this.npcHealthReader.query(
        this.npcConfirmedDead
      );
      if (serverHealth !== null) this.npcHealth = serverHealth;
      this.npcHealthDirty = false;
    }
    return this.observe();
  }

  override observe() {
    const state = super.observe();
    const npc = this.findNpcEntity();
    if (npc?.health !== undefined && npc.health !== null)
      this.npcHealth = Math.min(this.npcHealth, Math.max(0, npc.health));
    const npcAlive = !this.npcConfirmedDead && this.npcHealth > 0;
    const observedNpcPosition: [number, number] = npc && this.origin
      ? [
          npc.position.x - this.origin.x,
          npc.position.z - this.origin.z
        ]
      : [...this.npcPosition];
    const targetPosition = state.targetPosition as [number, number];
    const botPosition = state.botPosition as [number, number];
    const enemyNpcDistance = distance(targetPosition, observedNpcPosition);
    const interactionCenter: [number, number] = [
      (targetPosition[0] + observedNpcPosition[0]) / 2,
      (targetPosition[1] + observedNpcPosition[1]) / 2
    ];
    return {
      ...state,
      npcPosition: observedNpcPosition,
      npcHealth: this.npcHealth,
      npcAlive,
      npcUnderAttack:
        npcAlive && state.targetAlive && enemyNpcDistance <= NPC_THREAT_DISTANCE,
      botNpcDistance: distance(botPosition, observedNpcPosition),
      enemyNpcDistance,
      interactionDistance: distance(botPosition, interactionCenter)
    };
  }

  override async restoreWorldSettings(): Promise<void> {
    await super.restoreWorldSettings();
    await this.command("effect clear @s minecraft:invisibility");
  }

  private async healBotToFullHealth(): Promise<void> {
    const deadline = Date.now() + 2_000;
    await this.command("attribute @s minecraft:max_health base set 20");
    while (Math.abs(this.bot.health - 20) > 0.25 && Date.now() < deadline) {
      await this.command("effect give @s minecraft:instant_health 1 10 true");
      await sleep(75);
    }
    if (Math.abs(this.bot.health - 20) > 0.25)
      throw new Error(
        `Stage-six bot health setup failed: expected 20, observed ${this.bot.health}`
      );
  }

  protected findNpcEntity(): Entity | null {
    if (this.npcEntityId !== null) {
      const known = this.bot.entities[this.npcEntityId];
      if (known) return known;
    }
    if (!this.origin) return null;
    const expected = this.worldPosition(this.npcPosition);
    return Object.values(this.bot.entities).find(
      (entity) =>
        entity.name === "villager" &&
        Math.hypot(
          entity.position.x - expected.x,
          entity.position.z - expected.z
        ) <= 2
    ) ?? null;
  }
}

function distance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
