import type { Bot } from "mineflayer";
import type { Entity } from "prismarine-entity";
import { StageSixArena } from "./stageSixArena.js";
import { TargetHealthReader } from "./stageTwoSupport.js";
import {
  ACTION_DURATION_MS,
  ROOM_SIZE,
  sleep,
  validatePosition,
  type BridgeRequest
} from "./shared.js";
import type { Point } from "./stageFourCover.js";
import {
  THREAT_SPECS,
  hasThreatLineOfSight,
  nextThreatPosition,
  validateThreatTypes,
  type ThreatType
} from "./stageSevenThreats.js";

const ENEMY_TAG = "rl_stage7_enemy";
const ENEMY_TAGS = ["rl_stage7_enemy_0", "rl_stage7_enemy_1"] as const;

interface ThreatState {
  type: ThreatType;
  position: Point;
  entityId: number | null;
  health: number;
  alive: boolean;
  confirmedDead: boolean;
  nextAttackTime: number;
}

/** Two controlled Minecraft threats with explicit policy target selection. */
export class StageSevenArena extends StageSixArena {
  private configuringBase = true;
  private selectedEnemy = 0;
  private threats: ThreatState[] = [];
  private healthReaders: TargetHealthReader[] = [];

  constructor(bot: Bot) {
    super(bot);
    bot.on("entityHurt", (entity) => this.recordThreatHealth(entity));
    bot.on("entityDead", (entity) => {
      const index = this.threats.findIndex((threat) => threat.entityId === entity?.id);
      if (index < 0) return;
      this.threats[index].health = 0;
      this.threats[index].alive = false;
      this.threats[index].confirmedDead = true;
    });
  }

  override async reset(request: BridgeRequest) {
    const types = validateThreatTypes(request.enemyTypes);
    const positions = validateEnemyPositions(request.enemyPositions);
    this.configuringBase = true;
    await super.reset({
      ...request,
      targetPosition: positions[0],
      freezeTarget: true
    });
    if (!this.origin) throw new Error("Stage-seven arena origin is unavailable");

    await this.command(
      `kill @e[type=minecraft:zombie,tag=${this.options.targetTag}]`
    );
    this.threats = types.map((type, index) => ({
      type,
      position: positions[index],
      entityId: null,
      health: THREAT_SPECS[type].maxHealth,
      alive: true,
      confirmedDead: false,
      nextAttackTime: 0
    }));
    this.healthReaders = types.map(
      (type, index) => new TargetHealthReader(
        this.bot,
        `rl_s7_e${index}_hp`,
        ENEMY_TAGS[index],
        (command) => this.command(command),
        `minecraft:${type}`
      )
    );

    for (let index = 0; index < this.threats.length; index += 1) {
      await this.spawnThreat(index);
      const entity = this.findThreatEntity(index);
      if (!entity)
        throw new Error(`Stage-seven enemy ${index + 1} did not spawn`);
      this.threats[index].entityId = entity.id;
      await this.healthReaders[index].ensureObjective();
    }
    this.selectedEnemy = normalizeSelectedEnemy(request.selectedEnemy);
    this.configuringBase = false;
    this.syncSelectedToBase();
    return this.observe();
  }

  override async step(action: number) {
    if (!this.threats[this.selectedEnemy]?.alive) {
      const alive = this.threats.findIndex((threat) => threat.alive);
      if (alive >= 0) {
        this.selectedEnemy = alive;
        this.syncSelectedToBase();
      }
    }
    const elapsedBefore = this.elapsedSeconds;
    await super.step(action);
    this.syncSelectedFromBase();
    const duration = Math.max(0.1, this.elapsedSeconds - elapsedBefore);
    await this.advanceThreats(duration);
    if (this.npcHealthDirty) {
      const health = await this.npcHealthReader.query(this.npcConfirmedDead);
      if (health !== null) this.npcHealth = health;
      this.npcHealthDirty = false;
    }
    this.syncSelectedToBase();
    return this.observe();
  }

  override observe() {
    if (this.configuringBase || this.threats.length !== 2)
      return super.observe();
    const state = super.observe();
    this.syncSelectedFromBase();
    const botPosition = state.botPosition as Point;
    const npcPosition = state.npcPosition as Point;
    return {
      ...state,
      selectedEnemy: this.selectedEnemy,
      enemyStates: this.threats.map((threat, index) => {
        const entity = this.findThreatEntity(index);
        if (entity) {
          threat.position = this.localPosition(entity);
          if (entity.health !== undefined && entity.health !== null)
            threat.health = Math.min(threat.health, Math.max(0, entity.health));
        }
        const spec = THREAT_SPECS[threat.type];
        const focus = this.attackFocus(index, botPosition, npcPosition);
        const botDistance = distance(threat.position, botPosition);
        const npcDistance = distance(threat.position, npcPosition);
        return {
          type: threat.type,
          position: [...threat.position],
          health: threat.health,
          maxHealth: spec.maxHealth,
          alive: threat.alive,
          visible: entity !== null,
          ranged: spec.ranged,
          botDistance,
          npcDistance,
          attackingNpc:
            focus === "npc" && npcDistance <= spec.attackRange + 0.05 &&
            hasThreatLineOfSight(threat.position, npcPosition),
          attackingBot:
            focus === "bot" && botDistance <= spec.attackRange + 0.05 &&
            hasThreatLineOfSight(threat.position, botPosition)
        };
      })
    };
  }

  protected override async handleExtendedAction(
    action: number
  ): Promise<number | false> {
    if (action === 12 || action === 13) {
      const requested = action - 12;
      if (this.threats[requested]?.alive) {
        this.selectedEnemy = requested;
        this.syncSelectedToBase();
      }
      return ACTION_DURATION_MS;
    }
    return super.handleExtendedAction(action);
  }

  protected override getTargetEntity(): Entity | null {
    if (this.configuringBase || this.threats.length !== 2)
      return super.getTargetEntity();
    return this.findThreatEntity(this.selectedEnemy);
  }

  protected override async queryTargetHealth(
    confirmedDead: boolean
  ): Promise<number | null> {
    if (this.configuringBase || this.healthReaders.length !== 2)
      return super.queryTargetHealth(confirmedDead);
    const health = await this.healthReaders[this.selectedEnemy].query(
      confirmedDead
    );
    if (health !== null) {
      const threat = this.threats[this.selectedEnemy];
      threat.health = health;
      threat.alive = health > 0;
      threat.confirmedDead = health <= 0;
    }
    return health;
  }

  private async spawnThreat(index: number): Promise<void> {
    if (!this.origin) return;
    const threat = this.threats[index];
    const world = this.worldPosition(threat.position);
    const spec = THREAT_SPECS[threat.type];
    await this.command(
      `summon minecraft:${threat.type} ${world.x} ${this.origin.y + 1} ${world.z} ` +
      `{Tags:["${ENEMY_TAG}","${ENEMY_TAGS[index]}"],NoAI:1b,` +
      `PersistenceRequired:1b,Silent:1b,CanPickUpLoot:0b,` +
      `Health:${spec.maxHealth}.0f}`
    );
  }

  private async advanceThreats(duration: number): Promise<void> {
    const botPosition = this.botLocalPosition();
    let issuedCommand = false;
    for (let index = 0; index < this.threats.length; index += 1) {
      const threat = this.threats[index];
      if (!threat.alive) continue;
      const entity = this.findThreatEntity(index);
      if (entity) threat.position = this.localPosition(entity);
      const focus = this.attackFocus(index, botPosition, this.npcPosition);
      const victim = focus === "bot" ? botPosition : this.npcPosition;
      const spec = THREAT_SPECS[threat.type];
      const victimDistance = distance(threat.position, victim);
      const lineOfSight = hasThreatLineOfSight(threat.position, victim);
      if ((victimDistance > spec.attackRange || !lineOfSight) && victimDistance > 0) {
        const movement = Math.min(
          spec.moveSpeed * duration,
          Math.max(0, victimDistance - Math.min(spec.attackRange, 1.8))
        );
        threat.position = nextThreatPosition(threat.position, victim, movement);
        const world = this.worldPosition(threat.position);
        this.bot.chat(
          `/tp @e[tag=${ENEMY_TAGS[index]},limit=1] ` +
          `${world.x} ${this.origin!.y + 1} ${world.z}`
        );
        issuedCommand = true;
      }

      const newDistance = distance(threat.position, victim);
      if (
        newDistance <= spec.attackRange + 0.05 &&
        hasThreatLineOfSight(threat.position, victim) &&
        this.elapsedSeconds >= threat.nextAttackTime
      ) {
        threat.nextAttackTime = this.elapsedSeconds + spec.cooldownSeconds;
        const target = focus === "bot" ? "@s" : `@e[tag=rl_stage6_npc,limit=1]`;
        this.bot.chat(
          `/damage ${target} ${spec.attackDamage} minecraft:mob_attack ` +
          `by @e[tag=${ENEMY_TAGS[index]},limit=1]`
        );
        if (focus === "npc") {
          this.npcHealth = Math.max(0, this.npcHealth - spec.attackDamage);
          this.npcConfirmedDead = this.npcHealth <= 0;
        }
        issuedCommand = true;
      }
    }
    if (issuedCommand) await sleep(100);
  }

  private attackFocus(
    index: number,
    botPosition: Point,
    npcPosition: Point
  ): "bot" | "npc" {
    const threat = this.threats[index];
    const spec = THREAT_SPECS[threat.type];
    const intercepts =
      distance(threat.position, botPosition) <= spec.attackRange + 0.25 &&
      hasThreatLineOfSight(threat.position, botPosition);
    return intercepts ? "bot" : "npc";
  }

  private findThreatEntity(index: number): Entity | null {
    const threat = this.threats[index];
    if (!threat || threat.confirmedDead || !this.origin) return null;
    if (threat.entityId !== null) {
      const entity = this.bot.entities[threat.entityId];
      if (entity?.name === threat.type) return entity;
    }
    const expected = this.worldPosition(threat.position);
    const entity = Object.values(this.bot.entities)
      .filter((candidate) =>
        candidate.name === threat.type && this.insideArena(candidate)
      )
      .sort((a, b) =>
        Math.hypot(a.position.x - expected.x, a.position.z - expected.z) -
        Math.hypot(b.position.x - expected.x, b.position.z - expected.z)
      )[0] ?? null;
    if (entity) threat.entityId = entity.id;
    return entity;
  }

  private recordThreatHealth(entity: Entity | undefined): void {
    if (!entity) return;
    const threat = this.threats.find((candidate) => candidate.entityId === entity.id);
    if (!threat || entity.health === undefined || entity.health === null) return;
    threat.health = Math.min(threat.health, Math.max(0, entity.health));
  }

  private syncSelectedToBase(): void {
    const threat = this.threats[this.selectedEnemy];
    if (!threat) return;
    this.targetPosition = [...threat.position];
    this.targetEntityId = threat.entityId;
    this.lastKnownTargetHealth = threat.health;
    this.targetConfirmedDead = threat.confirmedDead || !threat.alive;
  }

  private syncSelectedFromBase(): void {
    const threat = this.threats[this.selectedEnemy];
    if (!threat) return;
    threat.position = [...this.targetPosition];
    threat.entityId = this.targetEntityId;
    threat.health = this.lastKnownTargetHealth;
    threat.alive = !this.targetConfirmedDead && threat.health > 0;
    threat.confirmedDead = this.targetConfirmedDead || threat.health <= 0;
  }

  private botLocalPosition(): Point {
    if (!this.origin) return [0, 0];
    return [
      this.bot.entity.position.x - this.origin.x,
      this.bot.entity.position.z - this.origin.z
    ];
  }

  private localPosition(entity: Entity): Point {
    if (!this.origin) return [0, 0];
    return [entity.position.x - this.origin.x, entity.position.z - this.origin.z];
  }

  private insideArena(entity: Entity): boolean {
    if (!this.origin) return false;
    return (
      entity.position.x >= this.origin.x &&
      entity.position.x <= this.origin.x + ROOM_SIZE &&
      entity.position.z >= this.origin.z &&
      entity.position.z <= this.origin.z + ROOM_SIZE
    );
  }
}

function validateEnemyPositions(value: unknown): [Point, Point] {
  if (!Array.isArray(value) || value.length !== 2)
    throw new Error("enemyPositions must contain two positions");
  return [
    validatePosition(value[0] as Point, "enemyPositions[0]"),
    validatePosition(value[1] as Point, "enemyPositions[1]")
  ];
}

function normalizeSelectedEnemy(value: unknown): number {
  return value === 1 ? 1 : 0;
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
