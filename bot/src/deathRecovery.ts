import type { Bot } from "mineflayer";
import { goals, Movements } from "mineflayer-pathfinder";
import { Vec3 } from "vec3";

interface DeathRecoveryOptions {
  onStart?: () => void;
  onEnd?: () => void;
  returnToPlayer?: () => void;
}

type BoundedPathfinder = Bot["pathfinder"] & { searchRadius: number };

const WAYPOINT_DISTANCE = 24;
const MAX_RECOVERY_TIME_MS = 4 * 60 * 1_000;
const DROP_SEARCH_RADIUS = 20;

/**
 * Remembers where the bot died and, after respawn, safely walks back to collect
 * item entities around that location. Dropped items normally disappear after
 * five minutes, so recovery has a four-minute deadline.
 */
export class DeathRecoveryController {
  private deathPosition: Vec3 | null = null;
  private deathDimension: string | null = null;
  private pending = false;
  private active = false;

  constructor(
    private readonly bot: Bot,
    private readonly options: DeathRecoveryOptions = {}
  ) {}

  get isBusy(): boolean {
    return this.pending || this.active;
  }

  recordDeath(): void {
    this.deathPosition = this.bot.entity.position.clone();
    this.deathDimension = this.bot.game.dimension;
    this.pending = true;
    this.bot.pathfinder.setGoal(null);
  }

  startAfterRespawn(): void {
    if (!this.pending || !this.deathPosition || this.active) return;

    const target = this.deathPosition.clone();
    const dimension = this.deathDimension;
    this.pending = false;
    this.active = true;
    this.options.onStart?.();
    void this.recover(target, dimension);
  }

  stop(): void {
    this.pending = false;
    this.active = false;
    this.deathPosition = null;
    this.bot.pathfinder.setGoal(null);
  }

  private async recover(target: Vec3, dimension: string | null): Promise<void> {
    let reachedDeathLocation = false;

    try {
      // Wait for the server to finish placing the bot at its respawn point.
      await sleep(750);

      if (dimension && this.bot.game.dimension !== dimension) {
        this.bot.chat("My items are in another dimension, so I can't recover them automatically.");
        return;
      }

      const pathfinder = this.bot.pathfinder as BoundedPathfinder;
      pathfinder.searchRadius = 64;
      pathfinder.thinkTimeout = 2_000;
      pathfinder.tickTimeout = 20;
      pathfinder.setMovements(createRecoveryMovements(this.bot));

      this.bot.chat("I respawned. Going back to recover my dropped items.");
      const deadline = Date.now() + MAX_RECOVERY_TIME_MS;
      reachedDeathLocation = await this.navigateInSegments(target, deadline);

      if (reachedDeathLocation) {
        await this.collectDropsNear(target, deadline);
        this.bot.chat("I finished recovering the items I could find.");
      }
    } catch (error) {
      console.error("Death-item recovery failed:", error);
    } finally {
      if (!reachedDeathLocation && this.bot.game.dimension === dimension) {
        this.bot.chat("I couldn't safely reach my dropped items.");
      }

      this.deathPosition = null;
      this.deathDimension = null;
      this.active = false;
      this.bot.pathfinder.setGoal(null);
      this.options.returnToPlayer?.();
      this.options.onEnd?.();
    }
  }

  private async navigateInSegments(target: Vec3, deadline: number): Promise<boolean> {
    if (this.bot.entity.position.distanceTo(target) <= 6) return true;

    while (Date.now() < deadline) {
      const current = this.bot.entity.position;
      const dx = target.x - current.x;
      const dz = target.z - current.z;
      const horizontalDistance = Math.hypot(dx, dz);

      if (horizontalDistance <= WAYPOINT_DISTANCE) break;

      const scale = WAYPOINT_DISTANCE / horizontalDistance;
      const waypointX = Math.floor(current.x + dx * scale);
      const waypointZ = Math.floor(current.z + dz * scale);

      try {
        await this.bot.pathfinder.goto(new goals.GoalNearXZ(waypointX, waypointZ, 4));
      } catch {
        return false;
      }
    }

    if (Date.now() >= deadline) return false;

    try {
      await this.bot.pathfinder.goto(
        new goals.GoalNear(
          Math.floor(target.x),
          Math.floor(target.y),
          Math.floor(target.z),
          4
        )
      );
      return true;
    } catch {
      return false;
    }
  }

  private async collectDropsNear(deathPosition: Vec3, deadline: number): Promise<void> {
    let lastItemSeenAt = Date.now();
    const inaccessibleItems = new Set<number>();

    while (Date.now() < deadline) {
      const item = Object.values(this.bot.entities)
        .filter((entity) =>
          entity.name === "item" &&
          !inaccessibleItems.has(entity.id) &&
          entity.position.distanceTo(deathPosition) <= DROP_SEARCH_RADIUS
        )
        .sort((a, b) =>
          a.position.distanceTo(this.bot.entity.position) -
          b.position.distanceTo(this.bot.entity.position)
        )[0];

      if (!item) {
        if (Date.now() - lastItemSeenAt >= 1_500) return;
        await sleep(200);
        continue;
      }

      lastItemSeenAt = Date.now();
      try {
        await this.bot.pathfinder.goto(
          new goals.GoalNear(
            Math.floor(item.position.x),
            Math.floor(item.position.y),
            Math.floor(item.position.z),
            1
          )
        );
        await sleep(250);
      } catch {
        // One inaccessible stack should not prevent attempting nearby stacks.
        inaccessibleItems.add(item.id);
      }
    }
  }
}

function createRecoveryMovements(bot: Bot): Movements {
  const movements = new Movements(bot);
  movements.canDig = false;
  movements.allow1by1towers = false;
  movements.allowParkour = false;
  movements.scafoldingBlocks = [];
  movements.maxDropDown = 2;
  movements.infiniteLiquidDropdownDistance = false;
  (movements as Movements & { liquidCost: number }).liquidCost = 20;

  for (const name of [
    "water",
    "lava",
    "fire",
    "soul_fire",
    "cactus",
    "powder_snow",
    "magma_block"
  ]) {
    const block = bot.registry.blocksByName[name];
    if (block) movements.blocksToAvoid.add(block.id);
  }

  return movements;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
