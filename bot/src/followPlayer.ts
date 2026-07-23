import type { Bot } from "mineflayer";
import { goals, Movements } from "mineflayer-pathfinder";
import type { Entity } from "prismarine-entity";
import { Vec3 } from "vec3";

const FOLLOW_DISTANCE = 4;
const FAR_DISTANCE = 20;
const RECOVERY_AFTER_MS = 5_000;
const TELEPORT_AFTER_MS = 12_000;
const TELEPORT_COOLDOWN_MS = 30_000;
const UPDATE_INTERVAL_MS = 500;
const NORMAL_SEARCH_RADIUS = 64;
const RECOVERY_SEARCH_RADIUS = 192;

const DANGEROUS_BLOCKS = [
  "lava", "fire", "soul_fire", "cactus", "sweet_berry_bush", "powder_snow",
  "magma_block", "campfire", "soul_campfire", "wither_rose"
];

export interface FollowOptions {
  isBusy?: () => boolean;
}

type PathfinderWithSearchRadius = Bot["pathfinder"] & { searchRadius: number };

export function startFollowingNearestPlayer(bot: Bot, options: FollowOptions = {}): () => void {
  const normalMovements = createMovements(bot, false);
  const recoveryMovements = createMovements(bot, true);
  const pathfinder = bot.pathfinder as PathfinderWithSearchRadius;

  pathfinder.thinkTimeout = 10_000;
  pathfinder.searchRadius = NORMAL_SEARCH_RADIUS;
  pathfinder.setMovements(normalMovements);

  let followedUsername: string | null = null;
  let followedEntityId: number | null = null;
  let recoveryActive = false;
  let lastProgressAt = Date.now();
  let lastUsablePathAt = Date.now();
  let lastKnownDistance = 0;
  let progressAnchor = bot.entity.position.clone();
  let bestDistance = Number.POSITIVE_INFINITY;
  let teleportCooldownUntil = 0;
  let pausedForBusy = false;

  const onPathUpdate = (result: { status: string; path: unknown[] }) => {
    if ((result.status === "success" || result.status === "partial") && result.path.length > 0)
      lastUsablePathAt = Date.now();
  };

  const onPathReset = (reason: string) => {
    if (reason === "stuck" || reason === "no_scaffolding_blocks")
      lastUsablePathAt = Math.min(lastUsablePathAt, Date.now() - TELEPORT_AFTER_MS);
  };

  bot.on("path_update", onPathUpdate);
  bot.on("path_reset", onPathReset);

  const interval = setInterval(() => {
    const now = Date.now();
    const busy = options.isBusy?.() ?? false;

    if (busy) {
      pausedForBusy = true;
      return;
    }

    if (!followedUsername) {
      const nearest = nearestPlayer(bot);
      if (!nearest?.username) return;
      followedUsername = nearest.username;
      followedEntityId = nearest.id;
      resetProgress(nearest);
      setFollowGoal(nearest);
      return;
    }

    const player = bot.players[followedUsername]?.entity;

    if (!player) {
      pathfinder.setGoal(null);
      if (
        !busy && lastKnownDistance > FAR_DISTANCE && now - lastProgressAt >= TELEPORT_AFTER_MS &&
        now - lastUsablePathAt >= TELEPORT_AFTER_MS && now >= teleportCooldownUntil
      )
        teleportToPlayer(followedUsername);
      return;
    }

    if (pausedForBusy) {
      pausedForBusy = false;
      recoveryActive = false;
      pathfinder.searchRadius = NORMAL_SEARCH_RADIUS;
      pathfinder.setMovements(normalMovements);
      resetProgress(player);
      setFollowGoal(player);
    }

    const distance = bot.entity.position.distanceTo(player.position);
    lastKnownDistance = distance;

    const moved = bot.entity.position.distanceTo(progressAnchor) >= 1.5;
    const gotCloser = distance <= bestDistance - 1;
    if (moved || gotCloser || distance <= FOLLOW_DISTANCE + 1) resetProgress(player);

    if (followedEntityId !== player.id) {
      followedEntityId = player.id;
      setFollowGoal(player);
    }

    if (distance <= FOLLOW_DISTANCE + 1) {
      if (recoveryActive) restoreNormalFollowing(player);
      return;
    }

    const noProgressFor = now - lastProgressAt;
    if (!recoveryActive && noProgressFor >= RECOVERY_AFTER_MS) {
      recoveryActive = true;
      pathfinder.searchRadius = RECOVERY_SEARCH_RADIUS;
      pathfinder.setMovements(recoveryMovements);
      setFollowGoal(player);
      console.log(`Follow recovery: searching a wider route to ${followedUsername}.`);
    }

    const pathfinderHasNoRoute = now - lastUsablePathAt >= TELEPORT_AFTER_MS;
    if (
      recoveryActive && distance > FAR_DISTANCE && noProgressFor >= TELEPORT_AFTER_MS &&
      pathfinderHasNoRoute && now >= teleportCooldownUntil
    ) {
      teleportToPlayer(followedUsername, player);
    }
  }, UPDATE_INTERVAL_MS);

  function resetProgress(player: Entity): void {
    lastProgressAt = Date.now();
    lastUsablePathAt = Date.now();
    progressAnchor = bot.entity.position.clone();
    bestDistance = bot.entity.position.distanceTo(player.position);
  }

  function restoreNormalFollowing(player: Entity): void {
    recoveryActive = false;
    pathfinder.searchRadius = NORMAL_SEARCH_RADIUS;
    pathfinder.thinkTimeout = 10_000;
    pathfinder.tickTimeout = 40;
    pathfinder.setMovements(normalMovements);
    setFollowGoal(player);
  }

  function setFollowGoal(player: Entity): void {
    pathfinder.setGoal(new goals.GoalFollow(player, FOLLOW_DISTANCE), true);
  }

  function teleportToPlayer(username: string, player?: Entity): void {
    if (!/^[A-Za-z0-9_]{1,16}$/.test(username)) return;

    const safePosition = player ? findSafePositionNear(bot, player.position) : null;
    const command = safePosition
      ? `/tp @s ${safePosition.x + 0.5} ${safePosition.y} ${safePosition.z + 0.5}`
      : `/tp @s ${username}`;

    console.log(`Follow recovery failed; teleporting to ${username}.`);
    bot.chat(command);
    teleportCooldownUntil = Date.now() + TELEPORT_COOLDOWN_MS;
    lastProgressAt = Date.now();
    lastUsablePathAt = Date.now();
  }

  return () => {
    clearInterval(interval);
    bot.removeListener("path_update", onPathUpdate);
    bot.removeListener("path_reset", onPathReset);
    pathfinder.setGoal(null);
  };
}

function createMovements(bot: Bot, recovery: boolean): Movements {
  const movements = new Movements(bot);
  movements.canDig = recovery; // Allows staircase mining during recovery
  movements.allow1by1towers = recovery; // Allows placing blocks to get down tall places
  movements.allowParkour = false;
  movements.maxDropDown = recovery ? 3 : 2;
  movements.infiniteLiquidDropdownDistance = recovery; // Allows dropping into water
  movements.dontCreateFlow = true;
  movements.dontMineUnderFallingBlock = true;

  for (const blockName of DANGEROUS_BLOCKS) {
    const block = bot.registry.blocksByName[blockName];
    if (block) movements.blocksToAvoid.add(block.id);
  }

  for (const entityName of ["creeper", "warden", "wither", "end_crystal"])
    movements.entitiesToAvoid.add(entityName);

  return movements;
}

function nearestPlayer(bot: Bot): Entity | null {
  return bot.nearestEntity((entity) =>
    entity.type === "player" && entity.username !== bot.username
  );
}

function findSafePositionNear(bot: Bot, center: Vec3): Vec3 | null {
  const origin = center.floored();

  for (let radius = 1; radius <= 4; radius++) {
    for (let x = -radius; x <= radius; x++) {
      for (let z = -radius; z <= radius; z++) {
        if (Math.abs(x) !== radius && Math.abs(z) !== radius) continue;

        for (let y = 2; y >= -2; y--) {
          const feet = origin.offset(x, y, z);
          const ground = bot.blockAt(feet.offset(0, -1, 0));
          const feetBlock = bot.blockAt(feet);
          const headBlock = bot.blockAt(feet.offset(0, 1, 0));

          if (!ground || !feetBlock || !headBlock) continue;
          if (ground.boundingBox === "empty") continue;
          if (feetBlock.boundingBox !== "empty" || headBlock.boundingBox !== "empty") continue;
          if ([ground.name, feetBlock.name, headBlock.name].some((name) =>
            DANGEROUS_BLOCKS.some((hazard) => name.includes(hazard)))) continue;

          return feet;
        }
      }
    }
  }

  return null;
}
