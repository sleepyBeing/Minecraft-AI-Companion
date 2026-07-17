import type { Bot } from "mineflayer";
import { goals, Movements } from "mineflayer-pathfinder";

const FOLLOW_DISTANCE = 4;
const RETARGET_INTERVAL_MS = 1_000;

/**
 * Pathfinding to keep the bot within 3-5 blocks of the closest visible player
 * Handles one block step ups, and keeps calculating new routes as the player moves
 */
export function startFollowingNearestPlayer(bot: Bot): () => void {
  const movements = new Movements(bot);
  movements.canDig = false;
  movements.allow1by1towers = false;
  movements.allowParkour = false;
  movements.maxDropDown = 2;
  movements.infiniteLiquidDropdownDistance = false;

  for (const blockName of [
    "lava", "fire", "soul_fire", "cactus", "sweet_berry_bush", "powder_snow",
    "magma_block", "campfire", "soul_campfire", "wither_rose"
  ]) {
    const block = bot.registry.blocksByName[blockName];
    if (block) movements.blocksToAvoid.add(block.id);
  }

  for (const entityName of ["creeper", "warden", "wither", "end_crystal"])
    movements.entitiesToAvoid.add(entityName);

  bot.pathfinder.setMovements(movements);

  let followedPlayerId: number | null = null;
  const interval = setInterval(() => {
    const player = bot.nearestEntity((entity) =>
      entity.type === "player" && entity.username !== bot.username
    );

    if (!player) {
      if (followedPlayerId !== null) bot.pathfinder.setGoal(null);
      followedPlayerId = null;
      return;
    }

    // Bot routes around obstacles and returns to player when separated
    if (followedPlayerId !== player.id) {
      bot.pathfinder.setGoal(new goals.GoalFollow(player, FOLLOW_DISTANCE), true);
      followedPlayerId = player.id;
    }
  }, RETARGET_INTERVAL_MS);

  return () => {
    clearInterval(interval);
    bot.pathfinder.setGoal(null);
  };
}
