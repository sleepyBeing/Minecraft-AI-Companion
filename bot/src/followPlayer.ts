import type { Bot } from "mineflayer";

const START_FOLLOWING_DISTANCE = 5;
const STOP_FOLLOWING_DISTANCE = 4;
const FOLLOW_INTERVAL_MS = 250;

/**
 * Keeps the bot near the closest visible player
 * Distances create a 3–5 block comfort zone without movement jitter

 */

export function startFollowingNearestPlayer(bot: Bot): () => void {
  let movingToPlayer = false;
  let looking = false;

  const interval = setInterval(() => {
    const player = bot.nearestEntity((entity) =>
      entity.type === "player" && entity.username !== bot.username
    );

    if (!player) {
      if (movingToPlayer) bot.setControlState("forward", false);
      movingToPlayer = false;
      return;
    }

    const distance = bot.entity.position.distanceTo(player.position);

    if (distance > START_FOLLOWING_DISTANCE) {
      movingToPlayer = true;
    } else if (distance <= STOP_FOLLOWING_DISTANCE) {
      movingToPlayer = false;
    }

    bot.setControlState("forward", movingToPlayer);

    if (movingToPlayer && !looking) {
      looking = true;
      void bot.lookAt(player.position.offset(0, player.height * 0.8, 0), true)
        .catch((error: unknown) => console.error("Could not look at player while following:", error))
        .finally(() => { looking = false; });
    }
  }, FOLLOW_INTERVAL_MS);

  return () => {
    clearInterval(interval);
    bot.setControlState("forward", false);
  };
}
