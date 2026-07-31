import type { Bot } from "mineflayer";
import { StageTwoArena } from "./stageTwoArena.js";

/** Live combat arena with normal zombie AI and knockback enabled. */
export class StageThreeArena extends StageTwoArena {
  constructor(bot: Bot) {
    super(bot, {
      stageName: "Stage-three",
      targetTag: "rl_stage3_target",
      healthObjective: "rl_stage3_health",
      stationaryTarget: false
    });
  }
}
