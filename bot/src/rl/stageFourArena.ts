import type { Bot } from "mineflayer";
import { StageTwoArena } from "./stageTwoArena.js";
import {
  chooseCoverPlan,
  hasSolidBlockBetween,
  isGeometricallyBlocked,
  retreatPoint,
  type Point
} from "./stageFourCover.js";
import type { BridgeRequest } from "./shared.js";

/** Moving-zombie arena with low-health retreat and solid cover mechanics. */
export class StageFourArena extends StageTwoArena {
  private survivalMode = false;

  constructor(bot: Bot) {
    super(bot, {
      stageName: "Stage-four",
      targetTag: "rl_stage4_target",
      healthObjective: "rl_stage4_health",
      stationaryTarget: false,
      naturalRegeneration: false
    });
  }

  override async reset(request: BridgeRequest) {
    this.survivalMode = request.botHealth === 5 || request.botHealth === 10;
    await super.reset(request);
    if (this.survivalMode) {
      await this.command(
        `data merge entity @e[type=minecraft:zombie,tag=${this.options.targetTag},limit=1] ` +
        "{Invulnerable:1b}"
      );
    }
    await this.buildCover();
    return this.observe();
  }

  override observe() {
    const state = super.observe();
    const botPosition = state.botPosition as Point;
    const targetPosition = state.targetPosition as Point;
    const plan = chooseCoverPlan(botPosition, targetPosition);
    const target = this.getTargetEntity();
    const geometricallyBlocked = isGeometricallyBlocked(
      botPosition,
      targetPosition
    );
    const blockLineOfSight = target
      ? hasSolidBlockBetween(this.bot, target)
      : false;
    const coverOccluded = geometricallyBlocked && blockLineOfSight;
    return {
      ...state,
      safePosition: plan.navigationPosition,
      protectedPosition: plan.protectedPosition,
      activeCover: plan.coverIndex,
      distanceToSafe: distance(botPosition, plan.navigationPosition),
      coverOccluded,
      inCover: coverOccluded,
      survivalMode: this.survivalMode
    };
  }

  async restoreWorldSettings(): Promise<void> {
    await this.command("gamerule naturalRegeneration true");
    await this.command("attribute @s minecraft:max_health base set 20");
    await this.command("effect give @s minecraft:instant_health 1 255 true");
  }

  protected override async handleExtendedAction(action: number): Promise<boolean> {
    if (action !== 8 && action !== 9) return false;

    const target = this.getTargetEntity();
    if (!target || !this.origin) return true;

    const botPosition: Point = [
      this.bot.entity.position.x - this.origin.x,
      this.bot.entity.position.z - this.origin.z
    ];
    const targetPosition: Point = [
      target.position.x - this.origin.x,
      target.position.z - this.origin.z
    ];
    const destination: Point =
      action === 8
        ? retreatPoint(botPosition, targetPosition)
        : chooseCoverPlan(botPosition, targetPosition).navigationPosition;
    const dx = destination[0] - botPosition[0];
    const dz = destination[1] - botPosition[1];
    if (Math.hypot(dx, dz) <= 0.1) return true;

    const yaw = Math.atan2(-dx, -dz);
    await this.bot.look(yaw, 0, true);
    this.bot.setControlState("forward", true);
    return true;
  }

  private async buildCover(): Promise<void> {
    if (!this.origin) throw new Error("Stage-four arena origin is unavailable");
    const { x, y, z } = this.origin;
    await this.command(
      `fill ${x + 7} ${y + 1} ${z + 1} ${x + 7} ${y + 2} ${z + 5} smooth_stone`
    );
    await this.command(
      `fill ${x + 7} ${y + 1} ${z + 9} ${x + 7} ${y + 2} ${z + 13} smooth_stone`
    );
  }
}

function distance(first: Point, second: Point): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}
