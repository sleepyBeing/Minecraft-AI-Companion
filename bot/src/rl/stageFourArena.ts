import type { Bot } from "mineflayer";
import { StageTwoArena } from "./stageTwoArena.js";
import type { BridgeRequest } from "./shared.js";

type Point = [number, number];

interface CoverRectangle {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

const COVER_RECTS: CoverRectangle[] = [
  { minX: 7, maxX: 8, minZ: 2, maxZ: 5 },
  { minX: 7, maxX: 8, minZ: 10, maxZ: 13 }
];
const SAFE_OFFSET = 1.75;

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
    const safePosition = nearestSafePosition(botPosition, targetPosition);
    return {
      ...state,
      safePosition,
      distanceToSafe: distance(botPosition, safePosition),
      inCover: isLineBlocked(botPosition, targetPosition),
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
    const destination =
      action === 8
        ? retreatPoint(botPosition, targetPosition)
        : nearestSafePosition(botPosition, targetPosition);
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
      `fill ${x + 7} ${y + 1} ${z + 2} ${x + 7} ${y + 2} ${z + 4} smooth_stone`
    );
    await this.command(
      `fill ${x + 7} ${y + 1} ${z + 10} ${x + 7} ${y + 2} ${z + 12} smooth_stone`
    );
  }
}

function nearestSafePosition(bot: Point, target: Point): Point {
  return COVER_RECTS
    .map((cover) => safePointBehind(cover, target))
    .sort((a, b) => distance(bot, a) - distance(bot, b))[0];
}

function safePointBehind(cover: CoverRectangle, target: Point): Point {
  const center: Point = [
    (cover.minX + cover.maxX) / 2,
    (cover.minZ + cover.maxZ) / 2
  ];
  let dx = center[0] - target[0];
  let dz = center[1] - target[1];
  const length = Math.hypot(dx, dz) || 1;
  dx /= length;
  dz /= length;
  return [
    clamp(center[0] + dx * SAFE_OFFSET, 0.5, 14.5),
    clamp(center[1] + dz * SAFE_OFFSET, 0.5, 14.5)
  ];
}

function retreatPoint(bot: Point, target: Point): Point {
  let dx = bot[0] - target[0];
  let dz = bot[1] - target[1];
  const length = Math.hypot(dx, dz) || 1;
  dx /= length;
  dz /= length;
  return [clamp(bot[0] + dx, 0.5, 14.5), clamp(bot[1] + dz, 0.5, 14.5)];
}

function isLineBlocked(start: Point, end: Point): boolean {
  return COVER_RECTS.some((cover) => segmentIntersectsRectangle(start, end, cover));
}

function segmentIntersectsRectangle(
  start: Point,
  end: Point,
  rectangle: CoverRectangle
): boolean {
  const dx = end[0] - start[0];
  const dz = end[1] - start[1];
  let minimum = 0;
  let maximum = 1;

  for (const [origin, delta, low, high] of [
    [start[0], dx, rectangle.minX, rectangle.maxX],
    [start[1], dz, rectangle.minZ, rectangle.maxZ]
  ] as const) {
    if (Math.abs(delta) < 1e-9) {
      if (origin < low || origin > high) return false;
      continue;
    }
    const first = (low - origin) / delta;
    const second = (high - origin) / delta;
    minimum = Math.max(minimum, Math.min(first, second));
    maximum = Math.min(maximum, Math.max(first, second));
    if (minimum > maximum) return false;
  }
  return true;
}

function distance(first: Point, second: Point): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
