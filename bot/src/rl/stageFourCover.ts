import type { Bot } from "mineflayer";
import type { Entity } from "prismarine-entity";

export type Point = [number, number];

export interface CoverRectangle {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

export interface CoverPlan {
  coverIndex: number;
  navigationPosition: Point;
  protectedPosition: Point;
}

export const COVER_RECTS: CoverRectangle[] = [
  { minX: 7, maxX: 8, minZ: 1, maxZ: 6 },
  { minX: 7, maxX: 8, minZ: 9, maxZ: 14 }
];

const ROOM_MIN = 0.5;
const ROOM_MAX = 14.5;
const SAFE_SIDE_CLEARANCE = 1.25;
const END_CLEARANCE = 0.75;
const END_MARGIN = 1.25;
const DYNAMIC_SHIFT_GAIN = 0.35;
const DYNAMIC_SHIFT_LIMIT = 1;
const LOS_SAMPLE_SPACING = 0.15;

export function chooseCoverPlan(bot: Point, target: Point): CoverPlan {
  return COVER_RECTS
    .map((cover, coverIndex) => planForRectangle(bot, target, cover, coverIndex))
    .sort((first, second) => planCost(bot, first) - planCost(bot, second))[0];
}

export function retreatPoint(bot: Point, target: Point): Point {
  let dx = bot[0] - target[0];
  let dz = bot[1] - target[1];
  const length = Math.hypot(dx, dz) || 1;
  dx /= length;
  dz /= length;
  return [clamp(bot[0] + dx, ROOM_MIN, ROOM_MAX), clamp(bot[1] + dz, ROOM_MIN, ROOM_MAX)];
}

export function isGeometricallyBlocked(start: Point, end: Point): boolean {
  return COVER_RECTS.some((cover) => segmentIntersectsRectangle(start, end, cover));
}

export function hasSolidBlockBetween(bot: Bot, target: Entity): boolean {
  const start = bot.entity.position.offset(0, 1.62, 0);
  const end = target.position.offset(0, target.height * 0.65, 0);
  const delta = end.minus(start);
  const distance = Math.max(delta.norm(), 1e-6);
  const samples = Math.max(2, Math.ceil(distance / LOS_SAMPLE_SPACING));
  let previousBlock = "";

  for (let index = 1; index < samples; index += 1) {
    const point = start.plus(delta.scaled(index / samples));
    const block = bot.blockAt(point);
    if (!block) continue;
    const key = `${block.position.x},${block.position.y},${block.position.z}`;
    if (key === previousBlock) continue;
    previousBlock = key;
    if (block.boundingBox !== "empty" && block.shapes.length > 0) return true;
  }
  return false;
}

function planForRectangle(
  bot: Point,
  target: Point,
  cover: CoverRectangle,
  coverIndex: number
): CoverPlan {
  const protectedPosition = protectedPoint(cover, target);
  return {
    coverIndex,
    protectedPosition,
    navigationPosition: navigationPoint(bot, target, protectedPosition, cover)
  };
}

function protectedPoint(cover: CoverRectangle, target: Point): Point {
  const centerX = (cover.minX + cover.maxX) / 2;
  const centerZ = (cover.minZ + cover.maxZ) / 2;
  const width = cover.maxX - cover.minX;
  const depth = cover.maxZ - cover.minZ;

  if (depth >= width) {
    const safeX =
      target[0] >= centerX
        ? cover.minX - SAFE_SIDE_CLEARANCE
        : cover.maxX + SAFE_SIDE_CLEARANCE;
    const availableShift = Math.max(0, depth / 2 - END_MARGIN);
    const limit = Math.min(DYNAMIC_SHIFT_LIMIT, availableShift);
    const shift = clamp((centerZ - target[1]) * DYNAMIC_SHIFT_GAIN, -limit, limit);
    return [clamp(safeX, ROOM_MIN, ROOM_MAX), clamp(centerZ + shift, ROOM_MIN, ROOM_MAX)];
  }

  const safeZ =
    target[1] >= centerZ
      ? cover.minZ - SAFE_SIDE_CLEARANCE
      : cover.maxZ + SAFE_SIDE_CLEARANCE;
  const availableShift = Math.max(0, width / 2 - END_MARGIN);
  const limit = Math.min(DYNAMIC_SHIFT_LIMIT, availableShift);
  const shift = clamp((centerX - target[0]) * DYNAMIC_SHIFT_GAIN, -limit, limit);
  return [clamp(centerX + shift, ROOM_MIN, ROOM_MAX), clamp(safeZ, ROOM_MIN, ROOM_MAX)];
}

function navigationPoint(
  bot: Point,
  target: Point,
  protectedPosition: Point,
  cover: CoverRectangle
): Point {
  const centerX = (cover.minX + cover.maxX) / 2;
  const centerZ = (cover.minZ + cover.maxZ) / 2;
  const verticalWall = cover.maxZ - cover.minZ >= cover.maxX - cover.minX;

  if (verticalWall) {
    const botSide = side(bot[0] - centerX);
    const protectedSide = side(protectedPosition[0] - centerX);
    if (botSide === protectedSide) return [...protectedPosition];

    const routeZ = clamp(
      target[1] < centerZ ? cover.maxZ + END_CLEARANCE : cover.minZ - END_CLEARANCE,
      ROOM_MIN,
      ROOM_MAX
    );
    const botLane =
      botSide < 0
        ? cover.minX - SAFE_SIDE_CLEARANCE
        : cover.maxX + SAFE_SIDE_CLEARANCE;
    return cover.minZ <= bot[1] && bot[1] <= cover.maxZ
      ? [clamp(botLane, ROOM_MIN, ROOM_MAX), routeZ]
      : [protectedPosition[0], routeZ];
  }

  const botSide = side(bot[1] - centerZ);
  const protectedSide = side(protectedPosition[1] - centerZ);
  if (botSide === protectedSide) return [...protectedPosition];

  const routeX = clamp(
    target[0] < centerX ? cover.maxX + END_CLEARANCE : cover.minX - END_CLEARANCE,
    ROOM_MIN,
    ROOM_MAX
  );
  const botLane =
    botSide < 0
      ? cover.minZ - SAFE_SIDE_CLEARANCE
      : cover.maxZ + SAFE_SIDE_CLEARANCE;
  return cover.minX <= bot[0] && bot[0] <= cover.maxX
    ? [routeX, clamp(botLane, ROOM_MIN, ROOM_MAX)]
    : [routeX, protectedPosition[1]];
}

function planCost(bot: Point, plan: CoverPlan): number {
  return (
    distance(bot, plan.navigationPosition) +
    0.25 * distance(plan.navigationPosition, plan.protectedPosition)
  );
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

function side(value: number): -1 | 1 {
  return value < 0 ? -1 : 1;
}

function distance(first: Point, second: Point): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
