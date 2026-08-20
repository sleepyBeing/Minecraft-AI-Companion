import { COVER_RECTS, isGeometricallyBlocked, type Point } from "./stageFourCover.js";

export const THREAT_TYPES = [
  "zombie",
  "skeleton",
  "enderman",
  "spider",
  "witch"
] as const;

export type ThreatType = typeof THREAT_TYPES[number];

export interface ThreatSpec {
  maxHealth: number;
  moveSpeed: number;
  attackRange: number;
  attackDamage: number;
  cooldownSeconds: number;
  ranged: boolean;
}

export const THREAT_SPECS: Record<ThreatType, ThreatSpec> = {
  zombie: { maxHealth: 20, moveSpeed: 2.3, attackRange: 1.8, attackDamage: 3, cooldownSeconds: 1, ranged: false },
  skeleton: { maxHealth: 20, moveSpeed: 2, attackRange: 7, attackDamage: 2, cooldownSeconds: 1.25, ranged: true },
  enderman: { maxHealth: 40, moveSpeed: 3, attackRange: 1.9, attackDamage: 4, cooldownSeconds: 1, ranged: false },
  spider: { maxHealth: 16, moveSpeed: 3.2, attackRange: 1.8, attackDamage: 2, cooldownSeconds: 0.8, ranged: false },
  witch: { maxHealth: 26, moveSpeed: 1.7, attackRange: 6, attackDamage: 3, cooldownSeconds: 1.5, ranged: true }
};

export function validateThreatTypes(value: unknown): [ThreatType, ThreatType] {
  if (!Array.isArray(value) || value.length !== 2)
    throw new Error("enemyTypes must contain exactly two enemies");
  const names = value.map((item) => String(item).toLowerCase());
  if (names[0] === names[1])
    throw new Error("Stage-seven enemies must be different types");
  if (!names.every((name) => THREAT_TYPES.includes(name as ThreatType)))
    throw new Error(`enemyTypes must come from ${THREAT_TYPES.join(", ")}`);
  return names as [ThreatType, ThreatType];
}

export function nextThreatPosition(
  current: Point,
  victim: Point,
  movementLength: number
): Point {
  const dx = victim[0] - current[0];
  const dz = victim[1] - current[1];
  const length = Math.hypot(dx, dz) || 1;
  const movement: Point = [dx / length * movementLength, dz / length * movementLength];
  const direct: Point = [current[0] + movement[0], current[1] + movement[1]];
  if (!insideCover(direct)) return clampPoint(direct);

  const alternatives = ([
    [current[0] + movement[0], current[1]],
    [current[0], current[1] + movement[1]]
  ] as Point[]).filter((point) => !insideCover(point));
  if (alternatives.length === 0) return [...current];
  return clampPoint(
    alternatives.sort((a, b) => distance(a, victim) - distance(b, victim))[0]
  );
}

export function hasThreatLineOfSight(start: Point, end: Point): boolean {
  return !isGeometricallyBlocked(start, end);
}

function insideCover(point: Point): boolean {
  return COVER_RECTS.some(
    (cover) =>
      point[0] >= cover.minX && point[0] <= cover.maxX &&
      point[1] >= cover.minZ && point[1] <= cover.maxZ
  );
}

function clampPoint(point: Point): Point {
  return [clamp(point[0], 0.5, 14.5), clamp(point[1], 0.5, 14.5)];
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
