export interface BridgeRequest {
  id: number;
  type:
    | "ping"
    | "stage1.reset"
    | "stage1.step"
    | "stage1.observe"
    | "stage2.reset"
    | "stage2.step"
    | "stage2.observe";
  action?: number;
  botPosition?: [number, number];
  targetPosition?: [number, number];
  botYaw?: number;
  rebuildArena?: boolean;
}

export interface ArenaOrigin {
  x: number;
  y: number;
  z: number;
}

export const ROOM_SIZE = 15;
export const ACTION_DURATION_MS = 100;
export const TURN_RADIANS = Math.PI * 0.1;
export const ATTACK_RANGE = 2.5;
export const CLOSE_ATTACK_RANGE = 2.0;
export const IRON_SWORD_COOLDOWN_SECONDS = 0.625;
export const ATTACK_AIM_SETTLE_MS = 75;
export const MOVEMENT_SETTLE_TIMEOUT_MS = 450;
export const SETTLED_HORIZONTAL_SPEED = 0.025;

export function validatePosition(
  position: [number, number],
  name: string
): [number, number] {
  if (
    !Array.isArray(position) ||
    position.length !== 2 ||
    !position.every(Number.isFinite)
  ) {
    throw new Error(`${name} must contain two finite coordinates`);
  }
  if (position.some((coordinate) => coordinate < 0.5 || coordinate > ROOM_SIZE - 0.5))
    throw new Error(`${name} must be within the 15x15 arena`);
  return [position[0], position[1]];
}

export function distanceBetween(
  a: [number, number],
  b: [number, number]
): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function parseCoordinate(value: string | undefined): number | null {
  if (value === undefined) return null;
  const coordinate = Number(value);
  if (!Number.isFinite(coordinate))
    throw new Error(`Invalid RL arena coordinate: ${value}`);
  return Math.floor(coordinate);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
