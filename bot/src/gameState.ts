import type { Bot } from "mineflayer";
import type { Entity } from "prismarine-entity";
import { Vec3 } from "vec3";

export interface GameState {
  timestamp: number;
  bot: BotState;
  players: PlayerState[];
  entities: NearbyEntity[];
  environment: EnvironmentState;
}

export interface BotState {
  position: Position;
  velocity: Position;
  yaw: number;
  pitch: number;
  health: number;
  food: number;
  foodSaturation: number;
  oxygenLevel: number;
  experience: { level: number; points: number };
  onGround: boolean;
  gameMode: string;
  heldItem: ItemStack | null;
  inventory: ItemStack[];
  activeControls: string[];
}

export interface PlayerState {
  username: string;
  position: Position | null;
  distance: number | null;
  yaw: number | null;
  pitch: number | null;
}

export interface NearbyEntity {
  id: number;
  name: string;
  kind: "player" | "hostile" | "passive" | "neutral" | "item" | "projectile" | "other";
  position: Position;
  velocity: Position;
  distance: number;
  health: number | null;
  isTargetingBot: boolean;
}

export interface EnvironmentState {
  dimension: string;
  timeOfDay: number;
  isDay: boolean;
  weather: { raining: boolean; thundering: boolean };
  nearbyBlocks: NearbyBlock[];
  hazards: Hazard[];
  resources: NearbyBlock[];
  cliffs: Cliff[];
}

export interface NearbyBlock {
  name: string;
  position: Position;
  distance: number;
}

export interface Hazard extends NearbyBlock {
  type: "lava" | "fire" | "cactus" | "berry_bush" | "powder_snow" | "fall";
}

export interface Cliff {
  position: Position;
  drop: number;
  distance: number;
}

export interface Position { x: number; y: number; z: number }
export interface ItemStack { name: string; count: number; slot: number | null }

export interface GameStateOptions {
  radius?: number;
  maxBlocks?: number;
}

const HOSTILE_MOBS = new Set([
  "blaze", "bogged", "breeze", "cave_spider", "creeper", "drowned", "elder_guardian",
  "endermite", "ender_dragon", "evoker", "ghast", "guardian", "hoglin", "husk",
  "magma_cube", "phantom", "piglin_brute", "pillager", "ravager", "shulker", "silverfish",
  "skeleton", "slime", "spider", "stray", "vex", "vindicator", "warden", "witch", "wither",
  "wither_skeleton", "zoglin", "zombie", "zombie_villager"
]);
const NEUTRAL_MOBS = new Set(["bee", "enderman", "goat", "iron_golem", "llama", "panda", "piglin", "polar_bear", "wolf", "zombified_piglin"]);
const HAZARD_TYPES: Array<[string, Hazard["type"]]> = [
  ["lava", "lava"], ["fire", "fire"], ["cactus", "cactus"], ["sweet_berry_bush", "berry_bush"], ["powder_snow", "powder_snow"]
];
const RESOURCE_NAMES = /_(ore|log|leaves|crop)$|^wheat$|^carrots$|^potatoes$|^beetroots$|^sugar_cane$|^bamboo$|^cobweb$/;


export function getGameState(bot: Bot, options: GameStateOptions = {}): GameState {
  const radius = options.radius ?? 16;
  const maxBlocks = options.maxBlocks ?? 32;
  const botPosition = bot.entity.position;
  const blockPositions = bot.findBlocks({
    matching: (block) => !["air", "cave_air", "void_air"].includes(block.name),
    maxDistance: radius,
    count: maxBlocks * 8,
    useExtraInfo: true
  });
  const blocks = blockPositions
    .map((position) => bot.blockAt(position))
    .filter((block): block is NonNullable<typeof block> => block !== null);

  const nearbyBlocks = blocks.slice(0, maxBlocks).map((block) => blockState(block, botPosition));
  const hazards = blocks
    .flatMap((block) => hazardState(block, botPosition))
    .concat(findCliffs(bot, radius).map((cliff) => ({ name: "cliff", position: cliff.position, distance: cliff.distance, type: "fall" as const })))
    .slice(0, maxBlocks);
  const resources = blocks.filter((block) => RESOURCE_NAMES.test(block.name)).slice(0, maxBlocks).map((block) => blockState(block, botPosition));

  return {
    timestamp: Date.now(),
    bot: {
      position: toPosition(botPosition), velocity: toPosition(bot.entity.velocity), yaw: bot.entity.yaw, pitch: bot.entity.pitch,
      health: bot.health, food: bot.food, foodSaturation: bot.foodSaturation, oxygenLevel: bot.oxygenLevel,
      experience: { level: bot.experience.level, points: bot.experience.points }, onGround: bot.entity.onGround,
      gameMode: bot.game.gameMode, heldItem: toItem(bot.heldItem), inventory: bot.inventory.items().map(toItem).filter((item): item is ItemStack => item !== null),
      activeControls: Object.entries(bot.controlState).filter(([, enabled]) => enabled).map(([control]) => control)
    },
    players: Object.values(bot.players).map((player) => ({
      username: player.username, position: player.entity ? toPosition(player.entity.position) : null,
      distance: player.entity ? botPosition.distanceTo(player.entity.position) : null,
      yaw: player.entity?.yaw ?? null, pitch: player.entity?.pitch ?? null
    })),
    entities: Object.values(bot.entities)
      .filter((entity) => entity.id !== bot.entity.id && botPosition.distanceTo(entity.position) <= radius)
      .map((entity) => entityState(entity, bot)),
    environment: {
      dimension: bot.game.dimension, timeOfDay: bot.time.timeOfDay, isDay: bot.time.isDay,
      weather: { raining: bot.isRaining, thundering: bot.thunderState > 0 }, nearbyBlocks, hazards, resources,
      cliffs: findCliffs(bot, radius)
    }
  };
}

function entityState(entity: Entity, bot: Bot): NearbyEntity {
  return {
    id: entity.id, name: entity.name ?? entity.type, kind: classifyEntity(entity), position: toPosition(entity.position),
    velocity: toPosition(entity.velocity), distance: bot.entity.position.distanceTo(entity.position), health: entity.health ?? null,
    isTargetingBot: false
  };
}

function classifyEntity(entity: Entity): NearbyEntity["kind"] {
  const name = entity.name ?? "";
  if (entity.type === "player") return "player";
  if (name === "item") return "item";
  if (name.includes("arrow") || name.includes("projectile")) return "projectile";
  if (entity.type === "hostile" || HOSTILE_MOBS.has(name)) return "hostile";
  if (NEUTRAL_MOBS.has(name)) return "neutral";
  return entity.type === "mob" || (entity.type as string) === "animal" ? "passive" : "other";
}

function findCliffs(bot: Bot, radius: number): Cliff[] {
  const origin = bot.entity.position.floored();
  const cliffs: Cliff[] = [];
  for (let x = -radius; x <= radius; x += 2) for (let z = -radius; z <= radius; z += 2) {
    const top = new Vec3(origin.x + x, origin.y, origin.z + z);
    const standing = bot.blockAt(top.offset(0, -1, 0));
    if (!standing || standing.boundingBox === "empty") continue;
    let drop = 0;
    for (let y = 1; y <= 8 && bot.blockAt(top.offset(0, -y, 0))?.boundingBox === "empty"; y++) drop = y;
    if (drop >= 3) cliffs.push({ position: toPosition(top), drop, distance: bot.entity.position.distanceTo(top) });
  }
  return cliffs.sort((a, b) => a.distance - b.distance).slice(0, 16);
}

function blockState(block: { name: string; position: Vec3 }, origin: Vec3): NearbyBlock {
  return { name: block.name, position: toPosition(block.position), distance: origin.distanceTo(block.position) };
}

function hazardState(block: { name: string; position: Vec3 }, origin: Vec3): Hazard[] {
  const match = HAZARD_TYPES.find(([name]) => block.name.includes(name));
  return match ? [{ ...blockState(block, origin), type: match[1] }] : [];
}

function toPosition(vector: Vec3): Position { return { x: vector.x, y: vector.y, z: vector.z }; }
function toItem(item: { name: string; count: number; slot?: number } | null): ItemStack | null {
  return item ? { name: item.name, count: item.count, slot: item.slot ?? null } : null;
}
