import type { Bot, EquipmentDestination } from "mineflayer";
import { goals, Movements } from "mineflayer-pathfinder";
import type { Entity } from "prismarine-entity";
import type { Item } from "prismarine-item";
import { Vec3 } from "vec3";

export type CombatState = "idle" | "approach" | "attack" | "retreat" | "block" | "eat" | "hide";

export interface CombatOptions {
  getProtectedPlayerUsername: () => string | null;
  onCombatStart?: () => void;
  onCombatEnd?: () => void;
}

const HOSTILES = new Set([
  "blaze", "bogged", "breeze", "cave_spider", "creeper", "drowned", "elder_guardian",
  "endermite", "ender_dragon", "evoker", "ghast", "guardian", "hoglin", "husk",
  "magma_cube", "phantom", "piglin_brute", "pillager", "ravager", "shulker", "silverfish",
  "skeleton", "slime", "spider", "stray", "vex", "vindicator", "warden", "witch", "wither",
  "wither_skeleton", "zoglin", "zombie", "zombie_villager"
]);
const DANGEROUS_FOOD = new Set([
  "pufferfish", "poisonous_potato", "rotten_flesh", "spider_eye", "raw_chicken"
]);
const BUILDING_BLOCKS = [
  "cobblestone", "cobbled_deepslate", "stone", "dirt", "netherrack", "oak_planks",
  "spruce_planks", "birch_planks"
];
const THREAT_RADIUS = 20;
const PROTECT_RADIUS = 7;
const MELEE_RANGE = 3.1;
const RETREAT_HEALTH = 7;
const SAFE_HEALTH = 13;
const THREAT_CLEAR_MS = 5_000;
const COMBAT_SEARCH_RADIUS = 32;
type BoundedPathfinder = Bot["pathfinder"] & { searchRadius: number };

/** Rule-based controller */
export class RuleBasedCombatController {
  private timer: NodeJS.Timeout | null = null;
  private ticking = false;
  private active = false;
  private state: CombatState = "idle";
  private targetId: number | null = null;
  private lastThreatAt = 0;
  private lastAttackAt = 0;
  private strafeLeft = false;
  private lastStrafeAt = 0;
  private eating = false;
  private placingBarrier = false;
  private retreatAnchor: Vec3 | null = null;
  private retreatStartedAt = 0;
  private lastRetreatGoalAt = 0;
  private lastArmorCheckAt = 0;
  private equippingArmor = false;

  constructor(private readonly bot: Bot, private readonly options: CombatOptions) {}

  get isBusy(): boolean { return this.active; }
  get currentState(): CombatState { return this.state; }

  start(): void {
    if (this.timer) return;
    void this.equipBestArmor(true);
    this.timer = setInterval(() => void this.tickSafely(), 200);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.leaveCombat();
  }

  private async tickSafely(): Promise<void> {
    if (this.ticking || !this.bot.entity || this.bot.health <= 0) return;
    this.ticking = true;
    try { await this.tick(); }
    catch (error) { console.error("Combat controller error:", error); }
    finally { this.ticking = false; }
  }

  private async tick(): Promise<void> {
    const protectedPlayer = this.getProtectedPlayer();
    const target = this.chooseThreat(protectedPlayer);
    const now = Date.now();

    if (target) {
      this.lastThreatAt = now;
      this.enterCombat();
      await this.equipBestArmor();
      const distance = this.bot.entity.position.distanceTo(target.position);

      if (this.bot.health <= RETREAT_HEALTH || (target.name === "creeper" && distance <= 5)) {
        if (distance > 7 && await this.tryEat()) return;
        await this.retreatFrom(target);
        return;
      }

      if ((this.bot.food <= 8 || this.bot.health <= 10) && distance > 7 && await this.tryEat()) return;
      if (distance > MELEE_RANGE) this.approach(target);
      else await this.meleeAttack(target);
      return;
    }

    // eating if low or hungry
    if ((this.bot.food <= 12 || this.bot.health <= 9) && this.findFood()) {
      this.enterCombat();
      if (await this.tryEat()) return;
    }

    if (!this.active) return;
    if (this.bot.health < SAFE_HEALTH && this.findFood()) {
      this.setState("hide");
      await this.tryEat();
      return;
    }

    if (now - this.lastThreatAt < THREAT_CLEAR_MS) {
      this.setState("hide");
      return;
    }

    this.leaveCombat();
  }

  private chooseThreat(player: Entity | null): Entity | null {
    const hostiles = Object.values(this.bot.entities).filter((entity) =>
      (entity.type === "hostile" || HOSTILES.has(entity.name ?? "")) &&
      this.bot.entity.position.distanceTo(entity.position) <= THREAT_RADIUS
    );

    return hostiles.sort((a, b) => {
      const score = (entity: Entity) => {
        const botDistance = this.bot.entity.position.distanceTo(entity.position);
        const playerDistance = player ? player.position.distanceTo(entity.position) : 100;
        const protecting = playerDistance <= PROTECT_RADIUS ? -30 : 0;
        const creeperUrgency = entity.name === "creeper" && botDistance <= 5 ? -15 : 0;
        return protecting + creeperUrgency + Math.min(botDistance, playerDistance);
      };
      return score(a) - score(b);
    })[0] ?? null;
  }

  private getProtectedPlayer(): Entity | null {
    const username = this.options.getProtectedPlayerUsername();
    return username ? this.bot.players[username]?.entity ?? null : null;
  }

  private enterCombat(): void {
    if (this.active) return;
    this.active = true;
    this.options.onCombatStart?.();
  }

  private leaveCombat(): void {
    if (!this.active) return;
    this.bot.pathfinder.setGoal(null);
    this.bot.clearControlStates();
    this.active = false;
    this.state = "idle";
    this.targetId = null;
    this.retreatAnchor = null;
    this.options.onCombatEnd?.();
  }

  private setState(next: CombatState): void {
    if (this.state === next) return;
    this.state = next;
    this.bot.clearControlStates();
    if (next === "hide" || next === "eat" || next === "attack" || next === "block")
      this.bot.pathfinder.setGoal(null);
  }

  private approach(target: Entity): void {
    const changedTarget = this.targetId !== target.id;
    if (this.state !== "approach" || changedTarget) {
      this.setState("approach");
      this.configureCombatPathfinder();
      this.bot.pathfinder.setMovements(createCombatMovements(this.bot));
      this.bot.pathfinder.setGoal(new goals.GoalFollow(target, 2.7), true);
    }
    this.targetId = target.id;
  }

  private async meleeAttack(target: Entity): Promise<void> {
    this.setState("attack");
    this.targetId = target.id;
    await this.equipStrongestWeapon();
    await this.bot.lookAt(target.position.offset(0, target.height * 0.65, 0), true);

    const now = Date.now();
    if (now - this.lastStrafeAt >= 800) {
      this.strafeLeft = !this.strafeLeft;
      this.lastStrafeAt = now;
    }
    this.bot.setControlState("left", this.strafeLeft);
    this.bot.setControlState("right", !this.strafeLeft);

    if (now - this.lastAttackAt >= attackDelay(this.bot.heldItem?.name)) {
      this.bot.attack(target);
      this.lastAttackAt = now;
    }
  }

  private async retreatFrom(target: Entity): Promise<void> {
    const now = Date.now();
    this.targetId = target.id;
    if (this.state !== "retreat") {
      this.setState("retreat");
      this.retreatAnchor = this.bot.entity.position.clone();
      this.retreatStartedAt = now;
    }

    if (now - this.lastRetreatGoalAt >= 1_500) {
      const away = this.bot.entity.position.minus(target.position);
      const length = Math.max(0.001, Math.sqrt(away.x * away.x + away.z * away.z));
      const x = Math.floor(this.bot.entity.position.x + away.x / length * 10);
      const z = Math.floor(this.bot.entity.position.z + away.z / length * 10);
      this.configureCombatPathfinder();
      this.bot.pathfinder.setMovements(createCombatMovements(this.bot));
      this.bot.pathfinder.setGoal(new goals.GoalNearXZ(x, z, 2));
      this.lastRetreatGoalAt = now;
    }

    const moved = this.retreatAnchor ? this.bot.entity.position.distanceTo(this.retreatAnchor) : 0;
    if (moved >= 2) {
      this.retreatAnchor = this.bot.entity.position.clone();
      this.retreatStartedAt = now;
    } else if (now - this.retreatStartedAt >= 2_500 && !this.placingBarrier) {
      await this.placeBarrierBetween(target);
      this.retreatStartedAt = now;
    }
  }

  private async placeBarrierBetween(target: Entity): Promise<void> {
    const blockItem = this.bot.inventory.items().find((item) => BUILDING_BLOCKS.includes(item.name));
    if (!blockItem) {
      this.setState("hide");
      return;
    }

    this.placingBarrier = true;
    this.setState("block");
    try {
      const delta = target.position.minus(this.bot.entity.position);
      const dx = Math.abs(delta.x) >= Math.abs(delta.z) ? Math.sign(delta.x) : 0;
      const dz = dx === 0 ? Math.sign(delta.z) : 0;
      const feet = this.bot.entity.position.floored().offset(dx, 0, dz);
      const ground = this.bot.blockAt(feet.offset(0, -1, 0));
      if (!ground || ground.boundingBox === "empty" || this.bot.blockAt(feet)?.boundingBox !== "empty") return;

      await this.bot.equip(blockItem, "hand");
      await this.bot.placeBlock(ground, new Vec3(0, 1, 0));

      const first = this.bot.blockAt(feet);
      const secondSpace = this.bot.blockAt(feet.offset(0, 1, 0));
      const nextItem = this.bot.inventory.items().find((item) => BUILDING_BLOCKS.includes(item.name));
      if (first && secondSpace?.boundingBox === "empty" && nextItem) {
        await this.bot.equip(nextItem, "hand");
        await this.bot.placeBlock(first, new Vec3(0, 1, 0));
      }
    } catch (error) {
      console.error("Could not place an emergency barrier:", error);
    } finally {
      this.placingBarrier = false;
      this.setState("hide");
    }
  }

  private async tryEat(): Promise<boolean> {
    if (this.eating) return true;
    const food = this.findFood();
    if (!food) return false;

    this.eating = true;
    this.setState("eat");
    try {
      await this.bot.equip(food, "hand");
      await this.bot.consume();
      return true;
    } catch (error) {
      console.error("Could not eat food:", error);
      return false;
    } finally {
      this.eating = false;
    }
  }

  private findFood(): Item | null {
    const lowHealth = this.bot.health <= 10;
    return this.bot.inventory.items()
      .filter((item) => {
        const special = item.name === "golden_apple" || item.name === "enchanted_golden_apple";
        return this.bot.registry.foodsByName[item.name] && !DANGEROUS_FOOD.has(item.name) &&
          (this.bot.food < 20 || special);
      })
      .sort((a, b) => foodScore(this.bot, b, lowHealth) - foodScore(this.bot, a, lowHealth))[0] ?? null;
  }

  private async equipStrongestWeapon(): Promise<void> {
    const weapon = this.bot.inventory.items()
      .filter((item) => weaponScore(item.name) > 0)
      .sort((a, b) => weaponScore(b.name) - weaponScore(a.name))[0];
    if (weapon && this.bot.heldItem?.slot !== weapon.slot) await this.bot.equip(weapon, "hand");
    else if (!weapon && this.bot.heldItem) await this.bot.unequip("hand");
  }

  private configureCombatPathfinder(): void {
    const pathfinder = this.bot.pathfinder as BoundedPathfinder;
    pathfinder.searchRadius = COMBAT_SEARCH_RADIUS;
    pathfinder.thinkTimeout = 1_000;
    pathfinder.tickTimeout = 15;
  }

  private async equipBestArmor(force = false): Promise<void> {
    const now = Date.now();
    if (this.equippingArmor || (!force && now - this.lastArmorCheckAt < 5_000)) return;
    this.equippingArmor = true;
    this.lastArmorCheckAt = now;

    try {
      const destinations: EquipmentDestination[] = ["head", "torso", "legs", "feet"];
      for (const destination of destinations) {
        const candidates = this.bot.inventory.items()
          .filter((item) => armorDestination(item.name) === destination && !hasBindingCurse(item))
          .sort((a, b) => armorScore(b) - armorScore(a));
        const best = candidates[0];
        if (!best) continue;

        const equippedSlot = this.bot.getEquipmentDestSlot(destination);
        const equipped = this.bot.inventory.slots[equippedSlot];
        if (equipped && hasBindingCurse(equipped)) continue;
        if (equipped?.slot === best.slot || (equipped && armorScore(equipped) >= armorScore(best))) continue;

        await this.bot.equip(best, destination);
      }
    } catch (error) {
      console.error("Could not equip the best armor:", error);
    } finally {
      this.equippingArmor = false;
    }
  }
}

function createCombatMovements(bot: Bot): Movements {
  const movements = new Movements(bot);
  movements.canDig = false;
  movements.allow1by1towers = false;
  movements.allowParkour = false;
  movements.maxDropDown = 2;
  movements.infiniteLiquidDropdownDistance = false;
  for (const name of ["lava", "fire", "soul_fire", "cactus", "powder_snow", "magma_block"]) {
    const block = bot.registry.blocksByName[name];
    if (block) movements.blocksToAvoid.add(block.id);
  }
  return movements;
}

function weaponScore(name: string): number {
  const material = name.startsWith("netherite_") ? 7 : name.startsWith("diamond_") ? 6 :
    name.startsWith("iron_") ? 5 : name.startsWith("copper_") ? 4 :
      name.startsWith("stone_") ? 3 : name.startsWith("golden_") ? 2 : name.startsWith("wooden_") ? 1 : 0;
  if (name === "mace") return 11;
  if (name === "trident") return 10;
  if (name.endsWith("_sword")) return 8 + material;
  if (name.endsWith("_axe")) return 7 + material;
  return 0;
}

function attackDelay(name?: string): number {
  if (!name) return 650;
  if (name.endsWith("_axe")) return 1_050;
  if (name === "mace") return 1_000;
  return 625;
}

function foodScore(bot: Bot, item: Item, lowHealth: boolean): number {
  const food = bot.registry.foodsByName[item.name];
  if (!food) return 0;
  const healingBonus = lowHealth && (item.name === "golden_apple" || item.name === "enchanted_golden_apple") ? 100 : 0;
  return healingBonus + food.foodPoints + food.saturation / 100;
}

function armorDestination(name: string): EquipmentDestination | null {
  if (name.endsWith("_helmet") || name === "turtle_helmet") return "head";
  if (name.endsWith("_chestplate")) return "torso";
  if (name.endsWith("_leggings")) return "legs";
  if (name.endsWith("_boots")) return "feet";
  return null;
}

function armorScore(item: Item): number {
  const name = item.name;
  const material = name.startsWith("netherite_") ? 70 : name.startsWith("diamond_") ? 60 :
    name.startsWith("iron_") ? 50 : name.startsWith("copper_") ? 45 :
      name.startsWith("chainmail_") ? 40 : name.startsWith("golden_") ? 30 :
        name.startsWith("leather_") ? 20 : name === "turtle_helmet" ? 52 : 0;
  const durabilityRatio = item.maxDurability > 0
    ? Math.max(0, (item.maxDurability - item.durabilityUsed) / item.maxDurability)
    : 1;
  const enchantment = item.enchants.reduce((score, enchant) => {
    if (enchant.name === "protection") return score + enchant.lvl * 4;
    if (["blast_protection", "projectile_protection", "fire_protection"].includes(enchant.name))
      return score + enchant.lvl * 1.5;
    if (enchant.name === "unbreaking") return score + enchant.lvl * 0.3;
    if (enchant.name === "mending") return score + 0.5;
    if (enchant.name === "thorns") return score + enchant.lvl * 0.5;
    return score;
  }, 0);
  return material + durabilityRatio * 3 + enchantment;
}

function hasBindingCurse(item: Item): boolean {
  return item.enchants.some((enchant) => enchant.name === "binding_curse" || enchant.name === "curse_of_binding");
}
