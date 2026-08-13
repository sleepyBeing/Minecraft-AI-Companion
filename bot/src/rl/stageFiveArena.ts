import type { Bot } from "mineflayer";
import { StageFourArena } from "./stageFourArena.js";
import {
  ACTION_DURATION_MS,
  ATTACK_RANGE,
  sleep,
  type BridgeRequest
} from "./shared.js";
import type { Point } from "./stageFourCover.js";

const STARTING_HEALTH = 15;
const FOOD_NAME = "golden_apple";
const FOOD_COUNT = 3;
const REENGAGE_MS = 500;
const REPLAN_MS = 75;

interface EatResult {
  selected: boolean;
  consumed: boolean;
  interrupted: boolean;
  noFood: boolean;
  healthBefore: number;
  healthAfter: number;
}

/** Live Minecraft arena for retreating, eating, and re-engaging. */
export class StageFiveArena extends StageFourArena {
  private hasEaten = false;
  private hasRecovered = false;
  private eatResult = emptyEatResult();

  constructor(bot: Bot) {
    super(bot);
  }

  override async reset(request: BridgeRequest) {
    this.hasEaten = false;
    this.hasRecovered = false;
    this.eatResult = emptyEatResult();
    await super.reset({ ...request, botHealth: 20 });
    await this.command("attribute @s minecraft:max_health base set 20");
    await this.command(`damage @s ${20 - STARTING_HEALTH} minecraft:generic`);
    await sleep(150);
    await this.command(`give @s minecraft:${FOOD_NAME} ${FOOD_COUNT}`);
    await sleep(150);
    await this.equipSword();
    if (Math.abs(this.bot.health - STARTING_HEALTH) > 0.75)
      throw new Error(
        `Stage-five health setup failed: expected 15, observed ${this.bot.health}`
      );
    if (this.foodCount() < FOOD_COUNT)
      throw new Error("Stage-five food was not received; ensure the bot is operator");
    this.hasRecovered = false;
    return this.observe();
  }

  override async step(action: number) {
    this.eatResult = emptyEatResult();
    return super.step(action);
  }

  override observe() {
    const state = super.observe();
    if (this.bot.health >= 19) this.hasRecovered = true;
    return {
      ...state,
      foodCount: this.foodCount(),
      hasEaten: this.hasEaten,
      hasRecovered: this.hasRecovered,
      eatResult: { ...this.eatResult }
    };
  }

  protected override async handleExtendedAction(
    action: number
  ): Promise<number | false> {
    if (action <= 9) return super.handleExtendedAction(action);
    if (action === 10) return this.eatFood();
    if (action === 11) return this.reengage();
    return false;
  }

  private async eatFood(): Promise<number> {
    const startedAt = Date.now();
    const healthBefore = this.bot.health;
    this.eatResult = {
      ...emptyEatResult(),
      selected: true,
      healthBefore,
      healthAfter: healthBefore
    };
    const food = this.bot.inventory.items().find((item) => item.name === FOOD_NAME);
    if (!food) {
      this.eatResult.noFood = true;
      return ACTION_DURATION_MS;
    }

    try {
      this.bot.clearControlStates();
      await this.bot.equip(food, "hand");
      await this.bot.consume();
      this.hasEaten = true;
      this.eatResult.consumed = true;
    } catch {
      this.eatResult.interrupted = true;
    } finally {
      this.eatResult.healthAfter = this.bot.health;
      await this.equipSword();
    }
    return Math.max(ACTION_DURATION_MS, Date.now() - startedAt);
  }

  private async reengage(): Promise<number> {
    const startedAt = Date.now();
    const deadline = startedAt + REENGAGE_MS;
    while (Date.now() < deadline && this.bot.health > 0) {
      const target = this.getTargetEntity();
      if (!target || !this.origin) break;
      const botPosition: Point = [
        this.bot.entity.position.x - this.origin.x,
        this.bot.entity.position.z - this.origin.z
      ];
      const targetPosition: Point = [
        target.position.x - this.origin.x,
        target.position.z - this.origin.z
      ];
      const dx = targetPosition[0] - botPosition[0];
      const dz = targetPosition[1] - botPosition[1];
      if (Math.hypot(dx, dz) <= ATTACK_RANGE) break;
      await this.bot.look(Math.atan2(-dx, -dz), 0, true);
      this.bot.setControlState("forward", true);
      await sleep(REPLAN_MS);
    }
    this.bot.clearControlStates();
    return Math.max(ACTION_DURATION_MS, Date.now() - startedAt);
  }

  private foodCount(): number {
    return this.bot.inventory.items()
      .filter((item) => item.name === FOOD_NAME)
      .reduce((total, item) => total + item.count, 0);
  }

  private async equipSword(): Promise<void> {
    const sword = this.bot.inventory.items().find(
      (item) => item.name === "iron_sword"
    );
    if (sword) await this.bot.equip(sword, "hand");
  }
}

function emptyEatResult(): EatResult {
  return {
    selected: false,
    consumed: false,
    interrupted: false,
    noFood: false,
    healthBefore: 0,
    healthAfter: 0
  };
}
