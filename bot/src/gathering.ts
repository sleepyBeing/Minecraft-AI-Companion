import type { Bot } from "mineflayer";
import { goals, Movements } from "mineflayer-pathfinder";
import type { Block } from "prismarine-block";
import type { Item } from "prismarine-item";
import { Vec3 } from "vec3";

const BLOCK_SEARCH_RADIUS = 48;
const ITEM_COLLECTION_RADIUS = 8;
const HAZARDS = [
  "lava", "fire", "soul_fire", "cactus", "sweet_berry_bush", "powder_snow",
  "magma_block", "campfire", "soul_campfire", "wither_rose"
];

interface MaterialSpec {
  displayName: string;
  blockNames: string[];
  itemNames: string[];
  underground: boolean;
  preferredY?: number;
}

interface GatherTask {
  id: number;
  requester: string;
  quantity: number;
  remainingToDeliver: number;
  spec: MaterialSpec;
  unreachable: Set<string>;
  explorationStep: number;
  stripDirection: number;
  stripSteps: number;
  awaitingDeliveryNotified: boolean;
}

const MATERIALS: Record<string, MaterialSpec> = {
  "oak_log": surface("oak logs", ["oak_log"]),
  "oak_logs": surface("oak logs", ["oak_log"]),
  "spruce_log": surface("spruce logs", ["spruce_log"]),
  "spruce_logs": surface("spruce logs", ["spruce_log"]),
  "birch_log": surface("birch logs", ["birch_log"]),
  "birch_logs": surface("birch logs", ["birch_log"]),
  "jungle_log": surface("jungle logs", ["jungle_log"]),
  "jungle_logs": surface("jungle logs", ["jungle_log"]),
  "acacia_log": surface("acacia logs", ["acacia_log"]),
  "acacia_logs": surface("acacia logs", ["acacia_log"]),
  "dark_oak_log": surface("dark oak logs", ["dark_oak_log"]),
  "dark_oak_logs": surface("dark oak logs", ["dark_oak_log"]),
  "mangrove_log": surface("mangrove logs", ["mangrove_log"]),
  "mangrove_logs": surface("mangrove logs", ["mangrove_log"]),
  "cherry_log": surface("cherry logs", ["cherry_log"]),
  "cherry_logs": surface("cherry logs", ["cherry_log"]),
  "pale_oak_log": surface("pale oak logs", ["pale_oak_log"]),
  "pale_oak_logs": surface("pale oak logs", ["pale_oak_log"]),
  "logs": surface("logs", [
    "oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log",
    "mangrove_log", "cherry_log", "pale_oak_log"
  ]),
  "coal": ore("coal", ["coal_ore", "deepslate_coal_ore"], ["coal"], 48),
  "coal_ore": ore("coal", ["coal_ore", "deepslate_coal_ore"], ["coal"], 48),
  "iron": ore("iron", ["iron_ore", "deepslate_iron_ore"], ["raw_iron"], 16),
  "iron_ore": ore("iron", ["iron_ore", "deepslate_iron_ore"], ["raw_iron"], 16),
  "copper": ore("copper", ["copper_ore", "deepslate_copper_ore"], ["raw_copper"], 48),
  "copper_ore": ore("copper", ["copper_ore", "deepslate_copper_ore"], ["raw_copper"], 48),
  "gold": ore("gold", ["gold_ore", "deepslate_gold_ore"], ["raw_gold"], -16),
  "gold_ore": ore("gold", ["gold_ore", "deepslate_gold_ore"], ["raw_gold"], -16),
  "diamond": ore("diamonds", ["diamond_ore", "deepslate_diamond_ore"], ["diamond"], -54),
  "diamond_ore": ore("diamonds", ["diamond_ore", "deepslate_diamond_ore"], ["diamond"], -54),
  "emerald": ore("emeralds", ["emerald_ore", "deepslate_emerald_ore"], ["emerald"], 96),
  "emerald_ore": ore("emeralds", ["emerald_ore", "deepslate_emerald_ore"], ["emerald"], 96),
  "redstone": ore("redstone", ["redstone_ore", "deepslate_redstone_ore"], ["redstone"], -54),
  "lapis": ore("lapis lazuli", ["lapis_ore", "deepslate_lapis_ore"], ["lapis_lazuli"], 0),
  "lapis_lazuli": ore("lapis lazuli", ["lapis_ore", "deepslate_lapis_ore"], ["lapis_lazuli"], 0),
  "ancient_debris": ore("ancient debris", ["ancient_debris"], ["ancient_debris"], 15),
  "nether_quartz": ore("quartz", ["nether_quartz_ore"], ["quartz"], 32),
  "quartz": ore("quartz", ["nether_quartz_ore"], ["quartz"], 32),
  "dirt": surface("dirt", ["dirt"], ["dirt"]),
  "sand": surface("sand", ["sand"], ["sand"]),
  "gravel": surface("gravel", ["gravel"], ["gravel"]),
  "cobblestone": ore("cobblestone", ["stone", "cobblestone"], ["cobblestone"], 32)
};

export class GatheringController {
  private task: GatherTask | null = null;
  private nextTaskId = 1;
  private paused = false;
  private restoreMovements = false;

  constructor(private readonly bot: Bot) {}

  get isBusy(): boolean {
    return this.task !== null;
  }

  get isPaused(): boolean {
    return this.paused;
  }

  setPaused(paused: boolean): void {
    if (this.paused === paused) return;
    this.paused = paused;
    if (paused) {
      this.bot.pathfinder.setGoal(null);
      this.bot.stopDigging();
    } else {
      this.restoreMovements = true;
    }
  }

  async handleCommand(requester: string, message: string): Promise<boolean> {
    if (!message.toLowerCase().startsWith("!gather")) return false;

    if (message.trim().toLowerCase() === "!gather stop") {
      if (this.task) this.stop("Gathering stopped.");
      else this.bot.chat("I am not currently gathering anything.");
      return true;
    }

    const match = message.trim().match(/^!gather\s+(\d+)\s+(.+)$/i);
    if (!match) {
      this.bot.chat("Usage: !gather <quantity> <material>, for example: !gather 12 oak logs");
      return true;
    }

    const quantity = Number(match[1]);
    if (!Number.isSafeInteger(quantity) || quantity < 1 || quantity > 2_304) {
      this.bot.chat("Gather quantity must be between 1 and 2304.");
      return true;
    }

    const spec = resolveMaterial(this.bot, match[2]);
    if (!spec) {
      this.bot.chat(`I don't recognize the material "${match[2]}".`);
      return true;
    }

    const toolProblem = requiredToolMessage(this.bot, spec);
    if (toolProblem) {
      this.bot.chat(`I can't gather ${spec.displayName}: ${toolProblem}.`);
      return true;
    }

    this.stop();
    const task: GatherTask = {
      id: this.nextTaskId++, requester, quantity, spec,
      remainingToDeliver: quantity,
      unreachable: new Set(), explorationStep: 0, stripDirection: 0, stripSteps: 0,
      awaitingDeliveryNotified: false
    };
    this.task = task;
    this.bot.chat(`Gathering ${quantity} ${spec.displayName} for ${requester}.`);
    void this.run(task);
    return true;
  }

  stop(notification?: string): void {
    if (!this.task) return;
    this.task = null;
    this.paused = false;
    this.restoreMovements = false;
    this.bot.pathfinder.stop();
    if (notification) this.bot.chat(notification);
  }

  private async run(task: GatherTask): Promise<void> {
    this.bot.pathfinder.setMovements(createGatherMovements(this.bot, false));

    try {
      while (this.isCurrent(task)) {
        if (this.paused) {
          await sleep(250);
          continue;
        }
        if (this.restoreMovements) {
          this.bot.pathfinder.setMovements(createGatherMovements(this.bot, false));
          this.restoreMovements = false;
        }

        const available = inventoryCount(this.bot, task.spec.itemNames);

        // Existing inventory counts toward the request. If the player dies,
        // keep the finished order and retry delivery after they respawn.
        if (available >= task.remainingToDeliver) {
          const delivered = await this.deliverGatheredItems(task, task.remainingToDeliver);
          if (!this.isCurrent(task)) return;

          if (delivered > 0) {
            task.remainingToDeliver -= delivered;
            task.awaitingDeliveryNotified = false;
          }

          if (task.remainingToDeliver <= 0) {
            this.task = null;
            this.bot.pathfinder.stop();
            this.bot.chat(`Delivered ${task.quantity} ${task.spec.displayName} to ${task.requester}.`);
            return;
          }

          if (!task.awaitingDeliveryNotified) {
            this.bot.chat(`I have the ${task.spec.displayName}; I will deliver them when ${task.requester} is reachable.`);
            task.awaitingDeliveryNotified = true;
          }
          await sleep(1_000);
          continue;
        }

        const toolProblem = requiredToolMessage(this.bot, task.spec);
        if (toolProblem) {
          this.bot.chat(`I had to stop gathering ${task.spec.displayName}: ${toolProblem}.`);
          this.task = null;
          this.bot.pathfinder.stop();
          return;
        }

        const block = this.findTargetBlock(task);
        if (block) {
          await this.harvestBlock(task, block);
        } else if (task.spec.underground) {
          await this.searchByStripMining(task);
        } else {
          await this.searchSurface(task);
        }
      }
    } catch (error) {
      if (!this.isCurrent(task)) return;
      if (this.paused) {
        void this.run(task);
        return;
      }
      console.error("Gathering failed:", error);
      this.bot.chat(`I couldn't continue gathering ${task.spec.displayName}.`);
      this.task = null;
      this.bot.pathfinder.stop();
    }
  }

  private findTargetBlock(task: GatherTask): Block | null {
    const ids = task.spec.blockNames
      .map((name) => this.bot.registry.blocksByName[name]?.id)
      .filter((id): id is number => id !== undefined);

    return this.bot.findBlock({
      matching: (block) => ids.includes(block.type) && !task.unreachable.has(positionKey(block.position)),
      maxDistance: BLOCK_SEARCH_RADIUS,
      useExtraInfo: true
    });
  }

  private async harvestBlock(task: GatherTask, block: Block): Promise<void> {
    try {
      await this.bot.pathfinder.goto(new goals.GoalLookAtBlock(block.position, this.bot.world, { reach: 4.5 }));
      if (!this.isCurrent(task)) return;

      const currentBlock = this.bot.blockAt(block.position);
      if (!currentBlock || !task.spec.blockNames.includes(currentBlock.name)) return;

      const tool = bestHarvestTool(this.bot, currentBlock);
      if (currentBlock.harvestTools && !tool) {
        throw new Error(`No adequate tool remains for ${currentBlock.name}`);
      }
      if (tool) await this.bot.equip(tool, "hand");

      await this.bot.dig(currentBlock, true, "raycast");
      await sleep(250);
      await this.collectNearbyDrops(task, block.position);
    } catch (error) {
      if (!this.isCurrent(task)) return;
      console.error(`Could not gather block at ${positionKey(block.position)}:`, error);
      task.unreachable.add(positionKey(block.position));
    }
  }

  private async collectNearbyDrops(task: GatherTask, origin: Vec3): Promise<void> {
    for (let attempt = 0; attempt < 3 && this.isCurrent(task); attempt++) {
      const item = Object.values(this.bot.entities)
        .filter((entity) => entity.name === "item" && entity.position.distanceTo(origin) <= ITEM_COLLECTION_RADIUS)
        .sort((a, b) => a.position.distanceTo(origin) - b.position.distanceTo(origin))[0];

      if (!item) {
        await sleep(200);
        continue;
      }

      try {
        await this.bot.pathfinder.goto(new goals.GoalNear(
          item.position.x, item.position.y, item.position.z, 1
        ));
        await sleep(200);
      } catch {
        return;
      }
    }
  }

  /** Walk to the requester and toss only the outstanding requested quantity. */
  private async deliverGatheredItems(task: GatherTask, amount: number): Promise<number> {
    const player = this.bot.players[task.requester]?.entity;
    if (!player) return 0;

    try {
      await this.bot.pathfinder.goto(new goals.GoalNear(
        player.position.x, player.position.y, player.position.z, 2
      ));
    } catch (error) {
      console.error(`Could not reach ${task.requester} to deliver gathered items:`, error);
      return 0;
    }

    let remaining = amount;
    let delivered = 0;
    for (const item of this.bot.inventory.items()) {
      if (remaining <= 0) break;
      if (!task.spec.itemNames.includes(item.name)) continue;

      const count = Math.min(item.count, remaining);
      try {
        await this.bot.toss(item.type, item.metadata, count);
        delivered += count;
        remaining -= count;
      } catch (error) {
        console.error(`Could not toss ${item.name} to ${task.requester}:`, error);
      }
    }
    return delivered;
  }

  private async searchSurface(task: GatherTask): Promise<void> {
    const angle = task.explorationStep++ * 2.399963;
    const distance = 24 + (task.explorationStep % 4) * 8;
    const targetX = Math.floor(this.bot.entity.position.x + Math.cos(angle) * distance);
    const targetZ = Math.floor(this.bot.entity.position.z + Math.sin(angle) * distance);

    try {
      await this.bot.pathfinder.goto(new goals.GoalNearXZ(targetX, targetZ, 3));
    } catch (error) {
      console.error("Surface gathering search could not reach its waypoint:", error);
    }
  }

  private async searchByStripMining(task: GatherTask): Promise<void> {
    const current = this.bot.entity.position.floored();
    const desiredY = task.spec.preferredY ?? current.y;
    const descending = current.y > desiredY + 2;
    const vectors = [new Vec3(1, 0, 0), new Vec3(0, 0, 1), new Vec3(-1, 0, 0), new Vec3(0, 0, -1)];
    const forward = vectors[task.stripDirection];
    const targetY = descending ? current.y - 1 : current.y;
    const target = current.offset(forward.x, targetY - current.y, forward.z);
    const feet = this.bot.blockAt(target);
    const head = this.bot.blockAt(target.offset(0, 1, 0));
    const ground = this.bot.blockAt(target.offset(0, -1, 0));

    if (!feet || !head || !ground || ground.boundingBox === "empty" ||
      isHazard(feet.name) || isHazard(head.name) || isHazard(ground.name)) {
      this.turnStripMine(task);
      return;
    }

    this.bot.pathfinder.setGoal(null);

    try {
      if (!await this.clearTunnelBlock(task, feet)) return;
      if (!await this.clearTunnelBlock(task, head)) return;
      if (!this.isCurrent(task)) return;

      await this.bot.pathfinder.goto(new goals.GoalNear(target.x, target.y, target.z, 0));
      task.stripSteps++;
      if (task.stripSteps >= 16) this.turnStripMine(task);
    } catch (error) {
      console.error("Strip-mining tunnel step failed:", error);
      this.turnStripMine(task);
    }
  }

  private async clearTunnelBlock(task: GatherTask, block: Block): Promise<boolean> {
    if (block.boundingBox === "empty") return true;
    if (!block.diggable || isHazard(block.name)) {
      this.turnStripMine(task);
      return false;
    }

    const tool = bestHarvestTool(this.bot, block);
    if (block.harvestTools && !tool) {
      this.bot.chat(`I need a better tool to mine ${block.name}.`);
      this.stop();
      return false;
    }

    if (tool) await this.bot.equip(tool, "hand");
    await this.bot.dig(block, true, "raycast");
    return true;
  }

  private turnStripMine(task: GatherTask): void {
    task.stripDirection = (task.stripDirection + 1) % 4;
    task.stripSteps = 0;
    task.explorationStep++;
  }

  private isCurrent(task: GatherTask): boolean {
    return this.task?.id === task.id;
  }
}

function surface(displayName: string, blockNames: string[], itemNames = blockNames): MaterialSpec {
  return { displayName, blockNames, itemNames, underground: false };
}

function ore(displayName: string, blockNames: string[], itemNames: string[], preferredY: number): MaterialSpec {
  return { displayName, blockNames, itemNames, underground: true, preferredY };
}

function resolveMaterial(bot: Bot, input: string): MaterialSpec | null {
  const normalized = input.trim().toLowerCase().replace(/[-\s]+/g, "_");
  const singular = normalized.endsWith("s") ? normalized.slice(0, -1) : normalized;
  const alias = MATERIALS[normalized] ?? MATERIALS[singular];
  if (alias) return alias;

  const block = bot.registry.blocksByName[normalized] ?? bot.registry.blocksByName[singular];
  if (!block) return null;
  return surface(block.displayName.toLowerCase(), [block.name], [block.name]);
}

function requiredToolMessage(bot: Bot, spec: MaterialSpec): string | null {
  const definitions = spec.blockNames
    .map((name) => bot.registry.blocksByName[name])
    .filter((block) => block?.harvestTools);
  if (definitions.length === 0) return null;

  const inventory = bot.inventory.items();
  const hasTool = inventory.some((item) =>
    definitions.some((block) => Boolean(block.harvestTools?.[String(item.type)]))
  );
  if (hasTool) return null;

  const toolNames = new Set<string>();
  for (const block of definitions) {
    for (const itemId of Object.keys(block.harvestTools ?? {})) {
      const item = bot.registry.items[Number(itemId)];
      if (item) toolNames.add(item.displayName);
    }
  }
  const examples = [...toolNames].slice(0, 3).join(", ");
  return examples ? `I need an adequate tool such as ${examples}` : "I don't have an adequate tool";
}

function bestHarvestTool(bot: Bot, block: Block): Item | null {
  const valid = bot.inventory.items().filter((item) => block.canHarvest(item.type));
  return valid.sort((a, b) => block.digTime(a.type, false, false, false) - block.digTime(b.type, false, false, false))[0] ?? null;
}

function createGatherMovements(bot: Bot, canMine: boolean): Movements {
  const movements = new Movements(bot);
  movements.canDig = canMine;
  movements.allow1by1towers = false;
  movements.allowParkour = false;
  movements.maxDropDown = 2;
  movements.infiniteLiquidDropdownDistance = false;
  movements.dontCreateFlow = true;
  movements.dontMineUnderFallingBlock = true;

  for (const blockName of HAZARDS) {
    const block = bot.registry.blocksByName[blockName];
    if (block) movements.blocksToAvoid.add(block.id);
  }
  return movements;
}

function inventoryCount(bot: Bot, itemNames: string[]): number {
  return bot.inventory.items()
    .filter((item) => itemNames.includes(item.name))
    .reduce((sum, item) => sum + item.count, 0);
}

function positionKey(position: Vec3): string {
  return `${position.x},${position.y},${position.z}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isHazard(blockName: string): boolean {
  return HAZARDS.some((hazard) => blockName.includes(hazard));
}
