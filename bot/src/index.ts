import "dotenv/config";
import mineflayer from "mineflayer";
import type { Entity } from "prismarine-entity";
import { Vec3 } from "vec3";

function getRequiredEnvironmentVariable(name: string): string {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing environment variable: ${name}`);
  }

  return value;
}

const host = getRequiredEnvironmentVariable("MINECRAFT_HOST");
const username = getRequiredEnvironmentVariable("MINECRAFT_USERNAME");
const version = getRequiredEnvironmentVariable("MINECRAFT_VERSION");
const port = Number(process.env.MINECRAFT_PORT ?? "25565");

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error(`Invalid Minecraft port: ${process.env.MINECRAFT_PORT}`);
}

console.log(`Connecting ${username} to ${host}:${port}...`);

const bot = mineflayer.createBot({
  host,
  port,
  username,
  version,
  auth: "offline"
});

bot.on("login", () => {
  console.log(`Logged in as ${bot.username}.`);
});

bot.once("spawn", () => {
  const { x, y, z } = bot.entity.position;

  console.log("Bot spawned successfully.");
  console.log(`Position: x=${x.toFixed(1)}, y=${y.toFixed(1)}, z=${z.toFixed(1)}`);

  bot.chat("CompanionBot is online.");
});

bot.on("kicked", (reason) => {
  console.error("Bot was kicked:", reason);
});

bot.on("error", (error) => {
  console.error("Bot error:", error);
});

bot.on("end", (reason) => {
  console.log("Bot disconnected:", reason);
});

// chat connection
bot.on("chat", async (username, message) => {
  if (username === bot.username) {
    return;
  }

  console.log(`<${username}> ${message}`);

  if (message.toLowerCase() === "!companion status") {
    const inventory = bot.inventory.items();
    const inventorySummary = inventory.length === 0
      ? "empty"
      : inventory.map((item) => `${item.count} ${item.name}`).join(", ");

    bot.chat(`Health: ${bot.health}/20 | Inventory: ${inventorySummary}`);
    return;
  }

  if (message.toLowerCase() === "!companion hello") {
    bot.chat(`Hello, ${username}.`);
  }

  if (message.toLowerCase() === "!companion dig") {
    const player = bot.players[username]?.entity;

    if (!player) {
      bot.chat(`I can't see you, ${username}. Move closer and try again.`);
      return;
    }

    // raycast from the player's eyes so the command targets the block they see
    const target = bot.blockAtEntityCursor(player, 64);

    if (!target) {
      bot.chat("I couldn't find a block in your line of sight.");
      return;
    }

    if (!bot.canDigBlock(target)) {
      bot.chat(`I can't dig that ${target.name}; it may be too far away or protected.`);
      return;
    }

    try {
      bot.chat(`Digging ${target.name}.`);
      await bot.dig(target, true);
    } catch (error) {
      console.error(`Could not dig ${target.name}:`, error);
      bot.chat(`I couldn't dig that ${target.name}.`);
    }

    return;
  }

  if (message.toLowerCase() === "!companion attack") {
    const player = bot.players[username]?.entity;

    if (!player) {
      bot.chat(`I can't see you, ${username}. Move closer and try again.`);
      return;
    }

    const target = findMobInPlayerSight(player);

    if (!target) {
      bot.chat("I can't see a mob in your line of sight.");
      return;
    }

    if (bot.entity.position.distanceTo(target.position) > 3.5) {
      bot.chat(`That ${target.name} is too far away for me to attack.`);
      return;
    }

    try {
      await bot.lookAt(target.position.offset(0, target.height / 2, 0), true);
      bot.attack(target);
      bot.chat(`Attacking the ${target.name}.`);
    } catch (error) {
      console.error(`Could not attack ${target.name}:`, error);
      bot.chat(`I couldn't attack that ${target.name}.`);
    }

    return;
  }
  
  // movement command testing
  if (message.toLowerCase() === "!companion forward") {
    bot.setControlState("forward", true);

    setTimeout(() => {
        bot.setControlState("forward", false);
    }, 100);   

    return;
  }

  if (message.toLowerCase() === "!companion jump") {
    bot.setControlState("jump", true);

    setTimeout(() => {
        bot.setControlState("jump", false);
    }, 1000);

    return;
  }

  if (message.toLowerCase() === "!companion stop") {
    bot.clearControlStates();
    bot.chat("Bot has stopped");
  } 
});

// returns the first mob in the player's line of sight
function findMobInPlayerSight(player: Entity): Entity | null {
  const eyePosition = player.position.offset(0, player.height, 0);
  const viewDirection = playerViewDirection(player.pitch, player.yaw);

  const candidates = Object.values(bot.entities)
    .filter((entity) => entity.type === "mob")
    .map((entity) => {
      const toEntity = entity.position.offset(0, entity.height / 2, 0).minus(eyePosition);
      const distanceAlongView = toEntity.dot(viewDirection);
      const distanceFromView = toEntity.minus(viewDirection.scaled(distanceAlongView)).norm();
      const hitRadius = Math.max(0.6, entity.width / 2);

      return { entity, distanceAlongView, distanceFromView, hitRadius };
    })
    .filter(({ distanceAlongView, distanceFromView, hitRadius }) =>
      distanceAlongView > 0 && distanceAlongView <= 64 && distanceFromView <= hitRadius
    )
    .sort((a, b) => a.distanceAlongView - b.distanceAlongView);

  return candidates[0]?.entity ?? null;
}

function playerViewDirection(pitch: number, yaw: number) {
  const cosPitch = Math.cos(pitch);
  return new Vec3(-Math.sin(yaw) * cosPitch, Math.sin(pitch), -Math.cos(yaw) * cosPitch);
}
