import "dotenv/config";
import mineflayer from "mineflayer";

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

// mc chat connection
bot.on("chat", (username, message) => {
  if (username === bot.username) {
    return;
  }

  console.log(`<${username}> ${message}`);

  if (message.toLowerCase() === "!companion hello") {
    bot.chat(`Hello, ${username}.`);
  }
  
});
