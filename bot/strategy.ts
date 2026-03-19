/**
 * Tradan — Drift + Ranger Strategy Bot
 *
 * Entry point. Run with:
 *   pnpm ts-node bot/strategy.ts
 *
 * Architecture:
 *   1. Connect to Drift
 *   2. Run loop: fetch price → compute signal → risk check → place/cancel orders
 *   3. Graceful shutdown on SIGINT / SIGTERM
 */

import "dotenv/config";
import {
  DriftClient,
  PositionDirection,
  OrderType,
  MarketType,
  BASE_PRECISION,
  PRICE_PRECISION,
  convertToNumber,
  getMarketOrderParams,
} from "@drift-labs/sdk";
import BN from "bn.js";

import { config } from "./config";
import { initDriftClient, teardownDriftClient, DriftConnection } from "./drift-client";
import {
  getMarkPrice,
  getPositionBase,
  checkRisk,
  recordTrade,
} from "./risk-manager";

// ============================================================
// STATE
// ============================================================

let running = true;
let entryPrice: number | null = null;

// ============================================================
// BASIC STRATEGY  (FOR TESTING / PLUMBING ONLY)
// This gives a tradeable signal so you can verify the full
// stack is wired up end-to-end.  Replace with your real logic
// in realStrategy() below.
// ============================================================

function basicStrategy(price: number): "long" | "short" | "none" {
  // Intentionally trivial: alternate direction based on price parity.
  const rounded = Math.round(price);
  if (rounded % 2 === 0) return "long";
  return "short";
}

// ============================================================
// TODO: IMPLEMENT REAL STRATEGY HERE
// ============================================================

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function realStrategy(_price: number, _history: number[]): "long" | "short" | "none" {
  // Replace the line below with your real signal logic.
  // Inputs you might want:
  //   - price history (moving averages, momentum, etc.)
  //   - funding rate
  //   - open interest
  //   - on-chain order-book data via Drift DLOB
  throw new Error("realStrategy not yet implemented — use basicStrategy for now");
}

// ============================================================
// ORDER HELPERS
// ============================================================

async function openPosition(
  client: DriftClient,
  direction: "long" | "short",
  marketIndex: number
): Promise<void> {
  const positionDirection =
    direction === "long" ? PositionDirection.LONG : PositionDirection.SHORT;

  // BASE_PRECISION for SOL-PERP is 1e9
  const baseAmount = new BN(config.baseAssetAmount * 1e9);

  const orderParams = getMarketOrderParams({
    marketIndex,
    direction: positionDirection,
    baseAssetAmount: baseAmount,
    marketType: MarketType.PERP,
  });

  console.log(
    `[strategy] placing ${direction} market order — ${config.baseAssetAmount} base on market ${marketIndex}`
  );

  const sig = await client.placePerpOrder(orderParams);
  console.log(`[strategy] order placed — tx: ${sig}`);

  entryPrice = getMarkPrice(client, marketIndex);
  recordTrade();
}

async function closePosition(
  client: DriftClient,
  marketIndex: number
): Promise<void> {
  console.log(`[strategy] closing position on market ${marketIndex}`);
  const sig = await client.closePosition(marketIndex);
  console.log(`[strategy] position closed — tx: ${sig}`);
  entryPrice = null;
  recordTrade();
}

// ============================================================
// MAIN LOOP
// ============================================================

async function loop(dc: DriftConnection): Promise<void> {
  const { client } = dc;
  const marketIndex = config.marketIndex;

  while (running) {
    try {
      const price = getMarkPrice(client, marketIndex);
      const positionBase = getPositionBase(client, marketIndex);
      const hasPosition = positionBase !== 0;

      console.log(
        `[loop] price=${price.toFixed(4)}  position=${positionBase.toFixed(4)}  leverage=${getLeverageStr(client)}`
      );

      // --- Risk checks first ---
      const risk = checkRisk(client, marketIndex, entryPrice);
      if (!risk.ok) {
        const reason = (risk as { ok: false; reason: string }).reason;
        console.warn(`[risk] blocked: ${reason}`);

        // Close position if stop-loss triggered
        if (reason.startsWith("stop loss hit") && hasPosition) {
          await closePosition(client, marketIndex);
        }

        await sleep(config.loopIntervalMs);
        continue;
      }

      // --- Compute signal ---
      const signal = basicStrategy(price);
      // TODO: swap to: const signal = realStrategy(price, priceHistory);

      if (signal === "none") {
        console.log("[strategy] signal: none — holding");
        await sleep(config.loopIntervalMs);
        continue;
      }

      // --- Already in opposite direction? Close first ---
      const inLong = positionBase > 0;
      const inShort = positionBase < 0;

      if ((signal === "short" && inLong) || (signal === "long" && inShort)) {
        console.log("[strategy] reversing position");
        await closePosition(client, marketIndex);
      }

      // --- Open if not already in the right direction ---
      const alreadyAligned =
        (signal === "long" && inLong) || (signal === "short" && inShort);

      if (!alreadyAligned) {
        await openPosition(client, signal, marketIndex);
      } else {
        console.log(`[strategy] already ${signal} — holding`);
      }
    } catch (err) {
      console.error("[loop] error:", err);
    }

    await sleep(config.loopIntervalMs);
  }
}

// ============================================================
// UTILITIES
// ============================================================

function getLeverageStr(client: DriftClient): string {
  try {
    const leverage = client.getUser(config.driftSubAccountId).getLeverage();
    return `${(leverage.toNumber() / 10_000).toFixed(2)}x`;
  } catch {
    return "n/a";
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================
// ENTRY POINT
// ============================================================

async function main(): Promise<void> {
  console.log("=== Tradan Strategy Bot ===");
  console.log(`env:          ${config.driftEnv}`);
  console.log(`market:       ${config.marketIndex}`);
  console.log(`position size:${config.baseAssetAmount}`);
  console.log(`max leverage: ${config.maxLeverage}x`);
  console.log(`stop loss:    ${(config.stopLossPct * 100).toFixed(1)}%`);
  console.log(`loop interval:${config.loopIntervalMs}ms`);
  console.log("===========================");

  if (config.killSwitch) {
    console.warn("[WARN] KILL_SWITCH is active — bot will not trade");
  }

  const dc = await initDriftClient();

  // Graceful shutdown
  const shutdown = async (signal: string) => {
    console.log(`\n[shutdown] received ${signal}`);
    running = false;
    await teardownDriftClient(dc);
    process.exit(0);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  await loop(dc);
}

main().catch((err) => {
  console.error("[fatal]", err);
  process.exit(1);
});
