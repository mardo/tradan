/**
 * Risk manager — enforces hard safety limits before any order is placed.
 *
 * Checks performed:
 *  1. Kill switch
 *  2. Max leverage cap
 *  3. Stop-loss on open position
 *  4. Cooldown between consecutive trades
 */

import { DriftClient, convertToNumber, PRICE_PRECISION, BASE_PRECISION } from "@drift-labs/sdk";
import { config } from "./config";

const COOLDOWN_MS = 60_000; // minimum time between opening new positions
let lastTradeTs = 0;

export type RiskCheckResult =
  | { ok: true }
  | { ok: false; reason: string };

/**
 * Get the current mark price for a perp market as a plain number.
 */
export function getMarkPrice(client: DriftClient, marketIndex: number): number {
  const market = client.getPerpMarketAccount(marketIndex);
  if (!market) throw new Error(`Perp market ${marketIndex} not found`);

  const oracleData = client.getOracleDataForPerpMarket(marketIndex);
  return convertToNumber(oracleData.price, PRICE_PRECISION);
}

/**
 * Returns the base amount currently held in a perp position (positive = long,
 * negative = short, 0 = no position).
 */
export function getPositionBase(
  client: DriftClient,
  marketIndex: number
): number {
  const user = client.getUser(config.driftSubAccountId);
  const position = user.getPerpPosition(marketIndex);
  if (!position) return 0;
  return convertToNumber(position.baseAssetAmount, BASE_PRECISION);
}

/**
 * Estimate current notional leverage for the sub-account.
 * (totalPerp notional / totalCollateral)
 */
export function estimateLeverage(client: DriftClient): number {
  try {
    const user = client.getUser(config.driftSubAccountId);
    const leverage = user.getLeverage();
    // getLeverage() returns leverage * 10_000
    return leverage.toNumber() / 10_000;
  } catch {
    return 0;
  }
}

/**
 * Run all risk checks. Returns { ok: true } if it's safe to trade.
 */
export function checkRisk(
  client: DriftClient,
  marketIndex: number,
  entryPrice: number | null
): RiskCheckResult {
  // 1. Kill switch
  if (config.killSwitch) {
    return { ok: false, reason: "kill switch active" };
  }

  // 2. Leverage cap
  const leverage = estimateLeverage(client);
  if (leverage > config.maxLeverage) {
    return {
      ok: false,
      reason: `leverage ${leverage.toFixed(2)}x exceeds max ${config.maxLeverage}x`,
    };
  }

  // 3. Stop-loss check on existing position
  if (entryPrice !== null) {
    const currentPrice = getMarkPrice(client, marketIndex);
    const positionBase = getPositionBase(client, marketIndex);

    if (positionBase > 0) {
      // Long position — stop loss triggers when price drops too far
      const drawdown = (entryPrice - currentPrice) / entryPrice;
      if (drawdown >= config.stopLossPct) {
        return {
          ok: false,
          reason: `stop loss hit — long entry ${entryPrice.toFixed(4)}, current ${currentPrice.toFixed(4)}, drawdown ${(drawdown * 100).toFixed(2)}%`,
        };
      }
    } else if (positionBase < 0) {
      // Short position — stop loss triggers when price rises too far
      const drawup = (currentPrice - entryPrice) / entryPrice;
      if (drawup >= config.stopLossPct) {
        return {
          ok: false,
          reason: `stop loss hit — short entry ${entryPrice.toFixed(4)}, current ${currentPrice.toFixed(4)}, drawup ${(drawup * 100).toFixed(2)}%`,
        };
      }
    }
  }

  // 4. Cooldown
  const elapsed = Date.now() - lastTradeTs;
  if (lastTradeTs > 0 && elapsed < COOLDOWN_MS) {
    return {
      ok: false,
      reason: `cooldown — ${((COOLDOWN_MS - elapsed) / 1000).toFixed(0)}s remaining`,
    };
  }

  return { ok: true };
}

/**
 * Record that a trade was just placed (resets cooldown timer).
 */
export function recordTrade(): void {
  lastTradeTs = Date.now();
}
