import "dotenv/config";

function requireEnv(key: string): string {
  const value = process.env[key];
  if (!value) throw new Error(`Missing required env var: ${key}`);
  return value;
}

function optionalEnv(key: string, fallback: string): string {
  return process.env[key] ?? fallback;
}

export const config = {
  rpcUrl: requireEnv("RPC_URL"),
  botPrivateKey: requireEnv("BOT_PRIVATE_KEY"),
  driftEnv: optionalEnv("DRIFT_ENV", "devnet") as "devnet" | "mainnet-beta",
  rangerVaultAddress: optionalEnv("RANGER_VAULT_ADDRESS", ""),
  driftSubAccountId: parseInt(optionalEnv("DRIFT_SUBACCOUNT_ID", "0"), 10),

  // Trading params
  marketIndex: parseInt(optionalEnv("MARKET_INDEX", "0"), 10),
  baseAssetAmount: parseFloat(optionalEnv("BASE_ASSET_AMOUNT", "1")),
  maxLeverage: parseFloat(optionalEnv("MAX_LEVERAGE", "3")),
  stopLossPct: parseFloat(optionalEnv("STOP_LOSS_PCT", "0.03")),
  loopIntervalMs: parseInt(optionalEnv("LOOP_INTERVAL_MS", "15000"), 10),

  killSwitch: optionalEnv("KILL_SWITCH", "false") === "true",
} as const;
