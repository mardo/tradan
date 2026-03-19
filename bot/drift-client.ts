/**
 * Drift client factory and connection helpers.
 *
 * Handles:
 *  - wallet creation from private key env var
 *  - DriftClient construction + subscription
 *  - graceful teardown
 */

import { Connection, Keypair } from "@solana/web3.js";
import { DriftClient, Wallet } from "@drift-labs/sdk";
import { config } from "./config";

export type DriftConnection = {
  client: DriftClient;
  wallet: Keypair;
  connection: Connection;
};

/**
 * Parse the BOT_PRIVATE_KEY env var into a Keypair.
 * Accepts a JSON array of numbers (from `cat bot-wallet.json`).
 */
function loadWallet(): Keypair {
  const raw = config.botPrivateKey.trim();
  const bytes = JSON.parse(raw) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(bytes));
}

/**
 * Create and subscribe a DriftClient.
 * Returns the client, the raw keypair, and the Connection.
 */
export async function initDriftClient(): Promise<DriftConnection> {
  const wallet = loadWallet();
  const connection = new Connection(config.rpcUrl, "confirmed");

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const driftWallet = new Wallet(wallet as any);

  const client = new DriftClient({
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    connection: connection as any,
    wallet: driftWallet,
    env: config.driftEnv,
    activeSubAccountId: config.driftSubAccountId,
    accountSubscription: {
      type: "websocket",
      resubTimeoutMs: 30_000,
    },
  });

  console.log(`[drift] connecting to ${config.driftEnv} …`);
  const ok = await client.subscribe();
  if (!ok) {
    throw new Error("DriftClient subscription failed");
  }

  console.log(`[drift] connected — authority: ${wallet.publicKey.toBase58()}`);
  return { client, wallet, connection };
}

/**
 * Tear down the DriftClient cleanly.
 */
export async function teardownDriftClient(dc: DriftConnection): Promise<void> {
  await dc.client.unsubscribe();
  console.log("[drift] unsubscribed");
}
