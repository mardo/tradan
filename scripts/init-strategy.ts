/**
 * Initialize a Drift strategy adapter inside a Ranger vault.
 *
 * Prerequisites:
 *   1. Vault already created via https://vaults.ranger.finance/create
 *   2. RANGER_VAULT_ADDRESS set in .env
 *   3. DRIFT_STRATEGY_ADDRESS set in .env  (Drift adapter address)
 *   4. DRIFT_ADAPTOR_PROGRAM set in .env   (Drift adapter program ID)
 *
 * Usage:
 *   pnpm ts-node scripts/init-strategy.ts
 *
 * After running, save the printed addresses to SETUP.md.
 *
 * NOTE: See client-scripts/ for the full reference implementation with all
 *       Drift-specific remaining accounts.
 */

import "dotenv/config";
import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import { VoltrClient } from "@voltr/vault-sdk";

function loadKeypairFromEnv(): Keypair {
  const raw = process.env.BOT_PRIVATE_KEY ?? "";
  if (!raw) throw new Error("BOT_PRIVATE_KEY not set");
  const bytes = JSON.parse(raw) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(bytes));
}

async function main() {
  const rpcUrl = process.env.RPC_URL ?? "https://api.devnet.solana.com";
  const vaultAddress = process.env.RANGER_VAULT_ADDRESS ?? "";
  const strategyAddress = process.env.DRIFT_STRATEGY_ADDRESS ?? "";
  const adaptorProgram = process.env.DRIFT_ADAPTOR_PROGRAM ?? "";

  if (!vaultAddress) throw new Error("RANGER_VAULT_ADDRESS not set in .env");
  if (!strategyAddress) throw new Error("DRIFT_STRATEGY_ADDRESS not set in .env");
  if (!adaptorProgram) throw new Error("DRIFT_ADAPTOR_PROGRAM not set in .env");

  const connection = new Connection(rpcUrl, "confirmed");
  const wallet = loadKeypairFromEnv();
  const voltrClient = new VoltrClient(connection, wallet);

  const vault = new PublicKey(vaultAddress);
  const strategy = new PublicKey(strategyAddress);

  console.log("Initializing strategy adapter…");
  console.log("  vault:         ", vault.toBase58());
  console.log("  strategy:      ", strategy.toBase58());
  console.log("  adaptorProgram:", adaptorProgram);
  console.log("  manager:       ", wallet.publicKey.toBase58());

  const ix = await voltrClient.createInitializeStrategyIx(
    {},
    {
      payer: wallet.publicKey,
      vault,
      manager: wallet.publicKey,
      strategy,
      adaptorProgram: new PublicKey(adaptorProgram),
      remainingAccounts: [],
    }
  );

  const tx = new Transaction().add(ix);
  const sig = await sendAndConfirmTransaction(connection, tx, [wallet], {
    commitment: "confirmed",
  });

  console.log("\n✅ Strategy initialized!");
  console.log("   tx:", sig);
  console.log("\n👉 Save these in SETUP.md:");
  console.log("   RANGER_VAULT_ADDRESS =", vaultAddress);
  console.log("   DRIFT_STRATEGY_ADDRESS =", strategyAddress);
}

main().catch((err) => {
  console.error("[error]", err);
  process.exit(1);
});
