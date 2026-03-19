/**
 * Allocate funds from the Ranger vault idle pool into the Drift strategy.
 *
 * Prerequisites:
 *   Set in .env:
 *     RANGER_VAULT_ADDRESS, DRIFT_STRATEGY_ADDRESS, DRIFT_ADAPTOR_PROGRAM,
 *     VAULT_ASSET_MINT
 *
 * Usage:
 *   DEPOSIT_AMOUNT_USDC=100 pnpm ts-node scripts/fund-strategy.ts
 *
 * DEPOSIT_AMOUNT_USDC defaults to 10 if not set.
 *
 * NOTE: See client-scripts/ for the full reference implementation with all
 *       Drift-specific remaining accounts (Drift user account, spot market etc.)
 */

import "dotenv/config";
import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import { TOKEN_PROGRAM_ID } from "@solana/spl-token";
import { VoltrClient } from "@voltr/vault-sdk";
import BN from "bn.js";

const USDC_DECIMALS = 6;

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
  const vaultAssetMint = process.env.VAULT_ASSET_MINT ?? "";
  const amountUsdc = parseFloat(process.env.DEPOSIT_AMOUNT_USDC ?? "10");

  if (!vaultAddress) throw new Error("RANGER_VAULT_ADDRESS not set in .env");
  if (!strategyAddress) throw new Error("DRIFT_STRATEGY_ADDRESS not set in .env");
  if (!adaptorProgram) throw new Error("DRIFT_ADAPTOR_PROGRAM not set in .env");
  if (!vaultAssetMint) throw new Error("VAULT_ASSET_MINT not set in .env");

  const amountRaw = new BN(Math.floor(amountUsdc * 10 ** USDC_DECIMALS));

  const connection = new Connection(rpcUrl, "confirmed");
  const wallet = loadKeypairFromEnv();
  const voltrClient = new VoltrClient(connection, wallet);

  const vault = new PublicKey(vaultAddress);
  const strategy = new PublicKey(strategyAddress);

  console.log(`Depositing ${amountUsdc} USDC (${amountRaw.toString()} raw) to strategy…`);

  const ix = await voltrClient.createDepositStrategyIx(
    { depositAmount: amountRaw },
    {
      manager: wallet.publicKey,
      vault,
      vaultAssetMint: new PublicKey(vaultAssetMint),
      strategy,
      assetTokenProgram: TOKEN_PROGRAM_ID,
      adaptorProgram: new PublicKey(adaptorProgram),
      remainingAccounts: [],
    }
  );

  const tx = new Transaction().add(ix);
  const sig = await sendAndConfirmTransaction(connection, tx, [wallet], {
    commitment: "confirmed",
  });

  console.log(`✅ Funded — tx: ${sig}`);
}

main().catch((err) => {
  console.error("[error]", err);
  process.exit(1);
});
