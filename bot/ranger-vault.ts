/**
 * Ranger vault helpers using the Voltr SDK.
 *
 * Covers:
 *  - reading vault state
 *  - building deposit-to-strategy instructions
 *  - building withdraw-from-strategy instructions
 *
 * NOTE: For the hackathon starter the vault is created via the Ranger UI at
 *       https://vaults.ranger.finance/create — this file handles post-creation
 *       programmatic interactions only.
 *
 * NOTE: The Drift adapter requires specific remainingAccounts and program IDs.
 *       See client-scripts/  (voltrxyz/client-scripts) for the full reference
 *       implementation used in production.
 */

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

export type VaultInfo = {
  address: string;
  assetMint: string;
  assetTotalValue: string;
};

/**
 * Create a VoltrClient connected to the given RPC and wallet.
 */
export function createVoltrClient(
  connection: Connection,
  wallet: Keypair
): VoltrClient {
  return new VoltrClient(connection, wallet);
}

/**
 * Fetch and log basic vault information.
 */
export async function fetchVaultInfo(
  voltrClient: VoltrClient,
  vaultAddress: string
): Promise<VaultInfo> {
  const vault = new PublicKey(vaultAddress);
  const account = await voltrClient.fetchVaultAccount(vault);

  const info: VaultInfo = {
    address: vaultAddress,
    assetMint: account.asset.mint.toBase58(),
    assetTotalValue: account.asset.totalValue.toString(),
  };

  console.log("[ranger] vault info:", JSON.stringify(info, null, 2));
  return info;
}

export type StrategyDepositParams = {
  connection: Connection;
  voltrClient: VoltrClient;
  wallet: Keypair;
  vaultAddress: string;
  strategyAddress: string;
  /** Drift strategy adapter program ID */
  adaptorProgram: string;
  /** The vault's asset mint (e.g. USDC) */
  vaultAssetMint: string;
  amount: BN;
  /** Any extra accounts required by the Drift adapter */
  remainingAccounts?: Array<{
    pubkey: PublicKey;
    isSigner: boolean;
    isWritable: boolean;
  }>;
};

/**
 * Deposit assets from the vault idle pool into the Drift strategy adapter.
 *
 * @param amount  Amount in base token decimals (e.g. USDC uses 6 decimals)
 */
export async function depositToStrategy(
  params: StrategyDepositParams
): Promise<string> {
  const {
    connection,
    voltrClient,
    wallet,
    vaultAddress,
    strategyAddress,
    adaptorProgram,
    vaultAssetMint,
    amount,
    remainingAccounts = [],
  } = params;

  const vault = new PublicKey(vaultAddress);
  const strategy = new PublicKey(strategyAddress);

  const ix = await voltrClient.createDepositStrategyIx(
    { depositAmount: amount },
    {
      manager: wallet.publicKey,
      vault,
      vaultAssetMint: new PublicKey(vaultAssetMint),
      strategy,
      assetTokenProgram: TOKEN_PROGRAM_ID,
      adaptorProgram: new PublicKey(adaptorProgram),
      remainingAccounts,
    }
  );

  const tx = new Transaction().add(ix);
  const sig = await sendAndConfirmTransaction(connection, tx, [wallet], {
    commitment: "confirmed",
  });

  console.log(`[ranger] deposited ${amount.toString()} to strategy — tx: ${sig}`);
  return sig;
}

export type StrategyWithdrawParams = {
  connection: Connection;
  voltrClient: VoltrClient;
  wallet: Keypair;
  vaultAddress: string;
  strategyAddress: string;
  adaptorProgram: string;
  vaultAssetMint: string;
  amount: BN;
  remainingAccounts?: Array<{
    pubkey: PublicKey;
    isSigner: boolean;
    isWritable: boolean;
  }>;
};

/**
 * Withdraw assets from the Drift strategy adapter back to the vault idle pool.
 */
export async function withdrawFromStrategy(
  params: StrategyWithdrawParams
): Promise<string> {
  const {
    connection,
    voltrClient,
    wallet,
    vaultAddress,
    strategyAddress,
    adaptorProgram,
    vaultAssetMint,
    amount,
    remainingAccounts = [],
  } = params;

  const vault = new PublicKey(vaultAddress);
  const strategy = new PublicKey(strategyAddress);

  const ix = await voltrClient.createWithdrawStrategyIx(
    { withdrawAmount: amount },
    {
      manager: wallet.publicKey,
      vault,
      vaultAssetMint: new PublicKey(vaultAssetMint),
      strategy,
      assetTokenProgram: TOKEN_PROGRAM_ID,
      adaptorProgram: new PublicKey(adaptorProgram),
      remainingAccounts,
    }
  );

  const tx = new Transaction().add(ix);
  const sig = await sendAndConfirmTransaction(connection, tx, [wallet], {
    commitment: "confirmed",
  });

  console.log(`[ranger] withdrew ${amount.toString()} from strategy — tx: ${sig}`);
  return sig;
}
