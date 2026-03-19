# Tradan — Setup Guide

> **Fill in every section as you complete the steps.  
> This file is the single source of truth for reproducibility, judging, and debugging.**

---

## 1. Prerequisites

| Tool | Min version | Install |
|------|-------------|---------|
| Node.js | 18+ | https://nodejs.org |
| pnpm | 8+ | `npm i -g pnpm` |
| Solana CLI | 1.18+ | https://docs.solanalabs.com/cli/install |
| ts-node | included | via pnpm |

---

## 2. Install dependencies

```bash
pnpm install
```

---

## 3. Create / import wallet

### Option A — generate a new keypair

```bash
solana-keygen new --outfile bot-wallet.json
solana address --keypair bot-wallet.json
```

### Option B — import existing wallet

```bash
# Convert your mnemonic or existing keypair file as needed.
# The bot reads BOT_PRIVATE_KEY as a JSON number array.
cat bot-wallet.json   # outputs: [12,34,56, ...]
```

> ⚠️  **Never commit `bot-wallet.json` or any private key to git.**

### Wallet addresses

| Label | Address |
|-------|---------|
| Bot wallet (authority) | *(fill in after keygen)* |

---

## 4. Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

| Variable | Description | Example |
|----------|-------------|---------|
| `RPC_URL` | Solana RPC endpoint | `https://mainnet.helius-rpc.com/?api-key=xxx` |
| `BOT_PRIVATE_KEY` | JSON array from `cat bot-wallet.json` | `[12,34,56,...]` |
| `DRIFT_ENV` | `devnet` or `mainnet-beta` | `devnet` |
| `RANGER_VAULT_ADDRESS` | Public key of your Ranger vault | *(see §6)* |
| `DRIFT_STRATEGY_ADDRESS` | Drift adapter program registered in Ranger | *(see §7)* |
| `DRIFT_SUBACCOUNT_ID` | Sub-account index (default 0) | `0` |
| `MARKET_INDEX` | Drift perp market index (0 = SOL-PERP) | `0` |
| `BASE_ASSET_AMOUNT` | Trade size in base units (e.g. 1 SOL) | `1` |
| `MAX_LEVERAGE` | Hard leverage cap | `3` |
| `STOP_LOSS_PCT` | Stop-loss fraction (0.03 = 3 %) | `0.03` |
| `LOOP_INTERVAL_MS` | Bot loop period | `15000` |
| `KILL_SWITCH` | Emergency stop: set `true` to halt all trading | `false` |

---

## 5. Devnet setup

```bash
# Point Solana CLI at devnet
solana config set --url https://api.devnet.solana.com

# Airdrop SOL for gas
solana airdrop 2 --keypair bot-wallet.json

# Get devnet USDC from Drift faucet
# https://app.drift.trade/?ref=devnet  (use the "Faucet" button in the UI)
```

---

## 6. Create Ranger Vault

1. Go to **https://vaults.ranger.finance/create**
2. Configure:
   - Name / description
   - Deposit token: **USDC**
   - Management fee, performance fee
   - Redemption period
3. Click **Create Vault** and approve the transaction
4. Save the vault address below

### Vault addresses

| Network | Vault Address | Created at (tx) |
|---------|---------------|-----------------|
| Devnet  | *(fill in)*   | *(fill in)*     |
| Mainnet | *(fill in)*   | *(fill in)*     |

Manage your vault at:
```
https://vaults.ranger.finance/manage/<VAULT_ADDRESS>
```

---

## 7. Initialize Drift Strategy Adapter

The Drift adapter bridges your Ranger vault to Drift's trading accounts.

### Using client-scripts (recommended)

```bash
# Clone the Ranger client-scripts reference repo (already done if you ran setup)
cd client-scripts

# Follow the README — roughly:
#   1. admin-init-vault.ts       (if vault was not yet registered on-chain)
#   2. admin-init-strategies.ts  (registers the Drift adapter with your vault)
#   3. manager-deposit-strategies.ts  (allocates capital)
```

### Or using the init-strategy script in this repo

```bash
# Set DRIFT_STRATEGY_ADDRESS in .env first, then:
pnpm init-strategy
```

### Strategy addresses

| Network | Strategy Address | Init tx |
|---------|-----------------|---------|
| Devnet  | *(fill in)*     | *(fill in)* |
| Mainnet | *(fill in)*     | *(fill in)* |

---

## 8. Fund the strategy

```bash
# Default: 10 USDC
pnpm fund-strategy

# Custom amount:
DEPOSIT_AMOUNT_USDC=100 pnpm fund-strategy
```

---

## 9. Run the bot

```bash
pnpm bot
```

Expected output:

```
=== Tradan Strategy Bot ===
env:          devnet
market:       0
position size:1
max leverage: 3x
stop loss:    3.0%
loop interval:15000ms
===========================
[drift] connecting to devnet …
[drift] connected — authority: <YOUR_PUBKEY>
[loop] price=145.3200  position=0.0000  leverage=n/a
[strategy] signal: long — placing order…
```

---

## 10. Mainnet deployment

1. Create a new vault on mainnet via the Ranger UI
2. Initialize a new strategy adapter (repeat §7 on mainnet)
3. Update `.env`:
   ```
   DRIFT_ENV=mainnet-beta
   RPC_URL=https://mainnet.helius-rpc.com/?api-key=<YOUR_KEY>
   RANGER_VAULT_ADDRESS=<MAINNET_VAULT>
   DRIFT_STRATEGY_ADDRESS=<MAINNET_STRATEGY>
   ```
4. Fund with real capital
5. Run bot: `pnpm bot`

---

## 11. Monitoring

| What | How |
|------|-----|
| Bot logs | stdout / pipe to a log file |
| Positions | https://app.drift.trade (connect your bot wallet) |
| Vault performance | https://vaults.ranger.finance/manage/<VAULT> |
| On-chain activity | Solscan / SolanaFM with your bot wallet address |

---

## 12. Reference repos

Cloned locally for reference (not part of the main codebase):

| Repo | Path | Purpose |
|------|------|---------|
| drift-labs/keeper-bots-v2 | `keeper-bots-v2/` | Bot architecture patterns |
| voltrxyz/client-scripts | `client-scripts/` | Ranger vault + strategy scripts |

---

## 13. Addresses index

*(Keep this table up-to-date as you set things up)*

| Label | Devnet | Mainnet |
|-------|--------|---------|
| Bot wallet | | |
| Ranger vault | | |
| Drift strategy adapter | | |
| Drift sub-account | | |
| Asset mint (USDC) | `4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU` (devnet) | `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` (mainnet) |

---

## 14. Known gotchas

- Free RPC endpoints often throttle or drop WebSocket connections — use a paid provider (Helius, QuickNode) for production.
- Ranger requires the strategy to be initialized **before** depositing — do not skip §7.
- Drift bots must handle reconnects — the SDK `resubTimeoutMs` config handles this.
- Leverage cap enforcement in the risk manager is approximate; Drift's on-chain margin system is the final authority.
- If `KILL_SWITCH=true`, the bot logs but does **not** trade.
