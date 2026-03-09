#!/usr/bin/env python3
"""
Replicate JPM CAIE ETF: European Phoenix Autocallable DCA on Vol-Targeted Index

Two-stage replication:
1. Reconstruct MQUSLVA index (SPX futures + 35% IV target + 6% annual deduction)
2. Simulate weekly DCA into European Phoenix autocallables on the reconstructed index

Data: SPX (CRSP+OptionMetrics), VIX (CBOE WRDS), Risk-free (OptionMetrics)
Author: Agentic Sciences
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json, time

t0 = time.time()
OUT = '/mnt/work/qr33/comewealth'

# ================================================================
# Stage 1: Reconstruct MQUSLVA (Vol Advantage Index)
# ================================================================
# Methodology (from JPM PDF p.15):
# - Track SPX futures
# - Target volatility: 35%
# - Rebalance weekly using IMPLIED volatility (VIX)
# - Leverage = 35% / VIX (capped at reasonable range)
# - Annual interest deduction: 6%

print("=" * 70)
print("STAGE 1: Reconstruct MQUSLVA Vol Advantage Index")
print("=" * 70)

# Load data
spx = pd.read_parquet(f'{OUT}/cache/spx_combined_daily.parquet')
spx['date'] = pd.to_datetime(spx['date'])
spx = spx.set_index('date').sort_index()

vix = pd.read_parquet(f'{OUT}/cache/vix_cboe_daily.parquet')
vix['date'] = pd.to_datetime(vix['date'])
vix = vix.set_index('date').sort_index()

# Merge
df = spx[['spx']].join(vix[['vix_close']], how='inner')
df = df.dropna()
df['spx_ret'] = df['spx'].pct_change()
print(f"Data: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")

# Vol targeting parameters
TARGET_VOL = 0.35  # 35%
ANNUAL_DEDUCTION = 0.06  # 6% p.a.
DAILY_DEDUCTION = ANNUAL_DEDUCTION / 252
MAX_LEVERAGE = 1.5  # Cap leverage
MIN_LEVERAGE = 0.1  # Floor

# Weekly rebalancing: compute leverage each Friday
df['weekday'] = df.index.weekday  # 0=Mon, 4=Fri
df['vix_decimal'] = df['vix_close'] / 100
df['leverage'] = (TARGET_VOL / df['vix_decimal']).clip(MIN_LEVERAGE, MAX_LEVERAGE)

# Only rebalance weekly (carry forward leverage on non-rebalance days)
# Rebalance on Friday (weekday=4), or last available day of the week
df['week'] = df.index.isocalendar().week.values
df['year_week'] = df.index.year * 100 + df['week']

# Mark last day of each week as rebalance day
df['is_rebal'] = False
for yw in df['year_week'].unique():
    mask = df['year_week'] == yw
    last_idx = df.loc[mask].index[-1]
    df.loc[last_idx, 'is_rebal'] = True

# Forward-fill leverage from rebalance days
df['leverage_active'] = np.nan
df.loc[df['is_rebal'], 'leverage_active'] = df.loc[df['is_rebal'], 'leverage']
df['leverage_active'] = df['leverage_active'].ffill()
df['leverage_active'] = df['leverage_active'].fillna(1.0)

# Compute MQUSLVA daily returns
# ret_mquslva = leverage * spx_ret - daily_deduction
df['mquslva_ret'] = df['leverage_active'].shift(1) * df['spx_ret'] - DAILY_DEDUCTION
df['mquslva_ret'] = df['mquslva_ret'].fillna(0)

# Build MQUSLVA index level
df['mquslva'] = (1 + df['mquslva_ret']).cumprod() * 100

# Stats
mq_total = df['mquslva'].iloc[-1] / df['mquslva'].iloc[0] - 1
mq_years = (df.index[-1] - df.index[0]).days / 365.25
mq_ann_ret = (1 + mq_total) ** (1 / mq_years) - 1
mq_ann_vol = df['mquslva_ret'].std() * np.sqrt(252)
mq_sharpe = mq_ann_ret / mq_ann_vol
mq_dd = ((df['mquslva'] - df['mquslva'].cummax()) / df['mquslva'].cummax()).min()

print(f"\nMQUSLVA Reconstruction:")
print(f"  Ann Return:  {mq_ann_ret:.2%}")
print(f"  Ann Vol:     {mq_ann_vol:.2%}")
print(f"  Sharpe:      {mq_sharpe:.2f}")
print(f"  Max DD:      {mq_dd:.2%}")
print(f"  Avg Leverage: {df['leverage_active'].mean():.2f}")

# JPM benchmark (from PDF p.15, as of Nov 2025):
# MQUSLVA: YTD 13.27%, 3Y 29.36%, 5Y 11.53%, 10Y 20.25%
print(f"\n  JPM Reference (10Y p.a.): 20.25%")
print(f"  Our Reconstruction:       {mq_ann_ret:.2%}")

# ================================================================
# Stage 2: European Phoenix Autocallable MC Simulation
# ================================================================
print("\n" + "=" * 70)
print("STAGE 2: European Phoenix Autocallable Monte Carlo")
print("=" * 70)

# Phoenix terms (from JPM PDF p.16):
TENOR_YEARS = 5
LOCKOUT_YEARS = 1
AUTOCALL_BARRIER = 1.00  # 100% of initial
COUPON_BARRIER = 0.60    # 60% of initial
KI_BARRIER = 0.60        # 60% European (maturity only)
ANNUAL_COUPON = 0.1433   # 14.33% p.a.
MONTHLY_COUPON = ANNUAL_COUPON / 12

# Use MQUSLVA realized parameters for simulation
mq_mu = mq_ann_ret
mq_sigma = mq_ann_vol
print(f"MQUSLVA params: mu={mq_mu:.2%}, sigma={mq_sigma:.2%}")

def price_phoenix(S0, mu, sigma, rf, n_paths=100000, seed=42):
    """Price a single European Phoenix autocallable via Monte Carlo."""
    rng = np.random.default_rng(seed)
    dt = 1/252
    n_steps = TENOR_YEARS * 252
    
    # Generate paths
    Z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_ret, axis=1))
    paths = np.column_stack([np.full(n_paths, S0), paths])
    
    # Monthly observation dates (every 21 trading days)
    monthly_days = [21 * m for m in range(1, TENOR_YEARS * 12 + 1)]
    monthly_days = [d for d in monthly_days if d <= n_steps]
    
    lockout_months = LOCKOUT_YEARS * 12
    
    # Track outcomes
    total_coupons = np.zeros(n_paths)
    exit_time = np.full(n_paths, TENOR_YEARS)  # Default: hold to maturity
    autocalled = np.zeros(n_paths, dtype=bool)
    principal_return = np.ones(n_paths)  # 1.0 = full principal
    
    for mi, day in enumerate(monthly_days):
        month_num = mi + 1
        if day > n_steps:
            break
            
        S_t = paths[:, day]
        alive = ~autocalled
        
        # Coupon: pay if S_t >= coupon_barrier * S0
        coupon_eligible = alive & (S_t >= COUPON_BARRIER * S0)
        total_coupons[coupon_eligible] += MONTHLY_COUPON
        
        # Autocall: after lockout, if S_t >= autocall_barrier * S0
        if month_num > lockout_months:
            autocall_hit = alive & (S_t >= AUTOCALL_BARRIER * S0)
            autocalled[autocall_hit] = True
            exit_time[autocall_hit] = day / 252
    
    # Maturity: European knock-in check
    S_T = paths[:, -1]
    at_maturity = ~autocalled
    
    # Knock-in: S_T < KI_barrier * S0 at maturity
    knocked_in = at_maturity & (S_T < KI_BARRIER * S0)
    principal_return[knocked_in] = S_T[knocked_in] / S0  # Lose principal
    
    # Not knocked in at maturity: full principal back
    not_ki = at_maturity & (S_T >= KI_BARRIER * S0)
    principal_return[not_ki] = 1.0
    
    # Total payoff = principal + coupons
    total_payoff = principal_return + total_coupons
    
    # Annualized return for each path
    ann_returns = (total_payoff) ** (1 / exit_time) - 1
    
    # Summary stats
    autocall_rate = autocalled.mean()
    ki_rate = knocked_in.mean()
    avg_coupon = total_coupons.mean() * 100
    avg_exit = exit_time.mean()
    avg_ann_ret = ann_returns.mean()
    
    return {
        'autocall_rate': autocall_rate,
        'ki_rate': ki_rate,
        'avg_total_coupon_pct': avg_coupon,
        'avg_exit_years': avg_exit,
        'avg_ann_return': avg_ann_ret,
        'median_ann_return': np.median(ann_returns),
        'q5_ann_return': np.percentile(ann_returns, 5),
        'q95_ann_return': np.percentile(ann_returns, 95),
        'avg_payoff': total_payoff.mean(),
        'principal_loss_rate': (principal_return < 1.0).mean(),
        'paths': paths,
        'total_coupons': total_coupons,
        'exit_time': exit_time,
        'autocalled': autocalled,
        'ann_returns': ann_returns,
    }

# Run MC with MQUSLVA parameters
rf = 0.04  # Approximate current risk-free
result = price_phoenix(100, mq_mu, mq_sigma, rf, n_paths=200000, seed=42)

print(f"\nMonte Carlo Results (200K paths):")
print(f"  Autocall rate:     {result['autocall_rate']:.1%}  (JPM ref: 97.2%)")
print(f"  Knock-in rate:     {result['ki_rate']:.1%}  (JPM ref: 0.0%)")
print(f"  Avg total coupon:  {result['avg_total_coupon_pct']:.1f}%")
print(f"  Avg exit (years):  {result['avg_exit_years']:.2f}")
print(f"  Avg ann return:    {result['avg_ann_return']:.2%}")
print(f"  Median ann return: {result['median_ann_return']:.2%}")
print(f"  5th pctl return:   {result['q5_ann_return']:.2%}")
print(f"  95th pctl return:  {result['q95_ann_return']:.2%}")
print(f"  Principal loss:    {result['principal_loss_rate']:.2%}")

# ================================================================
# Stage 3: DCA Simulation (Weekly Phoenix Investment)
# ================================================================
print("\n" + "=" * 70)
print("STAGE 3: Weekly DCA Phoenix Portfolio Simulation")
print("=" * 70)

# Use actual MQUSLVA paths for historical simulation
# Start DCA from 2020-06 (simulate CAIE ETF launch equivalent)
dca_start = '2020-06-01'
df_dca = df.loc[dca_start:].copy()

# Every Friday, invest $1 into a new Phoenix structure
# Track all live structures and their value

class PhoenixNote:
    def __init__(self, start_date, S0, coupon_rate=ANNUAL_COUPON):
        self.start_date = start_date
        self.S0 = S0
        self.coupon_rate = coupon_rate
        self.monthly_coupon = coupon_rate / 12
        self.total_coupons = 0
        self.months_elapsed = 0
        self.autocalled = False
        self.knocked_in = False
        self.alive = True
        self.principal = 1.0
        
    def observe_month(self, S_t):
        """Monthly observation."""
        if not self.alive:
            return 0
        self.months_elapsed += 1
        coupon_paid = 0
        
        # Coupon check
        if S_t >= COUPON_BARRIER * self.S0:
            coupon_paid = self.monthly_coupon
            self.total_coupons += coupon_paid
        
        # Autocall check (after 12 months)
        if self.months_elapsed > 12 and S_t >= AUTOCALL_BARRIER * self.S0:
            self.autocalled = True
            self.alive = False
        
        # Maturity check (60 months)
        if self.months_elapsed >= 60 and self.alive:
            if S_t < KI_BARRIER * self.S0:
                self.knocked_in = True
                self.principal = S_t / self.S0
            self.alive = False
        
        return coupon_paid
    
    def mtm_value(self, S_t):
        """Approximate mark-to-market value."""
        if not self.alive:
            return self.principal + self.total_coupons
        
        # Simplified MTM: 
        # Value ≈ principal PV + expected remaining coupons
        remaining_months = 60 - self.months_elapsed
        moneyness = S_t / self.S0
        
        # If well above barriers, close to par + accrued
        if moneyness > 0.8:
            # Principal safe, value near par
            pv = 0.98 + self.total_coupons
            # Add value of expected future coupons (discounted)
            expected_monthly = self.monthly_coupon * min(moneyness / 0.6, 1.0)
            future_coupon_pv = expected_monthly * remaining_months * 0.96
            return pv + future_coupon_pv * 0.3  # Conservative
        else:
            # Near or below barrier, principal at risk
            pv = max(moneyness, COUPON_BARRIER) + self.total_coupons
            return pv

# Run historical DCA
portfolio = []
nav_history = []
coupon_history = []

# Get monthly observation dates
months = pd.date_range(df_dca.index[0], df_dca.index[-1], freq='M')
fridays = df_dca[df_dca.index.weekday == 4].index

investment_count = 0
total_invested = 0
total_coupons_received = 0

for date in df_dca.index:
    S = df_dca.loc[date, 'mquslva']
    
    # Weekly investment on Fridays
    if date in fridays:
        portfolio.append(PhoenixNote(date, S))
        investment_count += 1
        total_invested += 1
    
    # Monthly observation (last trading day of month)
    if date in months or (date.month != df_dca.index[df_dca.index.get_loc(date) + 1].month 
                          if df_dca.index.get_loc(date) + 1 < len(df_dca) else False):
        monthly_coupons = 0
        for note in portfolio:
            if note.alive:
                c = note.observe_month(S)
                monthly_coupons += c
        total_coupons_received += monthly_coupons
        coupon_history.append({'date': date, 'coupons': monthly_coupons})
    
    # Calculate portfolio NAV
    live_notes = [n for n in portfolio if n.alive]
    dead_notes = [n for n in portfolio if not n.alive]
    
    nav_live = sum(n.mtm_value(S) for n in live_notes)
    nav_dead = sum(n.principal + n.total_coupons for n in dead_notes)
    total_nav = nav_live + nav_dead
    
    if total_invested > 0:
        nav_per_unit = total_nav / total_invested
    else:
        nav_per_unit = 1.0
    
    nav_history.append({
        'date': date,
        'nav': total_nav,
        'nav_per_unit': nav_per_unit,
        'live_notes': len(live_notes),
        'total_notes': len(portfolio),
        'total_invested': total_invested,
        'total_coupons': total_coupons_received,
        'mquslva': S,
    })

nav_df = pd.DataFrame(nav_history).set_index('date')

# Performance metrics
if len(nav_df) > 252:
    nav_df['ret'] = nav_df['nav_per_unit'].pct_change()
    total_ret = nav_df['nav_per_unit'].iloc[-1] / nav_df['nav_per_unit'].iloc[0] - 1
    n_yrs = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1/n_yrs) - 1
    ann_vol = nav_df['ret'].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = (nav_df['nav_per_unit'] - nav_df['nav_per_unit'].cummax()) / nav_df['nav_per_unit'].cummax()
    max_dd = dd.min()
else:
    ann_ret = 0; ann_vol = 0; sharpe = 0; max_dd = 0

# Count outcomes
autocalled_count = sum(1 for n in portfolio if n.autocalled)
ki_count = sum(1 for n in portfolio if n.knocked_in)
alive_count = sum(1 for n in portfolio if n.alive)
paying_count = sum(1 for n in portfolio if n.alive and not n.knocked_in)

print(f"\nDCA Portfolio Results:")
print(f"  Period: {nav_df.index[0].date()} to {nav_df.index[-1].date()}")
print(f"  Total notes invested:  {investment_count}")
print(f"  Live notes:            {alive_count}")
print(f"  Autocalled:            {autocalled_count} ({autocalled_count/max(investment_count,1):.1%})")
print(f"  Knocked in:            {ki_count} ({ki_count/max(investment_count,1):.1%})")
print(f"  Total coupons:         {total_coupons_received:.2f}")
print(f"  Coupon yield (total):  {total_coupons_received/max(total_invested,1)*100:.1f}%")
print(f"\n  NAV per unit:          {nav_df['nav_per_unit'].iloc[-1]:.4f}")
print(f"  Ann Return:            {ann_ret:.2%}  (CAIE ref: 14.59%)")
print(f"  Ann Volatility:        {ann_vol:.2%}")
print(f"  Sharpe Ratio:          {sharpe:.2f}")
print(f"  Max Drawdown:          {max_dd:.2%}")

# ================================================================
# Stage 4: Generate Figures
# ================================================================
print("\n" + "=" * 70)
print("STAGE 4: Generate Figures")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. MQUSLVA reconstruction vs SPX
ax = axes[0,0]
spx_norm = df['spx'] / df['spx'].iloc[0] * 100
ax.plot(df.index, df['mquslva'], 'b-', lw=1.5, label='MQUSLVA (reconstructed)')
ax.plot(df.index, spx_norm, 'gray', lw=1, alpha=0.6, label='SPX (normalized)')
ax.set_title('A. MQUSLVA Index Reconstruction')
ax.set_ylabel('Index Level (start=100)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 2. VIX and Leverage
ax = axes[0,1]
ax2 = ax.twinx()
ax.plot(df.index, df['vix_close'], 'r-', alpha=0.5, lw=0.8, label='VIX')
ax2.plot(df.index, df['leverage_active'], 'b-', alpha=0.7, lw=1, label='Leverage')
ax.set_title('B. VIX and Vol-Target Leverage')
ax.set_ylabel('VIX', color='r'); ax2.set_ylabel('Leverage', color='b')
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 3. MC Return Distribution
ax = axes[0,2]
ax.hist(result['ann_returns'] * 100, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(result['avg_ann_return'] * 100, color='red', ls='--', lw=2, 
           label=f'Mean: {result["avg_ann_return"]:.1%}')
ax.axvline(result['median_ann_return'] * 100, color='orange', ls='--', lw=2,
           label=f'Median: {result["median_ann_return"]:.1%}')
ax.set_title('C. Phoenix Annualized Return Distribution (200K MC)')
ax.set_xlabel('Annualized Return (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 4. DCA Portfolio NAV
ax = axes[1,0]
ax.plot(nav_df.index, nav_df['nav_per_unit'], 'b-', lw=1.5)
ax.set_title(f'D. DCA Portfolio NAV per Unit (Ann: {ann_ret:.1%})')
ax.set_ylabel('NAV per Unit')
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 5. Live Notes Count
ax = axes[1,1]
ax.fill_between(nav_df.index, nav_df['live_notes'], color='steelblue', alpha=0.4, label='Live')
ax.plot(nav_df.index, nav_df['total_notes'], 'k-', lw=1, alpha=0.5, label='Total')
ax.set_title('E. Phoenix Notes Count')
ax.set_ylabel('Count')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 6. Cumulative Coupons
ax = axes[1,2]
if coupon_history:
    coup_df = pd.DataFrame(coupon_history).set_index('date')
    ax.fill_between(coup_df.index, coup_df['coupons'].cumsum(), color='green', alpha=0.4)
    ax.set_title(f'F. Cumulative Coupons Received ({total_coupons_received:.1f})')
else:
    ax.set_title('F. Cumulative Coupons')
ax.set_ylabel('Cumulative Coupons')
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle('JPM CAIE Phoenix ETF Replication\nMQUSLVA Vol Targeting + European Phoenix DCA | Agentic Sciences',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/phoenix_dca_replication.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Figure: {OUT}/figures/phoenix_dca_replication.png")

# ================================================================
# Stage 5: Save Results
# ================================================================
nav_df.to_csv(f'{OUT}/results/phoenix_dca_daily.csv')
df[['spx','vix_close','mquslva','leverage_active','mquslva_ret']].to_csv(
    f'{OUT}/results/mquslva_reconstruction.csv')

summary = {
    'mquslva': {
        'period': f"{df.index[0].date()} to {df.index[-1].date()}",
        'ann_return_pct': round(mq_ann_ret * 100, 2),
        'ann_vol_pct': round(mq_ann_vol * 100, 2),
        'sharpe': round(mq_sharpe, 2),
        'max_dd_pct': round(mq_dd * 100, 2),
        'avg_leverage': round(df['leverage_active'].mean(), 2),
        'target_vol': TARGET_VOL,
        'annual_deduction': ANNUAL_DEDUCTION,
    },
    'phoenix_mc': {
        'n_paths': 200000,
        'autocall_rate_pct': round(result['autocall_rate'] * 100, 1),
        'ki_rate_pct': round(result['ki_rate'] * 100, 2),
        'avg_ann_return_pct': round(result['avg_ann_return'] * 100, 2),
        'median_ann_return_pct': round(result['median_ann_return'] * 100, 2),
        'avg_exit_years': round(result['avg_exit_years'], 2),
        'principal_loss_rate_pct': round(result['principal_loss_rate'] * 100, 2),
    },
    'dca_portfolio': {
        'period': f"{nav_df.index[0].date()} to {nav_df.index[-1].date()}",
        'total_notes': investment_count,
        'live_notes': alive_count,
        'autocalled': autocalled_count,
        'knocked_in': ki_count,
        'ann_return_pct': round(ann_ret * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'total_coupons': round(total_coupons_received, 2),
    },
    'caie_benchmark': {
        'ann_return': 14.59,
        'avg_coupon': 14.33,
        'autocall_rate': 97.2,
        'ki_rate': 0.0,
    }
}
with open(f'{OUT}/results/phoenix_dca_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

elapsed = time.time() - t0
print(f"\n⏱️ Total time: {elapsed:.1f}s")
print(f"📁 Results: {OUT}/results/phoenix_dca_*.csv/json")
print(f"📈 Figures: {OUT}/figures/phoenix_dca_replication.png")
