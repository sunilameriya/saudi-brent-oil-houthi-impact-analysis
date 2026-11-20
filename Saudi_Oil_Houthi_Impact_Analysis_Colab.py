# ================================================================
# SAUDI OIL / BRENT PRICE vs HOUTHI ATTACKS
# Six-month daily analytics | Google Colab compatible
#
# IMPORTANT:
# Saudi Aramco official selling prices (OSPs) are monthly.
# This project uses daily Brent spot price ($/barrel) as a market
# benchmark relevant to Saudi crude, NOT a daily Saudi Aramco price.
#
# Data source: FRED series DCOILBRENTEU (EIA Brent spot price)
# ================================================================

# In Google Colab, run:
# !pip -q install pandas numpy matplotlib openpyxl

import os
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# ------------------------- SETTINGS -------------------------
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.DateOffset(months=6)

OUTPUT_DIR = Path("Saudi_Oil_Houthi_Analysis")
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

# FRED CSV endpoint. FRED's Brent spot series is sourced from EIA.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

# ------------------------- EVENT TIMELINE -------------------------
# Event dates and wording are sourced from public reporting.
# "Claimed" means the Houthi claim was not independently confirmed
# in the source cited. Avoid treating every claim as a verified strike.
events = pd.DataFrame([
    {
        "event_date": "2026-07-13",
        "event": "First renewed Houthi attack in Saudi escalation",
        "event_type": "Saudi target / airport",
        "status": "Reported",
        "details": "Renewed campaign began with an intercepted missile attack targeting Abha airport; not an oil facility."
    },
    {
        "event_date": "2026-07-20",
        "event": "Announced blockade of Saudi-linked shipping",
        "event_type": "Shipping threat",
        "status": "Announced",
        "details": "Houthis announced a blockade of Saudi-affiliated shipping through Bab el-Mandeb."
    },
    {
        "event_date": "2026-07-23",
        "event": "Two Saudi oil tankers reportedly targeted",
        "event_type": "Oil shipping",
        "status": "Houthi claim",
        "details": "Houthis said they struck two Saudi oil tankers; reported by Reuters."
    },
    {
        "event_date": "2026-08-04",
        "event": "Faize Noore Oliya sunk",
        "event_type": "Commercial shipping",
        "status": "Reported",
        "details": "Commercial vessel attack reported in the Red Sea; four crew deaths were reported."
    },
    {
        "event_date": "2026-08-11",
        "event": "Tihama vessel attack",
        "event_type": "Commercial shipping",
        "status": "Reported / attribution context",
        "details": "Deadly attack on a ship near Bab el-Mandeb; Houthis later said they attacked a vessel."
    },
    {
        "event_date": "2026-08-24",
        "event": "Last corroborated maritime attack in ACLED reporting window",
        "event_type": "Commercial shipping",
        "status": "Reported",
        "details": "ACLED says no corroborated maritime attacks occurred after 24 August in its cited period."
    },
    {
        "event_date": "2026-09-17",
        "event": "VLCCs avoid Bab el-Mandeb amid Houthi threats",
        "event_type": "Shipping / freight risk",
        "status": "Market reporting",
        "details": "S&P Global reported large crude tankers avoiding the strait and Saudi crude rerouting via the Cape."
    },
    {
        "event_date": "2026-10-23",
        "event": "Brent jumps more than $3 after Houthi attacks",
        "event_type": "Oil price reaction reported",
        "status": "Reuters market report",
        "details": "Reuters reported Brent settled at $116.72, up $3.55 (3.14%) amid supply-disruption concerns after Houthi attacks."
    },
    {
        "event_date": "2026-10-29",
        "event": "Claims of strikes on Ras Tanura, Abqaiq and Petro Rabigh",
        "event_type": "Oil infrastructure",
        "status": "Houthi claim / not independently verified in source",
        "details": "Houthis claimed attacks on oil facilities; Critical Threats said it could not verify the claims at publication."
    },
    {
        "event_date": "2026-11-19",
        "event": "Further Houthi attack campaign reported in Saudi Arabia",
        "event_type": "Energy infrastructure / facilities",
        "status": "Reported claims and official statements",
        "details": "Reuters reported Houthi claims involving the East-West Pipeline and facilities; some refinery claims were not confirmed by Saudi authorities."
    }
])
events["event_date"] = pd.to_datetime(events["event_date"])

# ------------------------- FETCH DAILY PRICE DATA -------------------------
print("Fetching daily Brent spot price data from FRED / EIA...")
raw = pd.read_csv(FRED_URL)

# FRED CSV normally has DATE and DCOILBRENTEU columns.
date_col = "observation_date" if "observation_date" in raw.columns else "DATE"
value_col = "DCOILBRENTEU"

if date_col not in raw.columns or value_col not in raw.columns:
    raise ValueError(
        f"Unexpected FRED CSV columns: {list(raw.columns)}. "
        "Check the FRED download endpoint or update the column names."
    )

prices = raw[[date_col, value_col]].copy()
prices.columns = ["date", "brent_usd_per_barrel"]
prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
prices["brent_usd_per_barrel"] = pd.to_numeric(
    prices["brent_usd_per_barrel"], errors="coerce"
)
prices = prices.dropna().sort_values("date")
prices = prices[
    (prices["date"] >= START_DATE) &
    (prices["date"] <= END_DATE)
].reset_index(drop=True)

if prices.empty:
    raise ValueError("No Brent observations found for the selected six-month period.")

# Daily price features (business days with published observations)
prices["daily_change_usd"] = prices["brent_usd_per_barrel"].diff()
prices["daily_return_pct"] = prices["brent_usd_per_barrel"].pct_change() * 100
prices["abs_return_pct"] = prices["daily_return_pct"].abs()
prices["rolling_vol_5d"] = prices["daily_return_pct"].rolling(5).std()
prices["rolling_vol_20d"] = prices["daily_return_pct"].rolling(20).std()
prices["moving_avg_7d"] = prices["brent_usd_per_barrel"].rolling(7).mean()
prices["moving_avg_30d"] = prices["brent_usd_per_barrel"].rolling(30).mean()
prices["running_peak"] = prices["brent_usd_per_barrel"].cummax()
prices["drawdown_pct"] = (
    prices["brent_usd_per_barrel"] / prices["running_peak"] - 1
) * 100
prices["month"] = prices["date"].dt.strftime("%Y-%m")
prices["weekday"] = prices["date"].dt.day_name()

# Merge event labels onto exact calendar dates; prices may not exist on weekends.
prices["event_today"] = prices["date"].isin(events["event_date"])
prices["event_names"] = prices["date"].map(
    events.set_index("event_date")["event"]
).fillna("")

# Event proximity feature: nearest event within +/- 3 calendar days
event_dates = events["event_date"].sort_values().to_numpy(dtype="datetime64[ns]")
prices["days_from_nearest_event"] = prices["date"].apply(
    lambda d: int(np.min(np.abs((event_dates - np.datetime64(d)).astype("timedelta64[D]").astype(int))))
    if len(event_dates) else np.nan
)

# ------------------------- DAILY EVENT FLAGS / EVENT CALENDAR -------------------------
# Keep all event dates in a separate sheet, including weekend events.
events.to_csv(OUTPUT_DIR / "houthi_event_timeline.csv", index=False)

# Add event flags to price dates for the deliverable daily file.
# Event day marker will only be True if the event date is also a market observation date.
# The event calendar sheet preserves weekend/non-trading event dates.

# ------------------------- HELPER FUNCTIONS -------------------------
def save_chart(filename):
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

def add_event_markers(ax, labels=False):
    visible = events[
        (events["event_date"] >= prices["date"].min()) &
        (events["event_date"] <= prices["date"].max())
    ]
    for _, ev in visible.iterrows():
        ax.axvline(ev["event_date"], color="red", linestyle="--", alpha=0.45, linewidth=1)
        if labels:
            ax.text(
                ev["event_date"], 0.98, ev["event"],
                rotation=90, va="top", ha="right", fontsize=7,
                transform=ax.get_xaxis_transform()
            )

def event_window_return(event_date, before_days=1, after_days=1):
    """Close-to-close percentage change from last price on/before event-pre date
       to first available price on/after event+after_days."""
    event_date = pd.Timestamp(event_date)
    pre_rows = prices[prices["date"] < event_date]
    post_rows = prices[prices["date"] >= event_date + pd.Timedelta(days=after_days)]
    if pre_rows.empty or post_rows.empty:
        return None
    base = pre_rows.iloc[-1]
    target = post_rows.iloc[0]
    return {
        "event_date": event_date,
        "baseline_date": base["date"],
        "baseline_price": base["brent_usd_per_barrel"],
        "observation_date": target["date"],
        "observation_price": target["brent_usd_per_barrel"],
        "change_usd": target["brent_usd_per_barrel"] - base["brent_usd_per_barrel"],
        "change_pct": (
            target["brent_usd_per_barrel"] / base["brent_usd_per_barrel"] - 1
        ) * 100
    }

# ------------------------- GRAPH 1: DAILY BRENT PRICE + EVENTS -------------------------
plt.figure(figsize=(16, 6))
plt.plot(prices["date"], prices["brent_usd_per_barrel"], linewidth=1.4, label="Brent spot")
ax = plt.gca()
add_event_markers(ax, labels=True)
plt.title("1. Daily Brent Crude Price with Houthi-Related Event Markers")
plt.ylabel("USD per barrel")
plt.xlabel("Date")
plt.legend()
save_chart("01_daily_brent_price_events.png")

# ------------------------- GRAPH 2: PRICE + MOVING AVERAGES -------------------------
plt.figure(figsize=(15, 5))
plt.plot(prices["date"], prices["brent_usd_per_barrel"], label="Daily Brent")
plt.plot(prices["date"], prices["moving_avg_7d"], label="7-observation MA")
plt.plot(prices["date"], prices["moving_avg_30d"], label="30-observation MA")
add_event_markers(plt.gca())
plt.title("2. Brent Price with 7-Day and 30-Day Moving Averages")
plt.ylabel("USD per barrel")
plt.legend()
save_chart("02_moving_averages.png")

# ------------------------- GRAPH 3: DAILY PRICE CHANGE -------------------------
plt.figure(figsize=(15, 4))
plt.bar(prices["date"], prices["daily_change_usd"], width=1)
add_event_markers(plt.gca())
plt.axhline(0, color="black", linewidth=0.8)
plt.title("3. Daily Brent Price Change in USD per Barrel")
plt.ylabel("Daily change (USD)")
save_chart("03_daily_price_change.png")

# ------------------------- GRAPH 4: DAILY RETURNS -------------------------
plt.figure(figsize=(15, 4))
plt.bar(prices["date"], prices["daily_return_pct"], width=1)
add_event_markers(plt.gca())
plt.axhline(0, color="black", linewidth=0.8)
plt.title("4. Daily Brent Percentage Returns")
plt.ylabel("Daily return (%)")
save_chart("04_daily_returns.png")

# ------------------------- GRAPH 5: ABSOLUTE RETURN / SHOCKS -------------------------
plt.figure(figsize=(15, 4))
plt.plot(prices["date"], prices["abs_return_pct"], linewidth=1)
add_event_markers(plt.gca())
plt.title("5. Daily Absolute Return – Market Shock Intensity")
plt.ylabel("Absolute daily return (%)")
save_chart("05_absolute_returns.png")

# ------------------------- GRAPH 6: ROLLING VOLATILITY -------------------------
plt.figure(figsize=(15, 5))
plt.plot(prices["date"], prices["rolling_vol_5d"], label="5-observation volatility")
plt.plot(prices["date"], prices["rolling_vol_20d"], label="20-observation volatility")
add_event_markers(plt.gca())
plt.title("6. Rolling Brent Return Volatility")
plt.ylabel("Return standard deviation (%)")
plt.legend()
save_chart("06_rolling_volatility.png")

# ------------------------- GRAPH 7: DRAWDOWN -------------------------
plt.figure(figsize=(15, 4))
plt.fill_between(prices["date"], prices["drawdown_pct"], 0, alpha=0.35)
add_event_markers(plt.gca())
plt.title("7. Brent Drawdown from Six-Month Running High")
plt.ylabel("Drawdown (%)")
save_chart("07_drawdown.png")

# ------------------------- GRAPH 8: MONTHLY AVERAGE PRICE -------------------------
monthly = prices.groupby("month", as_index=False).agg(
    average_price=("brent_usd_per_barrel", "mean"),
    max_price=("brent_usd_per_barrel", "max"),
    min_price=("brent_usd_per_barrel", "min"),
    observations=("brent_usd_per_barrel", "count")
)
plt.figure(figsize=(12, 5))
plt.bar(monthly["month"], monthly["average_price"])
plt.title("8. Monthly Average Brent Price")
plt.ylabel("USD per barrel")
plt.xlabel("Month")
plt.xticks(rotation=35)
save_chart("08_monthly_average.png")

# ------------------------- GRAPH 9: MONTHLY PRICE RANGE -------------------------
plt.figure(figsize=(12, 5))
x = np.arange(len(monthly))
plt.vlines(x, monthly["min_price"], monthly["max_price"], linewidth=4)
plt.scatter(x, monthly["average_price"], label="Monthly average")
plt.xticks(x, monthly["month"], rotation=35)
plt.title("9. Monthly Brent High-Low Range and Average")
plt.ylabel("USD per barrel")
plt.legend()
save_chart("09_monthly_range.png")

# ------------------------- GRAPH 10: RETURN DISTRIBUTION -------------------------
valid_returns = prices["daily_return_pct"].dropna()
plt.figure(figsize=(10, 5))
plt.hist(valid_returns.clip(valid_returns.quantile(0.01), valid_returns.quantile(0.99)), bins=35)
plt.title("10. Distribution of Daily Brent Returns (1st–99th Percentile Clipped)")
plt.xlabel("Daily return (%)")
plt.ylabel("Frequency")
save_chart("10_return_distribution.png")

# ------------------------- GRAPH 11: RETURNS BY WEEKDAY -------------------------
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
weekday_groups = [
    prices.loc[prices["weekday"] == day, "daily_return_pct"].dropna()
    for day in weekday_order
]
plt.figure(figsize=(10, 5))
plt.boxplot(weekday_groups, tick_labels=weekday_order, showfliers=False)
plt.title("11. Daily Brent Return Distribution by Weekday")
plt.ylabel("Daily return (%)")
save_chart("11_weekday_returns.png")

# ------------------------- GRAPH 12: PRICE CHANGE VS DAILY RETURN -------------------------
plt.figure(figsize=(8, 6))
plt.scatter(prices["brent_usd_per_barrel"], prices["daily_return_pct"], alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("12. Brent Price Level vs Daily Percentage Return")
plt.xlabel("Brent price (USD/barrel)")
plt.ylabel("Daily return (%)")
save_chart("12_price_vs_return.png")

# ------------------------- GRAPH 13: RETURN VS ROLLING VOLATILITY -------------------------
plt.figure(figsize=(8, 6))
plt.scatter(prices["rolling_vol_20d"], prices["abs_return_pct"], alpha=0.7)
plt.title("13. 20-Day Volatility vs Absolute Daily Return")
plt.xlabel("20-observation volatility (%)")
plt.ylabel("Absolute daily return (%)")
save_chart("13_volatility_vs_shock.png")

# ------------------------- GRAPH 14: CUMULATIVE PRICE CHANGE -------------------------
prices["cumulative_change_pct"] = (
    prices["brent_usd_per_barrel"] / prices["brent_usd_per_barrel"].iloc[0] - 1
) * 100
plt.figure(figsize=(15, 4))
plt.plot(prices["date"], prices["cumulative_change_pct"])
add_event_markers(plt.gca())
plt.axhline(0, color="black", linewidth=0.8)
plt.title("14. Cumulative Brent Price Change from Start of Period")
plt.ylabel("Cumulative change (%)")
save_chart("14_cumulative_change.png")

# ------------------------- GRAPH 15: EVENT WINDOW PRICE PATHS -------------------------
# Normalize each event's nearest available pre-event close to 100.
plt.figure(figsize=(13, 6))
for _, ev in events.iterrows():
    event_date = ev["event_date"]
    subset = prices[
        (prices["date"] >= event_date - pd.Timedelta(days=5)) &
        (prices["date"] <= event_date + pd.Timedelta(days=10))
    ].copy()
    pre = subset[subset["date"] < event_date]
    if len(pre) and len(subset) >= 3:
        base = pre.iloc[-1]["brent_usd_per_barrel"]
        plt.plot(
            (subset["date"] - event_date).dt.days,
            subset["brent_usd_per_barrel"] / base * 100,
            alpha=0.65,
            label=event_date.strftime("%d %b")
        )
plt.axvline(0, color="red", linestyle="--", label="Event date")
plt.axhline(100, color="black", linewidth=0.8)
plt.title("15. Normalized Brent Price Paths Around Houthi-Related Events")
plt.xlabel("Calendar days from event")
plt.ylabel("Price index (pre-event close = 100)")
plt.legend(fontsize=8, ncol=2)
save_chart("15_event_window_paths.png")

# ------------------------- GRAPH 16: PRICE MOVE AFTER EACH EVENT -------------------------
event_rows = []
for _, ev in events.iterrows():
    result = event_window_return(ev["event_date"], after_days=1)
    if result:
        result["event"] = ev["event"]
        result["status"] = ev["status"]
        event_rows.append(result)
event_study = pd.DataFrame(event_rows)

plt.figure(figsize=(12, 6))
if not event_study.empty:
    labels = event_study["event"].str.slice(0, 34)
    plt.barh(labels, event_study["change_pct"])
    plt.axvline(0, color="black", linewidth=0.8)
plt.title("16. Brent Price Change from Pre-Event Close to First Price at Least 1 Day Later")
plt.xlabel("Price change (%)")
save_chart("16_event_price_changes.png")

# ------------------------- GRAPH 17: EVENT STATUS COUNTS -------------------------
status_counts = events["status"].value_counts()
plt.figure(figsize=(9, 5))
plt.bar(status_counts.index, status_counts.values)
plt.title("17. Event Timeline: Evidence / Attribution Status")
plt.ylabel("Number of timeline entries")
plt.xticks(rotation=25, ha="right")
save_chart("17_event_status_counts.png")

# ------------------------- GRAPH 18: DAILY PRICE + EVENT-DAY LABELS -------------------------
plt.figure(figsize=(15, 5))
plt.plot(prices["date"], prices["brent_usd_per_barrel"], label="Brent spot")
for _, ev in events.iterrows():
    if prices["date"].min() <= ev["event_date"] <= prices["date"].max():
        nearest = prices.iloc[(prices["date"] - ev["event_date"]).abs().argsort()[:1]]
        if not nearest.empty:
            row = nearest.iloc[0]
            plt.annotate(
                ev["event"][:28],
                xy=(row["date"], row["brent_usd_per_barrel"]),
                xytext=(0, 12),
                textcoords="offset points",
                rotation=45,
                fontsize=7
            )
plt.title("18. Brent Price with Nearest Trading-Day Event Annotations")
plt.ylabel("USD per barrel")
plt.legend()
save_chart("18_event_annotations.png")

# ------------------------- GRAPH 19: PRICE VS EVENT-PROXIMITY WINDOW -------------------------
prices["near_event_3d"] = prices["days_from_nearest_event"] <= 3
near = prices.loc[prices["near_event_3d"], "daily_return_pct"].dropna()
far = prices.loc[~prices["near_event_3d"], "daily_return_pct"].dropna()
plt.figure(figsize=(8, 5))
plt.boxplot([near, far], tick_labels=["Within 3 days of event", "More than 3 days away"], showfliers=False)
plt.title("19. Daily Return Distribution Near vs Away from Event Dates")
plt.ylabel("Daily return (%)")
save_chart("19_near_vs_far_event.png")

# ------------------------- GRAPH 20: DAILY PRICE WITH MONTHLY MEAN -------------------------
plt.figure(figsize=(15, 5))
plt.plot(prices["date"], prices["brent_usd_per_barrel"], alpha=0.55, label="Daily Brent")
monthly_map = monthly.set_index("month")["average_price"]
prices["monthly_average"] = prices["month"].map(monthly_map)
plt.plot(prices["date"], prices["monthly_average"], linewidth=2.5, label="Monthly average")
add_event_markers(plt.gca())
plt.title("20. Daily Brent Price vs Monthly Average with Event Markers")
plt.ylabel("USD per barrel")
plt.legend()
save_chart("20_daily_vs_monthly_average.png")

# ------------------------- EVENT STUDY TABLES -------------------------
# Multiple horizons, using last available pre-event close and first available
# published price at/after event+horizon. Weekend / holiday gaps are explicit.
horizon_rows = []
for _, ev in events.iterrows():
    pre = prices[prices["date"] < ev["event_date"]]
    if pre.empty:
        continue
    base = pre.iloc[-1]
    for days in [1, 3, 5, 10]:
        post = prices[prices["date"] >= ev["event_date"] + pd.Timedelta(days=days)]
        if post.empty:
            continue
        obs = post.iloc[0]
        horizon_rows.append({
            "event_date": ev["event_date"],
            "event": ev["event"],
            "status": ev["status"],
            "baseline_date": base["date"],
            "baseline_brent_usd_bbl": base["brent_usd_per_barrel"],
            "horizon_calendar_days": days,
            "observed_date": obs["date"],
            "observed_brent_usd_bbl": obs["brent_usd_per_barrel"],
            "change_usd_bbl": obs["brent_usd_per_barrel"] - base["brent_usd_per_barrel"],
            "change_pct": (
                obs["brent_usd_per_barrel"] / base["brent_usd_per_barrel"] - 1
            ) * 100
        })

event_horizons = pd.DataFrame(horizon_rows)
event_horizons.to_csv(OUTPUT_DIR / "event_study_horizons.csv", index=False)

# ------------------------- SUMMARY -------------------------
summary = pd.DataFrame({
    "Metric": [
        "Benchmark",
        "Source",
        "Period start",
        "Period end",
        "Number of published daily observations",
        "First observed Brent price (USD/barrel)",
        "Latest observed Brent price (USD/barrel)",
        "Overall change (%)",
        "Six-month high (USD/barrel)",
        "Six-month low (USD/barrel)",
        "Average daily absolute return (%)",
        "Maximum daily increase (USD/barrel)",
        "Maximum daily decrease (USD/barrel)"
    ],
    "Value": [
        "Europe Brent Spot Price FOB",
        "EIA data via FRED series DCOILBRENTEU",
        str(prices["date"].min().date()),
        str(prices["date"].max().date()),
        len(prices),
        prices["brent_usd_per_barrel"].iloc[0],
        prices["brent_usd_per_barrel"].iloc[-1],
        (prices["brent_usd_per_barrel"].iloc[-1] /
         prices["brent_usd_per_barrel"].iloc[0] - 1) * 100,
        prices["brent_usd_per_barrel"].max(),
        prices["brent_usd_per_barrel"].min(),
        prices["abs_return_pct"].mean(),
        prices["daily_change_usd"].max(),
        prices["daily_change_usd"].min()
    ]
})
summary.to_csv(OUTPUT_DIR / "key_summary.csv", index=False)

# Daily CSV includes all daily observed price data and derived fields.
prices.to_csv(OUTPUT_DIR / "Saudi_relevant_Brent_daily_6_months.csv", index=False)

# Excel workbook with multiple tabs.
excel_path = OUTPUT_DIR / "Saudi_Oil_Brent_Houthi_6_Month_Analysis.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    prices.to_excel(writer, sheet_name="Daily_Brent_Data", index=False)
    events.to_excel(writer, sheet_name="Houthi_Event_Timeline", index=False)
    event_horizons.to_excel(writer, sheet_name="Event_Study", index=False)
    monthly.to_excel(writer, sheet_name="Monthly_Summary", index=False)
    summary.to_excel(writer, sheet_name="Key_Summary", index=False)

# ------------------------- ZIP DOWNLOAD -------------------------
zip_path = Path("Saudi_Oil_Houthi_Analysis_Package.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            z.write(path, arcname=str(path.relative_to(OUTPUT_DIR)))

print("\n" + "=" * 65)
print("SAUDI OIL / HOUTHI IMPACT ANALYSIS COMPLETED")
print("=" * 65)
print("Benchmark: Daily Brent spot price, USD per barrel")
print("Observations:", len(prices))
print("Date range:", prices["date"].min().date(), "to", prices["date"].max().date())
print("Latest available price date:", prices["date"].max().date())
print("Charts generated:", len(list(CHART_DIR.glob("*.png"))))
print("\nKey summary:")
display(summary)
print("\nEvent study:")
display(event_horizons.head(20))
print("\nFiles created:")
print(excel_path)
print(OUTPUT_DIR / "Saudi_relevant_Brent_daily_6_months.csv")
print(zip_path)

# Google Colab download
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    print("Not running in Google Colab. ZIP saved in current working directory.")
