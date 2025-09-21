import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================
# Tunables
# =============================
EPS = 0.01                 # deadband for trend classification (1 percentage point)
ALERT_THRESHOLD = 0.10     # 10% alert threshold
EMERGENCY_THRESHOLD = 0.15 # 15% emergency threshold (optional override; see logic)
PLOT_ALERTS = True         # set False if you don’t want figures

# =============================
# Classifiers
# =============================
def classify(delta: float, eps: float = EPS) -> str:
    if pd.isna(delta):
        return "No Data"
    if delta > eps:
        return "Increasing"
    if delta < -eps:
        return "Decreasing"
    return "Stable"

def classify_with_ci(observed, lower, upper, eps: float = EPS) -> str:
    if pd.isna(observed) or pd.isna(lower) or pd.isna(upper):
        return "No Data"
    if lower > observed + eps:
        return "Increasing"
    if upper < observed - eps:
        return "Decreasing"
    return "Stable"

# =============================
# Core pipeline (runs once per target)
# =============================
def run_trend_pipeline_for_target(
    target_name: str,
    file_stub: str,
    col_prefix: str = ""
):
    """
    Inputs (CSV):
      data/Smoothed_{file_stub}_prediction_hb_1.csv
      data/Smoothed_{file_stub}_prediction_hb_2.csv
      data/Smoothed_{file_stub}_prediction_hb_3.csv

    Output CSVs:
      data/clean_trend_long_with_CI{_risk}.csv
      data/clean_trend_wide_with_CI{_risk}.csv
    """

    # === Load data ===
    hb_1 = pd.read_csv(f"data/Smoothed_{file_stub}_prediction_hb_1.csv")
    hb_2 = pd.read_csv(f"data/Smoothed_{file_stub}_prediction_hb_2.csv")
    hb_3 = pd.read_csv(f"data/Smoothed_{file_stub}_prediction_hb_3.csv")

    for df in (hb_1, hb_2, hb_3):
        df["time_period"] = pd.to_datetime(df["time_period"])
        df["Ward"] = df["Ward"].astype(str).str.strip()

    # Column mappings (handle "" vs "risk_" prefixes)
    obs_col  = f"{col_prefix}observed"
    p1_col   = f"{col_prefix}pred_1mo";  lb1_col = f"{col_prefix}lower_bound_1mo";  ub1_col = f"{col_prefix}upper_bound_1mo"
    p2_col   = f"{col_prefix}pred_2mo";  lb2_col = f"{col_prefix}lower_bound_2mo";  ub2_col = f"{col_prefix}upper_bound_2mo"
    p3_col   = f"{col_prefix}pred_3mo";  lb3_col = f"{col_prefix}lower_bound_3mo";  ub3_col = f"{col_prefix}upper_bound_3mo"

    # Normalize column names we actually use
    hb_1 = hb_1.rename(columns={obs_col: "observed", p1_col: "pred_1mo", lb1_col: "lower_bound_1mo", ub1_col: "upper_bound_1mo"})
    hb_2 = hb_2.rename(columns={p2_col: "pred_2mo",   lb2_col: "lower_bound_2mo",   ub2_col: "upper_bound_2mo"})
    hb_3 = hb_3.rename(columns={p3_col: "pred_3mo",   lb3_col: "lower_bound_3mo",   ub3_col: "upper_bound_3mo"})

    # Keep only needed cols
    obs = hb_1[["Ward", "County", "time_period", "observed"]].copy()

    # Shift predictions back so the forecast made at t aligns to its target month t
    def shift_predictions(df, months_back, horizon_label):
        return (
            df.assign(time_period=df["time_period"] - pd.DateOffset(months=months_back))[
                ["Ward", "time_period", f"pred_{horizon_label}", f"lower_bound_{horizon_label}", f"upper_bound_{horizon_label}"]
            ]
        )

    pred_1mo = shift_predictions(hb_1, 1, "1mo")
    pred_2mo = shift_predictions(hb_2, 2, "2mo")
    pred_3mo = shift_predictions(hb_3, 3, "3mo")

    # Merge all horizons
    merged = (
        obs.merge(pred_1mo, on=["Ward", "time_period"], how="left")
           .merge(pred_2mo, on=["Ward", "time_period"], how="left")
           .merge(pred_3mo, on=["Ward", "time_period"], how="left")
           .sort_values(["Ward", "time_period"])
    )

    # Keep rows where we have observed and the 3-month horizon (since alerts depend on 3mo)
    merged = merged.dropna(subset=["observed", "pred_3mo", "lower_bound_3mo", "upper_bound_3mo"])

    # Trends (simple diff vs observed)
    merged["trend_1mo"] = (merged.get("pred_1mo") - merged["observed"]).apply(classify) if "pred_1mo" in merged else "No Data"
    merged["trend_2mo"] = (merged.get("pred_2mo") - merged["observed"]).apply(classify) if "pred_2mo" in merged else "No Data"
    merged["trend_3mo"] = (merged["pred_3mo"] - merged["observed"]).apply(classify)

    # CI-based trends (the ones we actually use for alerts: 3mo)
    merged["trend_with_CI_1mo"] = merged.apply(lambda r: classify_with_ci(r["observed"], r.get("lower_bound_1mo"), r.get("upper_bound_1mo")), axis=1)
    merged["trend_with_CI_2mo"] = merged.apply(lambda r: classify_with_ci(r["observed"], r.get("lower_bound_2mo"), r.get("upper_bound_2mo")), axis=1)
    merged["trend_with_CI_3mo"] = merged.apply(lambda r: classify_with_ci(r["observed"], r["lower_bound_3mo"], r["upper_bound_3mo"]), axis=1)

    # Observed 2-month trend (simple and fast)
    merged["observed_slope_2mo"] = merged.groupby("Ward")["observed"].diff()
    merged["observed_trend_2mo"] = merged["observed_slope_2mo"].apply(classify)

    # -------- Long format (only the columns we use downstream) --------
    def to_long(h_label: str):
        cols = {
            f"trend_{h_label}": "predicted_trend",
            f"pred_{h_label}": "predicted_value",
            f"lower_bound_{h_label}": "lower_bound",
            f"upper_bound_{h_label}": "upper_bound",
            f"trend_with_CI_{h_label}": "predicted_trend_CI",
        }
        keep = ["Ward", "time_period"] + list(cols.keys())
        df = merged[[c for c in keep if c in merged.columns]].rename(columns=cols)
        df["horizon"] = h_label
        return df

    trend_long = pd.concat([to_long("1mo"), to_long("2mo"), to_long("3mo")], ignore_index=True, sort=False)

    # Attach observed & county once
    trend_long = trend_long.merge(
        merged[["Ward", "County", "time_period", "observed", "observed_trend_2mo"]],
        on=["Ward", "time_period"],
        how="left"
    )

    # Alert flag:
    #  - Only on 3mo horizon rows
    #  - observed >= ALERT_THRESHOLD
    #  - CI-based 3mo trend is Stable/Increasing
    #  - observed 2mo trend is Stable/Increasing
    #  - (optional) emergency override: observed >= EMERGENCY_THRESHOLD
    trend_long["alert_flag"] = False
    is_3mo = trend_long["horizon"].eq("3mo")
    ok_obs = trend_long["observed"].ge(ALERT_THRESHOLD)
    ok_ci  = trend_long["predicted_trend_CI"].isin(["Increasing", "Stable"])
    #ok_obs_trend = trend_long["observed_trend_2mo"].isin(["Increasing", "Stable"])

    #trend_long.loc[is_3mo & ok_obs & ok_ci & ok_obs_trend, "alert_flag"] = True
    trend_long.loc[is_3mo & ok_obs & ok_ci, "alert_flag"] = True


    # Sort & save (long)
    trend_long = trend_long.sort_values(["Ward", "time_period", "horizon"])
    long_path = f"data/clean_trend_long_with_CI{'_risk' if col_prefix=='risk_' else ''}.csv"
    trend_long.to_csv(long_path, index=False)
    print(f"✅ Saved {target_name} trend (long) with CI: {long_path}")

    # === Optional quick plots (only alerted rows, any horizon shown with 3mo logic) ===
    if PLOT_ALERTS:
        alerts_3mo = trend_long[(trend_long["horizon"] == "3mo") & (trend_long["alert_flag"])]
        wards = alerts_3mo["Ward"].unique()
        if len(wards) > 0:
            colors = plt.cm.get_cmap("tab10", len(wards))

            # Observed only
            fig, ax = plt.subplots(figsize=(14, 6))
            for i, wname in enumerate(wards):
                w = alerts_3mo[alerts_3mo["Ward"] == wname].sort_values("time_period")
                ax.plot(w["time_period"], w["observed"], label=wname, color=colors(i), marker="o", linewidth=2)
            ax.axhline(ALERT_THRESHOLD, linestyle="--", color="red", linewidth=1.5, label=f"{int(ALERT_THRESHOLD*100)}% Threshold")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f"Observed {target_name} in Wards That Triggered Alerts (3mo rule)", fontsize=14)
            ax.set_ylabel(target_name); ax.set_xlabel("Time")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(); plt.show()

    # -------- Wide format (keep it simple & correct) --------
    # 1) Pivot horizon-specific fields
    base = trend_long.sort_values(["Ward", "time_period", "horizon"]).drop_duplicates(["Ward", "time_period", "horizon"])
    pv_value = base.pivot(index=["Ward","time_period"], columns="horizon", values="predicted_value")
    pv_value.columns = [f"predicted_value_{c}" for c in pv_value.columns]

    pv_trend = base.pivot(index=["Ward","time_period"], columns="horizon", values="predicted_trend")
    pv_trend.columns = [f"predicted_trend_{c}" for c in pv_trend.columns]

    pv_trend_ci = base.pivot(index=["Ward","time_period"], columns="horizon", values="predicted_trend_CI")
    pv_trend_ci.columns = [f"predicted_trend_CI_{c}" for c in pv_trend_ci.columns]

    pv_lb = base.pivot(index=["Ward","time_period"], columns="horizon", values="lower_bound")
    pv_lb.columns = [f"lower_bound_{c}" for c in pv_lb.columns]

    pv_ub = base.pivot(index=["Ward","time_period"], columns="horizon", values="upper_bound")
    pv_ub.columns = [f"upper_bound_{c}" for c in pv_ub.columns]

    # 2) Observed / county / observed_trend_2mo (one row per ward-month)
    observed_block = (base[["Ward","County","time_period","observed","observed_trend_2mo"]]
                      .drop_duplicates(["Ward","time_period"])
                      .set_index(["Ward","time_period"]))

    # 3) Alert flag — ensure we carry the 3mo horizon result correctly
    alert_3mo = base[base["horizon"]=="3mo"][["Ward","time_period","alert_flag"]].set_index(["Ward","time_period"])

    # 4) Combine
    trend_wide = (
        pd.concat([observed_block, alert_3mo, pv_value, pv_trend, pv_trend_ci, pv_lb, pv_ub], axis=1)
          .reset_index()
          .sort_values(["Ward","time_period"])
    )

    # 5) Consecutive alert streaks (based on 3mo alert_flag)
    trend_wide["alert_flag"] = trend_wide["alert_flag"].fillna(False).astype(bool)
    trend_wide["consecutive_alerts"] = 0
    for ward, grp in trend_wide.groupby("Ward", sort=False):
        # increasing count within contiguous True blocks
        streak = (grp["alert_flag"].astype(int)
                  .groupby((~grp["alert_flag"]).cumsum()).cumsum())
        trend_wide.loc[grp.index, "consecutive_alerts"] = streak.values

    wide_path = f"data/clean_trend_wide_with_CI{'_risk' if col_prefix=='risk_' else ''}.csv"
    trend_wide.to_csv(wide_path, index=False)
    print(f"✅ Saved {target_name} trend (wide) with CI: {wide_path}")

# =============================
# Run for both targets
# =============================
# 1) Wasting prevalence (no prefix)
run_trend_pipeline_for_target(
    target_name="Wasting Prevalence",
    file_stub="wasting",
    col_prefix=""
)
# 2) Wasting risk (risk_ prefix)
run_trend_pipeline_for_target(
    target_name="Wasting Risk",
    file_stub="wasting_risk",
    col_prefix="risk_"
)



# compress_figures.py
from PIL import Image
from pathlib import Path

src = Path("assets/figures")   # put your images here
src.mkdir(parents=True, exist_ok=True)

for p in list(src.glob("*.png")) + list(src.glob("*.jpg")) + list(src.glob("*.jpeg")):
    out = p.with_suffix(".webp")
    if out.exists() and out.stat().st_mtime >= p.stat().st_mtime:
        continue
    img = Image.open(p).convert("RGB")
    w, h = img.size
    max_w = 1200
    if w > max_w:
        h = int(h * max_w / w)
        img = img.resize((max_w, h), Image.LANCZOS)
    img.save(out, "WEBP", quality=80, method=6)
    print("✓", out.name)
