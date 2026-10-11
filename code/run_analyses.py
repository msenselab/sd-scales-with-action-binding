"""Analyses for "Action Binding Gates Serial Dependence in Duration Perception".

Reproduces every statistic and Figures 2-4 of the manuscript and the
Supplementary Tables S1-S6 from the trial-level data in ../data/dataAll.csv:
  - central tendency (signed slope, duration centred at 1.1 s) and the
    following-task onset/offset latencies;
  - sequential dependence index (SDI) from the joint model
    B_n ~ T_n + T_{n-1}, the simple-slope specification B_n ~ T_n-1 for
    comparison, 2 x 2 ANOVAs, Bayes factors, and a linear mixed model;
  - decomposition of a previous following trial into its pre-cue (T1) and
    followed (T2) intervals, and the within-trial T1 effect;
  - participant-level regression on the three preceding durations.

Run from the repository root:
    python code/run_analyses.py
Results are written to results/, figures to figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analysis_utils import add_response_and_bias, add_task_transitions  # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

CENTER = 1.1
R_SCALE = 0.707
lines: list[str] = []
J: dict = {}


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def one_sample(x, label, family=1):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    t, p = stats.ttest_1samp(x, 0)
    d = x.mean() / x.std(ddof=1)
    bf = float(pg.bayesfactor_ttest(t, n, paired=True, r=R_SCALE))
    pc = min(p * family, 1.0)
    res = dict(label=label, mean=x.mean(), sd=x.std(ddof=1), se=x.std(ddof=1) / np.sqrt(n), n=n,
               t=t, df=n - 1, p=p, p_corr=pc, d=d, BF10=bf, BF01=1 / bf)
    say(f"  {label}: M = {x.mean():.3f} (SD {x.std(ddof=1):.3f}), t({n-1}) = {t:.2f}, p {fmt_p(p)}, "
        f"p_corr {fmt_p(pc)}, d = {d:.2f}, BF10 = {bf:.2f} (BF01 = {1/bf:.2f})")
    return res


def paired(a, b, label, family=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    return one_sample(diff, label, family)


def rm_2x2(df, dv, f1, f2, subject="nPar"):
    """2x2 rm-ANOVA plus JZS BF for each 1-df effect via participant-level contrasts."""
    aov = pg.rm_anova(data=df, dv=dv, within=[f1, f2], subject=subject, detailed=True, effsize="ng2")
    wide = df.pivot_table(index=subject, columns=[f1, f2], values=dv)
    l1, l2 = sorted(df[f1].unique()), sorted(df[f2].unique())
    c1 = wide[l1[0]].mean(axis=1) - wide[l1[1]].mean(axis=1)
    c2 = wide.xs(l2[0], axis=1, level=1).mean(axis=1) - wide.xs(l2[1], axis=1, level=1).mean(axis=1)
    cint = (wide[(l1[0], l2[0])] - wide[(l1[0], l2[1])]) - (wide[(l1[1], l2[0])] - wide[(l1[1], l2[1])])
    out = {}
    for src, contrast in zip(aov["Source"], [c1, c2, cint]):
        row = aov[aov["Source"] == src].iloc[0]
        t, _ = stats.ttest_1samp(contrast, 0)
        bf = float(pg.bayesfactor_ttest(t, len(contrast), paired=True, r=R_SCALE))
        out[src] = dict(F=row["F"], df1=int(row["ddof1"]), df2=int(row["ddof2"]), p=row["p_unc"],
                        ng2=row["ng2"], BF10=bf, BF01=1 / bf)
        say(f"  {src}: F({int(row['ddof1'])}, {int(row['ddof2'])}) = {row['F']:.2f}, p {fmt_p(row['p_unc'])}, "
            f"eta2_G = {row['ng2']:.3f}, BF10 = {bf:.2f} (BF01 = {1/bf:.2f})")
    return out


def per_participant_ols(df, formula, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        if len(g) < 10:
            continue
        m = smf.ols(formula, data=g).fit()
        row = dict(zip(group_cols if isinstance(keys, tuple) else [group_cols], keys if isinstance(keys, tuple) else [keys]))
        row.update({k: v for k, v in m.params.items()})
        row["n"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
df = (pd.read_csv(ROOT / "data" / "dataAll.csv")
      .pipe(add_response_and_bias)
      .pipe(add_task_transitions))
df["curDur_c"] = df["curDur"] - CENTER
df["preDur_c"] = df["preDur"] - CENTER
df["relPre"] = df["preDur"] - df["curDur"]
# previous trial's pre-cue (T1) interval when the previous trial was a following trial
df["preT1"] = df["prePreCueDur"]
df["preT1_c"] = df["preT1"] - CENTER
df["curT1_c"] = df["curPreCueDur"] - CENTER

df_valid = df.query("valid == 1").copy()
df_seq = df_valid.query('transition in ["RR","RF","FR","FF"]').copy()
df_follow = df_valid.query('curTask == "following"').dropna(subset=["flwOnLtc", "flwOffLtc"]).copy()

say("# Analysis results (generated by run_analyses.py)\n")
say(f"Trials: {len(df)} total, {len(df_valid)} valid, {len(df_seq)} with a preceding trial in the block.")
say("Transition counts: " + ", ".join(f"{k} = {v}" for k, v in df_seq['transition'].value_counts().sort_index().items()))
say(f"Following trials with T1 available: {df_valid.query('curTask==\"following\"')['curPreCueDur'].notna().sum()} of "
    f"{(df_valid.curTask=='following').sum()}; previous-following trials with T1_{{n-1}} available: "
    f"{df_seq.query('preTask==\"following\"')['preT1'].notna().sum()} of {(df_seq.preTask=='following').sum()}.")
r_t1t2 = np.corrcoef(df_valid.query('curTask=="following"')["curPreCueDur"], df_valid.query('curTask=="following"')["curDur"])[0, 1]
say(f"Correlation between T1 and T2 within following trials: r = {r_t1t2:.3f} (independent randomisation).\n")
J["design"] = dict(n_valid=int(len(df_valid)), n_seq=int(len(df_seq)), r_T1_T2=r_t1t2,
                   transitions=df_seq['transition'].value_counts().to_dict())

# --------------------------------------------------------------------------
# A2. Central tendency: signed, centred CTI; task tests; 2x2 CTI ANOVA
# --------------------------------------------------------------------------
say("## A2. Central tendency (signed slope, T_n centred at 1.1 s)\n")
cti_task = per_participant_ols(df_valid, "bias ~ curDur_c", ["nPar", "curTask"]).rename(columns={"curDur_c": "cti", "Intercept": "b0"})
wide = cti_task.pivot(index="nPar", columns="curTask", values="cti")
J["cti_task"] = {}
J["cti_task"]["reproduction"] = one_sample(wide["reproduction"], "CTI reproduction vs 0", family=3)
J["cti_task"]["following"] = one_sample(wide["following"], "CTI following vs 0", family=3)
J["cti_task"]["diff"] = paired(wide["reproduction"], wide["following"], "CTI reproduction − following", family=3)
b0 = cti_task.pivot(index="nPar", columns="curTask", values="b0")
say(f"  Intercept at 1.1 s (mean bias at the centre): reproduction {b0['reproduction'].mean():.3f} s, following {b0['following'].mean():.3f} s")
J["cti_task"]["intercept_repro"] = b0["reproduction"].mean()
J["cti_task"]["intercept_follow"] = b0["following"].mean()

say("\n2 × 2 rm-ANOVA on transition-level CTI (current task × switch):")
cti_tr = per_participant_ols(df_seq, "bias ~ curDur_c", ["nPar", "transition"]).rename(columns={"curDur_c": "cti"})
cti_tr["responseType"] = cti_tr["transition"].map({"RR": "reproduction", "FR": "reproduction", "RF": "following", "FF": "following"})
cti_tr["taskSwitch"] = cti_tr["transition"].map({"RR": "repeat", "FF": "repeat", "RF": "switch", "FR": "switch"})
J["cti_anova"] = rm_2x2(cti_tr, "cti", "responseType", "taskSwitch")
cti_means = cti_tr.groupby("transition")["cti"].agg(["mean", "std"]).round(3)
say("  Transition means:\n" + cti_means.to_string())
J["cti_transition_means"] = cti_means.to_dict()

# --------------------------------------------------------------------------
# Latency (family = 3)
# --------------------------------------------------------------------------
say("\n## Following-task latencies (per-participant slopes on T_n)\n")
lat = (df_follow.groupby("nPar").apply(lambda g: pd.Series({
    "onset": stats.linregress(g["curDur"], g["flwOnLtc"]).slope,
    "offset": stats.linregress(g["curDur"], g["flwOffLtc"]).slope}), include_groups=False).reset_index())
J["latency"] = {}
J["latency"]["onset"] = one_sample(lat["onset"], "onset-latency slope vs 0", family=3)
J["latency"]["offset"] = one_sample(lat["offset"], "offset-latency slope vs 0", family=3)
J["latency"]["diff"] = paired(lat["onset"], lat["offset"], "onset − offset slope", family=3)
pooled_on = stats.linregress(df_follow["curDur"], df_follow["flwOnLtc"])
pooled_off = stats.linregress(df_follow["curDur"], df_follow["flwOffLtc"])
say(f"  Pooled: onset slope = {pooled_on.slope:.3f} (R2 = {pooled_on.rvalue**2:.3f}, p {fmt_p(pooled_on.pvalue)}); "
    f"offset slope = {pooled_off.slope:.3f} (R2 = {pooled_off.rvalue**2:.3f}, p {fmt_p(pooled_off.pvalue)})")
say(f"  Mean onset latency {df_follow['flwOnLtc'].mean():.3f} s, mean offset latency {df_follow['flwOffLtc'].mean():.3f} s")
J["latency"]["pooled"] = dict(onset_slope=pooled_on.slope, onset_R2=pooled_on.rvalue**2, onset_p=pooled_on.pvalue,
                              offset_slope=pooled_off.slope, offset_R2=pooled_off.rvalue**2, offset_p=pooled_off.pvalue,
                              mean_onset=df_follow['flwOnLtc'].mean(), mean_offset=df_follow['flwOffLtc'].mean())

# --------------------------------------------------------------------------
# Simple-slope SDI (bias ~ preDur; Table S1), ANOVA first, then t-tests
# --------------------------------------------------------------------------
say("\n## Simple-slope SDI: slope of B_n on T_{n-1} without T_n (Supplementary Table S1)\n")
sdi = per_participant_ols(df_seq, "bias ~ preDur_c", ["nPar", "transition"]).rename(columns={"preDur_c": "sdi"})
sdi["responseType"] = sdi["transition"].map({"RR": "reproduction", "FR": "reproduction", "RF": "following", "FF": "following"})
sdi["taskSwitch"] = sdi["transition"].map({"RR": "repeat", "FF": "repeat", "RF": "switch", "FR": "switch"})
say("2 × 2 rm-ANOVA on SDI:")
J["sdi_anova"] = rm_2x2(sdi, "sdi", "responseType", "taskSwitch")
say("One-sample tests (Bonferroni family = 4):")
sdi_w = sdi.pivot(index="nPar", columns="transition", values="sdi")
J["sdi"] = {tr: one_sample(sdi_w[tr], f"SDI {tr}", family=4) for tr in ["RR", "FR", "FF", "RF"]}
J["sdi"]["RR_vs_FR"] = paired(sdi_w["RR"], sdi_w["FR"], "SDI RR − FR")
J["sdi"]["FF_vs_RF"] = paired(sdi_w["FF"], sdi_w["RF"], "SDI FF − RF")

# --------------------------------------------------------------------------
# A6. Relative-duration formulation: bias ~ T_n + T_{n-1}  (== bias ~ T_n + (T_{n-1}-T_n))
# --------------------------------------------------------------------------
say("\n## A6. T_n-adjusted SDI (B_n ~ T_n + T_{n-1}); slope on T_{n-1} == slope on (T_{n-1} − T_n) with T_n in the model\n")
adj = per_participant_ols(df_seq, "bias ~ curDur_c + preDur_c", ["nPar", "transition"]).rename(columns={"preDur_c": "sdi_adj", "curDur_c": "cti_adj"})
adj["responseType"] = adj["transition"].map({"RR": "reproduction", "FR": "reproduction", "RF": "following", "FF": "following"})
adj["taskSwitch"] = adj["transition"].map({"RR": "repeat", "FF": "repeat", "RF": "switch", "FR": "switch"})
say("2 × 2 rm-ANOVA on T_n-adjusted SDI:")
J["sdi_adj_anova"] = rm_2x2(adj, "sdi_adj", "responseType", "taskSwitch")
say("One-sample tests (Bonferroni family = 4):")
adj_w = adj.pivot(index="nPar", columns="transition", values="sdi_adj")
J["sdi_adj"] = {tr: one_sample(adj_w[tr], f"adjusted SDI {tr}", family=4) for tr in ["RR", "FR", "FF", "RF"]}
J["sdi_adj"]["RR_vs_FR"] = paired(adj_w["RR"], adj_w["FR"], "adjusted SDI RR − FR")
J["sdi_adj"]["FF_vs_RF"] = paired(adj_w["FF"], adj_w["RF"], "adjusted SDI FF − RF")
# sanity: identical coefficient from the explicit relative-duration parameterisation
chk = per_participant_ols(df_seq, "bias ~ curDur_c + relPre", ["nPar", "transition"])
say(f"  Check: max |slope(T_n-1) − slope(T_n-1 − T_n)| across participants = {np.abs(adj['sdi_adj'].values - chk['relPre'].values).max():.2e}")
say("Naive slope on (T_{n-1} − T_n) without T_n in the model (confounded with central tendency):")
naive = per_participant_ols(df_seq, "bias ~ relPre", ["nPar", "transition"])
naive_w = naive.pivot(index="nPar", columns="transition", values="relPre")
J["sdi_naive_rel"] = {tr: one_sample(naive_w[tr], f"naive relative slope {tr}") for tr in ["RR", "FR", "FF", "RF"]}

# --------------------------------------------------------------------------
# A1. Prior-trial T1 versus T2
# --------------------------------------------------------------------------
say("\n## A1. Decomposing the previous following trial: T1_{n-1} (pre-cue, not acted upon) versus T2_{n-1} (followed)\n")
prevF = df_seq.query('preTask == "following"').copy()   # FR and FF
dec = per_participant_ols(prevF, "bias ~ curDur_c + preT1_c + preDur_c", ["nPar", "transition"])
dec = dec.rename(columns={"preT1_c": "b_T1prev", "preDur_c": "b_T2prev", "curDur_c": "b_Tn"})
J["decomp"] = {}
for tr in ["FR", "FF"]:
    sub = dec.query("transition == @tr")
    say(f"  {tr} (n trials/participant ≈ {sub['n'].mean():.0f}):")
    J["decomp"][tr] = dict(
        T2prev=one_sample(sub["b_T2prev"], f"    {tr}: slope on T2_(n-1) [followed]"),
        T1prev=one_sample(sub["b_T1prev"], f"    {tr}: slope on T1_(n-1) [pre-cue, unacted]"),
        diff=paired(sub["b_T2prev"], sub["b_T1prev"], f"    {tr}: T2_(n-1) − T1_(n-1)"),
    )
# RR versus FR(T2) in the same T_n-adjusted specification
say("  RR (T_n-adjusted) versus FR slope on T2_(n-1) with T1_(n-1) partialled:")
J["decomp"]["RR_vs_FR_T2"] = paired(adj_w["RR"], dec.query("transition=='FR'").set_index("nPar")["b_T2prev"].reindex(adj_w.index),
                                    "    RR − FR(T2, T1 partialled)")
# does the current trial's own T1 matter for following responses? (RF, FF)
say("  Within-trial: does the current following trial's pre-cue T1_n bias the followed response?")
curF = df_seq.query('curTask == "following"').copy()
wt = per_participant_ols(curF, "bias ~ curDur_c + curT1_c + preDur_c", ["nPar", "transition"])
J["decomp"]["curT1"] = {}
for tr in ["RF", "FF"]:
    sub = wt.query("transition == @tr")
    J["decomp"]["curT1"][tr] = dict(
        curT1=one_sample(sub["curT1_c"], f"    {tr}: slope on current-trial T1_n"),
        T2prev=one_sample(sub["preDur_c"], f"    {tr}: slope on T_(n-1) with T1_n partialled"))
# interaction check: does T1_{n-1} modulate the T2_{n-1} effect in FR? (R1's interference hypothesis)
say("  Interaction T1_(n-1) × T2_(n-1) in FR (does the unacted interval modulate the carried-over effect?):")
inter = per_participant_ols(prevF.query("transition=='FR'"), "bias ~ curDur_c + preT1_c * preDur_c", ["nPar"])
J["decomp"]["FR_interaction"] = one_sample(inter["preT1_c:preDur_c"], "    FR: T1_(n-1) × T2_(n-1)")
# split by |T1-T2| distance in previous trial: does a discrepant T1 attenuate the T2 carry-over?
prevFR = prevF.query("transition=='FR'").copy()
prevFR["t1t2gap"] = np.where((prevFR["preT1"] - prevFR["preDur"]).abs() <= 0.2, "close", "far")
gap = per_participant_ols(prevFR, "bias ~ curDur_c + preDur_c", ["nPar", "t1t2gap"]).pivot(index="nPar", columns="t1t2gap", values="preDur_c")
say("  FR SDI on T2_(n-1) split by |T1_(n-1) − T2_(n-1)| (≤ 0.2 s vs > 0.2 s):")
J["decomp"]["FR_gap"] = dict(close=one_sample(gap["close"], "    close"), far=one_sample(gap["far"], "    far"),
                             diff=paired(gap["close"], gap["far"], "    close − far"))

# --------------------------------------------------------------------------
# A7. Lag analysis: participant-level multiple regression
# --------------------------------------------------------------------------
say("\n## A7. Lags 1–3: B_n ~ T_n + T_{n-1} + T_{n-2} + T_{n-3}, per participant and current task (nT > 3)\n")
# shift over all trials so that lag k is k trials back even when an intervening trial was excluded
lagdf = df.sort_values(["nPar", "nB", "nT"]).copy()
for k in (1, 2, 3):
    lagdf[f"lag{k}"] = lagdf.groupby(["nPar", "nB"])["curDur"].shift(k) - CENTER
lagdf = lagdf.query("valid == 1").dropna(subset=["lag1", "lag2", "lag3"])
lag = per_participant_ols(lagdf, "bias ~ curDur_c + lag1 + lag2 + lag3", ["nPar", "curTask"])
J["lags"] = {}
for task in ["reproduction", "following"]:
    sub = lag.query("curTask == @task")
    say(f"  {task}:")
    J["lags"][task] = {k: one_sample(sub[k], f"    {task} {k}", family=3) for k in ["lag1", "lag2", "lag3"]}
lw = lag.pivot(index="nPar", columns="curTask")
say("  Task comparison:")
for k in ["lag1", "lag2", "lag3"]:
    J["lags"][f"{k}_task_diff"] = paired(lw[(k, "reproduction")], lw[(k, "following")], f"    {k}: reproduction − following")
# pooled correlations as in the submission, for reference
say("  Pooled trial-level correlations (for reference):")
J["lags"]["pooled"] = {}
for task in ["reproduction", "following"]:
    sub = lagdf.query("curTask == @task")
    for k in ["lag1", "lag2", "lag3"]:
        r, p = stats.pearsonr(sub["bias"], sub[k])
        J["lags"]["pooled"][f"{task}_{k}"] = dict(r=r, p=p, n=len(sub))
        say(f"    {task} {k}: r = {r:.3f}, p {fmt_p(p)} (n = {len(sub)})")

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
sns.set_style("ticks")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
                     "legend.fontsize": 11, "savefig.dpi": 300, "savefig.bbox": "tight"})
TASK_COL = {"reproduction": "#C0392B", "following": "#2471A3"}          # red / blue (panel A)
LAT_COL = {"Onset latency": "#7F8C8D", "Offset latency": "#F39C12"}     # grey / amber (panel B)
SWITCH_COL = {"repeat": "#1B7837", "switch": "#762A83"}                 # green / purple (Fig 3, 4)


def points_and_fit(ax, data, x, y, hue, palette, xs=None):
    for lvl, col in palette.items():
        d = data[data[hue] == lvl]
        agg = d.groupby(x)[y].agg(["mean", "sem"]).reset_index()
        ax.errorbar(agg[x], agg["mean"], yerr=agg["sem"], fmt="o", color=col, ms=6, capsize=2, lw=1.4, label=lvl, zorder=3)
        # fit on participant-level means so the line matches the participant-level slope
        per = d.groupby(["nPar", x])[y].mean().reset_index()
        b = np.polyfit(per[x], per[y], 1)
        gx = np.linspace(per[x].min(), per[x].max(), 50) if xs is None else xs
        ax.plot(gx, np.polyval(b, gx), "--", color=col, lw=1.6, alpha=0.9, zorder=2)


# Figure 2 — distinct palettes for A and B
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
pa = df_valid.groupby(["nPar", "curTask", "curDur"], as_index=False)["bias"].mean()
points_and_fit(a1, pa, "curDur", "bias", "curTask", TASK_COL)
a1.axhline(0, color="grey", ls=":", lw=1)
a1.set(xlabel="Target duration (s)", ylabel="Response bias (s)", title="Central tendency", xlim=(0.75, 1.45))
a1.legend(frameon=False)
pl = (df_follow.groupby(["nPar", "curDur"], as_index=False).agg(onset=("flwOnLtc", "mean"), offset=("flwOffLtc", "mean"))
      .melt(id_vars=["nPar", "curDur"], var_name="lat", value_name="latency"))
pl["lat"] = pl["lat"].map({"onset": "Onset latency", "offset": "Offset latency"})
points_and_fit(a2, pl, "curDur", "latency", "lat", LAT_COL)
a2.set(xlabel="Target duration (s)", ylabel="Latency (s)", title="Following-task latencies", xlim=(0.75, 1.45))
a2.legend(frameon=False)
for ax, lab in zip((a1, a2), "AB"):
    ax.text(-0.14, 1.04, lab, transform=ax.transAxes, fontsize=15, fontweight="bold")
sns.despine(fig)
fig.tight_layout()
fig.savefig(FIG / "fig2_central_tendency.png"); fig.savefig(FIG / "fig2_central_tendency.pdf")
plt.close(fig)

# Figure 3 — bias adjusted for central tendency vs relative previous duration
# adjusted bias = bias − participant/transition CTI slope × (T_n − 1.1) − intercept  (removes the central-tendency component)
adj_lookup = adj.set_index(["nPar", "transition"])
df_seq["bias_adj"] = df_seq.apply(lambda r: r["bias"] - adj_lookup.loc[(r["nPar"], r["transition"]), "cti_adj"] * r["curDur_c"]
                                  - adj_lookup.loc[(r["nPar"], r["transition"]), "Intercept"], axis=1)
df_seq["relPre_r"] = df_seq["relPre"].round(1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, task in zip(axes, ["reproduction", "following"]):
    d = df_seq[df_seq.curTask == task]
    pa = d.groupby(["nPar", "taskSwitch", "relPre_r"], as_index=False)["bias_adj"].mean()
    points_and_fit(ax, pa, "relPre_r", "bias_adj", "taskSwitch", SWITCH_COL, xs=np.linspace(-0.6, 0.6, 50))
    ax.axhline(0, color="grey", ls=":", lw=1); ax.axvline(0, color="grey", ls=":", lw=1)
    ax.set(xlabel="Previous − current duration (s)", title=task.capitalize())
    labels = {"reproduction": {"repeat": "RR (repeat)", "switch": "FR (switch)"},
              "following": {"repeat": "FF (repeat)", "switch": "RF (switch)"}}[task]
    h, l = ax.get_legend_handles_labels(); ax.legend(h, [labels[x] for x in l], frameon=False, loc="upper left")
axes[0].set_ylabel("Bias adjusted for central tendency (s)")
for ax, lab in zip(axes, "AB"):
    ax.text(-0.12, 1.04, lab, transform=ax.transAxes, fontsize=15, fontweight="bold")
sns.despine(fig)
fig.tight_layout()
fig.savefig(FIG / "fig3_sequential_dependence.png"); fig.savefig(FIG / "fig3_sequential_dependence.pdf")
plt.close(fig)

# Figure 4 — adjusted SDI by transition (bars + participants)
fig, ax = plt.subplots(figsize=(5, 4))
order = ["RR", "FR", "FF", "RF"]
m = adj.groupby("transition")["sdi_adj"].agg(["mean", "sem"]).reindex(order)
cols = [SWITCH_COL["repeat"], SWITCH_COL["switch"], SWITCH_COL["repeat"], SWITCH_COL["switch"]]
ax.bar(order, m["mean"], yerr=m["sem"], color=cols, alpha=0.85, capsize=3, edgecolor="black", lw=0.6)
rng = np.random.default_rng(1)
for i, tr in enumerate(order):
    y = adj.query("transition == @tr")["sdi_adj"]
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, color="black", s=10, alpha=0.5, zorder=3)
ax.axhline(0, color="grey", ls=":", lw=1)
ax.set(ylabel="Sequential dependence index\n(slope on previous duration, s/s)", xlabel="Transition (previous → current)")
ax.axvline(1.5, color="lightgrey", lw=1)
ax.text(0.5, ax.get_ylim()[1] * 0.95, "current: reproduction", ha="center", fontsize=10)
ax.text(2.5, ax.get_ylim()[1] * 0.95, "current: following", ha="center", fontsize=10)
sns.despine(fig)
fig.tight_layout()
fig.savefig(FIG / "fig4_sdi_by_transition.png"); fig.savefig(FIG / "fig4_sdi_by_transition.pdf")
plt.close(fig)

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o

(OUT / "results.json").write_text(json.dumps(clean(J), indent=1))
(OUT / "results.md").write_text("\n".join(lines) + "\n")
sdi.to_csv(OUT / "sdi_by_participant.csv", index=False)
adj.to_csv(OUT / "sdi_adjusted_by_participant.csv", index=False)
dec.to_csv(OUT / "prior_T1_T2_decomposition.csv", index=False)
cti_tr.to_csv(OUT / "cti_by_transition.csv", index=False)
lag.to_csv(OUT / "lag_regression_by_participant.csv", index=False)
say(f"\nSaved results to {OUT} and figures to {FIG}")
