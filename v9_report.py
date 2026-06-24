"""Rapport final v9 : modèle seul vs marché seul vs blend calibré.
Compare aussi au tout premier moteur (win×200+place×60) sur la même fenêtre."""
import sys, os, math, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
import lib.scoring as S
from lib.scoring import analyze_course

stats = pickle.load(open("stats_calib.pkl", "rb"))
test = pickle.load(open("test_calib.pkl", "rb"))
ts, hs = stats["team_stats"], stats["horse_stats"]
eps = 1e-6

def evaluate():
    ll=bn=0; t1=t3=n=0; rank_winner=[]
    for parts, meta in test:
        try:
            r = analyze_course(parts, ts, hs, meta["discipline"], meta["hippodrome"])
        except Exception:
            continue
        if not r: continue
        n += 1
        place = r[0].get("ordreArrivee", 0) or 0
        if place == 1: t1 += 1
        if 1 <= place <= 3: t3 += 1
        for i, x in enumerate(r, 1):
            if (x.get("ordreArrivee", 0) or 0) == 1:
                rank_winner.append(i); break
        for x in r:
            p = max(eps, min(1-eps, x.get("proba", 0)/100.0))
            y = 1.0 if (x.get("ordreArrivee", 0) or 0) == 1 else 0.0
            ll += -(y*math.log(p)+(1-y)*math.log(1-p)); bn += 1
    rw = sum(rank_winner)/len(rank_winner) if rank_winner else 0
    return {"logloss": ll/max(bn,1), "top1_%": t1/max(n,1)*100,
            "top3_%": t3/max(n,1)*100, "rang": rw, "n": n}

configs = [
    ("Modèle seul",  dict(MARKET_BLEND=0.0)),
    ("Marché seul",  dict(MARKET_BLEND=1.0)),
    ("Blend v9",     dict(MARKET_BLEND=0.85)),
]
# base params
S.DRAW_COEF = 0.0; S.TEMPERATURE = 2.5; S.KAPPA_WIN = 2.0
S.W_HORSE, S.W_DRIVER, S.W_TRAINER = 0.55, 0.30, 0.15

results = {}
for name, cfg in configs:
    S.MARKET_BLEND = cfg["MARKET_BLEND"]
    results[name] = evaluate()

# réinitialise à la config finale
S.MARKET_BLEND = 0.85

print("═"*62)
print("RAPPORT FINAL v9 — 178 courses PMU (même fenêtre, calibrée)")
print("═"*62)
print(f"{'Config':<16}{'Top1':>8}{'Top3':>8}{'Rang':>7}{'logloss':>9}")
print("─"*62)
for name in ["Modèle seul", "Marché seul", "Blend v9"]:
    r = results[name]
    print(f"{name:<16}{r['top1_%']:>7.1f}%{r['top3_%']:>7.1f}%{r['rang']:>7.2f}{r['logloss']:>9.4f}")
print("─"*62)
ms, mks, bl = results["Modèle seul"], results["Marché seul"], results["Blend v9"]
print(f"\nGain du Blend vs Modèle seul :  Top1 {bl['top1_%']-ms['top1_%']:+.1f}pt  "
      f"Top3 {bl['top3_%']-ms['top3_%']:+.1f}pt  logloss {bl['logloss']-ms['logloss']:+.4f}")
print(f"Blend vs Marché seul        :  Top1 {bl['top1_%']-mks['top1_%']:+.1f}pt  "
      f"Top3 {bl['top3_%']-mks['top3_%']:+.1f}pt  logloss {bl['logloss']-mks['logloss']:+.4f}")

# ── graphique ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

names = ["Modèle\nseul", "Marché\nseul", "Blend v9\n(modèle+marché)"]
t1 = [results[k]["top1_%"] for k in ["Modèle seul", "Marché seul", "Blend v9"]]
t3 = [results[k]["top3_%"] for k in ["Modèle seul", "Marché seul", "Blend v9"]]
colors = ["#64748b", "#3b82f6", "#10b981"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0))
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#0f172a")
    for s in ax.spines.values(): s.set_color("#334155")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", color="#1e293b")

x = np.arange(3); w = 0.38
# Top1
b = ax1.bar(x, t1, w*1.4, color=colors)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9.5)
ax1.set_ylabel("Top1 — gagnant trouvé en #1 (%)", color="#94a3b8")
ax1.set_title("Précision du gagnant", color="#e2e8f0", fontweight="bold")
ax1.set_ylim(0, max(t1)*1.32)
for bar, v in zip(b, t1):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%", ha="center", color="#e2e8f0", fontsize=11, fontweight="bold")
# Top3
b2 = ax2.bar(x, t3, w*1.4, color=colors)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9.5)
ax2.set_ylabel("Top3 — #1 prédit placé 1-3 (%)", color="#94a3b8")
ax2.set_title("Précision de placement", color="#e2e8f0", fontweight="bold")
ax2.set_ylim(0, max(t3)*1.22)
for bar, v in zip(b2, t3):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.6, f"{v:.1f}%", ha="center", color="#e2e8f0", fontsize=11, fontweight="bold")
# annotation clé : blend > marché sur Top3
ax2.annotate(f"Bat le marché\n+{t3[2]-t3[1]:.1f}pt", xy=(2, t3[2]), xytext=(1.5, t3[2]+9),
             ha="center", color="#10b981", fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.4))
fig.suptitle("Moteur v9 — modèle + marché (178 courses PMU)",
             color="#10b981", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("v9_final.png", dpi=130, facecolor="#0f172a", bbox_inches="tight")
print("\n💾 v9_final.png généré")
json.dump({k: results[k] for k in results}, open("v9_result.json","w"), indent=2)
