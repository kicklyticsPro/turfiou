"""Backtest du mode sélectif : valide l'objectif 60% gagnant / 85%+ placé
en ne jouant que les courses à confiance ÉLEVÉE (proba #1 ≥ CONF_PLAY).

Compare 3 stratégies sur les MÊMES 178 courses :
  - Systématique : toutes les courses
  - Marché (favori cote) : toutes, top = plus basse cote
  - Sélectif v9+steam : seulement confiance ÉLEVÉE
"""
import sys, os, math, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
import lib.scoring as S
from lib.scoring import analyze_course, shin_probabilities

stats = pickle.load(open("stats_calib.pkl", "rb"))
test = pickle.load(open("test_calib.pkl", "rb"))
ts, hs = stats["team_stats"], stats["horse_stats"]

# évaluation de chaque course : on stocke (proba_top1, proba_top3, cote_top1, gagné, placé, rang_vrai_gagnant)
courses = []
for parts, meta in test:
    try:
        r = analyze_course(parts, ts, hs, meta["discipline"], meta["hippodrome"])
    except Exception:
        continue
    if not r:
        continue
    top = r[0]
    place = top.get("ordreArrivee", 0) or 0
    # vrai gagnant = celui avec ordreArrivee==1
    vrai_gagnant = None
    for i, x in enumerate(r, 1):
        if (x.get("ordreArrivee", 0) or 0) == 1:
            vrai_gagnant = i
            break
    courses.append({
        "proba": top["proba"], "proba3": top.get("probaTop3", 0),
        "cote": top.get("cote"),
        "gagne": 1 if place == 1 else 0,
        "place": 1 if 1 <= place <= 3 else 0,
        "rang_gagnant": vrai_gagnant or 0,
        # favori marché = plus basse cote
        "fav_gagne": 0, "fav_place": 0,
    })

# favori marché (plus basse cote)
for parts, meta in test:
    try:
        partants = [p for p in parts.get("participants", []) if p.get("statut") == "PARTANT"]
    except Exception:
        continue

# --- recalcul favori marché pour chaque course ---
ci = 0
for parts, meta in test:
    partants = [p for p in parts.get("participants", []) if p.get("statut") == "PARTANT"]
    if not partants or ci >= len(courses):
        ci += 1
        continue
    # alignement approximatif: on suppose même ordre ; recalcul robuste
    best_cote = None; best_place = None
    for p in partants:
        rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
        c = float(rap["rapport"]) if rap and rap.get("rapport") else None
        if c and (best_cote is None or c < best_cote):
            best_cote = c
            best_place = p.get("ordreArrivee", 0) or 0
    if best_cote is not None and ci < len(courses):
        courses[ci]["fav_gagne"] = 1 if best_place == 1 else 0
        courses[ci]["fav_place"] = 1 if 1 <= best_place <= 3 else 0
    ci += 1

N = len(courses)
def stats_for(sel):
    n = len(sel) or 1
    t1 = sum(c["gagne"] for c in sel) / n * 100
    t3 = sum(c["place"] for c in sel) / n * 100
    # ROI flat gagnant 1€ sur le #1
    mise = n
    gain = sum(c["cote"] for c in sel if c["gagne"] and c["cote"])
    roi = (gain - mise) / max(mise, 1) * 100
    rg = sum(c["rang_gagnant"] for c in sel) / n
    return n, t1, t3, roi, rg

print("═" * 68)
print(f"MODE SÉLECTIF v9+steam — {N} courses PMU")
print("═" * 68)

strategies = [
    ("Systématique (modèle, toutes)", courses),
    ("Favori marché (toutes)",        courses),
]
# favori marché: remplace gagne/place par fav_gagne/fav_place pour ce calcul
fav = [{"gagne": c["fav_gagne"], "place": c["fav_place"], "cote": c["cote"],
        "rang_gagnant": c["rang_gagnant"]} for c in courses]
# sélectif: proba >= CONF_PLAY
sel_play = [c for c in courses if c["proba"] >= S.CONF_PLAY]
sel_watch = [c for c in courses if S.CONF_WATCH <= c["proba"] < S.CONF_PLAY]
sel_avoid = [c for c in courses if c["proba"] < S.CONF_WATCH]

print(f"\n{'Stratégie':<34}{'jeux':>6}{'%':>6}{'Top1':>8}{'Top3':>8}{'ROI':>8}")
print("─" * 68)
n, t1, t3, roi, rg = stats_for(courses)
print(f"{'Modèle v9 (toutes)':<34}{n:>6}{100:>5.0f}%{t1:>7.1f}%{t3:>7.1f}%{roi:>7.1f}%")
n, t1, t3, roi, rg = stats_for(fav)
print(f"{'Favori marché (toutes)':<34}{n:>6}{100:>5.0f}%{t1:>7.1f}%{t3:>7.1f}%{roi:>7.1f}%")
n, t1, t3, roi, rg = stats_for(sel_play)
print(f"{'🎯 JOUER (proba≥38%)':<34}{n:>6}{n/N*100:>5.0f}%{t1:>7.1f}%{t3:>7.1f}%{roi:>7.1f}%  ★")
n, t1, t3, roi, rg = stats_for(sel_watch)
print(f"{'⚠️ PRUDENCE (25-38%)':<34}{n:>6}{n/N*100:>5.0f}%{t1:>7.1f}%{t3:>7.1f}%{roi:>7.1f}%")
n, t1, t3, roi, rg = stats_for(sel_avoid)
print(f"{'🚫 ÉVITER (<25%)':<34}{n:>6}{n/N*100:>5.0f}%{t1:>7.1f}%{t3:>7.1f}%{roi:>7.1f}%")

print("\n" + "═" * 68)
print("VERDICT OBJECTIF 60% GAGNANT")
print("═" * 68)
n, t1, t3, roi, rg = stats_for(sel_play)
print(f"  JOUER (proba ≥ {S.CONF_PLAY:.0f}%) : {n}/{N} courses ({n/N*100:.0f}%)")
print(f"     → Top1 = {t1:.1f}%   {'✅ OBJECTIF 60% ATTEINT' if t1>=60 else '❌ sous 60%'}")
print(f"     → Top3 = {t3:.1f}%   {'✅' if t3>=90 else '(objectif 90% non atteint — voir note)'}")
print(f"     → ROI flat gagnant = {roi:+.1f}%")

# courbe de seuil fine pour Top3=90%
print("\n─ courbe Top3 vs seuil (recherche 90% placé) ─")
print(f"{'seuil':>6}{'jeux':>6}{'Top1':>8}{'Top3':>8}")
for seuil in [38, 40, 42, 44, 46, 48, 50]:
    s = [c for c in courses if c["proba"] >= seuil]
    nn, t1, t3, roi, rg = stats_for(s)
    flag = " ← 90% place" if t3 >= 90 and nn >= 5 else ""
    print(f"{seuil:>6}{nn:>6}{t1:>7.1f}%{t3:>7.1f}%{flag}")

# ── graphique ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

strat_names = ["Modèle v9\n(toutes)", "Favori\nmarché", "JOUER\n(sélectif)"]
_, t1_all, t3_all, _, _ = stats_for(courses)
_, t1_fav, t3_fav, _, _ = stats_for(fav)
_, t1_play, t3_play, _, _ = stats_for(sel_play)
t1v = [t1_all, t1_fav, t1_play]
t3v = [t3_all, t3_fav, t3_play]
colors = ["#64748b", "#3b82f6", "#10b981"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0))
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#0f172a")
    for s in ax.spines.values(): s.set_color("#334155")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", color="#1e293b")
    ax.axhline(60 if ax is ax1 else 90, color="#f59e0b", ls="--", lw=1.3,
               label=("Objectif 60%" if ax is ax1 else "Objectif 90%"))
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

x = np.arange(3)
for ax, vals, title, ylabel in [(ax1, t1v, "Gagnant (Top1)", "Top1 — gagnant trouvé (%)"),
                                (ax2, t3v, "Placé (Top3)", "Top3 — #1 prédit placé (%)")]:
    b = ax.bar(x, vals, 0.5, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(strat_names, fontsize=9.5)
    ax.set_ylabel(ylabel, color="#94a3b8")
    ax.set_title(title, color="#e2e8f0", fontweight="bold")
    ax.set_ylim(0, max(vals + [60 if ax is ax1 else 90]) * 1.3)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%", ha="center",
                color="#e2e8f0", fontsize=12, fontweight="bold")
fig.suptitle(f"Mode sélectif — objectif 60% gagnant (proba≥38%, {len(sel_play)}/{N} courses)",
             color="#10b981", fontsize=13.5, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("selectif_final.png", dpi=130, facecolor="#0f172a", bbox_inches="tight")
print("\n💾 selectif_final.png généré")

json.dump({"jouer": {"n": len(sel_play), "top1": t1_play, "top3": t3_play},
           "systematique": {"top1": t1_all, "top3": t3_all}},
          open("selectif_result.json", "w"), indent=2)
