"""
Tests du moteur de calcul v8 (lib/scoring.py).
Lancables :  python -m pytest tests/test_scoring.py -v
ou simplement :  python tests/test_scoring.py
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib import scoring as S


# ───────────────────────── helpers ─────────────────────────

def b(c, v, p, dw=None, dp=None):
    """bucket avec difficultés optionnelles."""
    d = {"c": c, "v": v, "p": p}
    if dw is not None: d["dw"] = dw
    if dp is not None: d["dp"] = dp
    return d


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def run_all():
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ✅ {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"  ❌ {name}: {e}")
            except Exception as e:
                failures.append((name, repr(e)))
                print(f"  ❌ {name}: {repr(e)}")
    print(f"\n{'='*50}")
    if failures:
        print(f"💥 {len(failures)} échec(s)")
        for n, msg in failures:
            print(f"   - {n}: {msg}")
        sys.exit(1)
    print("🎉 Tous les tests passent")
    return 0


# ─────────────────────── multiplicateurs ───────────────────

def test_win_mult_average_is_one():
    # 12 courses, champ moyen 12 → dw = 12*(1/12) = 1 ; gagné 1 → moyen
    m = S.win_multiplier(12, 1, 1.0)
    assert approx(m, 1.0, 0.01), m

def test_win_mult_dominant_above_one():
    m = S.win_multiplier(20, 10, 20 / 12.0)   # 10 victoires en 20 courses
    assert m > 2.0, m

def test_win_mult_weak_below_one():
    m = S.win_multiplier(30, 0, 30 / 12.0)
    assert m < 1.0, m

def test_win_mult_no_data_is_average():
    assert S.win_multiplier(0, 0, 0) == 1.0

def test_shrinkage_toward_one_small_sample():
    # 1 course, gagnée, champ 12 → dw=1/12. Sans shrinkage m=12 (absurde).
    # Avec κ=1 → m = (1+1)/(0.083+1) ≈ 1.85 (raisonnable, pas 12)
    m = S.win_multiplier(1, 1, 1 / 12.0, kappa=1.0)
    assert 1.5 < m < 2.5, m


# ───────────── normalisation par taille de champ ───────────

def test_field_size_normalization():
    """MÊME taux de victoire brut (6/60 = 10 %), champs de tailles différentes
    → multiplicateurs différents (corrige le biais du système actuel).

    Gagner 10 % dans des champs de 6  = SOUS la moyenne (moyenne = 1/6 ≈ 16,7 %)
    Gagner 10 % dans des champs de 18 = AU-DESSUS de la moyenne (moyenne = 5,6 %)
    """
    mA = S.win_multiplier(60, 6, 60 / 6.0)    # dw = 10  → m = 6/10  = 0.60
    mB = S.win_multiplier(60, 6, 60 / 18.0)   # dw = 3.3 → m = 6/3.3 = 1.80
    assert mA < 1.0, mA            # sous la moyenne dans des petits champs
    assert mB > 1.2, mB            # au-dessus de la moyenne dans des gros champs
    assert mB > mA                 # B mieux noté : mêmes 10 % mais champs plus durs


def test_legacy_formula_cannot_distinguish_fields():
    """L'ancienne formule ignore dw → MÊME score malgré des champs différents."""
    sa = S.legacy_bucket_score(b(60, 6, 18, dw=60 / 6.0))   # champ 6
    sb = S.legacy_bucket_score(b(60, 6, 18, dw=60 / 18.0))  # champ 18
    assert sa == sb               # ancien système : taille de champ ignorée ❌


# ───────────────────── probabilités calibrées ──────────────

def test_shin_removes_overround():
    # cotes avec marge : 2.0 / 2.0 → implicites 0.5+0.5=1.0 (pas de marge)
    p = S.shin_probabilities([2.0, 2.0])
    assert approx(sum(p), 100.0, 0.5)
    assert approx(p[0], 50.0, 1.0)

def test_shin_overround_realistic():
    # bookmaker : 1.8 / 1.8 → implicites 1.11 (marge 11%) → vraies ~50/50 < implicites
    raw = [100 / 1.8, 100 / 1.8]
    p = S.shin_probabilities([1.8, 1.8])
    assert approx(sum(p), 100.0, 0.5)
    assert p[0] < raw[0]          # proba vraie < proba implicite (marge retirée)

def test_shin_handles_missing_odds():
    p = S.shin_probabilities([2.0, None, 4.0])
    assert approx(sum(p), 100.0, 0.5)
    assert p[1] == 0.0
    assert p[0] > p[2]            # cote 2 > cote 4

def test_shin_all_missing():
    assert S.shin_probabilities([None, None]) == [0.0, 0.0]


# ─────────────────────── Harville top 3 ────────────────────

def test_harville_sums_to_expected():
    # 3 chevaux équilibrés → chacun P(top3) ≈ 1 (ils occupent les 3 places)
    p = S.harville_top3([1.0, 1.0, 1.0])
    for x in p:
        assert x > 0.95, p

def test_harville_favorite_higher_top3():
    p = S.harville_top3([5.0, 1.0, 1.0, 1.0])
    assert p[0] > p[1]            # favori a plus de chance d'être top3

def test_harville_large_field():
    p = S.harville_top3([1.0] * 12)
    assert approx(sum(p), 3.0, 0.2)   # 3 chevaux dans le top3 → somme ≈ 3


# ──────────────────────── musique ──────────────────────────

def test_musique_winning_form():
    m = S.musique_form_multiplier("1a 1a 2a 1a")
    assert m > 1.2, m

def test_musique_bad_form():
    m = S.musique_form_multiplier("Da Da 0a 9a")
    assert m < 0.7, m

def test_musique_empty():
    assert S.musique_form_multiplier("") == 1.0
    assert S.musique_form_multiplier(None) == 1.0


# ───────────────────── power score & blend ─────────────────

def test_power_score_average_is_50():
    assert S.power_score(1.0) == 50.0

def test_power_score_monotone():
    assert S.power_score(2.0) > S.power_score(1.0) > S.power_score(0.5)

def test_power_score_bounded():
    assert 0 <= S.power_score(0.001) <= 100
    assert 0 <= S.power_score(1000) <= 100

def test_blend_ignores_none():
    m = S.blend_multipliers([(2.0, 0.5), (None, 0.5)])
    assert approx(m, 2.0)        # seul le présent compte

def test_blend_average_when_empty():
    assert S.blend_multipliers([]) == 1.0


# ════════════════ analyse de course end-to-end ═════════════

def _synth_stats():
    """Stats où le cheval 'GAGNANT' est objectivement le meilleur."""
    # GAGNANT : 40% de victoires en champ 10 ; dw = 20*(1/10)=2, v=8
    horse = {
        "GAGNANT": b(20, 8, 15, dw=20 / 10.0, dp=20 * 3 / 10.0),
        "MOYEN": b(20, 2, 8, dw=20 / 10.0, dp=20 * 3 / 10.0),
        "FAIBLE": b(20, 0, 3, dw=20 / 10.0, dp=20 * 3 / 10.0),
    }
    team = {
        "drivers": {"BON PILOTE": b(60, 12, 30, dw=60 / 12.0, dp=60 * 3 / 12.0)},
        "drivers_short": {},
        "drivers_disc": {}, "drivers_hippo": {},
        "entraineurs": {"BON ENT": b(60, 10, 25, dw=60 / 12.0, dp=60 * 3 / 12.0)},
        "entraineurs_short": {}, "entraineurs_disc": {},
    }
    return {"team_stats": team, "horse_stats": {"global": horse}}


def test_analyze_ranks_best_horse_first():
    stats = _synth_stats()
    parts = {"participants": [
        {"statut": "PARTANT", "nom": "FAIBLE", "driver": "X", "entraineur": "Y",
         "nombreCourses": 20, "nombreVictoires": 0, "nombrePlaces": 3, "musique": "0a"},
        {"statut": "PARTANT", "nom": "GAGNANT", "driver": "BON PILOTE", "entraineur": "BON ENT",
         "nombreCourses": 20, "nombreVictoires": 8, "nombrePlaces": 15, "musule": "1a",
         "musique": "1a"},
        {"statut": "PARTANT", "nom": "MOYEN", "driver": "X", "entraineur": "Y",
         "nombreCourses": 20, "nombreVictoires": 2, "nombrePlaces": 8, "musique": "3a"},
    ]}
    out = S.analyze_course(parts, stats["team_stats"], stats["horse_stats"], "TROT_ATTELE", "VINCENNES")
    assert out[0]["nom"] == "GAGNANT", [r["nom"] for r in out]
    assert out[0]["rang"] == 1

def test_analyze_probabilities_sum_to_100():
    stats = _synth_stats()
    parts = {"participants": [
        {"statut": "PARTANT", "nom": "GAGNANT", "driver": "BON PILOTE", "entraineur": "BON ENT",
         "nombreCourses": 20, "nombreVictoires": 8, "nombrePlaces": 15, "musique": "1a"},
        {"statut": "PARTANT", "nom": "MOYEN", "driver": "X", "entraineur": "Y",
         "nombreCourses": 20, "nombreVictoires": 2, "nombrePlaces": 8, "musique": "3a"},
        {"statut": "PARTANT", "nom": "FAIBLE", "driver": "X", "entraineur": "Y",
         "nombreCourses": 20, "nombreVictoires": 0, "nombrePlaces": 3, "musique": "0a"},
    ]}
    out = S.analyze_course(parts, stats["team_stats"], stats["horse_stats"], "TROT_ATTELE", "VINCENNES")
    s = sum(r["proba"] for r in out)
    assert approx(s, 100.0, 0.5), s
    assert out[0]["proba"] > out[-1]["proba"]
    # champs UI présents
    for k in ("rang", "nom", "driver", "entraineur", "nbCourses", "cote",
              "edge", "ordreArrivee", "scores", "proba", "probaTop3", "fairOdds"):
        assert k in out[0], k
    for k in ("cheval", "driver", "entraineur", "composite"):
        assert k in out[0]["scores"], k

def test_analyze_edge_meaningful():
    stats = _synth_stats()
    parts = {"participants": [
        {"statut": "PARTANT", "nom": "GAGNANT", "driver": "BON PILOTE", "entraineur": "BON ENT",
         "nombreCourses": 20, "nombreVictoires": 8, "nombrePlaces": 15, "musique": "1a",
         "dernierRapportDirect": {"rapport": 3.0}},   # marché sous-estime
        {"statut": "PARTANT", "nom": "FAIBLE", "driver": "X", "entraineur": "Y",
         "nombreCourses": 20, "nombreVictoires": 0, "nombrePlaces": 3, "musique": "0a",
         "dernierRapportDirect": {"rapport": 1.5}},   # marché sur-estime
    ]}
    out = S.analyze_course(parts, stats["team_stats"], stats["horse_stats"], "TROT_ATTELE", "VINCENNES")
    by = {r["nom"]: r for r in out}
    assert by["GAGNANT"]["edge"] > 0     # value bet positif (modèle > marché)
    assert by["FAIBLE"]["edge"] < 0


# ═════════════ backtest synthétique : nouveau vs ancien ═════

def test_backtest_new_beats_legacy():
    """Scénario où la taille de champ décide : un cheval qui gagne 50 % mais
    seulement dans des champs de 3 (faciles) contre un qui gagne 15 % dans des
    champs de 14 (durs). Le "vrai" meilleur est le second (force ajustée).

    Nouveau modèle (normalisé par champ) doit le classer #1 → ~60 % de réussite.
    Ancienne formule (champ ignoré) classe le faux favori #1 → ~40 %.
    """
    import random
    random.seed(42)

    # buckets historiques avec difficultés réelles
    horse = {
        # PETIT : 15 victoires / 30 courses en champ 3 → dw=30/3=10, m=15/10=1.5
        "PETIT": b(30, 15, 24, dw=30 / 3.0, dp=30 * 3 / 3.0),
        # GRAND : 5 victoires / 30 courses en champ 14 → dw=30/14=2.14, m=5/2.14=2.3
        "GRAND": b(30, 5, 20, dw=30 / 14.0, dp=30 * 3 / 14.0),
    }
    team = {"drivers": {}, "drivers_short": {}, "drivers_disc": {},
            "drivers_hippo": {}, "entraineurs": {}, "entraineurs_short": {},
            "entraineurs_disc": {}}
    stats = {"team_stats": team, "horse_stats": {"global": horse}}

    names = ["PETIT", "GRAND"]
    true_p = {"PETIT": 0.40, "GRAND": 0.60}        # GRAND = vrai favori

    new_top1 = legacy_top1 = 0
    N_COURSES = 400
    for _ in range(N_COURSES):
        winner = random.choices(names, weights=[true_p[n] for n in names])[0]
        parts = {"participants": []}
        for nm in names:
            parts["participants"].append({
                "statut": "PARTANT", "nom": nm, "driver": "X", "entraineur": "Y",
                "nombreCourses": 0,           # carrière désactivée → bucket historique seul
                "nombreVictoires": 0, "nombrePlaces": 0, "musique": "",
                "ordreArrivee": 1 if nm == winner else 2,
            })

        out = S.analyze_course(parts, stats["team_stats"], stats["horse_stats"], "TROT_ATTELE", "V")
        if out[0]["nom"] == winner:
            new_top1 += 1

        legacy_ranked = sorted(names, key=lambda n: -(S.legacy_bucket_score(horse[n]) or 0))
        if legacy_ranked[0] == winner:
            legacy_top1 += 1

    new_rate = new_top1 / N_COURSES
    leg_rate = legacy_top1 / N_COURSES
    print(f"      nouveau: {new_rate:.1%}  vs  ancien: {leg_rate:.1%}")
    assert new_rate > leg_rate + 0.10, (new_rate, leg_rate)


if __name__ == "__main__":
    run_all()
