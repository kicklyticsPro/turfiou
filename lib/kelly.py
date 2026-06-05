"""
v4 - Kelly Criterion pour le sizing optimal des paris.

Formule : f* = (p × b - (1-p)) / b
où :
  - f* : fraction du capital à miser
  - p : probabilité estimée de gagner
  - b : gain net si on gagne (cote - 1)

On utilise Kelly fractionnaire (1/4 ou 1/2) pour réduire la volatilité.
"""


def kelly_fraction(p, cote, kelly_mult=0.25):
    """
    Retourne la fraction du capital à miser.
    p : proba de gagner (0-1)
    cote : cote européenne (ex: 5.0 = quintuple le mise)
    kelly_mult : 0.25 = quart-Kelly (recommandé, plus conservateur)
    """
    if not p or not cote or cote <= 1:
        return 0
    b = cote - 1  # gain net
    q = 1 - p     # proba perdre
    f = (p * b - q) / b
    f = max(0, f) * kelly_mult
    # On cap à 5% du capital pour éviter les positions trop grosses
    return min(f, 0.05)


def kelly_amount(p, cote, capital, kelly_mult=0.25):
    """Retourne le montant à miser en €."""
    f = kelly_fraction(p, cote, kelly_mult)
    return round(capital * f, 2)


def expected_value(p, cote):
    """EV = p × (cote - 1) - (1 - p). Positif = pari +EV."""
    if not p or not cote:
        return 0
    return p * (cote - 1) - (1 - p)


def expected_roi(p, cote):
    """ROI espéré en % par pari."""
    if not p or not cote:
        return 0
    return expected_value(p, cote) * 100
