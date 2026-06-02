# 🚀 Déploiement sur Render.com

Render est **gratuit** pour les apps Flask, **supporte Python nativement**, et déploie automatiquement à chaque `git push`.

## ⚡ Méthode 1 — Déploiement express (5 minutes)

### Étape 1. Pousser le code sur GitHub

```bash
cd turf-analyzer
git init
git add .
git commit -m "Initial commit - Turf Analyzer"

# Crée un repo vide sur https://github.com/new (nommé "turf-analyzer" par ex.)
git branch -M main
git remote add origin https://github.com/TON-USERNAME/turf-analyzer.git
git push -u origin main
```

### Étape 2. Créer le service sur Render

1. Va sur **https://render.com** → **Sign up** (avec GitHub, c'est plus simple)
2. Clique **New +** → **Web Service**
3. **Connect a repository** → autorise Render à accéder à ton GitHub
4. Sélectionne ton repo **turf-analyzer**
5. Render détecte automatiquement le fichier `render.yaml` → **Apply**

C'est tout ! Render va :
- Installer les dépendances (`pip install -r requirements.txt`)
- Lancer le serveur avec `gunicorn`
- Te donner une URL publique du type `https://turf-analyzer-xxxx.onrender.com`

⏱️ Le premier build prend ~2-3 minutes.

---

## 🔧 Méthode 2 — Configuration manuelle (sans render.yaml)

Si tu préfères tout configurer toi-même dans l'interface :

| Champ | Valeur |
|---|---|
| **Name** | `turf-analyzer` |
| **Region** | `Frankfurt` (le plus proche pour la France) |
| **Branch** | `main` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Instance Type** | `Free` |

Et dans **Environment** :
- `PYTHON_VERSION` = `3.11.9`

Clique **Create Web Service**.

---

## ⚠️ Limitations du plan gratuit Render

| Limite | Détail | Impact pour Turf Analyzer |
|---|---|---|
| ⏰ **Sleep après 15 min d'inactivité** | L'app "s'endort" si personne ne l'utilise pendant 15 min | Premier chargement après sleep = **30-60s** (cold start) |
| 🔄 **750h/mois gratuites** | Largement suffisant pour un usage perso | OK |
| 💾 **Pas de disque persistant** | `/tmp` est volatile, reset à chaque redémarrage | Le cache stats team se recalcule (4-30s) après redémarrage |
| 🧠 **512 MB RAM** | Suffisant pour ton app | OK (l'app utilise ~100 MB) |
| 🚀 **CPU partagé** | Plus lent qu'un VPS | Le backtest sur 30 jours peut prendre ~1 min au lieu de 30s |

### Solutions pour éviter le cold start
1. **Plan Starter à 7$/mois** : pas de sleep, disque persistant 1 GB
2. **Service de ping gratuit** comme [cron-job.org](https://cron-job.org) : ping ton URL toutes les 10 min pour la garder éveillée
3. **UptimeRobot** (gratuit) : monitoring + ping

---

## 🐛 Dépannage

### "Application failed to respond"
- Vérifie les logs dans Render : Dashboard → ton service → **Logs**
- Cause fréquente : le `Start Command` n'utilise pas `$PORT`. Render impose d'écouter sur la variable d'env `PORT`.

### Build échoue avec "Could not find a version that satisfies the requirement..."
- Vérifie que `PYTHON_VERSION` = `3.11.9` (numpy nécessite Python ≥ 3.9)

### Le site fonctionne mais l'API PMU est lente
- Normal sur le plan gratuit (CPU partagé). Passe en Starter (7$/mois) ou attends — l'API met en cache.

### Erreur 502 après plusieurs minutes
- Render redémarre l'app si elle ne répond pas en 30s → augmente `--timeout 120` (déjà fait dans `render.yaml`).

---

## 🔄 Mise à jour de l'app

Une fois déployée, à chaque modification :
```bash
git add .
git commit -m "Update"
git push
```
Render redéploie automatiquement en ~2 minutes. ✨

---

## 🎯 Alternatives

| Hébergeur | Gratuit ? | Cold start | Avantage |
|---|---|---|---|
| **Render** ⭐ | ✅ 750h/mois | 30-60s | Le plus simple, render.yaml |
| **Railway** | ✅ 5$ crédit/mois | Aucun | Plus rapide, mais quota limité |
| **Fly.io** | ✅ 3 VMs gratuites | Aucun | Plus technique, CLI requise |
| **PythonAnywhere** | ✅ 1 app gratuite | Aucun | Pas de Git auto-deploy |
| **Heroku** | ❌ Plus de free tier | — | Payant uniquement |

---

## 🔒 Notes légales

- Cette app utilise l'API publique PMU. Vérifie les conditions d'utilisation si tu monétises.
- Affiche bien le disclaimer sur le jeu responsable (déjà en bas de page).
- Ne stocke aucune donnée utilisateur → pas de RGPD à gérer.
