"""Paramètres communs et paramètres du comparateur IF historique.

Tous les seuils configurables sont centralisés ici pour que les notebooks,
``app.py`` et les scripts partagent les mêmes valeurs de référence.
"""

# ======================
# Intervalles et features
# ======================
DEFAULT_INTERVAL = "15T"  # 15 minutes - correspond aux données brutes
DEFAULT_WINDOW_BASELINE = 24  # fenêtre du rolling robust z-score

# Note: Intervalles disponibles dans l'UI
# - 15T : 15 minutes (recommandé si données brutes = 15 min)
# - 30T : 30 minutes
# - 1H  : 1 heure
# - 2H  : 2 heures
# - 4H  : 4 heures

# ======================
# Isolation Forest (détection)
# ======================
DEFAULT_CONTAMINATION = 0.06
DEFAULT_BASELINE_RATIO = 0.6  # fit IF sur la partie baseline (evite l'effet quota trop rigide)
DEFAULT_RANDOM_STATE = 42

# ======================
# Règles métier (épisodes)
# ======================
DEFAULT_PERSIST_HOURS = 7      # durée de la fenêtre glissante du comparateur IF
DEFAULT_ALERT_MIN = 2          # min anomalies dans fenêtre
DEFAULT_MIX_MODE = "MIX"       # "MIX" ou "IF-ONLY"
DEFAULT_MIX_RATE_THR = 0.24    # taux minimum anomalies
DEFAULT_Z_LOW_THR = -2.0       # seuil z-score bas
DEFAULT_Z_HIGH_THR = 2.0       # seuil z-score haut

# ======================
# Comparateur historique
# ======================
DEFAULT_COOLDOWN_HOURS = 12    # période refroidissement notifications
DEFAULT_MI_Z_HIGH_THR = 2.2    # seuil spike Motion Index

# ======================
# Isolation Forest (pré-traitement)
# ======================
DEFAULT_SENSOR_WARMUP_BINS = 3   # bins ignores en debut de serie (stabilisation capteur)

# ======================
# Score de confiance boiterie
# ======================
# Poids empiriques pour lame_confidence (somme = 100).
# Non calibres contre des labels cliniques : le score sert au
# classement relatif, pas a l'interpretation comme probabilite.
CONFIDENCE_W_FAMILY   = 30       # coherence multi-familles
CONFIDENCE_W_ANOMRATE = 30       # taux d'anomalies dans la fenetre
CONFIDENCE_W_MI_SPIKE = 20       # proportion de spikes Motion Index
CONFIDENCE_W_IF_SCORE = 20       # score brut Isolation Forest

# ======================
# Qualité données
# ======================
DEFAULT_COVERAGE_MIN_PCT = 25.0  # % minimum couverture requise
