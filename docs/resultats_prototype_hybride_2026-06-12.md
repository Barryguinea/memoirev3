# Résultats initiaux du prototype mémoire final

> **Document historique remplacé.** Ces résultats correspondent à la première
> fusion `OR`. La campagne corrigée et la validation SLS sont rapportées dans
> `resultats_hybrides_corriges_2026-06-12.md`.

Date d'exécution: 12 juin 2026.

## Protocole

- 11 vaches admissibles du corpus IceQube 2023;
- 10 événements physiquement distincts par vache, soit 110 événements;
- événements placés après la baseline de 60 %;
- trois profils HYPO, trois profils INSTABILITE;
- deux contrôles négatifs brefs;
- deux confondants adversariaux: activité de type œstrus et hypoactivité non
  locomotrice;
- métriques attribuables après soustraction de l'exécution propre;
- cinq configurations définies avant l'analyse complète.

## Résultats

| Configuration | Détection HYPO | Détection INSTABILITE | Détection positive globale | Alertes contrôles | Alertes confondants | IoU >= 0,20 | Fond/vache-jour |
|---|---:|---:|---:|---:|---:|---:|---:|
| Référence | 54,5 % | 42,4 % | 48,5 % | 4,5 % | 72,7 % | 27,3 % | 0,999 |
| Persistance 3 h | 48,5 % | 42,4 % | 45,5 % | 4,5 % | 54,5 % | 28,8 % | 0,839 |
| Seuils stricts | 51,5 % | 39,4 % | 45,5 % | 4,5 % | 68,2 % | 28,8 % | 0,964 |
| Agrégation 4 h | 51,5 % | 36,4 % | 43,9 % | 4,5 % | 63,6 % | 25,8 % | 0,830 |
| Strict + cooldown 24 h | 48,5 % | 24,2 % | 36,4 % | 4,5 % | 54,5 % | 25,8 % | 0,607 |

## Lecture honnête

1. Le prototype rejette généralement le pic capteur isolé et l'exercice bref,
   mais un cas sur vingt-deux de contrôle déclenche encore une alerte.
2. L'ajout de la branche INSTABILITE ne produit pas actuellement une meilleure
   performance globale que le détecteur HYPO de version antérieure.
3. La fusion `OR` augmente fortement la charge de fond.
4. L'activité de type œstrus reste fréquemment confondue avec l'instabilité.
5. L'hypoactivité non locomotrice reste fréquemment confondue avec la branche
   HYPO, ce qui est attendu sans information clinique ou contextuelle.
6. Le relèvement uniforme des seuils ne résout pas le problème.
7. Le système hybride ne doit pas encore devenir la méthode principale du
   mémoire ni être décrit comme une amélioration validée.

## Sorties reproductibles

- `data/validation/hybrid_sensitivity_full/sensitivity_summary.csv`;
- `data/validation/hybrid_sensitivity_full/events_*.csv`;
- `logs/hybrid_sensitivity_full_2026-06-12.log`.

## Décision recommandée

Conserver mémoire final comme branche expérimentale. La prochaine amélioration doit
porter sur la représentation de l'instabilité et l'ajout de contexte externe,
pas sur une recherche extensive de seuils sur les mêmes profils synthétiques.
Sans progrès mesurable, version antérieure reste scientifiquement plus défendable.
