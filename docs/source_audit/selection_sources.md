# Sélection des sources de mémoire final

## Base retenue: version antérieure

version antérieure fournit la base active parce qu'elle contient:

- la séparation chronologique baseline/futur;
- les injections exclusivement post-baseline;
- la soustraction de l'exécution propre;
- l'analyse avec la vache comme unité statistique;
- les profils hypoactifs graduels;
- l'analyse de sensibilité et la cohorte IceTag--SLS synchronisée;
- l'interface Streamlit et les tests logiciels actuels;
- le cadrage non diagnostique du mémoire.

## Éléments utiles du premier mémoire

Les sources du premier mémoire ont été consultées lors de la construction de
mémoire final, puis retirées du dépôt final pour éviter de conserver une archive
redondante avec le manuscrit actif. Après audit, seuls les éléments compatibles
avec le cadrage actuel ont été réintégrés dans `memoire/`:

- les explications générales d'Isolation Forest et de LOF;
- les diagrammes d'architecture et éléments de traçabilité;
- les sections opérationnelles, éthiques et économiques;
- certaines figures et certains tableaux de revue de littérature;
- la description détaillée de l'interface.

Le moteur IF + règles est déjà présent dans le code hérité de version antérieure. Il
n'est donc pas dupliqué une seconde fois.

## Éléments explicitement exclus comme preuves

- les injections alternant des multiplicateurs extrêmes;
- les résultats des 9 504 exécutions de l'ancien protocole;
- la calibration clinique du score sans labels synchronisés;
- la validation IceTag d'automne 2019 comparée aux SLS d'hiver 2019;
- les formulations « diagnostic » ou « boiterie probable »;
- les seuils anciens considérés comme biologiquement validés;
- les fichiers compilés, caches et sorties intermédiaires historiques.

Ces éléments peuvent être étudiés hors dépôt pour comprendre l'historique du
projet, mais ils ne doivent pas être copiés dans les résultats scientifiques de
mémoire final.

## Règle de réécriture

Le manuscrit actif sous `memoire/` constitue la version final déposée dans ce dépôt:
il conserve le module HYPO comme résultat principal et présente l'extension
bidirectionnelle comme exploratoire. Les archives de travail antérieures ne sont
pas nécessaires à la compilation, aux tests ni à la reproduction des résultats.
