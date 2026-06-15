# Sélection des sources de MemoireV3

## Base retenue: MemoireV2

MemoireV2 fournit la base active parce qu'elle contient:

- la séparation chronologique baseline/futur;
- les injections exclusivement post-baseline;
- la soustraction de l'exécution propre;
- l'analyse avec la vache comme unité statistique;
- les profils hypoactifs graduels;
- l'analyse de sensibilité et la cohorte IceTag--SLS synchronisée;
- l'interface Streamlit et les tests logiciels actuels;
- le cadrage non diagnostique du mémoire.

## Éléments utiles du premier mémoire

Les sources du premier mémoire sont conservées sous
`sources_memoire/memoire_initial` pour récupérer après audit:

- les explications générales d'Isolation Forest et de LOF;
- les diagrammes d'architecture et éléments de traçabilité;
- les sections opérationnelles, éthiques et économiques;
- certaines figures et certains tableaux de revue de littérature;
- la description détaillée de l'interface.

Le moteur IF + règles est déjà présent dans le code hérité de MemoireV2. Il
n'est donc pas dupliqué une seconde fois.

## Éléments explicitement exclus comme preuves

- les injections alternant des multiplicateurs extrêmes;
- les résultats des 9 504 exécutions de l'ancien protocole;
- la calibration clinique du score sans labels synchronisés;
- la validation IceTag d'automne 2019 comparée aux SLS d'hiver 2019;
- les formulations « diagnostic » ou « boiterie probable »;
- les seuils anciens considérés comme biologiquement validés;
- les fichiers compilés, caches et sorties intermédiaires historiques.

Ces éléments peuvent être étudiés pour comprendre l'historique du projet, mais
ils ne doivent pas être copiés dans les résultats scientifiques de MemoireV3.

## Règle de réécriture

Le manuscrit actif sous `memoire/` est actuellement une base V2. Il ne doit pas
être réécrit comme mémoire hybride avant qu'une architecture V3 techniquement
acceptable soit identifiée. À défaut, l'hybride devra rester une perspective de
recherche et MemoireV2 demeurera la version déposée.
