# Portée de l'alerte comportementale

## Formulation défendable

Le système analyse des changements persistants par rapport à la baseline
individuelle et produit une alerte destinée à prioriser une vérification sur le
terrain. Sans score locomoteur daté, examen du pied ou diagnostic vétérinaire,
la sortie ne doit pas être appelée diagnostic, probabilité ou détection
clinique de boiterie.

## Base des scénarios synthétiques

Paudyal et al. (2021) ont rapporté, la veille d'un parage thérapeutique, environ
1 810 pas/jour contre 2 542 chez les témoins, soit près de 29 % de moins, ainsi
que 631 contre 581 minutes couchées, soit près de 9 % de plus. Ces valeurs
ancrent le profil `gradual_moderate`. Les profils `gradual_mild` et
`gradual_marked` encadrent cette amplitude pour une analyse de sensibilité; ils
ne correspondent pas à des classes cliniques validées.

La littérature montre aussi que le piétinement peut augmenter lors d'une
station immobile en salle de traite. Le système ne suppose donc pas que toute
hausse ponctuelle des pas est une boiterie. Le contrôle
`isolated_short_variation` vérifie qu'une hausse brève et isolée ne suffit pas à
déclencher une alerte multivariée.

Sources primaires:

- Paudyal et al. (2021), *Lying and stepping behaviors around corrective or
  therapeutic claw trimming*: https://pmc.ncbi.nlm.nih.gov/articles/PMC9623814/
- Oehm et al. (2023), *Leveraging Accelerometer Data for Lameness Detection in
  Dairy Cows*: https://www.mdpi.com/2076-2615/13/23/3681
- Schönberger et al. (2023), *Analysis of Dairy Cow Behavior during Milking
  Associated with Lameness*: https://www.mdpi.com/2624-862X/4/4/38

## Ce que mesure la revalidation

- la couverture attribuable d'une fenêtre synthétique après retrait de l'exécution propre;
- l'IoU attribuable entre la fenêtre et les intervalles nouvellement alertés;
- l'apparition d'un nouveau départ d'alerte absent de la série non injectée;
- le délai du nouveau départ lorsqu'il existe;
- l'évolution des résultats entre niveaux synthétiques;
- le nombre d'alertes de fond sur les séries réelles non étiquetées.

Une alerte de fond n'est pas qualifiée de faux positif clinique, car l'absence
de label ne prouve pas l'absence de trouble. De même, les taux obtenus sur les
injections ne sont pas des sensibilités ou spécificités cliniques.

## Validation clinique manquante

Pour revendiquer une détection de boiterie, il faudra ultérieurement disposer de
scores locomoteurs datés, idéalement multi-évaluateurs, d'examens des pieds ou
diagnostics vétérinaires, et séparer entraînement et test par vache, période et
si possible par ferme.
