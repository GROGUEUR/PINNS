# Fiches de lecture (sprint 0)

> Objectif : que A et B puissent tous les deux répondre, à l'oral, aux questions de `ROADMAP.md` § 7.
> Cocher quand c'est lu, puis compléter les réponses en 2–3 phrases chacune.

## Raissi, Perdikaris, Karniadakis (2019) — sections 2 et 3

Lu par : [ ] A  [ ] B

- Comment la perte combine-t-elle données (IC/BC) et résidu de l'EDP ? Lien avec l'Éq. 4 du sujet.
- Pourquoi les auteurs dérivent-ils le réseau par rapport à ses **entrées** (autograd) ?
- Quel optimiseur utilisent-ils et pourquoi L-BFGS convient-il quand la perte est déterministe ?
- Que se passe-t-il quand l'IC est raide (exemple de l'équation de Burgers) ?

## Wang, Sankaran, Wang, Perdikaris (2023) — *An Expert's Guide to Training PINNs*, section 2

Lu par : [ ] A  [ ] B

- Pourquoi adimensionner avant d'entraîner ? Quel problème d'échelle rencontre-t-on sinon avec α = 2·10⁻⁵ ?
- Pondération par normes de gradients : formule de λ̂ᵢ, rôle de la moyenne mobile.
- Pourquoi tanh, une init Xavier et des entrées dans [−1, 1] ?
- Quelle est l'idée de l'entraînement causal (bonus ROADMAP § 5) ?

## Wu, Zhu, Tan, Kartha, Lu (2023) — RAD (lecture rapide, sprint 3)

Lu par : [ ] A  [ ] B

- Formule p ∝ |r|ᵏ / E[|r|ᵏ] + c ; rôle de k et de c ; défauts k = 1, c = 1.
