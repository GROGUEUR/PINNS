# Adimensionnement du problème

> À rédiger au sprint 1 par A (relecture B). Source de vérité pour les conventions
> résumées dans `AGENTS.md` § 2. Toute modification ici doit y être reportée.

## Plan attendu

1. Équation dimensionnée (Éq. 1 du sujet), IC (Éq. 2), BC (Éq. 3).
2. Changement de variables : $x^* = x/L$, $t^* = \alpha t / L^2$, $\theta = (T - T_{amb}) / (T_{obj} - T_{amb})$.
3. Équation adimensionnée en $t^*$ : $\partial_{t^*}\theta = \theta_{xx} + \theta_{yy}$.
4. Second changement $\tau = t^*/t^*_{max}$ (entrée du réseau) et résidu
   $r = \partial_\tau \theta - t^*_{max}(\theta_{xx} + \theta_{yy})$.
5. IC lissée $\theta_0^\varepsilon = \tfrac12\left[1 - \tanh\frac{d - R}{\varepsilon}\right]$ et justification.
6. Retour aux unités physiques : $T = T_{amb} + (T_{obj} - T_{amb})\,\theta$, $t = t^* L^2/\alpha$.
7. Ordres de grandeur : $L^2/\alpha = 50\,000$ s, $t_{max} = 5\,000$ s, table $\theta_{max}(t^*)$ (ROADMAP § 0).
