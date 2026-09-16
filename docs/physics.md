# Adimensionnement du problème

> Source de vérité pour les conventions résumées dans `AGENTS.md` § 2. Toute modification
> ici doit y être reportée, ainsi que dans `src/config.py`.
> Rédigé au sprint 1 par A ; relecture B.

## 1. Problème dimensionné (sujet § 2)

Domaine $\Omega = [0, L]^2$ avec $L = 1$ m, temps $t \in [0, t_{max}]$.

$$
\frac{\partial T}{\partial t} - \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right) = 0
\qquad \text{(Éq. 1)}, \quad \alpha = 2 \cdot 10^{-5}\ \mathrm{m^2/s}
$$

- IC (Éq. 2) : $T(x, y, 0) = T_{obj} = 80$ °C dans l'objet, $T_{amb} = 20$ °C ailleurs.
- BC (Éq. 3) : $T = T_{amb}$ sur les quatre murs $\partial\Omega$, pour tout $t$.

Pourquoi ne pas entraîner directement là-dessus ? Les entrées vaudraient $t \sim 5000$ s
et la sortie $T \sim 20$ à $80$ °C, avec $\alpha = 2 \cdot 10^{-5}$ en facteur du laplacien.
Un MLP à activation tanh sature pour des entrées de cet ordre, et les trois termes de la
perte (Éq. 4) auraient des échelles incomparables. On ramène donc tout à l'ordre 1
(Wang et al. 2023, *Expert's Guide*, § 2).

## 2. Variables sans dimension

$$
x^* = \frac{x}{L}, \qquad y^* = \frac{y}{L}, \qquad
t^* = \frac{\alpha\, t}{L^2}, \qquad
\theta = \frac{T - T_{amb}}{T_{obj} - T_{amb}} .
$$

- $L^2/\alpha = 50\,000$ s ($\approx 13{,}9$ h) est le **temps caractéristique de diffusion** :
  la durée pour que la chaleur traverse la pièce.
- $\theta \in [0, 1]$ : $0$ à l'ambiante, $1$ à la température de l'objet.
  On note $\Delta T = T_{obj} - T_{amb} = 60$ °C.

**Dérivées.** Avec $T = T_{amb} + \Delta T\,\theta$ et la règle de dérivation en chaîne :

$$
\frac{\partial T}{\partial t} = \Delta T \,\frac{\partial \theta}{\partial t^*}\,\frac{\partial t^*}{\partial t}
= \frac{\Delta T\,\alpha}{L^2}\,\frac{\partial \theta}{\partial t^*},
\qquad
\frac{\partial^2 T}{\partial x^2} = \frac{\Delta T}{L^2}\,\frac{\partial^2 \theta}{\partial x^{*2}} .
$$

## 3. Équation adimensionnée en $t^*$

En injectant dans l'Éq. 1 et en simplifiant par le facteur commun $\Delta T\,\alpha / L^2$ :

$$
\boxed{\;\frac{\partial \theta}{\partial t^*} = \frac{\partial^2 \theta}{\partial x^{*2}} + \frac{\partial^2 \theta}{\partial y^{*2}}\;}
\qquad \text{sur } [0,1]^2 \times ]0, t^*_{max}] .
$$

Plus aucun paramètre physique n'apparaît : $\alpha$, $L$, $T_{amb}$, $T_{obj}$ sont absorbés
dans les variables. C'est le **solveur DF** (`src/fd_solver.py`) qui travaille dans cette
forme, avec le pas $dt^* = 0{,}2\,dx^2$ (condition de stabilité FTCS 2D : $dt^* \le dx^2/4$).

- IC : $\theta(x^*, y^*, 0) = 1$ dans l'objet, $0$ ailleurs (lissée, voir § 5).
- BC : $\theta = 0$ sur $\partial\Omega$.

## 4. Second changement de temps : $\tau$, l'entrée du réseau

L'horizon retenu est $t^*_{max} = 0{,}1$ (`T_STAR_MAX`), soit $t_{max} = 5\,000$ s $\approx 1{,}4$ h.
Pour que la troisième entrée du réseau couvre aussi $[0, 1]$, on pose

$$
\tau = \frac{t^*}{t^*_{max}} \in [0, 1], \qquad
\frac{\partial \theta}{\partial t^*} = \frac{1}{t^*_{max}}\,\frac{\partial \theta}{\partial \tau} .
$$

L'équation devient $\partial_\tau \theta = t^*_{max}\,(\theta_{xx} + \theta_{yy})$, d'où le
**résidu** minimisé par le PINN (`src/physics.py`) :

$$
\boxed{\; r(x, y, \tau) = \frac{\partial \theta}{\partial \tau} - t^*_{max}\left( \frac{\partial^2 \theta}{\partial x^2} + \frac{\partial^2 \theta}{\partial y^2} \right)\;}
$$

Piège (AGENTS.md § 7) : oublier le facteur $t^*_{max}$ revient à simuler une diffusion
dix fois trop rapide.

## 5. Condition initiale lissée $\theta_0^\varepsilon$

L'échelon de l'Éq. 2 n'est pas dérivable, et un MLP à activation tanh est $C^\infty$ : il ne
peut pas le représenter, et la perte $L_{IC}$ resterait bloquée sur la discontinuité.
On remplace l'échelon par une marche lissée (`src/geometry.py`) :

$$
\theta_0^\varepsilon(x, y) = \frac{1}{2}\left[ 1 - \tanh\frac{d(x, y) - R}{\varepsilon} \right],
\qquad
d(x, y) =
\begin{cases}
\sqrt{(x - c_x)^2 + (y - c_y)^2} & \text{disque (rayon } R) \\
\max(|x - c_x|,\ |y - c_y|) & \text{pavé (demi-côté } R,\ \text{distance de Chebyshev)}
\end{cases}
$$

- $\theta_0^\varepsilon = 1/2$ exactement sur le bord $d = R$, $\approx 0{,}88$ à $d = R - \varepsilon$,
  $\approx 0{,}12$ à $d = R + \varepsilon$ : la transition a une largeur $\approx 2\varepsilon$.
- $\varepsilon = 0{,}01$ (`EPS_IC`) est un compromis : petit, on retrouve l'échelon mais le
  réseau peine ; grand, l'IC s'éloigne du sujet. À ajuster entre $0{,}005$ et $0{,}02$.
- **Le solveur DF utilise la même $\theta_0^\varepsilon$** (`initial_condition_grid`), sinon
  l'erreur mesurée près de $t = 0$ serait celle du lissage, pas celle du PINN.

## 6. Retour aux unités physiques

$$
T = T_{amb} + \Delta T\,\theta, \qquad
t = \frac{L^2}{\alpha}\,t^* = \frac{L^2}{\alpha}\,t^*_{max}\,\tau = 5\,000\,\tau \ \text{s}, \qquad
x = L\,x^* .
$$

Fonctions : `theta_to_celsius`, `celsius_to_theta` (`src/geometry.py`), `CFG.t_max_s`.

## 7. Ordres de grandeur

Mesure faite avec un solveur DF de contrôle, disque $R = 0{,}1$ (ROADMAP § 0 ; à reconfirmer
avec `src/fd_solver.py` au sprint 1) :

| $t^*$ | $\tau$ | $\theta_{max}$ | $T_{max}$ | temps réel |
|---|---|---|---|---|
| 0,001 | 0,01 | 0,92 | 75,1 °C | 50 s |
| 0,005 | 0,05 | 0,39 | 43,6 °C | 4 min |
| 0,01 | 0,1 | 0,22 | 33,3 °C | 8 min |
| 0,05 | 0,5 | 0,05 | 22,8 °C | 42 min |
| 0,1 | 1 | 0,017 | 21,0 °C | 1,4 h |

Deux conclusions :

1. À $\tau = 1$ la pièce est revenue à l'équilibre : l'horizon $t^*_{max} = 0{,}1$ suffit.
2. **Toute la dynamique se joue pour $\tau < 0{,}05$.** C'est la « raideur » du sujet.

## 8. Conséquences pour l'échantillonnage (`src/sampling.py`)

- **Résidu** : $x, y$ uniformes ; $\tau = u^2$ avec $u \sim U[0,1]$, de sorte que
  $P(\tau < 0{,}05) = \sqrt{0{,}05} \approx 22\,\%$ au lieu de $5\,\%$.
- **IC** : la moitié des points est tirée dans la bande $|d - R| \le 3\varepsilon$, là où
  $\theta_0^\varepsilon$ varie ; l'autre moitié est uniforme pour ancrer l'intérieur ($\theta = 1$)
  et le fond ($\theta = 0$).
- **BC** : un quart des points par mur, $\tau$ uniforme (la BC n'est pas plus dure près de $\tau = 0$).

## 9. Solution analytique de contrôle (tests)

Pour l'équation du § 3 avec BC homogènes, la fonction

$$
\theta(x, y, t^*) = \sin(\pi x)\,\sin(\pi y)\,e^{-2\pi^2 t^*}
\quad\Longleftrightarrow\quad
\theta(x, y, \tau) = \sin(\pi x)\,\sin(\pi y)\,e^{-2\pi^2 t^*_{max}\,\tau}
$$

est une solution exacte : $\partial_{t^*}\theta = -2\pi^2\theta$ et $\theta_{xx} + \theta_{yy} = -2\pi^2\theta$.
Elle sert à valider le solveur DF (`tests/test_fd.py`, erreur $L^2$ relative $< 10^{-3}$) et le
résidu autograd (`tests/test_physics.py`, résidu nul à $10^{-5}$ près avec le facteur $t^*_{max}$).

## 10. Résumé des conventions

| Grandeur | Symbole | Plage | Où |
|---|---|---|---|
| Entrées du réseau | $(x, y, \tau)$ | $[0,1]^3$ (puis $[-1,1]^3$ dans `forward`) | `model.py` |
| Sortie du réseau | $\theta$ | $\approx [0, 1]$ | `model.py` |
| Résidu | $r = \theta_\tau - t^*_{max}(\theta_{xx} + \theta_{yy})$ | | `physics.py` |
| Temps du DF | $t^* \in [0, t^*_{max}]$ | | `fd_solver.py` |
| Constantes | `T_STAR_MAX`, `EPS_IC`, `OBJECT_*` | | `config.py` |
