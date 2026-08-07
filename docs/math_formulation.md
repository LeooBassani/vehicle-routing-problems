# Mathematical Formulations

Exact MILP formulations solved with CBC (via PuLP). This document states
each formulation in full: sets, parameters, decision variables, objective,
and constraints. Implementations are in `src/vrp/models/`.

All three formulations share the same capacity convention: $u_i$ tracks
**remaining vehicle capacity** immediately after serving customer $i$
(decreasing along the route), rather than cumulative load delivered
(increasing) matching the physical picture more directly: a vehicle leaves
the depot full and empties out as it delivers.

---

## 1. CVRP — Capacitated Vehicle Routing Problem

**File:** `src/vrp/models/cvrp.py`

### Sets

- $V = \{0, 1, \dots, n\}$ — all nodes, where $0$ is the depot and $\{1, \dots, n\}$ are customers

### Parameters

$$d_{ij}$ distance from node $i$ to node $j$$ 
$$q_i$ demand of customer $i$ (with $q_0 = 0$)$
$$Q$ vehicle capacity$
$$K$ number of available vehicles$

### Decision variables

$$x_{ij} \in \{0,1\} \qquad \forall i,j \in V,\ i \ne j$$
$$x_{ij} = 1 \text{ if some vehicle travels directly from node } i \text{ to node } j$$

$$u_i \ge 0 \qquad \forall i \in V \setminus \{0\}$$
$$u_i = \text{remaining vehicle capacity immediately after serving customer } i$$

### Objective

$$\min \sum_{i \in V} \sum_{\substack{j \in V \\ j \ne i}} d_{ij}\, x_{ij}$$

### Constraints

**(1) Each customer is visited exactly once:**

$$\sum_{\substack{i \in V \\ i \ne h}} x_{ih} = 1 \qquad \forall h \in V $$

$$\sum_{\substack{j \in V \\ j \ne h}} x_{hj} = 1 \qquad \forall h \in V $$

($h$ is the index for "the customer this constraint is about," distinct
from $i$/$j$, which range over that customer's possible neighbors.)

**(2) Fleet size at the depot:**

$$\sum_{j \ne 0} x_{0j} \le K, \qquad \sum_{i \ne 0} x_{i0} \le K, \qquad \sum_{j \ne 0} x_{0j} = \sum_{i \ne 0} x_{i0}$$

**(3) Capacity + subtour elimination:**

$$u_j \le u_i - q_j + Q(1 - x_{ij}) \qquad \forall i,j \in V \setminus \{0\},\ i \ne j$$

$$0 \le u_i \le Q - q_i \qquad \forall i \in V \setminus \{0\}$$

Whenever $x_{ij}=1$, this forces $u_j \le u_i - q_j$: remaining capacity
must strictly decrease at every step. 

---

## 2. VRPTW — VRP with Time Windows

**File:** `src/vrp/models/vrptw.py`

Extends CVRP with time windows. Same $x_{ij}$, visit, and fleet
constraints as above, plus:

### Additional parameters

$$[e_i, l_i]$ time window at node $i$ (earliest/latest allowed start of service)$
$$s_i$ | service duration at node $i$$
$$\tau_{ij}$ | travel time from $i$ to $j$ ($\tau_{ij} = d_{ij} / \text{speed}$)$
$$[e_0, l_0]$ | depot operating window$

### Additional decision variable

$$t_i \in [e_i, l_i] \qquad \forall i \in V$$
$$t_i = \text{arrival time at node } i$$

### Additional constraint: big-M time propagation

$$t_j \ge t_i + s_i + \tau_{ij} - M_{ij}(1 - x_{ij}) \qquad \forall i \in V,\ j \ne 0,\ i \ne j$$

where $M_{ij} = l_0 + \tau_{ij}$.

Capacity is still enforced via the same remaining-capacity $u_i$
variables and constraint (3) from the CVRP section, applied identically
among customer nodes.

---

## 3. MDVRP — Multi-Depot VRP

**File:** `src/vrp/models/mdvrp.py`

### Additional sets & parameters

- $D = \{0, \dots, |D|-1\}$ — depot nodes; 
- $N$ — customer nodes ($V = D \cup N$)
- $K_d$ — vehicles available at depot $d$

### Additional decision variable

$$z_{id} \in \{0,1\} \qquad \forall i \in N,\ d \in D$$
$$z_{id} = 1 \text{ if customer } i \text{ is served out of depot } d$$

### Why $z$ is needed

With a single depot, flow conservation alone guarantees every route both
starts and ends at the (only) depot. With multiple depots this is no
longer automatic: a flow model could route a vehicle out of depot
$A$, through a chain of customers, and back into depot $B$ — not a valid
vehicle route (a vehicle belongs to one depot and must return to it).
$z_{id}$ fixes each customer to one depot, and the constraints below force
every arc on a route to connect nodes with the *same* depot assignment.

### Additional constraints

**(1) Each customer assigned to exactly one depot:**

$$\sum_{d \in D} z_{id} = 1 \qquad \forall i \in N$$

**(2) Depot arcs only to/from the customer's assigned depot:**

$$x_{di} \le z_{id}, \qquad x_{id} \le z_{id} \qquad \forall i \in N,\ d \in D$$

**(3) Customer-customer arcs require identical depot assignment** (both
directions, every depot):

$$z_{id} - z_{jd} \le 1 - x_{ij}, \qquad z_{jd} - z_{id} \le 1 - x_{ij} \qquad \forall i,j \in N,\ i\ne j,\ d \in D$$

**(4) Fleet size per depot:**

$$\sum_{i \in N} x_{di} \le K_d, \qquad \sum_{i \in N} x_{id} \le K_d, \qquad \sum_i x_{di} = \sum_i x_{id} \qquad \forall d \in D$$

Capacity is enforced via the same remaining-capacity $u_i$ variables as
CVRP, applied among customer nodes only.

---
