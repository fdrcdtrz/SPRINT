from gurobipy import Model
from gurobipy import GRB
from initialization import *
import pandas as pd
import matplotlib.pyplot as plt

# script per definizione funzione di salvataggio risultati, problema di ottimizzazione per calcolo di Q^I, V^I, Q^N, V^N
def save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, filename="results.csv"):
    results = []

    for s in services:
        for r in resources:
            assigned = round(x[s.id, r.id].x)  # 1 se assegnato, 0 altrimenti
            list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
            list_s_kvi_service = [float(kvi) for kvi in s.kvi_service]
            list_r_kpi_resource = [float(kpi) for kpi in r.kpi_resource]
            list_r_kvi_resource = [float(kvi) for kvi in r.kvi_resource]

            results.append([
                s.id, r.id, assigned,
                normalized_kpi.get((r.id, s.id), 0),
                normalized_kvi.get((r.id, s.id), 0),
                s.min_kpi, s.min_kvi,
                list_s_kpi_service, list_s_kvi_service,
                list_r_kpi_resource, list_r_kvi_resource
            ])

    df = pd.DataFrame(results, columns=[
        "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource"
    ])

    df.to_csv(filename, index=False)
    print(f"\nSaved in {filename}")

def plot_pareto_front(pareto_solutions):
    pareto_solutions.sort()  # Ordina le soluzioni per KPI
    kpi_values, kvi_values = zip(*pareto_solutions)  # Separazione in due liste

    plt.figure(figsize=(8, 6))
    plt.plot(kpi_values, kvi_values, marker='o', linestyle='-', color='b', label="Fronte di Pareto")
    plt.xlabel("KPI Totale")
    plt.ylabel("KVI Totale")
    plt.title("Fronte di Pareto KPI-KVI")
    plt.grid(True)
    plt.legend()
    plt.show()


def optimize_kpi(services, resources, normalized_kpi, normalized_kvi):

    # Creazione del modello
    model = Model("Maximize_KPI")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(s.id, r.id) for s in services for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
                f"kvi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 3: Ogni servizio è assegnato a una sola risorsa
    for s in services:
        model.addConstr(sum(x[s.id, r.id] for r in resources) == 1, f"assign_service_{s.id}")

    # Vincolo 4: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[s.id, r.id] * s.demand for s in services) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI totali
    model.setObjective(
        sum(normalized_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()

    # Risultati
    print("\nSoluzione Ottima:")
    for s in services:
        for r in resources:
            if round(x[s.id, r.id].x) == 1:
                print(f"Servizio {s.id} assegnato a risorsa {r.id}")

    # Valore ottimo dell'obiettivo
    print(f"\nValore ottimale di KPI: {model.ObjVal}")

    Q_I = model.ObjVal

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, filename="results_optimization_qi.csv")

    return Q_I


def optimize_kvi(services, resources, normalized_kpi, normalized_kvi):

    # Creazione del modello
    model = Model("Maximize_KVI")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(s.id, r.id) for s in services for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
                f"kvi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 3: Ogni servizio è assegnato a una sola risorsa
    for s in services:
        model.addConstr(sum(x[s.id, r.id] for r in resources) == 1, f"assign_service_{s.id}")

    # Vincolo 4: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[s.id, r.id] * s.demand for s in services) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI totali
    model.setObjective(
        sum(normalized_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()

    # Risultati
    print("\nSoluzione Ottima:")
    for s in services:
        for r in resources:
            if round(x[s.id, r.id].x) == 1:
                print(f"Servizio {s.id} assegnato a risorsa {r.id}")

    # Valore ottimo dell'obiettivo
    print(f"\nValore ottimale di KVI: {model.ObjVal}")

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, filename="results_optimization_vi.csv")

    V_I = model.ObjVal

    return V_I


def q_nadir(services, resources, normalized_kpi, normalized_kvi, V_I):
    # Creazione del modello
    model = Model("Maximize_KPI_constraining_V")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(s.id, r.id) for s in services for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: il valore massimo che l'obiettivo V(X) può assumere è pari a V_I
    model.addConstr(
        sum(normalized_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) == V_I,
        "kvi_equals_max_kvi_value"
    )

    # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
                f"kvi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 4: Ogni servizio è assegnato a una sola risorsa
    for s in services:
        model.addConstr(sum(x[s.id, r.id] for r in resources) == 1, f"assign_service_{s.id}")

    # Vincolo 5: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[s.id, r.id] * s.demand for s in services) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI totali
    model.setObjective(
        sum(normalized_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()

    # Risultati
    print("\nSoluzione Ottima:")
    for s in services:
        for r in resources:
            if round(x[s.id, r.id].x) == 1:
                print(f"Servizio {s.id} assegnato a risorsa {r.id}")

    # Valore ottimo dell'obiettivo
    print(f"\nValore ottimale di KPI: {model.ObjVal}")

    Q_N = model.ObjVal

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, filename="results_optimization_qn.csv")

    return Q_N


def v_nadir(services, resources, normalized_kpi, normalized_kvi, Q_I):
    # Creazione del modello
    model = Model("Maximize_KVI_constraining_Q")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(s.id, r.id) for s in services for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: il valore massimo che l'obiettivo Q(X) può assumere è pari a Q_I
    model.addConstr(
        sum(normalized_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) == Q_I,
        "kpi_equals_max_kpi_value"
    )

    # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
                f"kvi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 4: Ogni servizio è assegnato a una sola risorsa
    for s in services:
        model.addConstr(sum(x[s.id, r.id] for r in resources) == 1, f"assign_service_{s.id}")

    # Vincolo 5: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[s.id, r.id] * s.demand for s in services) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI totali
    model.setObjective(
        sum(normalized_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()

    # Risultati
    print("\nSoluzione Ottima:")
    for s in services:
        for r in resources:
            if round(x[s.id, r.id].x) == 1:
                print(f"Servizio {s.id} assegnato a risorsa {r.id}")

    # Valore ottimo dell'obiettivo
    print(f"\nValore ottimale di KVI: {model.ObjVal}")

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, filename="results_optimization_vn.csv")

    V_N = model.ObjVal

    return V_N


def epsilon_constraint_exact(services, resources, normalized_kpi, normalized_kvi, Q_N, Q_I, delta=0.01):

    pareto_solutions = []
    epsilon = Q_N - delta  # Valore iniziale di epsilon

    while epsilon <= Q_I:
        # Creazione del modello
        model = Model("Epsilon_Constraint_Exact")

        # Variabili di decisione x[s, r] ∈ {0,1}
        x = model.addVars(
            [(s.id, r.id) for s in services for r in resources],
            vtype=GRB.BINARY,
            name="x"
        )

        # Vincoli:

        # Vincolo epsilon-constraint sul KPI
        model.addConstr(
            sum(normalized_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= epsilon,
            "epsilon_kpi"
        )

        # Vincolo su KPI e KVI minimo da soddisfare
        for s in services:
            for r in resources:
                model.addConstr(
                    (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                    f"kpi_threshold_{s.id}_{r.id}"
                )
                model.addConstr(
                    (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
                    f"kvi_threshold_{s.id}_{r.id}"
                )

        # Ogni servizio è assegnato a una sola risorsa
        for s in services:
            model.addConstr(sum(x[s.id, r.id] for r in resources) == 1, f"assign_service_{s.id}")

        # Capacità della risorsa non deve essere superata
        for r in resources:
            model.addConstr(
                sum(x[s.id, r.id] * s.demand for s in services) <= r.availability,
                f"capacity_{r.id}"
            )

        # Funzione obiettivo: massimizzare KVI
        model.setObjective(
            sum(normalized_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
            GRB.MAXIMIZE
        )

        # Risolvi il modello
        model.optimize()

        # Salva la soluzione
        if model.status == GRB.OPTIMAL:
            kpi_value = sum(
                normalized_kpi[(r.id, s.id)] * x[s.id, r.id].x for s in services for r in resources
            )
            kvi_value = model.ObjVal
            pareto_solutions.append((kpi_value, kvi_value))
            print(f"Epsilon: {epsilon}, KPI: {kpi_value}, KVI: {kvi_value}")

        # Incrementa epsilon
        epsilon += delta

    return pareto_solutions

# tolgo soluzioni dominate (i hope it works)
def filter_pareto_solutions(solutions):
    pareto_front = []

    for candidate in solutions:
        q_candidate, v_candidate = candidate
        dominated = False

        for other in solutions:
            q_other, v_other = other

            if (q_other >= q_candidate and v_other >= v_candidate) and (q_other > q_candidate or v_other > v_candidate):
                dominated = True
                break
        if not dominated:
            pareto_front.append(candidate)

    return pareto_front


def cut_and_solve(services, resources, normalized_kpi, normalized_kvi, max_iters=100, tolerance=1e-5):

    # Modello rilassato, quindi con vincolo di integrità rilassato
    model = Model("Cut_and_Solve")

    # Variabili binarie x[s, r] ∈ {0,1} → inizialmente rilassate a continue [0,1]
    x = model.addVars(
        [(s.id, r.id) for s in services for r in resources],
        vtype=GRB.CONTINUOUS,
        lb=0, ub=1,  # [0,1]
        name="x"
    )

    # Vincoli standard
    for s in services:
        model.addConstr(sum(x[s.id, r.id] for r in resources) == 1, f"assign_service_{s.id}")

    for r in resources:
        model.addConstr(
            sum(x[s.id, r.id] * s.demand for s in services) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI
    model.setObjective(
        sum(normalized_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
        GRB.MAXIMIZE
    )

    # Risolvi il problema rilassato
    model.optimize()

    print(f"Lower Bound iniziale: {model.ObjVal}")

    for _ in range(max_iters):
        # Variabili frazionarie
        fractional_vars = [(var, var.x) for var in model.getVars() if tolerance < var.x < 1 - tolerance]

        if not fractional_vars:
            print("Soluzione intera trovata!")
            break

        # Aggiunta taglio
        var, value = fractional_vars[0]  # Prima variabile frazionaria trovata
        model.addConstr(var <= round(value), name=f"cut_{var.VarName}")  # Da rivedere reduced cut!!!

        # Risoluzione problema con cut
        model.optimize()

        print(f"Iterazione {_ + 1}: KPI = {model.ObjVal}")

    print("\nSoluzione finale:")
    for s in services:
        for r in resources:
            if round(x[s.id, r.id].x) == 1:
                print(f"Servizio {s.id} assegnato a risorsa {r.id}")

    return x


