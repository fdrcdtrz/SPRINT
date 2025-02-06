from gurobipy import Model, GRB
from initialization import *
import pandas as pd
import matplotlib.pyplot as plt


# script per definizione funzione di salvataggio risultati, problema di ottimizzazione per calcolo di Q^I, V^I, Q^N, V^N
def save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, filename="results.csv"):
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
                normalized_kvi.get((r.id, s.id), 0), # chiave più nulla se non esiste, ovvero zero
                weighted_sum_kpi.get((r.id, s.id), 0),
                weighted_sum_kvi.get((r.id, s.id), 0),
                s.min_kpi, s.min_kvi,
                list_s_kpi_service, list_s_kvi_service,
                list_r_kpi_resource, list_r_kvi_resource
            ])

    df = pd.DataFrame(results, columns=[
        "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource",
        "Global_KPI", "Global_KVI"
    ])

    df.to_csv(filename, index=False)
    print(f"\nSaved in {filename}")


def plot_pareto_front(pareto_solutions):
    pareto_solutions.sort()  # Ordina le soluzioni per KPI
    kpi_values, kvi_values = zip(*pareto_solutions)  # Separazione in due liste

    plt.figure(figsize=(8, 6))
    plt.plot(kpi_values, kvi_values, marker='o', linestyle='-', color='b', label="Pareto Optimal-Set")
    plt.xlabel("KPI Totale")
    plt.ylabel("KVI Totale")
    plt.title("Pareto Front")
    plt.grid(True)
    plt.legend()
    plt.show()


def optimize_kpi(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi):

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
                (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
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
        sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
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

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, filename="results_optimization_qi.csv")

    return Q_I


def optimize_kvi(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi):

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
                (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
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
        sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
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

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, filename="results_optimization_vi.csv")

    V_I = model.ObjVal

    return V_I


def q_nadir(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I):
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
        sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= V_I - 0.01, #== V_I,
        "kvi_equals_max_kvi_value"
    )

    # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
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
        sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()
    print(f"DEBUG: Q_N calcolato = {model.ObjVal}")

    # Risultati
    print("\nSoluzione Ottima:")

    for s in services:
        for r in resources:
            if round(x[s.id, r.id].x) == 1:
                print(f"Servizio {s.id} assegnato a risorsa {r.id}")

    # Valore ottimo dell'obiettivo
    print(f"\nValore ottimale di KPI: {model.ObjVal}")

    Q_N = model.ObjVal

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, filename="results_optimization_qn.csv")

    return Q_N


def v_nadir(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_I):
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
        sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) == Q_I,
        "kpi_equals_max_kpi_value"
    )

    # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            model.addConstr(
                (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
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
        sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
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

    save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, filename="results_optimization_vn.csv")

    V_N = model.ObjVal

    return V_N


def exact_epsilon_constraint(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta):

    pareto_solutions = []
    epsilon = Q_N - delta  # Inizializza epsilon con il nadir point

    while epsilon >= Q_I:
        model = Model("Exact_Epsilon_Constraint")

        # Variabili di decisione
        x = model.addVars(
            [(s.id, r.id) for s in services for r in resources],
            vtype=GRB.BINARY, name="x"
        )

        # Vincolo epsilon sul KPI
        model.addConstr(
            sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= epsilon
        )

        # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for s in services:
            for r in resources:
                model.addConstr(
                    (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                    f"kpi_threshold_{s.id}_{r.id}"
                )

        # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for s in services:
            for r in resources:
                model.addConstr(
                    (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
                    f"kvi_threshold_{s.id}_{r.id}"
                )

        # Vincoli standard (assegnazione, capacità)
        for s in services:
            model.addConstr(sum(x[s.id, r.id] for r in resources) == 1)

        for r in resources:
            model.addConstr(sum(x[s.id, r.id] * s.demand for s in services) <= r.availability)


        # Funzione obiettivo: massimizzare KVI
        model.setObjective(
            sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
            GRB.MAXIMIZE
        )

        # Risoluzione
        model.optimize()

        if model.status == GRB.OPTIMAL:
            kpi_value = sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id].x for s in services for r in resources)
            kvi_value = model.ObjVal
            pareto_solutions.append((kpi_value, kvi_value))

        # Aggiornato epsilon al valore di KVI trovato
        epsilon = kvi_value - delta

    return pareto_solutions


# tolgo soluzioni dominate (IT DOESNT WORK DA CAMBIAREEEEE)
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

def cut_and_solve(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta, max_iters=10,
                                     tolerance=1e-5, cost_threshold=0.01):

    pareto_solutions = []
    epsilon = Q_N - delta  # epsilon inizialmente parte da qui. gradualmente portato verso Q_I ovvero V_N

    while epsilon <= Q_I:

        # Risolvo problema denso rilassato per 1) cercare UB 2) identificare variabili dello sparse problem.
        dense_model = Model(f"Dense_{epsilon}")
        x_dense = dense_model.addVars(
            [(s.id, r.id) for s in services for r in resources],
            vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x"
        )

        # Vincolo epsilon su KPI
        dense_model.addConstr(
            sum(weighted_sum_kpi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources) >= epsilon,
            "epsilon_kpi"
        )

        # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for s in services:
            for r in resources:
                dense_model.addConstr(
                    (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x_dense[s.id, r.id] >= 0,
                    f"kpi_threshold_{s.id}_{r.id}"
                )

        # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for s in services:
            for r in resources:
                dense_model.addConstr(
                    (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_dense[s.id, r.id] >= 0,
                    f"kvi_threshold_{s.id}_{r.id}"
                )

        # Vincolo associazione un servizio -> una risorsa
        for s in services:
            dense_model.addConstr(sum(x_dense[s.id, r.id] for r in resources) == 1)

        # Vincolo disponibilità non violata
        for r in resources:
            dense_model.addConstr(sum(x_dense[s.id, r.id] * s.demand for s in services) <= r.availability)

        # Obiettivo: Massimizzare KVI
        dense_model.setObjective(
            sum(weighted_sum_kvi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources),
            GRB.MAXIMIZE
        )

        dense_model.optimize()
        UB = dense_model.ObjVal  # Upper Bound iniziale

        # Seleziono solo variabili con costo ridotto entro una soglia:
        # tra tutte le varibili del modello denso prese con getVars(), prendo il costo ridotto e verifico se è al
        # di sopra di una soglia
        selected_vars = [var.VarName for var in dense_model.getVars() if var.RC >= cost_threshold]
        remaining_vars = [var.VarName for var in dense_model.getVars() if var.RC < cost_threshold]

        # fractional_vars = [(var, var.x, var.RC) for var in dense_model.getVars() if tolerance < var.x < 1 - tolerance]
        # fractional_vars.sort(key=lambda v: v[2], reverse=True)  # Ordina per costo ridotto
        # selected_vars = [var.VarName for var, _, rc in fractional_vars if abs(rc) >= cost_threshold]

        # Sparse model con le variabili selezionate
        # sparse_model = Model(f"Sparse_{epsilon}")
        # x_sparse = sparse_model.addVars(
        #     selected_vars,
        #     vtype=GRB.BINARY, name="x"
        # )

        sparse_model = Model(f"Sparse_{epsilon}")
        x_sparse = sparse_model.addVars(
            selected_vars,
            vtype=GRB.BINARY, name="x"
        )

        # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for s in services:
            for r in resources:
                var_name = f"x[{s.id},{r.id}]"
                if var_name in x_sparse:
                    sparse_model.addConstr(
                        (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_sparse[var_name] >= 0
                    )

        # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for s in services:
            for r in resources:
                var_name = f"x[{s.id},{r.id}]"
                if var_name in x_sparse:
                    sparse_model.addConstr(
                        (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_sparse[var_name] >= 0,
                        f"kvi_threshold_{s.id}_{r.id}"
                    )

        # Stessi vincoli di capacità e assegnazione
        for s in services:
            sparse_model.addConstr(
                sum(x_sparse[f"x[{s.id},{r.id}]"] for r in resources if f"x[{s.id},{r.id}]" in x_sparse) == 1)

        for r in resources:
            sparse_model.addConstr(
                sum(x_sparse[f"x[{s.id},{r.id}]"] * s.demand for s in services if
                    f"x[{s.id},{r.id}]" in x_sparse) <= r.availability
            )

        # Vincolo epsilon su KPI
        sparse_model.addConstr(
            sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"] for s in services if
                f"x[{s.id},{r.id}]" in x_sparse) >= epsilon
        )

        # Forza le variabili non selezionate a 0
        for s in services:
            for r in resources:
                var_name = f"x[{s.id},{r.id}]"
                if var_name not in x_sparse:
                    sparse_model.addConstr(x_dense[s.id, r.id] == 0, name=f"fix_{var_name}")

        # Obiettivo: Massimizzare KVI
        sparse_model.setObjective(
            sum(weighted_sum_kvi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"] for s in services if
                f"x[{s.id},{r.id}]" in x_sparse),
            GRB.MAXIMIZE
        )

        # Iterazioni del Cut-and-Solve, procedo fino a che non trovo convergenza
        for _ in range(max_iters):
            sparse_model.optimize()
            LB = sparse_model.ObjVal  # Lower Bound aggiornato

            # Arresto quando trovo convergenza
            if UB <= LB:
                print(f"Convergenza per epsilon={epsilon}")
                break

        # Salvo soluzione
        kpi_value = sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"].x for s in services if
                        f"x[{s.id},{r.id}]" in x_sparse)
        kvi_value = sparse_model.ObjVal
        pareto_solutions.append((kpi_value, kvi_value))

        # Update epsilon: Q(X*) - delta
        epsilon = kvi_value - delta

    return pareto_solutions


# def cut_and_solve(services, resources, normalized_kpi, normalized_kvi, Q_N, Q_I, delta, max_iters=10,
#                                      tolerance=1e-5, cost_threshold=0.01):
#
#     pareto_solutions = []
#     epsilon = Q_N - delta  # epsilon inizialmente parte da qui. gradualmente portato verso Q_I ovvero V_N
#
#     while epsilon <= Q_I:
#
#         # Risolvo problema denso rilassato per 1) cercare UB 2) identificare variabili dello sparse problem. in
#         # quanto rilassato, vincolo di integrità scompare e x[s,r] in [0,1] invece che {0,1}
#         dense_model = Model(f"Dense_{epsilon}")
#         x_dense = dense_model.addVars(
#             [(s.id, r.id) for s in services for r in resources],
#             vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x"
#         )
#
#         # Vincolo epsilon su KPI
#         dense_model.addConstr(
#             sum(normalized_kpi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources) >= epsilon
#         )
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 dense_model.addConstr(
#                     (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x_dense[s.id, r.id] >= 0,
#                     f"kpi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 dense_model.addConstr(
#                     (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x_dense[s.id, r.id] >= 0,
#                     f"kvi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincolo associazione un servizio -> una risorsa
#         for s in services:
#             dense_model.addConstr(sum(x_dense[s.id, r.id] for r in resources) == 1)
#
#         # Vincolo disponibilità non violata
#         for r in resources:
#             dense_model.addConstr(sum(x_dense[s.id, r.id] * s.demand for s in services) <= r.availability)
#
#
#         # Obiettivo: Massimizzare KVI
#         dense_model.setObjective(
#             sum(normalized_kvi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources),
#             GRB.MAXIMIZE
#         )
#
#         dense_model.optimize()
#         UB = dense_model.ObjVal  # Upper Bound iniziale
#
#         # Seleziono solo variabili con costo ridotto entro una soglia /
#         # da rivedere tolleranza. alfa nei paper è un valore positivo
#
#         fractional_vars = [(var, var.x, var.RC) for var in dense_model.getVars() if tolerance < var.x < 1 - tolerance]
#         fractional_vars.sort(key=lambda v: v[2], reverse=True)  # Ordina per costo ridotto
#         selected_vars = [var.VarName for var, _, rc in fractional_vars if abs(rc) >= cost_threshold]
#
#
#         # Una volta trovate queste variabili, le posso utilizzare nel problema sparso da risolvere esattamente. Le
#         # varibili in questione sono solo quelle selezionate.
#
#         sparse_model = Model(f"Sparse_{epsilon}")
#         x_sparse = sparse_model.addVars(
#             selected_vars,
#             vtype=GRB.BINARY, name="x"
#         )
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 sparse_model.addConstr(
#                     (normalized_kpi[(r.id, s.id)] - s.min_kpi) * x_sparse[s.id, r.id] >= 0,
#                     f"kpi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 sparse_model.addConstr(
#                     (normalized_kvi[(r.id, s.id)] - s.min_kvi) * x_sparse[s.id, r.id] >= 0,
#                     f"kvi_threshold_{s.id}_{r.id}"
#                 )
#
#
#         # Stessi vincoli di capacità e assegnazione
#         for s in services:
#             sparse_model.addConstr(
#                 sum(x_sparse[f"x[{s.id},{r.id}]"] for r in resources if f"x[{s.id},{r.id}]" in x_sparse) == 1)
#
#         for r in resources:
#             sparse_model.addConstr(
#                 sum(x_sparse[f"x[{s.id},{r.id}]"] * s.demand for s in services if
#                     f"x[{s.id},{r.id}]" in x_sparse) <= r.availability
#             )
#
#         # Vincolo epsilon su KPI
#         sparse_model.addConstr(
#             sum(normalized_kpi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"] for s in services if
#                 f"x[{s.id},{r.id}]" in x_sparse) >= epsilon
#         )
#
#         # Forza le variabili non selezionate a 0 (ovvero il vincolo 26)
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{s.id},{r.id}]"
#                 if var_name not in x_sparse:
#                     sparse_model.addConstr(x_dense[s.id, r.id] == 0, name=f"fix_{var_name}")
#
#         # Obiettivo: Massimizzare KVI
#         sparse_model.setObjective(
#             sum(normalized_kvi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"] for s in services if
#                 f"x[{s.id},{r.id}]" in x_sparse),
#             GRB.MAXIMIZE
#         )
#
#         # Iterazioni del Cut-and-Solve, procedo fino a che non trovo convergenza
#         for _ in range(max_iters):
#             sparse_model.optimize()
#             LB = sparse_model.ObjVal  # Lower Bound aggiornato
#
#             # Arresto quando trovo convergenza
#             if UB <= LB:
#             #if abs(UB - LB) < tolerance:
#                 print(f"Convergenza per epsilon={epsilon}")
#                 break
#
#         # Salvo soluzione
#         kpi_value = sum(normalized_kpi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"].x for s in services if
#                         f"x[{s.id},{r.id}]" in x_sparse)
#         kvi_value = sparse_model.ObjVal
#         pareto_solutions.append((kpi_value, kvi_value))
#
#         # Update epsilon: Q(X*) - delta
#         epsilon = kvi_value - delta
#
#     return pareto_solutions
#
def epsilon_constraint_exact(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.01):

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
            sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= epsilon,
            "epsilon_kpi"
        )

        # Vincolo su KPI e KVI minimo da soddisfare
        for s in services:
            for r in resources:
                model.addConstr(
                    (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
                    f"kpi_threshold_{s.id}_{r.id}"
                )
                model.addConstr(
                    (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
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
            sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
            GRB.MAXIMIZE
        )

        # Risolvi il modello
        model.optimize()

        # Salva la soluzione
        if model.status == GRB.OPTIMAL:
            kpi_value = sum(
                weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id].x for s in services for r in resources
            )
            kvi_value = model.ObjVal
            pareto_solutions.append((kpi_value, kvi_value))
            print(f"Epsilon: {epsilon}, KPI: {kpi_value}, KVI: {kvi_value}")

        # Incrementa epsilon
        epsilon += delta

    return pareto_solutions


