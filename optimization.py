import gurobipy
from gurobipy import Model, GRB
from initialization import *
import pandas as pd
import matplotlib.pyplot as plt
import random
import csv


# script per definizione funzione di salvataggio risultati, problema di ottimizzazione per calcolo di Q^I, V^I, Q^N, V^N
def save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                     filename="results.csv"):
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
                normalized_kvi.get((r.id, s.id), 0),  # chiave più nulla se non esiste, ovvero zero
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

    df = df[df['Assigned'] != 0]

    df.to_csv(filename, index=False)
    print(f"\nSaved in {filename}")


def save_epsilon_constraint(services, resources, x, normalized_kpi, normalized_kvi,
                            weighted_sum_kpi, weighted_sum_kvi, epsilon):
    filename = f"epsilon_{epsilon:.6f}.csv"

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
                weighted_sum_kpi.get((r.id, s.id), 0),
                weighted_sum_kvi.get((r.id, s.id), 0),
                s.min_kpi, s.min_kvi,
                list_s_kpi_service, list_s_kvi_service,
                list_r_kpi_resource, list_r_kvi_resource
            ])

    df = pd.DataFrame(results, columns=[
        "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Weighted_Sum_KPI", "Weighted_Sum_KVI",  # Fix colonna mancata
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource"
    ])

    df = df[df['Assigned'] != 0]

    df.to_csv(filename, index=False)
    print(f"\nSaved in {filename}")
    return


def save_pareto_solutions(pareto_solutions, filename="pareto_solutions.csv"):
    pareto_solutions.sort()

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["KPI_Totale", "KVI_Totale"])
        writer.writerows(pareto_solutions)


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
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for s in services:
            for r in resources:
                if round(x[s.id, r.id].x) == 1:
                    print(f"Servizio {s.id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KPI: {model.ObjVal}")

        Q_I = model.ObjVal

        save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         filename="results_optimization_qi.csv")
    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible. Analizzo il conflitto...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità
        Q_I = 0

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
            # model.addConstr(
            #     (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
            #     f"kpi_threshold_{s.id}_{r.id}"
            # )
            model.addConstr(
                weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] >= s.min_kpi * x[s.id, r.id],
                f"kpi_threshold_{s.id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for s in services:
        for r in resources:
            # model.addConstr(
            #     (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
            #     f"kvi_threshold_{s.id}_{r.id}"
            # )
            model.addConstr(
                weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] >= s.min_kvi * x[s.id, r.id],
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
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for s in services:
            for r in resources:
                if round(x[s.id, r.id].x) == 1:
                    print(f"Servizio {s.id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KVI: {model.ObjVal}")

        save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         filename="results_optimization_vi.csv")

        V_I = model.ObjVal
    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible. Analizzo il conflitto...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità
        V_I = 0

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
        sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= V_I - 0.01,
        # == V_I,
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
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")

        for s in services:
            for r in resources:
                if round(x[s.id, r.id].x) == 1:
                    print(f"Servizio {s.id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KPI: {model.ObjVal}")

        Q_N = model.ObjVal

        save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         filename="results_optimization_qn.csv")

    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible.")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità
        Q_N = 0

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
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for s in services:
            for r in resources:
                if round(x[s.id, r.id].x) == 1:
                    print(f"Servizio {s.id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KVI: {model.ObjVal}")

        save_results_csv(services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         filename="results_optimization_vn.csv")

        V_N = model.ObjVal

    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible. Analizzo il conflitto...")

        model.computeIIS()

        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità

        V_N = 0

    return V_N


# def exact_epsilon_constraint(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
#                              Q_N, Q_I, delta):
#     pareto_solutions = []
#     epsilon = Q_N - delta  # Inizializza epsilon con il nadir point
#
#     while epsilon >= Q_I:
#         model = Model("Exact_Epsilon_Constraint")
#
#         # Variabili di decisione
#         x = model.addVars(
#             [(s.id, r.id) for s in services for r in resources],
#             vtype=GRB.BINARY, name="x"
#         )
#
#         # Vincolo epsilon sul KPI
#         model.addConstr(
#             sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= epsilon
#         )
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 model.addConstr(
#                     (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x[s.id, r.id] >= 0,
#                     f"kpi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 model.addConstr(
#                     (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x[s.id, r.id] >= 0,
#                     f"kvi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincoli standard (assegnazione, capacità)
#         for s in services:
#             model.addConstr(sum(x[s.id, r.id] for r in resources) == 1)
#
#         for r in resources:
#             model.addConstr(sum(x[s.id, r.id] * s.demand for s in services) <= r.availability)
#
#         # Funzione obiettivo: massimizzare KVI
#         model.setObjective(
#             sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources),
#             GRB.MAXIMIZE
#         )
#
#         # Risoluzione
#         model.optimize()
#
#         if model.status == GRB.OPTIMAL:
#             kpi_value = sum(weighted_sum_kvi[(r.id, s.id)] * x[s.id, r.id].x for s in services for r in resources)
#             kvi_value = model.ObjVal
#             pareto_solutions.append((kpi_value, kvi_value))
#
#         # Salvo
#
#         save_epsilon_constraint(services, resources, x, normalized_kpi, normalized_kvi,
#                                 weighted_sum_kpi, weighted_sum_kvi, epsilon)
#
#         # Aggiornato epsilon al valore di KVI trovato
#         epsilon = kvi_value - delta
#
#     return pareto_solutions

def branch_and_bound_pareto(services, resources, normalized_kpi, normalized_kvi,
                            weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.01):
    """
    Risolve il problema MILP con Branch-and-Bound classico per generare il fronte di Pareto.
    """
    pareto_solutions = []
    epsilon = Q_N - delta  # Valore iniziale di epsilon

    while epsilon >= Q_I:
        # Creazione del modello
        model = gurobipy.Model("Branch-and-Bound_Pareto")

        # Variabili di decisione binarie
        x = model.addVars(
            [(s.id, r.id) for s in services for r in resources],
            vtype=GRB.BINARY,
            name="x"
        )

        # Vincoli:
        model.addConstr(
            sum(weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id] for s in services for r in resources) >= epsilon,
            "epsilon_kpi"
        )

        # Vincolo KPI e KVI minimo da soddisfare
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

        model.setParam(GRB.Param.Heuristics, 0.0)  # Disattiva tutte le euristiche
        model.setParam(GRB.Param.Cuts, 0)  # Nessun taglio
        model.setParam(GRB.Param.Presolve, 0)  # Nessun pre-processing
        model.setParam(GRB.Param.MIPFocus, 0)  # Nessuna priorità nella ricerca
        model.setParam(GRB.Param.BranchDir, 1)  # Forza Gurobi a fare branching sempre nella stessa direzione

        # Risoluzione del modello
        model.optimize()

        # Salva la soluzione se ottimale
        if model.status == GRB.OPTIMAL:
            kpi_value = sum(
                weighted_sum_kpi[(r.id, s.id)] * x[s.id, r.id].x for s in services for r in resources
            )
            kvi_value = model.ObjVal
            pareto_solutions.append((kpi_value, kvi_value))
            print(f"Epsilon: {epsilon}, KPI: {kpi_value}, KVI: {kvi_value}")

        # Aggiorna epsilon per la prossima iterazione
        epsilon -= delta

    return pareto_solutions


# def cut_and_solve(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I,
#                   delta, max_iters=10,
#                   tolerance=1e-5, cost_threshold=0.01):
#     pareto_solutions = []
#     epsilon = Q_N - delta  # epsilon inizialmente parte da qui. gradualmente portato verso Q_I ovvero V_N
#     print(f"Q_N: {Q_N}, Q_I: {Q_I}")
#     UBest = float("inf")
#     n = 0
#
#     while epsilon <= Q_I:
#
#         # Risolvo problema denso rilassato per 1) cercare UB 2) identificare variabili dello sparse problem.
#         dense_model = Model(f"Dense_{epsilon}")
#         x_dense = dense_model.addVars(
#             [(s.id, r.id) for s in services for r in resources],
#             vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x"
#         )
#
#         # Vincolo epsilon su KPI
#         dense_model.addConstr(
#             sum(weighted_sum_kpi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources) >= epsilon,
#             "epsilon_kpi"
#         )
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 dense_model.addConstr(
#                     (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x_dense[s.id, r.id] >= 0,
#                     f"kpi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 dense_model.addConstr(
#                     (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_dense[s.id, r.id] >= 0,
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
#         # Vincolo del taglio se l'iterazione è successiva alla prima
#         if n > 1:
#             dense_model.update()
#             for s in services:
#                 for r in resources:
#                     var = x_dense[s.id, r.id]
#                     var_name = f"x[{r.id},{s.id}]"
#                     print(f"Checking var: {var}")  # Debug: stampa la variabile
#                     print(f"Selected vars: {selected_vars_values}")  # Lista dei nomi delle variabili
#
#                     if var_name in selected_vars_values:
#                         print(f"VarName: {var_name}")  # Debug: stampa il nome della variabile
#                         selected_value = selected_vars_values[var_name]  # Recupera il valore dal dizionario
#
#                         # Imposta il valore come punto di partenza per la variabile
#                         var.setAttr('Start', selected_value)
#                         print(f"Setting Start for {var_name} to {selected_value}")
#
#                     dense_model.addConstr(var <= 0)  # Aggiungi la restrizione
#
#         # Obiettivo: Massimizzare KVI
#         dense_model.setObjective(
#             sum(weighted_sum_kvi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources),
#             GRB.MAXIMIZE
#         )
#
#         dense_model.optimize()
#
#         dense_model.optimize()  # Risolvi il modello
#         if dense_model.status == GRB.OPTIMAL:
#             print(f"Posso avere Selected Vars.")
#         else:
#             print(f"Il modello non è ottimale. Status: {dense_model.status}")
#             dense_model.computeIIS()
#             dense_model.write("infeasible_model.ilp")
#
#         LB = dense_model.ObjVal  # Upper Bound iniziale
#         print(f"Lower bound: {LB}")
#         n += 1
#
#         # Seleziono solo variabili con costo ridotto entro una soglia:
#         # tra tutte le variabili del modello denso prese con getVars(), prendo il costo ridotto e verifico se è al
#         # di sopra di una soglia. Se il costo è >=0, quando posta pari a 1, la var. può aumentare il mio obiettivo.
#         # Dal momento che il mio problema è un problema di massimizzazione, il RC dice quanto migliora il mio obiettivo.
#         # Le seleziono e le pongo pari a zero nel mio problema sparso da risolvere esattamente: in questo modo, riduco
#         # lo spazio di ricerca delle soluzioni e identifico un LB al mio problema.
#
#
#         sparse_model = Model(f"Sparse_{epsilon}")
#         x_sparse = sparse_model.addVars(
#             [(s.id, r.id) for s in services for r in resources],
#             vtype=GRB.BINARY, name="x"
#         )
#
#         # Estrai le variabili dal modello 'dense_model'
#         selected_vars = [v for v in dense_model.getVars() if v.RC > cost_threshold] #contiene gurobi vars
#         print(f"Selected Vars: {selected_vars}")
#         # Durante il modello sparso, dopo aver risolto il problema sparso
#         selected_vars_values = {v.VarName: v.X for v in dense_model.getVars() if v.RC > cost_threshold}
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 sparse_model.addConstr(
#                     (weighted_sum_kvi[r.id, s.id] - s.min_kvi) * x_sparse[s.id, r.id] >= 0
#                 )
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 sparse_model.addConstr(
#                     (weighted_sum_kvi[r.id, s.id] - s.min_kvi) * x_sparse[s.id, r.id] >= 0,
#                     f"kvi_threshold_{r.id}_{s.id}"
#                 )
#
#         # Stessi vincoli di capacità e assegnazione
#         for s in services:
#             sparse_model.addConstr(sum(x_sparse[s.id, r.id] for r in resources) == 1)
#
#         sparse_model.addConstr(
#             sum(x_sparse[s.id, r.id] * s.demand for r in resources) <= r.availability
#         )
#
#         # Vincolo epsilon su KPI
#
#         sparse_model.addConstr(
#             sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[s.id, r.id] for r in resources) >= epsilon
#         )
#
#         # Forza le variabili selezionate a 0
#         sparse_model.addConstrs((var == 1) for var in selected_vars if var in sparse_model.getVars())
#
#         # Obiettivo: Massimizzare KVI
#         sparse_model.setObjective(
#             sum(weighted_sum_kvi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"] for s in services if
#                 f"x[{s.id},{r.id}]" in x_sparse),
#             GRB.MAXIMIZE
#         )
#
#         # Iterazioni del Cut-and-Solve, procedo fino a che non trovo convergenza
#         sparse_model.optimize()
#
#         if sparse_model.status == GRB.OPTIMAL:
#             UB = sparse_model.ObjVal  # Accedi al valore dell'obiettivo solo se il modello è ottimale
#             UB = min(UB, UBest)  # Lower Bound aggiornato
#         else:
#             print(f"Il modello non ha trovato una soluzione ottimale. Status: {sparse_model.status}")
#             UB = UBest
#
#         # Arresto quando trovo convergenza
#         if abs(UB - LB) < tolerance:
#             print(f"Convergenza per epsilon={epsilon}")
#         else:
#             continue  # torno all'inizio del while
#
#         # Salvo soluzione
#         kpi_value = sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[s.id, r.id].x for s in services for r in resources)
#         kvi_value = sparse_model.ObjVal
#         pareto_solutions.append((kpi_value, kvi_value))
#
#         # Update epsilon: Q(X*) - delta
#         epsilon = max(kvi_value - delta, Q_I)
#
#
#     return pareto_solutions


# def cut_and_solve(services, resources, normalized_kpi, normalized_kvi,
#                   weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta,
#                   max_iters=10, tolerance=1e-5, cost_threshold=0.01, max_inner_iters=5):

# pareto_solutions = []
# epsilon = min(Q_I, Q_N - delta)  # Parto dal nadir point e scendo verso Q_I
# stagnant_lb_count = 0  # Conta quante iterazioni LB rimane invariato
# stagnant_epsilon_count = 0  # Conta quante iterazioni epsilon rimane invariato
# prev_lb = float('-inf')
# prev_epsilon = epsilon
# zero_rc_count = 0
#
# iter_count = 0
#
# while epsilon <= Q_I or iter_count < max_iters:
#     print(f"Attuale epsilon = {epsilon}")
#
#     inner_iter = 0
#     UB, LB = float('inf'), float('-inf')
#
#     while inner_iter < max_inner_iters and abs(UB - LB) >= tolerance:
#
#         # inizio con il problema denso rilassato per primo UB e per identificare costo ridotto delle variabili
#         # x in [0,1] e non più (0,1)
#
#         dense_model = Model(f"Dense_{iter_count}_{epsilon:.6f}")
#         unique_keys = list(set((s.id, r.id) for s in services for r in resources))
#
#         x_dense = dense_model.addVars(
#             unique_keys,
#             vtype=GRB.CONTINUOUS, lb=0, ub=1,
#             name=f"x_dense_{iter_count}_{epsilon:.6f}"
#         )
#
#         # Vincolo epsilon
#         dense_model.addConstr(
#             sum(weighted_sum_kpi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources) >= epsilon
#         )
#
#         # Vincoli kpi e kvi minimo
#         for s in services:
#             for r in resources:
#                 for (s.id, r.id) in x_dense:
#                     dense_model.addConstr((weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x_dense[s.id, r.id] >= 0)
#                     dense_model.addConstr((weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_dense[s.id, r.id] >= 0)
#
#         # Vincoli di assegnazione e capacità
#         for s in services:
#             dense_model.addConstr(sum(x_dense[s.id, r.id] for r in resources) == 1)
#         for r in resources:
#             dense_model.addConstr(sum(x_dense[s.id, r.id] * s.demand for s in services) <= r.availability)
#
#         # Funzione obiettivo: massimizzare KVI
#         dense_model.setObjective(
#             sum(weighted_sum_kvi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources),
#             GRB.MAXIMIZE
#         )
#
#         dense_model.optimize()
#         if dense_model.status != GRB.OPTIMAL:
#             print(f"Status non ottimale: {dense_model.status}")
#             break
#
#         if dense_model.status == GRB.INFEASIBLE:
#             print("Problem infeasible, skipping iteration.")
#             dense_model.computeIIS()
#             dense_model.write("dense_model.ilp")
#             epsilon -= delta
#             continue
#
#         UB = dense_model.ObjVal  # Upper Bound iniziale
#
#         save_epsilon_constraint(services, resources, x_dense, normalized_kpi, normalized_kvi,
#                                 weighted_sum_kpi, weighted_sum_kvi, epsilon)
#
#
#         # Costo ridotto
#         selected_vars = [(s.id, r.id) for s in services for r in resources if x_dense[s.id, r.id].RC >= cost_threshold]
#         for s in services:
#             for r in resources:
#                 print(x_dense[s.id, r.id].RC)
#
#         if not selected_vars:
#             zero_rc_count += 1
#             print(f"Variabili hanno RC = 0!")
#
#             # Opzione 1: Selezionare un sottoinsieme casuale**
#             if zero_rc_count <= 3:  # Riproviamo 3 volte
#                 selected_vars = random.sample([(s.id, r.id) for s in services for r in resources],
#                                               min(10, len(services) * len(resources)))
#
#
#         # Se selected_vars è vuoto, riprovo fino a 3 volte dimezzando cost_threshold
#         retries = 0
#         while not selected_vars and retries < 3:
#             cost_threshold /= 2
#             selected_vars = [(s.id, r.id) for s in services for r in resources if x_dense[s.id, r.id].RC >= cost_threshold]
#             retries += 1
#
#         if not selected_vars:
#             print("Nessuna variabile selezionata, salto questa iterazione.")
#             break
#
#         selected_vars = list(set(selected_vars))
#
#         # verifico se ci sono duplicati
#         if len(selected_vars) != len(set(selected_vars)):
#             print("Risoluzione duplicati.")
#
#         # verifico lunghezza dopo il fix
#         if len(selected_vars) == 0:
#             print("selected_vars è ancora vuoto. salto iterazione.")
#             break  # Salta iterazione
#
#         # Con queste variabili, risolvo il problema sparso
#         sparse_model = Model(f"Sparse_{iter_count}_{epsilon:.6f}")
#         x_sparse = sparse_model.addVars(
#             selected_vars, vtype=GRB.BINARY,
#             name=f"x_sparse_{iter_count}_{epsilon:.6f}"
#         )
#
#         # Vincoli solo sulle variabili selezionate
#         for s in services:
#             for r in resources:
#                 if (s.id, r.id) in selected_vars:
#                     sparse_model.addConstr((weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x_sparse[s.id, r.id] >= 0)
#                     sparse_model.addConstr((weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_sparse[s.id, r.id] >= 0)
#
#         # Vincoli di assegnazione e capacità
#         for s in services:
#             sparse_model.addConstr(sum(x_sparse[s.id, r.id] for r in resources if (s.id, r.id) in selected_vars) == 1)
#
#         for r in resources:
#             sparse_model.addConstr(
#                 sum(x_sparse[s.id, r.id] * s.demand for s in services if (s.id, r.id) in selected_vars) <= r.availability
#             )
#         # Vincolo epsilon su KPI
#         sparse_model.addConstr(
#             sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[s.id, r.id] for s in services for r in resources if (s.id, r.id) in selected_vars) >= epsilon
#         )
#
#         sparse_model.optimize()
#
#         if sparse_model.status == GRB.OPTIMAL:
#             LB = sparse_model.ObjVal  # Aggiorno Lower Bound
#
#             # per salvare e plottare
#             kpi_value = sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[s.id, r.id].x
#                             for s in services for r in resources if (s.id, r.id) in selected_vars)
#             kvi_value = sparse_model.ObjVal
#
#             pareto_solutions.append((kpi_value, kvi_value))
#
#             # salvo
#             save_results_csv(services, resources, x_sparse, normalized_kpi, normalized_kvi,
#                              weighted_sum_kpi, weighted_sum_kvi, filename=f"results_sparse_{epsilon:.6f}.csv")
#
#             # aggiorno
#             epsilon = max(kvi_value - delta, Q_I)
#         elif (sparse_model.status == GRB.INFEASIBLE):
#                 print("Sparse Problem infeasible, skipping iteration.")
#                 sparse_model.computeIIS()
#                 sparse_model.write("sparse_model.ilp")
#                 LB = Q_I
#                 epsilon = max(LB - delta, Q_I)
#                 continue
#
#         if abs(UB - LB) < tolerance:
#             print(f"convergence for epsilon={epsilon}")
#             break
#
#         LB = sparse_model.ObjVal
#
#         # Se LB non aggiornato, assegno un valore minimo
#         if LB == float('-inf'):
#             print("LB non è aggiornato, quindi LB = Q_I")
#             LB = Q_I
#
#         # Aggiorno epsilon in base a Q(X*) - delta
#         epsilon = max(LB - delta, Q_I)
#
#         # Controllo se LB rimane invariato
#         if LB == prev_lb:
#             stagnant_lb_count += 1
#         else:
#             stagnant_lb_count = 0
#         prev_lb = LB
#
#         # Controllo se epsilon rimane invariato
#         if epsilon == prev_epsilon:
#             stagnant_epsilon_count += 1
#         else:
#             stagnant_epsilon_count = 0
#         prev_epsilon = epsilon
#
#         inner_iter += 1
#
#         # Se LB, epsilon o zero_rc_count non cambiano da 5 iterazioni, forzo l'uscita
#         if stagnant_lb_count >= 5:
#             print("LB non cambia da 5 iterazioni, fermo il ciclo.")
#             return pareto_solutions
#
#         if stagnant_epsilon_count >= 5:
#             print("Epsilon non cambia da 5 iterazioni, fermo il ciclo.")
#             return pareto_solutions
#
#         # iter totale
#         iter_count += 1
#
# return pareto_solutions


# def cut_and_solve(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta, max_iters=10,
#                                      tolerance=1e-5, cost_threshold=0.01):
#
#     pareto_solutions = []
#     epsilon = Q_N - delta  # epsilon inizialmente parte da qui. gradualmente portato verso Q_I ovvero V_N
#
#     while epsilon >= Q_I:
#
#         # Risolvo problema denso rilassato per 1) cercare UB 2) identificare variabili dello sparse problem.
#         dense_model = Model(f"Dense_{epsilon}")
#         x_dense = dense_model.addVars(
#             [(s.id, r.id) for s in services for r in resources],
#             vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x"
#         )
#
#         # Vincolo epsilon su KPI
#         dense_model.addConstr(
#             sum(weighted_sum_kpi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources) >= epsilon,
#             "epsilon_kpi"
#         )
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 dense_model.addConstr(
#                     (weighted_sum_kpi[(r.id, s.id)] - s.min_kpi) * x_dense[s.id, r.id] >= 0,
#                     f"kpi_threshold_{s.id}_{r.id}"
#                 )
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 dense_model.addConstr(
#                     (weighted_sum_kvi[(r.id, s.id)] - s.min_kvi) * x_dense[s.id, r.id] >= 0,
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
#         # Obiettivo: Massimizzare KVI
#         dense_model.setObjective(
#             sum(weighted_sum_kvi[(r.id, s.id)] * x_dense[s.id, r.id] for s in services for r in resources),
#             GRB.MAXIMIZE
#         )
#
#         dense_model.optimize()
#         UB = dense_model.ObjVal  # Upper Bound iniziale
#
#         # Seleziono solo variabili con costo ridotto entro una soglia:
#         # tra tutte le varibili del modello denso prese con getVars(), prendo il costo ridotto e verifico se è al
#         # di sopra di una soglia
#         selected_vars = [var.VarName for var in dense_model.getVars() if var.RC >= cost_threshold]
#
#         # fractional_vars = [(var, var.x, var.RC) for var in dense_model.getVars() if tolerance < var.x < 1 - tolerance]
#         # fractional_vars.sort(key=lambda v: v[2], reverse=True)  # Ordina per costo ridotto
#         # selected_vars = [var.VarName for var, _, rc in fractional_vars if abs(rc) >= cost_threshold]
#
#         # Sparse model con le variabili selezionate
#         # sparse_model = Model(f"Sparse_{epsilon}")
#         # x_sparse = sparse_model.addVars(
#         #     selected_vars,
#         #     vtype=GRB.BINARY, name="x"
#         # )
#
#         sparse_model = Model(f"Sparse_{epsilon}")
#         x_sparse = sparse_model.addVars(
#             [(s.id, r.id) for s in services for r in resources],
#             vtype=GRB.BINARY, name="x"
#         )
#
#         # Vincolo KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{r.id},{s.id}]"
#                 if var_name in selected_vars:
#                     sparse_model.addConstr(
#                         (weighted_sum_kvi[(var_name)] - s.min_kvi) * x_sparse[s.id, r.id] >= 0
#                     )
#
#
#         # Vincolo KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{r.id},{s.id}]"
#                 if var_name in selected_vars:
#                     sparse_model.addConstr(
#                         (weighted_sum_kvi[(var_name)] - s.min_kvi) * x_sparse[s.id, r.id]>= 0,
#                         f"kvi_threshold_{r.id}_{s.id}"
#                     )
#
#         # Stessi vincoli di capacità e assegnazione
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{r.id},{s.id}]"
#                 if var_name in selected_vars:
#                     sparse_model.addConstr(
#                         sum(x_sparse[s.id, r.id]) == 1
#                     )
#         # old
#
#         # for s in services:
#         #     var_name = f"x[{s.id},{r.id}]"
#         #     if var_name in selected_vars:
#         #         sparse_model.addConstr(
#         #             sum(x_sparse[f"x[{s.id},{r.id}]"] for r in resources if f"x[{s.id},{r.id}]" in x_sparse) == 1)
#
#
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{r.id},{s.id}]"
#                 if var_name in selected_vars:
#                     sparse_model.addConstr(
#                         sum(x_sparse[s.id, r.id] * s.demand) <= r.availability
#                     )
#
#         # old
#
#         # for r in resources:
#         #     var_name = f"x[{s.id},{r.id}]"
#         #     if var_name in selected_vars:
#         #         sparse_model.addConstr(
#         #             sum(x_sparse[var_name] * s.demand for s in services if
#         #                 f"x[{s.id},{r.id}]" in x_sparse) <= r.availability
#         #         )
#
#         # Vincolo epsilon su KPI
#
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{r.id},{s.id}]"
#                 if var_name in selected_vars:
#                     sparse_model.addConstr(
#                         sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[s.id, r.id]) >= epsilon
#                     )
#
#         # old
#
#         # var_name = f"x[{s.id},{r.id}]"
#         # if var_name in selected_vars:
#         #     sparse_model.addConstr(
#         #         sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[var_name] for s in services if
#         #             f"x[{s.id},{r.id}]" in x_sparse) >= epsilon
#         #     )
#
#         # Forza le variabili non selezionate a 0
#
#         for s in services:
#             for r in resources:
#                 var_name = f"x[{r.id},{s.id}]"
#                 if var_name not in selected_vars:
#                     sparse_model.addConstr(
#                         x_sparse[s.id, r.id] == 0
#                     )
#
#         # for s in services:
#         #     for r in resources:
#         #         var_name = f"x[{s.id},{r.id}]"
#         #         if var_name not in selected_vars:
#         #             sparse_model.addConstr(x_sparse[s.id, r.id] == 0, name=f"fix_{var_name}")
#
#         # Obiettivo: Massimizzare KVI
#         sparse_model.setObjective(
#             sum(weighted_sum_kvi[(r.id, s.id)] * x_sparse[f"x[{s.id},{r.id}]"] for s in services if
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
#             if abs(UB - LB) < tolerance:
#                 print(f"Convergenza per epsilon={epsilon}")
#                 break
#
#         # Salvo soluzione
#         kpi_value = sum(weighted_sum_kpi[(r.id, s.id)] * x_sparse[s.id, r.id].x for s in services for r in resources)
#         kvi_value = sparse_model.ObjVal
#         pareto_solutions.append((kpi_value, kvi_value))
#
#         # Update epsilon: Q(X*) - delta
#         epsilon = kvi_value - delta
#
#     return pareto_solutions

def epsilon_constraint_exact(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                             Q_N, Q_I, delta=0.01):
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

            save_epsilon_constraint(services, resources, x, normalized_kpi, normalized_kvi,
                                    weighted_sum_kpi, weighted_sum_kvi, epsilon)

        # Incrementa epsilon
        epsilon += delta

    return pareto_solutions
