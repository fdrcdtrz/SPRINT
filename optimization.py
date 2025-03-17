import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import Model, GRB

from initialization import *


# script per definizione funzione di salvataggio risultati, problema di ottimizzazione per calcolo di Q^I, V^I, Q^N, V^N

def save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                     results_dir, filename):

    filepath = os.path.join(results_dir, filename)  # Percorso corretto
    results = []

    for request_id, service_id in enumerate(service_requests):  # Iteriamo sulle richieste
        s = services[service_id]  # Otteniamo l'oggetto Service corrispondente
        for r in resources:
            assigned = round(x[request_id, r.id].x)  # 1 se assegnato, 0 altrimenti
            if assigned == 1:  # Salviamo solo assegnazioni valide
                list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
                list_s_kvi_service = []
                list_r_kpi_resource = [float(kpi) for kpi in r.kpi_resource]
                list_r_kvi_resource = []

                results.append([
                    request_id, service_id, r.id, assigned,
                    normalized_kpi.get((r.id, service_id), 0),
                    0,
                    weighted_sum_kpi.get((r.id, service_id), 0),
                    weighted_sum_kvi.get((r.id, service_id), 0),
                    s.min_kpi, 0,
                    list_s_kpi_service, list_s_kvi_service,
                    list_r_kpi_resource, list_r_kvi_resource
                ])

    df = pd.DataFrame(results, columns=[
        "Request_ID", "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Weighted_Sum_KPI", "Weighted_Sum_KVI",
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource"
    ])

    df.to_csv(filepath, index=False)
    print(f"Salvato: {filepath}")



def save_epsilon_constraint(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                            weighted_sum_kpi, weighted_sum_kvi, results_dir, epsilon):

    filename = f"epsilon_{epsilon:.6f}.csv"
    filepath = os.path.join(results_dir, filename)  # Percorso corretto
    results = []

    for request_id, service_id in enumerate(service_requests):  # Iteriamo sulle richieste
        s = services[service_id]  # Otteniamo l'oggetto Service corrispondente
        for r in resources:
            assigned = round(x[request_id, r.id].x)  # 1 se assegnato, 0 altrimenti
            if assigned == 1:  # Salviamo solo assegnazioni valide
                list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
                list_s_kvi_service = []
                list_r_kpi_resource = [float(kpi) for kpi in r.kpi_resource]
                list_r_kvi_resource = []

                results.append([
                    request_id, service_id, r.id, assigned,
                    normalized_kpi.get((r.id, service_id), 0),
                    0,
                    weighted_sum_kpi.get((r.id, service_id), 0),
                    weighted_sum_kvi.get((r.id, service_id), 0),
                    s.min_kpi, 0,
                    list_s_kpi_service, list_s_kvi_service,
                    list_r_kpi_resource, list_r_kvi_resource
                ])

    df = pd.DataFrame(results, columns=[
        "Request_ID", "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Weighted_Sum_KPI", "Weighted_Sum_KVI",
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource"
    ])

    df.to_csv(filepath, index=False)
    print(f"Salvato: {filepath}")

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


def optimize_kpi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, results_dir):
    # Creazione del modello
    model = Model("Maximize_KPI")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 3: Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Vincolo 4: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI totali
    model.setObjective(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()
    if model.IsMIP == 1:
        print("The model is a MIP.")


    # Risultati
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Servizio {request_id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KPI: {model.ObjVal}")

        Q_I = model.ObjVal

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_qi.csv")

    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible. Analizzo il conflitto...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità
        Q_I = 0

    return Q_I


def optimize_kvi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, results_dir):
    # Creazione del modello
    model = Model("Maximize_KVI")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 2: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                (weighted_sum_kvi[(r.id, service_id)]) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 3: Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Vincolo 4: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KVI totali
    model.setObjective(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()

    if model.IsMIP == 1:
        print("The model is a MIP.")

    # Risultati
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Servizio {request_id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KVI: {model.ObjVal}")

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_vi.csv")

        V_I = model.ObjVal
    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible. Analizzo il conflitto...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità
        V_I = 0

    return V_I


def q_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I, results_dir):
    # Creazione del modello
    model = Model("Maximize_KPI_constraining_V")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: il valore massimo che l'obiettivo V(X) può assumere è pari a V_I
    model.addConstr(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources)
        >= V_I - 0.01,
        "kvi_equals_max_kvi_value"
    )

    # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 4: Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Vincolo 5: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KPI totali
    model.setObjective(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.optimize()
    print(f"DEBUG: Q_N calcolato = {model.ObjVal}")

    if model.IsMIP == 1:
        print("The model is a MIP.")

    # Risultati
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Servizio {request_id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KPI: {model.ObjVal}")

        Q_N = model.ObjVal

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_qn.csv")

    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible.")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità
        Q_N = 0

    return Q_N


def v_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_I, results_dir):
    # Creazione del modello
    model = Model("Maximize_KVI_constraining_Q")

    # Creazione delle variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincolo 1: il valore massimo che l'obiettivo Q(X) può assumere è pari a Q_I
    model.addConstr(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources)
        >= Q_I,
        "kpi_equals_max_kpi_value"
    )

    # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
        s = services[service_id]  # Oggetto Service corrispondente

        for r in resources:
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Vincolo 4: Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Vincolo 5: Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Funzione obiettivo: massimizzare KVI totali
    model.setObjective(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )


    model.optimize()
    if model.IsMIP == 1:
        print("The model is a MIP.")

    # Risultati
    if model.status == GRB.OPTIMAL:
        print("\nSoluzione Ottima:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Servizio {request_id} assegnato a risorsa {r.id}")

        # Valore ottimo dell'obiettivo
        print(f"\nValore ottimale di KVI: {model.ObjVal}")

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_vn.csv")

        V_N = model.ObjVal

    if model.Status == GRB.INFEASIBLE:
        print("Il modello è infeasible. Analizzo il conflitto...")

        model.computeIIS()

        model.write("infeasible_model.ilp")  # Scrive il file con i vincoli responsabili dell'infeasibilità

        V_N = 0

    return V_N


def epsilon_constraint_exact(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                             Q_N, Q_I, delta, results_dir):
    pareto_solutions = []
    epsilon = Q_N - delta  # Valore iniziale di epsilon

    while epsilon <= Q_I:
        # Creazione del modello
        model = Model("Epsilon_Constraint_Exact")

        # Variabili di decisione x[s, r] ∈ {0,1}
        x = model.addVars(
            [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
            vtype=GRB.BINARY,
            name="x"
        )

        # Vincoli:

        # Vincolo epsilon-constraint sul KPI
        model.addConstr(
            sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
                for request_id in range(len(service_requests)) for r in resources)
            >= epsilon,
            "epsilon_kpi"
        )

        # Vincolo 2: KPI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for request_id in range(len(service_requests)):
            service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
            s = services[service_id]  # Oggetto Service corrispondente

            for r in resources:
                model.addConstr(
                    (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                    f"kpi_threshold_{request_id}_{r.id}"
                )

        # Vincolo 3: KVI offerto dalla risorsa a cui è assegnato servizio deve essere > minimo desiderato
        for request_id in range(len(service_requests)):
            service_id = service_requests[request_id]  # ID del servizio associato alla richiesta
            s = services[service_id]  # Oggetto Service corrispondente

            for r in resources:
                model.addConstr(
                    weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                    f"kpi_threshold_{request_id}_{r.id}"
                )

        # Vincolo 4: Ogni servizio è assegnato a una sola risorsa
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

        # Vincolo 5: Capacità della risorsa non deve essere superata
        for r in resources:
            model.addConstr(
                sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                    for request_id in range(len(service_requests))) <= r.availability,
                f"capacity_{r.id}"
            )

        # Funzione obiettivo: massimizzare KVI totali
        model.setObjective(
            sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
                for request_id in range(len(service_requests)) for r in resources),
            GRB.MAXIMIZE
        )

        model.setParam("MIPFocus", 3)  # meno nodi
        model.setParam("VarBranch", 2)  # branching aggressivo
        model.setParam("MIPGap", 0.02)  # ferma ricerca quando il gap è inferiore al 2 percento

        # Risolvi il modello
        model.optimize()
        if model.IsMIP == 1:
            print("The model is a MIP.")

        # Salva la soluzione
        if model.status == GRB.OPTIMAL:
            kpi_value = sum(
                weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id].x
                for request_id in range(len(service_requests)) for r in resources
            )
            kvi_value = model.ObjVal
            pareto_solutions.append((kpi_value, kvi_value))
            print(f"Epsilon: {epsilon}, KPI: {kpi_value}, KVI: {kvi_value}")

            save_epsilon_constraint(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                                    weighted_sum_kpi, weighted_sum_kvi, results_dir, epsilon)

        # Incrementa epsilon
        epsilon += delta

    return pareto_solutions

def optimize_multiobj_kpi_kvi(service_requests, services, resources, normalized_kpi, normalized_kvi,
                              weighted_sum_kpi, weighted_sum_kvi, results_dir):

    solutions = []

    model = Model("MultiObj_KPI_KVI")

    # Variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincoli: KPI e KVI minimi
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]
        s = services[service_id]

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )
            model.addConstr(
                (weighted_sum_kvi[(r.id, service_id)]) * x[request_id, r.id] >= 0,
                f"kvi_threshold_{request_id}_{r.id}"
            )

    # Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # La capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Definizione degli obiettivi multipli
    model.ModelSense = GRB.MAXIMIZE

    # Primo obiettivo: Massimizzare KPI
    model.setObjectiveN(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        index=0,  # Primo obiettivo
        priority=1,  # Priorità più alta
        name="Maximize_KPI"
    )

    # Secondo obiettivo: Massimizzare KVI
    model.setObjectiveN(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        index=1,  # Secondo obiettivo
        priority=1,  # Stessa priorità per il metodo della somma pesata
        name="Maximize_KVI"
    )

    # Ottimizzazione multi-obiettivo
    model.optimize()

    # Verifica se la soluzione è ottimale
    if model.status == GRB.OPTIMAL:
        print(f"\nSoluzione Ottima: KPI={model.ObjNVal[0]}, KVI={model.ObjNVal[1]}")
        kpi_value = model.ObjNVal[0]
        kvi_value = model.ObjNVal[1]
        solutions.append((kpi_value, kvi_value))

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                         weighted_sum_kpi, weighted_sum_kvi, results_dir, filename=f"results_both_obj.csv")

    else:
        print("Il modello non ha trovato una soluzione ottima.")
        return None, None



def optimize_multiobj_weighted_ip(service_requests, services, resources, normalized_kpi, normalized_kvi,
                                  weighted_sum_kpi, weighted_sum_kvi, alpha, results_dir):
    model = Model("MultiObj_Weighted_KPI_KVI_IP")

    # Variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincoli di KPI e KVI minimi
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]
        s = services[service_id]
        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kvi_threshold_{request_id}_{r.id}"
            )

    # Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Definizione della funzione obiettivo combinata con pesi
    model.setObjective(
        sum((alpha * weighted_sum_kpi[(r.id, service_requests[request_id])] +
             (1 - alpha) * weighted_sum_kvi[(r.id, service_requests[request_id])]) * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    # Impostazioni per IP puro senza simplesso o branch-and-bound
    model.setParam("Heuristics", 0.0)   # Disabilita euristiche
    model.setParam("Presolve", 0)       # Disabilita pre-elaborazione
    model.setParam("Cuts", 0)           # Disabilita tutti i tagli
    model.setParam("Method", 0)        # Disabilita il simplesso
    model.setParam("NodeMethod", -1)    # Nessun rilassamento continuo nei nodi
    model.setParam("LPWarmStart", 0)    # Evita l'uso di LP per inizializzare
    model.setParam("SolutionLimit", 1e9) # Nessun limite artificiale
    model.setParam("IntegralityFocus", 1) # Forza a lavorare solo con interi
    model.setParam("Threads", 1)        # Un solo thread
    model.setParam("MIPGap", 0.0)       # Nessuna tolleranza
    model.setParam("Symmetry", 0)       # Disabilita riduzione simmetrica
    model.setParam("VarBranch", 0)      # Nessun branching avanzato
    model.setParam("RINS", 0)           # Nessuna ricerca su soluzioni parziali
    model.setParam("NoRelHeurTime", 0)  # Disabilita le euristiche di rilassamento
    model.setParam("DualReductions", 0) # Disabilita riduzioni duali
    model.setParam("BarConvTol", 1e-9)  # Maggiore precisione numerica
    model.setParam("NumericFocus", 3)   # Massima attenzione alla precisione numerica
    model.setParam('MIPFocus', 1)  # Ottimizzazione senza rilassamento (focalizzato sulla ricerca esaustiva)

    # Ottimizza il modello
    model.optimize()

    # Verifica se la soluzione è ottimale
    if model.status == GRB.OPTIMAL:
        kpi_value = sum(
            weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id].X
            for request_id in range(len(service_requests)) for r in resources
        ) * alpha

        kvi_value = sum(
            weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id].X
            for request_id in range(len(service_requests)) for r in resources
        ) * (1 - alpha)

        print(f"\nSoluzione trovata per α={alpha}: KPI={kpi_value}, KVI={kvi_value}")

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                         weighted_sum_kpi, weighted_sum_kvi, results_dir,
                         filename=f"results_weighted_alpha_{alpha}.csv")
        return kpi_value, kvi_value
    else:
        print(f"Il modello non ha trovato una soluzione ottima per α={alpha}.")
        return None, None


def optimize_branch_and_bound(service_requests, services, resources, normalized_kpi, normalized_kvi,
                              weighted_sum_kpi, weighted_sum_kvi, alpha, results_dir):

    model = Model("Branch_And_Bound")

    # Variabili di decisione x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Vincoli: KPI e KVI minimi
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]
        s = services[service_id]

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )
            model.addConstr(
                (weighted_sum_kvi[(r.id, service_id)]) * x[request_id, r.id] >= 0,
                f"kvi_threshold_{request_id}_{r.id}"
            )

    # Ogni servizio è assegnato a una sola risorsa
    for request_id, service_id in enumerate(service_requests):
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # La capacità della risorsa non deve essere superata
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Definizione della funzione obiettivo combinata
    model.setObjective(
        sum((alpha * weighted_sum_kpi[(r.id, service_requests[request_id])] +
             (1 - alpha) * weighted_sum_kvi[(r.id, service_requests[request_id])]) * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    # Forza l'uso del Branch-and-Bound puro
    model.setParam("Heuristics", 0.0)  # Disabilita tutte le euristiche
    model.setParam("Presolve", 0)  # Disabilita la pre-elaborazione
    model.setParam("Cuts", 0)  # Disabilita i tagli di Gomory e altri
    model.setParam("MIPFocus", 2)  # Focalizza sulla qualità dei bound
    model.setParam("VarBranch", 0)  # Usa il branching standard
    model.setParam("Symmetry", 0)  # Disabilita la riduzione della simmetria
    model.setParam("RINS", 0)  # Disabilita la ricerca su soluzioni parziali
    model.setParam("NoRelHeurTime", 0)  # Disabilita le euristiche di rilassamento

    # Ottimizza il modello
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        kpi_value = sum(
            weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id].X
            for request_id in range(len(service_requests)) for r in resources
        ) * alpha
        kvi_value = sum(
            weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id].X
            for request_id in range(len(service_requests)) for r in resources
        ) * (1 - alpha)

        print(f"\nSoluzione trovata con Branch-and-Bound puro: KPI={kpi_value}, KVI={kvi_value}")

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                         weighted_sum_kpi, weighted_sum_kvi, results_dir,
                         filename=f"results_branch_bound_alpha_{alpha}.csv")

        return kpi_value, kvi_value

    else:
        print("Il modello non ha trovato una soluzione ottima.")
        return None, None

