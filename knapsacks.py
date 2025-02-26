import numpy as np
import os
import pandas as pd


# multi knapsack con dynamic programming. total value è valore funzione obiettivo del problema rilassato. quindi UB
def multi_knapsack_dp(services, resources, weighted_sum_kpi, weighted_sum_kvi, lambda_, alpha=0.5):
    N = len(resources)  # Numero di risorse (zaini)
    J = len(services)  # Numero di servizi (oggetti)

    # Capacità di ogni risorsa
    availabilities = np.array([r.availability for r in resources], dtype=int)

    # Domanda di ogni servizio
    demands = np.array([s.demand for s in services], dtype=int)

    # Matrice dei valori dei servizi per ogni risorsa
    values = np.zeros((J, N))
    for j in range(J):
        for n in range(N):
            kpi_value = weighted_sum_kpi.get((services[j].id, resources[n].id), 0)
            kvi_value = weighted_sum_kvi.get((services[j].id, resources[n].id), 0)
            values[j, n] = (
                    alpha * (kpi_value - services[j].min_kpi) + (1 - alpha) * (kvi_value - services[j].min_kvi) + lambda_[j]
            )

    # Tabella DP: dp[j][n][w] registra il valore massimo con j servizi, n-esima risorsa e capacità w
    dp = np.zeros((J + 1, N, max(availabilities) + 1))

    # Traccia delle assegnazioni (servizio → risorsa), vettore lungo J di -1 (non assegnato)
    item_assignment = [-1] * J  # avrò praticamente un vettore di J elementi. l'entry j-esima rappresenta il matching
    # del servizio j-esimo. sono praticamente ordinate. se voglio usarlo in futuro sulla linea dell'ottimizzatore
    # dovrò aggiungere id etc

    # Costruzione tabella DP
    for j in range(1, J + 1):
        for n in range(N):
            for w in range(availabilities[n] + 1):
                dp[j][n][w] = dp[j - 1][n][w]  # Non assegnare il servizio j

                # Caso: il servizio j è assegnato alla risorsa n (se c'è spazio)
                if demands[j - 1] <= w:
                    dp[j][n][w] = max(
                        dp[j][n][w],
                        dp[j - 1][n][w - demands[j - 1]] + values[j - 1, n]
                        # servizio assegnato una sola volta. agg demand
                    )

    # Ricostruzione della soluzione
    remaining_capacities = availabilities.copy()
    for j in range(J, 0, -1):
        best_knapsack = -1
        best_value = -float("inf")

        for n in range(N):
            if remaining_capacities[n] >= demands[j - 1] and dp[j][n][remaining_capacities[n]] > best_value:
                best_value = dp[j][n][remaining_capacities[n]]
                best_knapsack = n

        if best_knapsack != -1:
            item_assignment[j - 1] = best_knapsack
            remaining_capacities[best_knapsack] -= demands[j - 1]

    # Valore totale ottimale -> ma non è quello mio lagrangiano
    total_value = sum(dp[J, n, availabilities[n]] for n in range(N))

    return total_value, item_assignment


def compute_total_value_lagrangian(services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi,
                                   lambda_, total_value, alpha=0.5):
    # penalty_kpi = 0
    # penalty_kvi = 0

    # for j, assigned_r in enumerate(item_assignment):
    #     if assigned_r != -1:  # Se il servizio è assegnato
    #         s = services[j]
    #         r = resources[assigned_r]
    #         kpi_value = weighted_sum_kpi.get((s.id, r.id), 0)
    #         kvi_value = weighted_sum_kvi.get((s.id, r.id), 0)
    #
    #         # Valore della funzione obiettivo con KPI e KVI
    #         service_value = alpha * kpi_value + (1 - alpha) * kvi_value
    #         total_value_lagrangian += service_value
    #
    #         # Penalizzazioni per KPI e KVI violati
    #         penalty_kpi += lambda_[j, assigned_r] * (s.min_kpi - kpi_value)
    #         penalty_kvi += mu_[j, assigned_r] * (s.min_kvi - kvi_value)

    # Penalizzazione vincolo di assegnazione, quindi la sommatoria finale
    penalty_lambda = np.sum(lambda_)

    # Valore totale considerando tutte le penalizzazioni
    total_value += penalty_lambda

    return total_value


def save_results_csv_lagrangian(services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi, results_dir,
                                filename):

    filepath = os.path.join(results_dir, filename)
    results = []

    for j, assigned_r in enumerate(item_assignment):
        if assigned_r != -1:
            s = services[j]
            r = resources[assigned_r]

            results.append([
                s.id, r.id, 1,
                weighted_sum_kpi.get((s.id, r.id), 0),
                weighted_sum_kvi.get((s.id, r.id), 0),
                s.min_kpi, s.min_kvi
            ])

    df = pd.DataFrame(results, columns=[
        "Service_ID", "Resource_ID", "Assigned",
        "Weighted_Sum_KPI", "Weighted_Sum_KVI",
        "Min_KPI", "Min_KVI"
    ])

    df.to_csv(filepath, index=False)
    print(f"Risultati salvati in {filepath}")


###########

#  algoritmo 2: riparazione soluzione. per ogni servizio, si verifica se è stato assegnato a più risorse, mantenendo
#  quello con valore migliore. si controlla se sono stati rispettati vincoli minimi ed eventualmente si ri-assegna. si
#  assegna un servizio che non è stato mai assegnato in modo analogo

def repair_solution(services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi, lambda_, alpha):

    repaired_assignment = item_assignment.copy()
    remaining_capacities = {r.id: r.availability for r in resources}  # Capacità residue per risorsa

    for j, assigned_r in enumerate(item_assignment):  # j è l'indice del servizio, assigned_r è la risorsa assegnata
        s = services[j]

        # Se il servizio è assegnato a una risorsa
        if assigned_r == -1:

            best_resource = -1
            best_value = -float("inf")

            # Ricerca risorsa migliore
            for new_r in resources:
                if remaining_capacities[new_r.id] >= s.demand:
                    new_kpi = weighted_sum_kpi.get((s.id, new_r.id), 0)
                    new_kvi = weighted_sum_kvi.get((s.id, new_r.id), 0)

                    total_value = (lambda_[j] - alpha * (new_kpi - s.min_kpi) +
                                   (1 - alpha) * ((new_kvi - s.min_kvi) / s.demand))

                    if total_value > best_value:
                        best_value = total_value
                        best_resource = new_r.id

                # Se abbiamo trovato una nuova risorsa migliore, aggiorniamo l'assegnazione
                if best_resource != -1:
                    remaining_capacities[assigned_r] += s.demand  # Ripristiniamo la capacità
                    remaining_capacities[best_resource] -= s.demand  # Aggiorniamo la nuova capacità
                    repaired_assignment[j] = best_resource
                else:
                    # Se non esiste una risorsa valida, il servizio rimane non assegnato
                    repaired_assignment[j] = -1

    return repaired_assignment


#  mi serve valore funzione obiettivo di partenza per lower bound

def compute_total_value(services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi, alpha=0.5):
    total_value = 0

    for j, assigned_r in enumerate(item_assignment):
        if assigned_r != -1:  # Se il servizio è assegnato
            s = services[j]
            r = resources[assigned_r]
            kpi_value = weighted_sum_kpi.get((s.id, r.id), 0)
            kvi_value = weighted_sum_kvi.get((s.id, r.id), 0)

            # Valore totale per questa assegnazione
            service_value = alpha * (kpi_value - s.min_kpi) + (1 - alpha) * (kvi_value - s.min_kvi)
            total_value += service_value  # Somma al valore totale

    return total_value


#########


def update_lagrangian_multipliers(services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi,
                                  lambda_, UB, LB, total_value_lagrangian, total_value_feasible, z=0.1):
    # Calcolo dei subgradienti
    gamma_lambda = np.zeros(len(services))
    # np.zeros((len(services), len(resources)))
    # gamma_mu = np.zeros((len(services), len(resources)))
    # gamma_nu = np.zeros(len(services))

    for j, assigned_r in enumerate(item_assignment):
        if assigned_r != -1:  # Se il servizio è stato assegnato
            s = services[j]
            r = resources[assigned_r]

            # gamma_lambda[j, assigned_r] = s.min_kpi - weighted_sum_kpi.get((s.id, r.id), 0)
            # gamma_mu[j, assigned_r] = s.min_kvi - weighted_sum_kvi.get((s.id, r.id), 0)
            gamma_lambda[j] = 1 - sum(
                1 for _ in item_assignment if _ == assigned_r)  # Numero di assegnazioni sulla stessa risorsa

    # Concatenazione dei subgradienti per la norma euclidea
    gamma = np.concatenate([gamma_lambda.flatten()])
    norm_gamma = np.linalg.norm(gamma)

    # Aggiornamento dei bound
    UB = min(UB, total_value_lagrangian)
    LB = max(LB, total_value_feasible)

    # Calcolo dello step size
    step_size = z * (UB - LB) / (norm_gamma ** 2) if norm_gamma > 0 else 0

    # Aggiornamento dei moltiplicatori lagrangiani con proiezione a valori non negativi
    lambda_ = np.maximum(0, lambda_ + step_size * gamma_lambda)
    # mu_ = np.maximum(0, mu_ + step_size * gamma_mu)
    # nu_ = np.maximum(0, nu_ + step_size * gamma_nu)

    return lambda_, UB, LB

## check

def is_feasible_solution(services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi):

    # Controllo che ogni servizio sia assegnato esattamente una volta
    if item_assignment.count(-1) != 0:
        return False # Se ci sono servizi non assegnati, la soluzione non è feasible


    # Controllo che nessuna risorsa sia sovraccarica
    resource_load = {r.id: 0 for r in resources}

    for j, assigned_r in enumerate(item_assignment):
        if assigned_r != -1:
            s = services[j]
            resource_load[assigned_r] += s.demand

    for r in resources:
        if resource_load[r.id] > r.availability:
            return False  # Se una risorsa è sovraccarica, la soluzione non è feasible

    return True  # Se tutti i vincoli sono soddisfatti, la soluzione è feasible