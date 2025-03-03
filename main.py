from benchmark import *
from initialization import compute_computation_time
from knapsacks import *
from optimization import *
from initialization import *
from timeit import default_timer as timer
import time, sys, datetime
from datetime import date
import numpy as np
import random
import csv
import os


class Service:
    def __init__(self, id, demand, min_kpi, min_kvi, kpi_service_req, kvi_service_req, kpi_service, kvi_service,
                 weights_kpi, weights_kvi, flops, p_s):
        self.id = id
        self.demand = demand
        self.min_kpi = 0  # valore minimo globale tollerabile kpi
        self.min_kvi = 0  # valore minimo globale tollerabile kvi
        self.kpi_service_req = np.array(kpi_service_req)  # requested minimum
        self.kvi_service_req = np.array(kvi_service_req)  # requested minimum
        self.kpi_service = np.array(kpi_service)  # 4 KPI, valore desiderato
        self.kvi_service = np.array(kvi_service)  # 3 KVI, valore desiderato
        self.weights_kpi = np.array(weights_kpi)  # per calcolo kpi globale
        self.weights_kvi = np.array(weights_kvi)  # per calcolo kvi globale
        self.flops = flops
        self.p_s = p_s

    def __getitem__(self, key):
        return getattr(self, key, None)  # Restituisce l'attributo se esiste, altrimenti None


class Resource:
    def __init__(self, id, availability, kpi_resource, kvi_resource, n_c, P_c, u_c, n_m, P_m, speed, fcp, N0, lmbd):
        self.id = id
        self.availability = availability
        self.kpi_resource = np.array(kpi_resource)
        self.kvi_resource = np.array(kvi_resource)
        self.n_c = n_c
        self.P_c = P_c
        self.u_c = u_c
        self.n_m = n_m  # memory available GBytes
        self.P_m = P_m
        self.speed = speed
        self.fpc = fcp
        self.N0 = N0
        self.lmbd = lmbd

    def __getitem__(self, key):
        return getattr(self, key, None)  # Restituisce l'attributo se esiste, altrimenti None


if __name__ == '__main__':

    num_services_list = [5]
    delta = 0.1
    num_resources = 5
    min_kpi = 0
    min_kvi = 0
    weights_kpi = [1 / 3, 1 / 3, 1 / 3]
    weights_kvi = [1 / 3, 1 / 3, 1 / 3]
    services = [
        Service(id=0, demand=2, min_kpi=0, min_kvi=0,
                kpi_service_req=[0.8, 80, 40], kvi_service_req=[2000, 8, 0.4],
                kpi_service=[0.6, 100, 30], kvi_service=[800, 10, 0.2],
                weights_kpi=weights_kpi, weights_kvi=weights_kvi,
                flops=2e6, p_s=2),

        Service(id=1, demand=1, min_kpi=0, min_kvi=0,
                kpi_service_req=[0.05, 70, 50], kvi_service_req=[900, 8, 0.3],
                kpi_service=[0.02, 100, 40], kvi_service=[500, 15, 0.5],
                weights_kpi=weights_kpi, weights_kvi=weights_kvi,
                flops=2e6, p_s=3),

        Service(id=2, demand=3, min_kpi=0, min_kvi=0,
                kpi_service_req=[0.2, 180, 65], kvi_service_req=[1100, 4, 0.5],
                kpi_service=[0.150, 250, 25], kvi_service=[900, 5, 0.2],
                weights_kpi=weights_kpi, weights_kvi=weights_kvi,
                flops=5e6, p_s=4),

        Service(id=3, demand=2, min_kpi=0, min_kvi=0,
                kpi_service_req=[0.08, 1000, 60], kvi_service_req=[1000, 6, 0.2],
                kpi_service=[0.01, 250, 30], kvi_service=[900, 5, 0.2],
                weights_kpi=weights_kpi, weights_kvi=weights_kvi,
                flops=1e7, p_s=5)
    ]

    for num_services in num_services_list:
        results_dir = f"test_benchmark_{num_services}_200_{delta}_1"
        path_onedrive = (r"C:\Users\Federica\OneDrive - Politecnico di Bari\phd\works\comnet\Simulazioni")
        full_path = os.path.join(path_onedrive, results_dir)
        os.makedirs(full_path, exist_ok=True)

        # Probabilità assegnate ai servizi (esempio: servizio 2 e 3 più richiesti)
        probabilities = [0.1, 0.2, 0.4, 0.3]  # Somma = 1

        # Generazione delle richieste basate sulla distribuzione
        service_requests = np.random.choice(range(len(services)), size=num_services, p=probabilities)
        print("Distribuzione delle richieste di servizio:", service_requests)

        # start = timer()
        start = time.time()
        # random.seed(30)

        # Inizializzo J servizi e N risorse.
        # Questi sono liste di oggetti Resource e Service: per .get() ogni elemento (index) di una singola istanza,
        # la ind-esima, posso usare anche services[ind].parametro[index].

        # demand_values = np.random.randint(1, 4, num_services)
        # flops_values = np.random.randint(1000000, 100000000, num_services)
        # p_s_values = np.random.randint(1, 6, num_services)

        availability_values = np.random.randint(5, 20, num_resources)
        n_c_values = np.random.uniform(1, 4, num_resources)
        P_c_values = np.random.uniform(0.01, 0.04, num_resources)
        u_c_values = np.random.uniform(0.001, 1, num_resources)
        n_m_values = np.random.randint(2000000, 6000000, num_resources)
        P_m_values = np.random.uniform(0.01, 0.02, num_resources)
        speed_values = np.random.uniform(50e9, 150e9, num_resources)  # Hz
        fcp_values = np.random.randint(1000, 1000000, num_resources)
        N0 = 10e-10
        lmbd_values = np.random.randint(1, 100, num_resources)

        # Indicators offered by the resources

        deadlines_off = np.random.uniform(0.01, 1, num_resources)
        data_rates_off = np.random.uniform(10, 350, num_resources)
        plr_off = np.random.uniform(1, 100, num_resources)

        # KPI servizio

        # for j in range(num_services):
        #     deadlines = np.random.uniform(0.01, 1, 2)
        #     data_rates = np.random.uniform(10, 250, 2)
        #     plr = np.random.uniform(1, 100, 2)
        #     env_sustainability = np.random.randint(900, 4000, 2)
        #     trust = np.random.randint(1, 20, 2)
        #     inclusiveness = np.random.uniform(0.1, 1)
        #
        #     services.append(Service(
        #         j, demand=demand_values[j], min_kpi=min_kpi, min_kvi=min_kvi,
        #         kpi_service_req=[np.max(deadlines), np.min(data_rates), np.max(plr)],
        #         kvi_service_req=[np.max(env_sustainability), np.min(trust), np.max(inclusiveness)],
        #         kpi_service=[np.min(deadlines), np.max(data_rates), np.min(plr)],
        #         kvi_service=[np.min(env_sustainability), np.max(trust), np.min(inclusiveness)],
        #         weights_kpi=weights_kpi, weights_kvi=weights_kvi,
        #         flops=flops_values[j], p_s=p_s_values[j]
        #     ))
        #     print(f"Service id: {services[j].id}, {services[j].demand}, {services[j].min_kpi}, {services[j].min_kvi}, "
        #           f"{services[j].kpi_service}, {services[j].kpi_service_req}, {services[j].kvi_service}, {services[j].kvi_service_req}, "
        #           f"{services[j].flops}, {services[j].p_s}")

        resources = [Resource(n, availability=availability_values[n],
                              kpi_resource=[deadlines_off[n], data_rates_off[n], plr_off[n]], kvi_resource=[0, 0, 0],
                              n_c=n_c_values[n], P_c=P_c_values[n], u_c=u_c_values[n],
                              n_m=n_m_values[n], P_m=P_m_values[n],
                              speed=speed_values[n], fcp=fcp_values[n], N0=N0,
                              lmbd=lmbd_values[n]) for n in range(num_resources)]

        # for resource in resources:
        #     print(resource.id, resource.availability, resource.kpi_resource, resource.n_c, resource.n_m, resource.fpc,
        #           resource.P_m, resource.P_c, resource.speed, resource.lmbd)

        # test: da cambiare ogni normalized_kpi con weighted_sum_kpi e stessa cosa per kvi

        q_v_big_req(services, [-1, 1, -1], [1, -1, -1])
        # for s in services:
        #     print(f"min q rec total {s.min_kpi}, min v rec total {s.min_kvi}")

        for service in services:
            for resource in resources:
                computation_time = compute_computation_time(service, resource)
                # print(computation_time)

        # TIS
        normalized_kvi, weighted_sum_kvi = compute_normalized_kvi(services, resources, CI=475, signs=[1, -1,
                                                                                                      -1])  #
        # trustworthiness inclusiveness sustainability


        # for k, v in weighted_sum_kvi.items():
        #     print(f"service: {k[0]}, resource: {k[1]}: {v}")

        normalized_kpi, weighted_sum_kpi = compute_normalized_kpi(services, resources, signs=[-1, 1,
                                                                                              -1])  # latenza,
        # (utilizzo banda,) data rate e plr

        # for k, v in weighted_sum_kpi.items():
        #     print(f"service: {k[0]}, resource: {k[1]}: {v}")

        # normalized_kpi, weighted_sum_kpi = normalized_kpi(services, resources, [-1, -1, 1, -1])
        # normalized_kvi, weighted_sum_kvi = normalized_kvi(services, resources, [1, -1, -1])
        #
        # for (res_id, serv_id), q_x in weighted_sum_kpi.items():
        #     print(f"Resource {res_id} takes on service {serv_id} with a global kpi of {q_x}")
        #
        # for (res_id, serv_id), norm_kpi in normalized_kpi.items():
        #     print(f"Resource {res_id} takes on service {serv_id} with normalized kpis of {norm_kpi}")

        # for (res_id, serv_id), v_x in weighted_sum_kvi.items():
        #     print(f"Resource {res_id} takes on service {serv_id} with a global kvi of {v_x}")
        #
        # for (res_id, serv_id), norm_kvi in normalized_kvi.items():
        #     print(f"Resource {res_id} takes on service {serv_id} with normalized kvis of {norm_kvi}")
        #
        # V_I = optimize_kvi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
        #                    results_dir)
        #
        # Q_I = optimize_kpi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi,
        #                    weighted_sum_kvi, results_dir)
        #
        # V_N = v_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi,
        #               weighted_sum_kpi, weighted_sum_kvi, Q_I, results_dir)
        #
        # Q_N = q_nadir(service_requests, services, resources, normalized_kpi,
        #               normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I, results_dir)
        #
        #
        #
        # pareto_solutions_exact = epsilon_constraint_exact(service_requests, services, resources, normalized_kpi, normalized_kvi,
        # weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=delta, results_dir=results_dir)

        # plot_pareto_front(pareto_solutions_exact)
        # save_pareto_solutions(service_requests, pareto_solutions_exact, filename="pareto_solutions.csv")
        assignment, total_kpi, total_kvi = greedy_assignment_kpi(service_requests, services, resources,
                                                                 weighted_sum_kpi,
                                                                 weighted_sum_kvi, max_assignments=10)

        save_assignment_results(service_requests, assignment, services, resources,
                                weighted_sum_kpi, weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi,
                                total_kvi,
                                results_dir=results_dir, filename="greedy_kpi_results.csv")

        assignment, total_kpi, total_kvi = random_assignment(service_requests, services, resources, weighted_sum_kpi,
                                                             weighted_sum_kvi)

        print(
            f"Assignment: {assignment}, Total KPI: {total_kpi}, Total KVI: {total_kvi}, service_requests: {service_requests}")

        save_assignment_results(service_requests, assignment, services, resources, weighted_sum_kpi,
                                weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi, total_kvi,
                                results_dir=results_dir,
                                filename="random_results.csv")

        # end = timer()
        # pareto_filename = os.path.join(results_dir, "pareto_solutions.csv")
        # save_pareto_solutions(pareto_solutions_exact, filename=pareto_filename)

        # Parametri del metodo subgradiente
        max_iterations = 20  # Numero massimo di iterazioni
        tolerance = 1e-3  # Soglia di convergenza
        z = 0.5  # Parametro per lo step size

        # Inizializzazione dei moltiplicatori lagrangiani
        lambda_ = np.ones(len(service_requests)) * 0.1

        # Inizializzazione dei bound
        UB = float("inf")  # Upper Bound iniziale
        LB = float("-inf")  # Lower Bound iniziale

        # Loop iterativo per il metodo subgradiente
        for k in range(max_iterations):
            # Zaini
            total_value_not_lagrangian, item_assignment = multi_knapsack_dp(
                service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi, lambda_
            )

            #  Total value lagrangian
            total_value_lagrangian = compute_total_value_lagrangian(services, resources,
                                                                    item_assignment,
                                                                    weighted_sum_kpi, weighted_sum_kvi,
                                                                    lambda_, total_value_not_lagrangian, alpha=0.5)

            print("Valore totale lagrangiano:", total_value_not_lagrangian)
            print("Valore totale lagrangiano corretto:", total_value_lagrangian)
            print("Assegnazione lagrangiana:", item_assignment)

            if is_feasible_solution(service_requests, services, resources, item_assignment, weighted_sum_kpi,
                                    weighted_sum_kvi):
                print(f"Soluzione feasible trovata all'iterazione {k + 1}, interrompo l'ottimizzazione.")
                save_results_csv_lagrangian(service_requests,
                                            services, resources, item_assignment, weighted_sum_kpi, weighted_sum_kvi,
                                            results_dir=results_dir, filename=f"iteration_{k + 1}.csv"
                                            )
                break

            # Riparazione
            item_assignment_repaired = repair_solution(service_requests,
                                                       services, resources, item_assignment, weighted_sum_kpi,
                                                       weighted_sum_kvi, min_kpi, min_kvi
                                                       )

            # Valore f. obiettivo con soluzione feasible (riparata)
            total_value_feasible = compute_total_value(service_requests,
                                                       services, resources, item_assignment_repaired, weighted_sum_kpi,
                                                       weighted_sum_kvi
                                                       )

            print("Valore totale riparato:", total_value_feasible)
            print("Assegnazione riparata:", item_assignment_repaired)

            # Aggiorna i moltiplicatori di Lagrange, lo step size, UB e LB
            lambda_, UB, LB = update_lagrangian_multipliers(service_requests,
                                                            services, resources, item_assignment_repaired,
                                                            weighted_sum_kpi, weighted_sum_kvi,
                                                            lambda_, UB, LB, total_value_lagrangian,
                                                            total_value_feasible, z
                                                            )

            save_results_csv_lagrangian(service_requests,
                                        services, resources, item_assignment_repaired, weighted_sum_kpi,
                                        weighted_sum_kvi,
                                        results_dir=results_dir, filename=f"iteration_{k + 1}.csv"
                                        )

            # Convergenza
            gap = (UB - LB) / max(1, abs(LB))
            print(f"Iterazione {k + 1}: UB = {UB:.4f}, LB = {LB:.4f}, Gap = {gap:.6f}")

            if gap < tolerance:
                print("The covergence was reached.")
                break

            print(f"Valore finale UB: {UB}, LB: {LB}")

        # Tempo di esecuzione
        end_time = time.time()
        time_elapsed = end_time - start

        # with open(os.path.join(results_dir, "execution_time.txt"), "w") as file:
        #     file.write(f"Servizi: {num_services}, Tempo: {time_elapsed:.6f} sec\n")

        print(f"Completato per {num_services} servizi. Tempo: {time_elapsed:.6f} sec")
