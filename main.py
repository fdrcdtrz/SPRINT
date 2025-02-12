from benchmark import *
from optimization import *
from initialization import *
from timeit import default_timer as timer
import time, sys, datetime
from datetime import date
import numpy as np
import random
import csv

class Service:
    def __init__(self, id, demand, min_kpi, min_kvi, kpi_service_req, kvi_service_req, kpi_service, kvi_service, weights_kpi, weights_kvi, flops, p_s):
        self.id = id
        self.demand = demand
        self.min_kpi = min_kpi  # valore minimo globale tollerabile kpi
        self.min_kvi = min_kvi  # valore minimo globale tollerabile kvi
        self.kpi_service_req = np.array(kpi_service_req) # requested minimum
        self.kvi_service_req = np.array(kvi_service_req) # requested minimum
        self.kpi_service = np.array(kpi_service)  # 4 KPI, valore desiderato
        self.kvi_service = np.array(kvi_service)  # 3 KVI, valore desiderato
        self.weights_kpi = np.array(weights_kpi)  # per calcolo kpi globale
        self.weights_kvi = np.array(weights_kvi)  # per calcolo kvi globale
        self.flops = flops
        self.p_s = p_s

class Resource:
    def __init__(self, id, availability, kpi_resource, kvi_resource, n_c, P_c, u_c, n_m, P_m, speed, fcp, N0, lmbd):
        self.id = id
        self.availability = availability
        self.kpi_resource = np.array(kpi_resource)
        self.kvi_resource = np.array(kvi_resource)
        self.n_c = n_c
        self.P_c = P_c
        self.u_c = u_c
        self.n_m = n_m #memory available GBytes
        self.P_m = P_m
        self.speed = speed
        self.fpc = fcp
        self.N0 = N0
        self.lmbd = lmbd


if __name__ == '__main__':

    # start = timer()
    start =  time.time()
    #random.seed(30)

    # inizializzo J servizi e N risorse.
    # Questi sono liste di oggetti Resource e Service: per .get() ogni elemento index di una singola istanza, la ind,
    # mi serve services[ind].parametro[index]

    services = [Service(j, demand=random.randint(1, 3), min_kpi=0, min_kvi=0, kpi_service_req=[random.uniform(200e-3, 600e-3), random.uniform(1e-6, 10), random.randint(1, 10), random.uniform(80, 100)], kvi_service_req=[random.randint(1, 5), random.uniform(0.8,1), random.randint(400,1000)], kpi_service=[random.uniform(100e-3, 200e-3), random.uniform(0.1, 100), random.randint(1, 50), random.uniform(0.1, 100)],
                        kvi_service=[random.randint(1, 30), random.uniform(1e-6,1), random.randint(200,400)], weights_kpi=[0.25, 0.25, 0.25, 0.25], weights_kvi=[0.33, 0.33, 0.33], flops=random.uniform(1E3,1E6), p_s=random.randint(1,5)) for j in range(170)]

    resources = [Resource(n, availability=random.randint(25, 85), kpi_resource=[random.uniform(10e-3, 300e-3), random.randint(5, 50), random.randint(40, 200), random.randint(1, 20)], kvi_resource=[0,0,0], n_c=random.uniform(1, 4), P_c=random.randint(10, 50), u_c=random.uniform(0.001,1), n_m=random.randint(2e3, 6e3), P_m=random.randint(10, 50), speed=random.uniform(50e9, 150e9), fcp=random.uniform(1e6,1e9), N0=10e-10, lmbd=random.uniform(0.001,1)) for n in range(200)]

    # test: da cambiare ogni normalized_kpi con weighted_sum_kpi e stessa cosa per kvi

    q_v_big_req(services, [-1, -1, 1, -1], [1, -1, -1])
    # for s in services:
    #     print(f"min q rec total {s.min_kpi}, min v rec total {s.min_kvi}")

    for service in services:
        for resource in resources:
            computation_time = compute_computation_time(service, resource)
            #print(computation_time)

    # TIS
    normalized_kvi, weighted_sum_kvi = compute_normalized_kvi(services, resources, CI=475, signs=[1, -1, -1]) # trustworthiness inclusiveness sustainability
    # for k, v in weighted_sum_kvi.items():
    #     print(f"service: {k[0]}, resource: {k[1]}: {v}")

    normalized_kpi, weighted_sum_kpi = compute_normalized_kpi(services, resources, signs=[-1, -1, 1, -1]) # latenza, utilizzo banda, data rate e plr
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
    V_I = optimize_kvi(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi)
    Q_I = optimize_kpi(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi)
    V_N = v_nadir(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_I)
    Q_N = q_nadir(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I)
    # #
    # #
    pareto_solutions_exact = epsilon_constraint_exact(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.01)

    plot_pareto_front(pareto_solutions_exact)
    save_pareto_solutions(pareto_solutions_exact, filename="pareto_solutions.csv")
    #
    # pareto_solutions_filtered = filter_pareto_solutions(pareto_solutions_exact)
    # plot_pareto_front(pareto_solutions_filtered)
    #
    # final_solution = cut_and_solve(services, resources, normalized_kpi, normalized_kvi)
    #
    # pareto_solutions = cut_and_solve(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.001, max_iters=10, tolerance=1e-5, cost_threshold=0.1)
    # print(pareto_solutions)
    #plot_pareto_front(pareto_solutions)

    # pareto_solutions = cut_and_solve(services, resources, normalized_kpi, normalized_kvi,
    #                   weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.1,
    #                   max_iters=10, tolerance=1e-5, cost_threshold=1, max_inner_iters=5)

    # #pareto_solutions = cut_and_solve(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I,
    #                   #delta=0.01, max_iters=10, tolerance=1e-5, cost_threshold=0.0001)
    #
    # pareto_solutions = branch_and_bound_pareto(services, resources, normalized_kpi, normalized_kvi,
    #                             weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.01)

    #print(pareto_solutions)

    # assignment, total_kpi, total_kvi = greedy_assignment_kpi(services, resources, weighted_sum_kpi)
    # save_assignment_results(assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi,
    #                         normalized_kpi, normalized_kvi, total_kpi, total_kvi, "greedy_kpi_results.csv")
    #
    # assignment, total_kpi, total_kvi = greedy_assignment_kvi(services, resources, weighted_sum_kvi)
    # save_assignment_results(assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi,
    #                         normalized_kpi, normalized_kvi, total_kpi, total_kvi, "greedy_kvi_results.csv")
    #
    # assignment, total_kpi, total_kvi = random_assignment(services, resources, weighted_sum_kpi, weighted_sum_kvi)
    # save_assignment_results(assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi,
    #                         normalized_kpi, normalized_kvi, total_kpi, total_kvi, "random_results.csv")

    # end = timer()
    end = time.time()
    time_elapsed = end - start

    print(f"Time elapsed: {time_elapsed}")
    with open("execution_time.txt", "a") as file:
        file.write(f"Servizi: {len(services)}, Risorse: {len(resources)}, Tempo: {time_elapsed:.6f} sec\n")
