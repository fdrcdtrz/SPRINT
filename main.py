from optimization import *
from initialization import *
import numpy as np
import random


class Service:
    def __init__(self, id, demand, min_kpi, min_kvi, kpi_service, kvi_service, weights_kpi, weights_kvi, flops, p_s):
        self.id = id
        self.demand = demand
        self.min_kpi = min_kpi  # valore minimo globale tollerabile kpi
        self.min_kvi = min_kvi  # valore minimo globale tollerabile kvi
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
        self.n_m = n_m
        self.P_m = P_m
        self.speed = speed
        self.fpc = fcp
        self.N0 = N0
        self.lmbd = lmbd

    # da aggiunere i parametri per il calcolo di KVI (n_c, P_c, u_c, n_m, P_m, speed, fpc, N0, lambda)

# da aggiungere da qualche parte la definizione di questi due parametri: g_sj^{r_n}, g_sj^{r_n, e}, guadagno canale che dipende da servizi e risorse


if __name__ == '__main__':

    # inizializzo J servizi e N risorse.
    # Questi sono liste di oggetti Resource e Service: per .get() ogni elemento index di una singola istanza, la ind,
    # mi serve services[ind].parametro[index]

    services = [Service(j, demand=random.randint(1, 5), min_kpi=0.001, min_kvi=0.001, kpi_service=[random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)],
                        kvi_service=[random.randint(10, 50), random.randint(1, 5), random.uniform(1e-6,1)], weights_kpi=[0.25, 0.25, 0.25, 0.25], weights_kvi=[0.33, 0.33, 0.33], flops=2000, p_s=50) for j in range(5)]

    resources = [Resource(n, availability=random.randint(10, 50), kpi_resource=[random.randint(1, 20), random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)], kvi_resource=[random.randint(1, 5), random.randint(1, 5), random.uniform(1e-6,1)], n_c=20, P_c=50, u_c=20, n_m=20, P_m=50, speed=500, fcp=20000, N0=40, lmbd=0.4) for n in range(10)]

    # test: da cambiare ogni normalized_kpi con weighted_sum_kpi e stessa cosa per kvi

    normalized_kpi, weighted_sum_kpi = normalized_kpi(services, resources, [1, -1, 1, -1])
    normalized_kvi, weighted_sum_kvi = normalized_kvi(services, resources, [1, -1, 1])
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

    # V_I = optimize_kvi(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi)
    # Q_I = optimize_kpi(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi)
    # V_N = v_nadir(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_I)
    # Q_N = q_nadir(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I)


    #pareto_solutions_exact = epsilon_constraint_exact(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.01)

    #plot_pareto_front(pareto_solutions_exact)
    #
    # pareto_solutions_filtered = filter_pareto_solutions(pareto_solutions_exact)
    # plot_pareto_front(pareto_solutions_filtered)
    #
    # final_solution = cut_and_solve(services, resources, normalized_kpi, normalized_kvi)
    #
    # pareto_solutions = cut_and_solve(services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=0.001, max_iters=10, tolerance=1e-5, cost_threshold=0.1)
    # print(pareto_solutions)
    #plot_pareto_front(pareto_solutions)