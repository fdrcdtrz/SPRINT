import os
import time

import numpy as np

from benchmark import *
from initialization import *
from knapsacks import *
from optimization import *


class Service:

    def __init__(self, id, demand, min_kpi, min_kvi, impact, kpi_service_req, kvi_service_req, kpi_service, kvi_service,
                 weights_kpi, weights_kvi, size):
        self.id = id
        self.demand = demand
        self.min_kpi = 0  # valore minimo globale tollerabile kpi
        self.min_kvi = 0
        self.impact = impact  # valore minimo globale tollerabile kvi
        self.kpi_service_req = np.array(kpi_service_req)  # requested minimum
        self.kvi_service_req = np.array(kvi_service_req)  # requested minimum
        self.kpi_service = np.array(kpi_service)  # 4 KPI, valore desiderato
        self.kvi_service = np.array(kvi_service)  # 3 KVI, valore desiderato
        self.weights_kpi = np.array(weights_kpi)  # per calcolo kpi globale
        self.weights_kvi = np.array(weights_kvi)  # per calcolo kvi globale
        self.size = size
        # self.risk_appetite = risk_appetite

    # property            # first decorate the getter method
    def get_id(self):  # This getter method name is *the* name
        return self.id

    def get_demand(self):  # This getter method name is *the* name
        return self.demand

    def get_min_kpi(self):  # This getter method name is *the* name
        return self.min_kpi

    def get_min_kvi(self):  # This getter method name is *the* name
        return self.min_kvi

    def get_impact(self):  # This getter method name is *the* name
        return self.impact

    def get_kpi_service_req(self):  # This getter method name is *the* name
        return self.kpi_service_req

    def get_kvi_service_req(self):  # This getter method name is *the* name
        return self.kvi_service_req

    def get_kpi_service(self):  # This getter method name is *the* name
        return self.kpi_service

    def get_kvi_service(self):  # This getter method name is *the* name
        return self.kvi_service

    def get_weights_kpi(self):  # This getter method name is *the* name
        return self.weights_kpi

    def get_weights_kvi(self):  # This getter method name is *the* name
        return self.weights_kvi

    def get_size(self):  # This getter method name is *the* name
        return self.size

    # def get_risk_appetite(self):  # This getter method name is *the* name
    #     return self.risk_appetite

    def set_id(self, value):  # This getter method name is *the* name
        self.id = value

    def set_demand(self, value):  # This setter method name is *the* name
        self.demand = value

    def set_min_kpi(self, value):  # This setter method name is *the* name
        self.min_kpi = value

    def set_min_kvi(self, value):  # This setter method name is *the* name
        self.min_kvi = value

    def set_impact(self, value):  # This setter method name is *the* name
        self.impact = value

    def set_kpi_service_req(self, value):  # This setter method name is *the* name
        self.kpi_service_req = value

    def set_kvi_service_req(self, value):  # This setter method name is *the* name
        self.kvi_service_req = value

    def set_kpi_service(self, value):  # This setter method name is *the* name
        self.kpi_service = value

    def set_kvi_service(self, value):  # This setter method name is *the* name
        self.kvi_service = value

    def set_weights_kpi(self, value):  # This setter method name is *the* name
        self.weights_kpi = value

    def set_weights_kvi(self, value):  # This setter method name is *the* name
        self.weights_kvi = value

    def set_size(self, value):  # This setter method name is *the* name
        self.size = value

    # def set_risk_appetite(self, value):  # This setter method name is *the* name
    #     self.risk_appetite = value


class Resource:
    def __init__(self, id, availability, kpi_resource, kvi_resource, carbon_offset, P_c, u_c, P_m, fcp, N0,
                 lambda_failure, lambda_services_per_hour, likelihood):
        self.id = id
        self.availability = availability
        self.kpi_resource = np.array(kpi_resource)
        self.kvi_resource = np.array(kvi_resource)
        self.carbon_offset = carbon_offset
        self.P_c = P_c
        self.u_c = u_c
        self.P_m = P_m
        self.fpc = fcp
        self.N0 = N0
        self.lambda_failure = lambda_failure
        self.lambda_services_per_hour = lambda_services_per_hour
        self.likelihood = likelihood

    def get_availability(self):  # This setter method name is *the* name
        return self.availability

    def get_kpi_resource(self):  # This setter method name is *the* name
        return self.kpi_resource

    def get_kvi_resource(self):  # This setter method name is *the* name
        return self.kvi_resource

    def get_carbon_offset(self):  # This setter method name is *the* name
        return self.carbon_offset

    def get_P_c(self):  # This setter method name is *the* name
        return self.P_c

    def get_u_c(self):  # This setter method name is *the* name
        return self.u_c

    def get_P_m(self):  # This setter method name is *the* name
        return self.P_m

    def get_fpc(self):  # This setter method name is *the* name
        return self.fpc

    def get_N0(self):  # This setter method name is *the* name
        return self.N0

    def get_lambda_failure(self):  # This setter method name is *the* name
        return self.lambda_failure

    def get_lambda_services_per_hour(self):  # This setter method name is *the* name
        return self.lambda_services_per_hour

    def get_likelihood(self):
        return self.likelihood

    def set_availability(self, value):  # This setter method name is *the* name
        self.availability = value

    def set_kpi_resource(self, value):  # This setter method name is *the* name
        self.kpi_resource = value

    def set_kvi_resource(self, value):  # This setter method name is *the* name
        self.kvi_resource = value

    def set_carbon_offset(self, value):  # This setter method name is *the* name
        self.carbon_offset = value

    def set_P_c(self, value):  # This setter method name is *the* name
        self.P_c = value

    def set_u_c(self, value):  # This setter method name is *the* name
        self.u_c = value

    def set_P_m(self, value):  # This setter method name is *the* name
        self.P_m = value

    def set_fpc(self, value):  # This setter method name is *the* name
        self.fpc = value

    def set_N0(self, value):  # This setter method name is *the* name
        self.N0 = value

    def set_lambda_failure(self, value):  # This setter method name is *the* name
        self.lambda_failure = value

    def set_lambda_services_per_hour(self, value):  # This setter method name is *the* name
        self.lambda_services_per_hour = value

    def set_likelihood(self, value):
        self.likelihood = value


if __name__ == '__main__':

    num_services_list = [100] # [100], [80, 85, 90, 95, 100, 105, 110, 115, 120]
    num_services_type = 8
    delta = 0.1
    num_resources = [50, 55, 60, 68, 70, 75, 80, 85, 90] #[50, 55, 60, 65, 70, 75, 80, 85, 90], [80]
    #num_resources = [95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 170, 175, 180, 185, 190, 195, 200]
    weights_kpi = [0.2, 0.2, 0.6]
    weights_kvi = [0.2, 0.2, 0.6]  # tis

    deadlines = [0.002, 0.5, 1, 10, 15]
    deadlines_req = [0.02, 0.6, 1.2, 50]
    plrs = [20.0, 20.0, 30.0, 40.0]
    plrs_req = [35.0, 45.0, 45.0, 50.0]
    data_rates = [70.0, 100.0, 100.0, 250.0]
    data_rates_req = [45.0, 60.0, 80.0, 95.0]
    sizes = [600e6, 1e9, 1e9, 1.2e9]  # Mb
    #p_s_values = [2, 2, 3, 4]
    demand_values = [2, 4, 4, 5]
    impact_values = [0.25, 0.5, 0.75, 1]
    # risk_appetite_values = [1, 2, 2, 3]


    services = []
    for i in range(num_services_type):
        chosen_index = i % len(demand_values)

        service = Service(i, 0, 0, 0,0, 0, 0, 0,
                          0, weights_kpi, weights_kvi, 0)

        deadline = deadlines[chosen_index]
        plr = plrs[chosen_index]
        plr_req = plrs_req[chosen_index]
        data_rate = data_rates[chosen_index]
        impact = impact_values[chosen_index]


        if deadline > 9:
            deadline_req = deadlines_req[len(deadlines_req) - 1]
        else:
            deadline_req = deadlines_req[chosen_index]

        if data_rate == 70:
            data_rate_req = data_rates_req[0]
        else:
            data_rate_req = data_rates_req[chosen_index]

        service.set_kpi_service([deadline, data_rate, plr])
        service.set_kpi_service_req([deadline_req, data_rate_req, plr_req])
        service.set_demand(demand_values[chosen_index])
        service.set_impact(impact)
        service.set_size(sizes[chosen_index])

        services.append(service)

        print(f"Service id: {services[i].id}, {services[i].demand}, {services[i].min_kpi}, {services[i].impact}, "
              f"{services[i].kpi_service}, {services[i].kpi_service_req}, {services[i].kvi_service}, {services[i].kvi_service_req}, "
              f"{services[i].size}")


    for num_services in num_services_list:
        for num_resource in num_resources:
            results_dir = f"prova_benchmark_{num_services}_{num_resource}_{delta}_WI_WKPI" #f"test_results_{num_services}_{num_resource}_{delta}_WT_WKPI"
            # path_onedrive = r"C:\Users\Federica\OneDrive - Politecnico di Bari\phd\works\comnet\Simulazioni"
            path_locale = r"C:\Users\Federica de Trizio\PycharmProjects\CutAndSolve"
            full_path = os.path.join(path_locale, results_dir)
            os.makedirs(full_path, exist_ok=True)

            # Probabilità assegnate ai servizi
            probabilities = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
            service_requests = []

            # Generazione delle richieste basate sulla distribuzione
            for i in range(num_services):
                chosen_index = i % len(probabilities)
                service_requests.append(probabilities[chosen_index])
            print("Distribuzione delle richieste di servizio:", service_requests)

            start = time.time()

            availability_values = [10, 20, 50, 50]
            carbon_offset_values = [(1.5*1e6) / 365, (2*1e6) / 365, (2*1e6) / 365, (2.5*1e6) / 365]  # x grammi * 10^6 (ton) /365 gg as avg, con x = [1.5, 2, 2.5]
            P_c_values = [0.01, 0.02, 0.02, 0.04]
            u_c_values = [0.1, 0.5, 0.8, 1]
            P_m_values = [0.1, 0.15, 0.15, 0.2]
            fcp_values = [40e9, 100e9, 100e9, 150e9]
            N0 = 10e-10
            lambda_failure_values = [8760, 8760, 8760, 45000, 45000]
            lambda_services_per_hour_values = [150, 200, 200, 250]  # avg servizi al giorno
            likelihood_values = [0.25, 0.5, 0.75, 1]

            # congiunti

            # gain_values_eavesdropper = np.random.uniform(0.05, 0.5, num_resources * num_services)
            # gain_values = np.random.uniform(1, 6, num_resources * num_services)

            # Indicators offered by the resources

            deadlines_off = [0.001, 0.4, 0.8, 20]
            data_rates_off = [85.0, 110.0, 110.0, 250.0]
            plr_off = [10.0, 20.0, 20.0, 40.0]

            resources = []

            for i in range(num_resource):
                chosen_index = i % len(availability_values)
                resource = Resource(i, 0, 0, [0, 0, 0], 0, 0, 0, 0, 0, N0, 0, 0, 0)

                availability_value = availability_values[chosen_index]
                carbon_offset_value = carbon_offset_values[chosen_index]
                P_c_value = P_c_values[chosen_index]
                u_c_value = u_c_values[chosen_index]
                P_m_value = P_m_values[chosen_index]
                fcp_value = fcp_values[chosen_index]
                lambda_failure_value = lambda_failure_values[chosen_index]
                lambda_services_per_hour_value = lambda_services_per_hour_values[chosen_index]
                likelihood_value = likelihood_values[chosen_index]

                deadline_off = deadlines_off[chosen_index]
                data_rate_off = data_rates_off[chosen_index]
                plr_off_res = plr_off[chosen_index]

                resource.set_availability(availability_value)
                resource.set_kpi_resource([deadline_off, data_rate_off, plr_off_res])
                resource.set_carbon_offset(carbon_offset_value)
                resource.set_P_c(P_c_value)
                resource.set_u_c(u_c_value)
                resource.set_P_m(P_m_value)
                resource.set_fpc(fcp_value)
                resource.set_lambda_failure(lambda_failure_value)
                resource.set_lambda_services_per_hour(lambda_services_per_hour_value)
                resource.set_likelihood(likelihood_value)

                resources.append(resource)


            for resource in resources:
                print(resource.id, resource.availability, resource.kpi_resource, resource.fpc,
                      resource.P_m, resource.P_c, resource.lambda_services_per_hour, resource.likelihood)

            # Calcolo Q_MIN e computation time

            q_v_big_req(services, [-1, 1, -1], [1, -1, -1])  # qui dentro set
            # for s in services:
            #     print(f"min q rec total {s.min_kpi}, min v rec total {s.min_kvi}")

            for service in services:
                for resource in resources:
                    computation_time = compute_computation_time(service, resource)
                    print(computation_time)

            # TIS:  trustworthiness inclusiveness sustainability
            normalized_kvi, weighted_sum_kvi, energy_sustainability_values, trustworthiness_values, failure_probability_values = compute_normalized_kvi(services, resources, CI=475, signs=[1, -1, -1])  #

            normalized_kpi, weighted_sum_kpi = compute_normalized_kpi(services, resources, signs=[-1, 1, -1])  # latenza, data rate e plr

            ############# RISOLUZIONE CLASSICA ############################

            # solutions_alpha = []
            #
            # for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:  # Test con diversi pesi
            #     kpi_value, kvi_value =  optimize_multiobj_weighted_ip(service_requests, services, resources, normalized_kpi, normalized_kvi,
            #                           weighted_sum_kpi, weighted_sum_kvi, alpha, results_dir)
            #
            #     if kpi_value is not None:
            #         solutions_alpha.append((kpi_value, kvi_value))
            #
            # # Plotta il fronte di Pareto
            # plot_pareto_front(solutions_alpha)
            # print(solutions_alpha)

            # solutions_BB = []
            # for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:  # Test con diversi pesi
            #     kpi_value, kvi_value = optimize_branch_and_bound(service_requests, services, resources, normalized_kpi, normalized_kvi,
            #                       weighted_sum_kpi, weighted_sum_kvi, alpha, results_dir)
            #     if kpi_value is not None:
            #         solutions_BB.append((kpi_value, kvi_value))
            #
            # # Plotta il fronte di Pareto
            # plot_pareto_front(solutions_BB)
            # print(solutions_BB)



            ############## METODO EPSILON-CONSTRAINT: CALCOLO IDEAL E NADIR POINTS E IMPLEMENTAZIONE DEL METODO ESATTO

            V_I = optimize_kvi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                               results_dir)

            Q_I = optimize_kpi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi,
                               weighted_sum_kvi, results_dir)

            V_N = v_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi,
                          weighted_sum_kpi, weighted_sum_kvi, Q_I, results_dir)

            Q_N = q_nadir(service_requests, services, resources, normalized_kpi,
                          normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I, results_dir)



            pareto_solutions_exact = epsilon_constraint_exact(service_requests, services, resources, normalized_kpi, normalized_kvi,
            weighted_sum_kpi, weighted_sum_kvi, Q_N, Q_I, delta=delta, results_dir=results_dir)

            plot_pareto_front(pareto_solutions_exact)
            pareto_filename = os.path.join(results_dir, "pareto_solutions.csv")
            save_pareto_solutions(pareto_solutions_exact, filename=pareto_filename)


            ############ APPROCCI BENCHMARK: GREEDY ASSIGNMENT KPI E RANDOM ASSIGNMENT

            assignment, total_kpi, total_kvi = greedy_assignment_kpi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi)

            save_assignment_results(service_requests, assignment, services, resources,
                                    weighted_sum_kpi, weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi,
                                    total_kvi,
                                    results_dir=results_dir, filename="greedy_kpi_results.csv")
            ### TRUSTWORTHINESS ###
            # assignment, total_kpi, total_kvi =  greedy_kvi_trustworthiness(service_requests, services, resources, trustworthiness_values,
            #                    weighted_sum_kpi, weighted_sum_kvi)

            ### INCLUSIVENESS ###
            # assignment, total_kpi, total_kvi = greedy_kvi_failure_probability(service_requests, services, resources, failure_probability_values,
            #                        weighted_sum_kpi, weighted_sum_kvi)

            ### SUSTAINABILITY ###
            # assignment, total_kpi, total_kvi = greedy_kvi_sustainability(service_requests, services, resources, energy_sustainability_values,
            #                   weighted_sum_kpi, weighted_sum_kvi)

            assignment, total_kpi, total_kvi = greedy_assignment_kvi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi)

            save_assignment_results(service_requests, assignment, services, resources,
                                    weighted_sum_kpi, weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi,
                                    total_kvi,
                                    results_dir=results_dir, filename="greedy_kvi_results.csv")


            assignment, total_kpi, total_kvi = random_assignment(service_requests, services, resources, weighted_sum_kpi,
                                                                 weighted_sum_kvi)

            print(f"Assignment: {assignment}, Total KPI: {total_kpi}, Total KVI: {total_kvi}, service_requests: {service_requests}")

            save_assignment_results(service_requests, assignment, services, resources, weighted_sum_kpi,
                                    weighted_sum_kvi, normalized_kpi, normalized_kvi, total_kpi, total_kvi,
                                    results_dir=results_dir,
                                    filename="random_results.csv")

            ############ APPROCCIO LAGRANGIAN-HEURISTIC BASED

            #Parametri del metodo subgradiente
            # max_iterations = 20  # Numero massimo di iterazioni
            # tolerance = 1e-3  # Soglia di convergenza
            # z = 0.5  # Parametro per lo step size
            #
            # # Loop iterativo per il metodo subgradiente
            # for alpha in [i / 10 for i in range(11)]:
            #     lambda_ = np.ones(len(service_requests)) * 0.1
            #     UB = float("inf")  # Upper Bound iniziale
            #     LB = float("-inf")  # Lower Bound iniziale
            #
            #     for k in range(max_iterations):
            #         # Zaini
            #         total_value_not_lagrangian, item_assignment = multi_knapsack_dp(
            #             service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi, lambda_, alpha
            #         )
            #
            #         #  Total value lagrangian
            #         total_value_lagrangian = compute_total_value_lagrangian(services, resources,
            #                                                                 item_assignment,
            #                                                                 weighted_sum_kpi, weighted_sum_kvi,
            #                                                                 lambda_, total_value_not_lagrangian, alpha)
            #
            #         print("Valore totale lagrangiano:", total_value_not_lagrangian)
            #         print("Valore totale lagrangiano corretto:", total_value_lagrangian)
            #         print("Assegnazione lagrangiana:", item_assignment)
            #
            #         if is_feasible_solution(service_requests, services, resources, item_assignment, weighted_sum_kpi,
            #                                 weighted_sum_kvi):
            #             print(f"Soluzione feasible trovata all'iterazione {k + 1}, interrompo l'ottimizzazione.")
            #             save_results_csv_lagrangian(service_requests,
            #                                         services, resources, item_assignment, weighted_sum_kpi,
            #                                         weighted_sum_kvi,
            #                                         results_dir=results_dir, filename=f"alpha_{alpha}_iteration_{k}.csv"
            #                                         )
            #             suboptimal_solutions = compute_total_value_comparabile(service_requests, services, resources, item_assignment,
            #                                             weighted_sum_kpi, weighted_sum_kvi)
            #
            #             save_suboptimal_solutions(suboptimal_solutions, filename=f"alpha_{alpha}_iteration_{k}.csv")
            #             plot_pareto_front(suboptimal_solutions)
            #
            #
            #             break
            #
            #         # Riparazione
            #         item_assignment_repaired = repair_solution(service_requests,
            #                                                    services, resources, item_assignment, weighted_sum_kpi,
            #                                                    weighted_sum_kvi, lambda_, alpha
            #                                                    )
            #
            #         # Valore f. obiettivo con soluzione feasible (riparata)
            #         total_value_feasible = compute_total_value(service_requests,
            #                                                    services, resources, item_assignment_repaired,
            #                                                    weighted_sum_kpi,
            #                                                    weighted_sum_kvi, alpha
            #                                                    )
            #
            #         print("Valore totale riparato:", total_value_feasible)
            #         print("Assegnazione riparata:", item_assignment_repaired)
            #
            #         # Aggiorna i moltiplicatori di Lagrange, lo step size, UB e LB
            #         lambda_, UB, LB = update_lagrangian_multipliers(service_requests,
            #                                                         services, resources, item_assignment_repaired,
            #                                                         weighted_sum_kpi, weighted_sum_kvi,
            #                                                         lambda_, UB, LB, total_value_lagrangian,
            #                                                         total_value_feasible, z
            #                                                         )
            #
            #         save_results_csv_lagrangian(service_requests,
            #                                     services, resources, item_assignment_repaired, weighted_sum_kpi,
            #                                     weighted_sum_kvi,
            #                                     results_dir=results_dir, filename=f"alpha_{alpha}_iteration_{k}.csv"
            #                                     )
            #
            #         suboptimal_solutions = compute_total_value_comparabile(service_requests, services, resources,
            #                                                                item_assignment_repaired,
            #                                                                weighted_sum_kpi, weighted_sum_kvi)
            #
            #         save_suboptimal_solutions(suboptimal_solutions, filename=f"alpha_{alpha}_iteration_{k}.csv")
            #         plot_pareto_front(suboptimal_solutions)
            #
            #         # Convergenza
            #         gap = (UB - LB) / max(1, abs(LB))
            #         print(f"Iterazione {k + 1}: UB = {UB:.4f}, LB = {LB:.4f}, Gap = {gap:.6f}")
            #
            #         if gap < tolerance:
            #             print("The covergence was reached.")
            #             break
            #
            #         print(f"Valore finale UB: {UB}, LB: {LB}")

            # Tempo di esecuzione
            end_time = time.time()
            time_elapsed = end_time - start
            with open(os.path.join(results_dir, "execution_time.txt"), "w") as file:
                file.write(f"Servizi: {num_services}, Tempo: {time_elapsed:.6f} sec\n")

            print(f"Completato per {num_services} servizi. Tempo: {time_elapsed:.6f} sec")
