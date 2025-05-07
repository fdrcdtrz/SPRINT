import csv
import random

from optimization import *

def greedy_assignment_kvi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi):
    total_kpi_sum = 0
    total_kvi_sum = 0
    final_assignment = {}

    assignment = {}
    total_kpi = 0
    total_kvi = 0
    availability = {r.id: r.availability for r in resources}

    # Ordina le richieste per KvI massimo
    sorted_requests = sorted(range(len(service_requests)),
                             key=lambda req_id: -max(
                                 weighted_sum_kvi.get((n, service_requests[req_id]), 0)
                                 for n in range(len(resources))))

    for request_id in sorted_requests:
        service_id = service_requests[request_id]
        demand = services[service_id].demand

        # Seleziona risorse con availability sufficiente
        candidates = [r for r in resources if availability[r.id] >= demand]
        if candidates:
            best_resource = max(candidates, key=lambda r: weighted_sum_kvi.get((resources.index(r), service_id), 0))
            ridx = resources.index(best_resource)

            assignment[request_id] = best_resource.id
            availability[best_resource.id] -= demand
            total_kpi += weighted_sum_kpi.get((ridx, service_id), 0)
            total_kvi += weighted_sum_kvi.get((ridx, service_id), 0)

    final_assignment = dict(sorted(assignment.items()))
    total_kpi_sum += total_kpi
    total_kvi_sum += total_kvi

    return final_assignment, total_kpi_sum, total_kvi_sum

def greedy_kvi_sustainability(service_requests, services, resources, energy_sustainability_values,
                              weighted_sum_kpi, weighted_sum_kvi):

    # availability = {r.id: r.availability for r in resources}
    # assignment = {}
    # total_kpi = 0
    # total_kvi = 0
    #
    # for req_id, service_id in enumerate(service_requests):
    #     demand = services[service_id].demand
    #     # Ordina risorse per il kvi decrescente
    #     sorted_resources = sorted(
    #         resources,
    #         key=lambda r: -energy_sustainability_values.get((r.id, service_id), 0)
    #     )
    #
    #     for r in sorted_resources:
    #         if availability[r.id] >= demand:
    #             assignment[req_id] = r.id
    #             availability[r.id] -= demand
    #             total_kpi += weighted_sum_kpi.get((r.id, service_id), 0)
    #             total_kvi += weighted_sum_kvi.get((r.id, service_id), 0)
    #             break
    #
    # return assignment, total_kpi, total_kvi

    total_kpi_sum = 0
    total_kvi_sum = 0
    final_assignment = {}

    assignment = {}
    total_kpi = 0
    total_kvi = 0
    availability = {r.id: r.availability for r in resources}

    sorted_requests = sorted(range(len(service_requests)),
                             key=lambda req_id: -max(
                                 weighted_sum_kvi.get((n, service_requests[req_id]), 0)
                                 for n in range(len(resources))))

    for request_id in sorted_requests:
        service_id = service_requests[request_id]
        demand = services[service_id].demand

        # Seleziona solo le risorse con availability sufficiente
        candidates = [r for r in resources if availability[r.id] >= demand]
        if candidates:
            # Ordina i candidati per probabilità di fallimento decrescente
            sorted_candidates = sorted(
                candidates,
                key=lambda r: -energy_sustainability_values.get((r.id, service_id), 0)
            )
            best_resource = sorted_candidates[0]

            assignment[request_id] = best_resource.id
            availability[best_resource.id] -= demand
            total_kpi += weighted_sum_kpi.get((best_resource.id, service_id), 0)
            total_kvi += weighted_sum_kvi.get((best_resource.id, service_id), 0)

    final_assignment = dict(sorted(assignment.items()))
    total_kpi_sum += total_kpi
    total_kvi_sum += total_kvi

    return final_assignment, total_kpi_sum, total_kvi_sum


def greedy_kvi_trustworthiness(service_requests, services, resources, trustworthiness_values,
                               weighted_sum_kpi, weighted_sum_kvi):

    availability = {r.id: r.availability for r in resources}
    assignment = {}
    total_kpi = 0
    total_kvi = 0

    for req_id, service_id in enumerate(service_requests):
        demand = services[service_id].demand
        # Ordina risorse per failure probability decrescente
        sorted_resources = sorted(
            resources,
            key=lambda r: -trustworthiness_values.get((r.id, service_id), 0)
        )

        for r in sorted_resources:
            if availability[r.id] >= demand:
                assignment[req_id] = r.id
                availability[r.id] -= demand
                total_kpi += weighted_sum_kpi.get((r.id, service_id), 0)
                total_kvi += weighted_sum_kvi.get((r.id, service_id), 0)
                break

    return assignment, total_kpi, total_kvi

def greedy_kvi_failure_probability(service_requests, services, resources, failure_probability_values,
                                   weighted_sum_kpi, weighted_sum_kvi):
    # availability = {r.id: r.availability for r in resources}
    # assignment = {}
    # total_kpi = 0
    # total_kvi = 0
    #
    # for req_id, service_id in enumerate(service_requests):
    #     demand = services[service_id].demand
    #     # Ordina risorse per incl decrescente
    #     sorted_resources = sorted(
    #         resources,
    #         key=lambda r: -failure_probability_values.get((r.id, service_id), 0)
    #     )
    #
    #     for r in sorted_resources:
    #         if availability[r.id] >= demand:
    #             assignment[req_id] = r.id
    #             availability[r.id] -= demand
    #             total_kpi += weighted_sum_kpi.get((r.id, service_id), 0)
    #             total_kvi += weighted_sum_kvi.get((r.id, service_id), 0)
    #             break
    #
    # return assignment, total_kpi, total_kvi


    total_kpi_sum = 0
    total_kvi_sum = 0
    final_assignment = {}

    assignment = {}
    total_kpi = 0
    total_kvi = 0
    availability = {r.id: r.availability for r in resources}

    # Ordina le richieste per massima probabilità di fallimento sulle risorse
    sorted_requests = sorted(
        range(len(service_requests)),
        key=lambda req_id: -max(
            failure_probability_values.get((resources[n].id, service_requests[req_id]), 0)
            for n in range(len(resources))
        )
    )

    # Ordina le risorse per probabilità di fallimento decrescente per ciascuna richiesta
    for request_id in sorted_requests:
        service_id = service_requests[request_id]
        demand = services[service_id].demand

        # Seleziona solo le risorse con availability sufficiente
        candidates = [r for r in resources if availability[r.id] >= demand]
        if candidates:
            # Ordina i candidati per probabilità di fallimento decrescente
            sorted_candidates = sorted(
                candidates,
                key=lambda r: -failure_probability_values.get((r.id, service_id), 0)
            )
            best_resource = sorted_candidates[0]
            ridx = resources.index(best_resource)

            assignment[request_id] = best_resource.id
            availability[best_resource.id] -= demand
            total_kpi += weighted_sum_kpi.get((best_resource.id, service_id), 0)
            total_kvi += weighted_sum_kvi.get((best_resource.id, service_id), 0)

    final_assignment = dict(sorted(assignment.items()))
    total_kpi_sum += total_kpi
    total_kvi_sum += total_kvi

    return final_assignment, total_kpi_sum, total_kvi_sum


def greedy_assignment_kpi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi):
    total_kpi_sum = 0
    total_kvi_sum = 0
    final_assignment = {}

    assignment = {}
    total_kpi = 0
    total_kvi = 0
    availability = {r.id: r.availability for r in resources}

    # Ordina le richieste per KPI massimo
    sorted_requests = sorted(range(len(service_requests)),
                             key=lambda req_id: -max(
                                 weighted_sum_kpi.get((n, service_requests[req_id]), 0)
                                 for n in range(len(resources))))

    for request_id in sorted_requests:
        service_id = service_requests[request_id]
        demand = services[service_id].demand

        # Seleziona risorse con availability sufficiente
        candidates = [r for r in resources if availability[r.id] >= demand]
        if candidates:
            best_resource = max(candidates, key=lambda r: weighted_sum_kpi.get((resources.index(r), service_id), 0))
            ridx = resources.index(best_resource)

            assignment[request_id] = best_resource.id
            availability[best_resource.id] -= demand
            total_kpi += weighted_sum_kpi.get((ridx, service_id), 0)
            total_kvi += weighted_sum_kvi.get((ridx, service_id), 0)

    final_assignment = dict(sorted(assignment.items()))
    total_kpi_sum += total_kpi
    total_kvi_sum += total_kvi

    return final_assignment, total_kpi_sum, total_kvi_sum

# def greedy_assignment_kpi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi, num_seeds=500,
#                           max_assignments=10):
#     total_kpi_sum = 0
#     total_kvi_sum = 0
#
#     for _ in range(num_seeds):
#         assignment = {}
#         total_kpi = 0
#         total_kvi = 0
#
#         # Mappa per tracciare quante volte ogni risorsa è stata assegnata
#         resource_usage = {n: 0 for n, r in enumerate(resources)}
#
#         # Ordino le richieste in base al miglior KPI ottenibile
#         sorted_requests = sorted(range(len(service_requests)),
#                                  key=lambda req_id: -max(
#                                      weighted_sum_kpi.get((n, service_requests[req_id]), 0) for n, r in enumerate(resources)))
#
#         for request_id in sorted_requests:
#             service_id = service_requests[request_id]
#             s = services[service_id]
#
#             # Seleziona solo risorse con meno di max_assignments assegnazioni
#             valid_resources = [r for n, r in enumerate(resources) if resource_usage[n] < max_assignments]
#
#             if valid_resources:
#                 best_resource = max(valid_resources,
#                                     key=lambda r: weighted_sum_kpi.get((resources.index(r), service_id), 0))
#             else:
#                 best_resource = None  # Nessuna risorsa valida trovata
#
#             if best_resource:
#                 assignment[request_id] = best_resource.id
#                 resource_usage[best_resource.id] += 1
#             else:
#                 print(
#                     f"Nessuna risorsa disponibile per la richiesta {request_id} (service_id: {service_id})")
#
#             # Aggiorna i KPI totali
#             total_kpi += weighted_sum_kpi.get((best_resource.id, service_id), 0)
#             total_kvi += weighted_sum_kvi.get((best_resource.id, service_id), 0)
#
#         total_kpi_sum += total_kpi
#         total_kvi_sum += total_kvi
#         assignment = dict(sorted(assignment.items()))
#
#     # Restituisce la media dei KPI/KVI sui num_seeds tentativi
#     return assignment, total_kpi_sum / num_seeds, total_kvi_sum / num_seeds


def random_assignment(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi, num_seeds=100,
                      max_assignments=10):
    total_kpi_sum = 0
    total_kvi_sum = 0

    for _ in range(num_seeds):
        assignment = {}
        total_kpi = 0
        total_kvi = 0

        # Mappa per tracciare quante volte ogni risorsa è stata assegnata
        resource_usage = {r.id: 0 for r in resources}

        for request_id in range(len(service_requests)):
            service_id = service_requests[request_id]
            s = services[service_id]

            # Seleziona solo risorse con meno di max_assignments assegnazioni
            valid_resources = [r for r in resources if resource_usage[r.id] < max_assignments]

            if valid_resources:
                chosen_resource = random.choice(valid_resources)
                assignment[request_id] = chosen_resource.id
                resource_usage[chosen_resource.id] += 1
            else:
                print(
                    f"Attenzione: nessuna risorsa disponibile per la richiesta {request_id} (service_id: {service_id})")

            total_kpi += weighted_sum_kpi.get((chosen_resource.id, service_id), 0) if chosen_resource else 0
            total_kvi += weighted_sum_kvi.get((chosen_resource.id, service_id), 0) if chosen_resource else 0

        total_kpi_sum += total_kpi
        total_kvi_sum += total_kvi
        assignment = dict(sorted(assignment.items()))

    return assignment, total_kpi_sum / num_seeds, total_kvi_sum / num_seeds


def save_assignment_results(service_requests, assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi, normalized_kpi,
                            normalized_kvi, total_kpi, total_kvi, results_dir, filename):
    # Crea la cartella se non esiste
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    print(f"Cartella: {os.path.abspath(results_dir)}")
    print(f"Percorso completo del file: {filepath}")

    results = []

    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # id servizio
        s = services[service_id]  # obj Service corrispondente
        r_id = assignment.get(s.id, None)
        if r_id is not None:
            assigned = 1
            kpi_value = weighted_sum_kpi.get((r_id, s.id), 0)
            kvi_value = weighted_sum_kvi.get((r_id, s.id), 0)
            norm_kpi = normalized_kpi.get((r_id, s.id), 0)
            norm_kvi = 0
            min_kpi_value = s.min_kpi
            min_kvi_value = 0

            list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
            list_s_kvi_service = []

            resource = next((r for r in resources if r.id == r_id), None)
            list_r_kpi_resource = [float(kpi) for kpi in resource.kpi_resource] if resource else []
            list_r_kvi_resource = []

            results.append([
                s.id, r_id, assigned,
                norm_kpi, norm_kvi,
                kpi_value, kvi_value,
                min_kpi_value, min_kvi_value,
                list_s_kpi_service, list_s_kvi_service,
                list_r_kpi_resource, list_r_kvi_resource
            ])

    try:
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Service_ID", "Resource_ID", "Assigned",
                             "Normalized_KPI", "Normalized_KVI",
                             "Weighted_Sum_KPI", "Weighted_Sum_KVI",
                             "Min_KPI", "Min_KVI",
                             "KPI_Service", "KVI_Service",
                             "KPI_Resource", "KVI_Resource"])
            writer.writerows(results)

            writer.writerow([])
            writer.writerow(["Total", "", "", "", "", total_kpi, total_kvi, "", "", "", "", "", ""])

        print(f"File salvato in: {filepath}")

    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")

# def save_assignment_results(service_requests, assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi, normalized_kpi,
#                             normalized_kvi, total_kpi, total_kvi, results_dir, filename):
#     # Crea la cartella se non esiste
#     os.makedirs(results_dir, exist_ok=True)
#     filepath = os.path.join(results_dir, filename)
#
#     print(f"Cartella: {os.path.abspath(results_dir)}")
#     print(f"Percorso completo del file: {filepath}")
#
#     results = []
#
#     for request_id in range(len(service_requests)):
#         service_id = service_requests[request_id]  # id servizio
#         s = services[service_id]  # obj Service corrispondente
#         r_id = assignment.get(s.id, None)
#         if r_id is not None:
#             assigned = 1
#             kpi_value = weighted_sum_kpi.get((r_id, s.id), 0)
#             kvi_value = weighted_sum_kvi.get((r_id, s.id), 0)
#             norm_kpi = normalized_kpi.get((r_id, s.id), 0)
#             norm_kvi = normalized_kvi.get((r_id, s.id), 0)
#             min_kpi_value = s.min_kpi
#             min_kvi_value = s.min_kvi
#
#             list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
#             list_s_kvi_service = [float(kvi) for kvi in s.kvi_service]
#
#             resource = next((r for r in resources if r.id == r_id), None)
#             list_r_kpi_resource = [float(kpi) for kpi in resource.kpi_resource] if resource else []
#             list_r_kvi_resource = [float(kvi) for kvi in resource.kvi_resource] if resource else []
#
#             results.append([
#                 s.id, r_id, assigned,
#                 norm_kpi, norm_kvi,
#                 kpi_value, kvi_value,
#                 min_kpi_value, min_kvi_value,
#                 list_s_kpi_service, list_s_kvi_service,
#                 list_r_kpi_resource, list_r_kvi_resource
#             ])
#
#     try:
#         with open(filepath, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["Service_ID", "Resource_ID", "Assigned",
#                              "Normalized_KPI", "Normalized_KVI",
#                              "Weighted_Sum_KPI", "Weighted_Sum_KVI",
#                              "Min_KPI", "Min_KVI",
#                              "KPI_Service", "KVI_Service",
#                              "KPI_Resource", "KVI_Resource"])
#             writer.writerows(results)
#
#             writer.writerow([])
#             writer.writerow(["Total", "", "", "", "", total_kpi, total_kvi, "", "", "", "", "", ""])
#
#         print(f"File salvato in: {filepath}")
#
#     except Exception as e:
#         print(f"Errore nel salvataggio del file: {e}")


