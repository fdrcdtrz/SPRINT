from main import *
from optimization import *
from initialization import *
import random
import csv
import time
def greedy_assignment_kpi(services, resources, weighted_sum_kpi):
    assignment = {}
    total_kpi = 0
    total_kvi = 0

    sorted_services = sorted(services, key=lambda s: -max(weighted_sum_kpi.get((r.id, s.id), 0) for r in resources))

    for s in sorted_services:
        best_resource = max(resources, key=lambda r: weighted_sum_kpi.get((r.id, s.id), 0))
        assignment[s.id] = best_resource.id
        total_kpi += weighted_sum_kpi.get((best_resource.id, s.id), 0)
        total_kvi += weighted_sum_kvi.get((best_resource.id, s.id), 0)  # Per tracciare anche il KVI ottenuto

    return assignment, total_kpi, total_kvi



def greedy_assignment_kvi(services, resources, weighted_sum_kvi):
    assignment = {}
    total_kpi = 0  # Per coerenza, ma non è l'obiettivo principale qui
    total_kvi = 0

    sorted_services = sorted(services, key=lambda s: -max(weighted_sum_kvi.get((r.id, s.id), 0) for r in resources))

    for s in sorted_services:
        best_resource = max(resources, key=lambda r: weighted_sum_kvi.get((r.id, s.id), 0))
        assignment[s.id] = best_resource.id
        total_kpi += weighted_sum_kpi.get((best_resource.id, s.id), 0)  # Per tracciare anche il KPI ottenuto
        total_kvi += weighted_sum_kvi.get((best_resource.id, s.id), 0)

    return assignment, total_kpi, total_kvi



def random_assignment(services, resources, weighted_sum_kpi, weighted_sum_kvi):
    assignment = {}
    total_kpi = 0
    total_kvi = 0

    for s in services:
        chosen_resource = random.choice(resources)
        assignment[s.id] = chosen_resource.id
        total_kpi += weighted_sum_kpi.get((chosen_resource.id, s.id), 0)
        total_kvi += weighted_sum_kvi.get((chosen_resource.id, s.id), 0)

    return assignment, total_kpi, total_kvi


def save_assignment_results(assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi, normalized_kpi,
                            normalized_kvi, total_kpi, total_kvi, filename):

    results = []

    for s in services:
        r_id = assignment.get(s.id, None)
        if r_id is not None:
            assigned = 1  # Il servizio è assegnato
            kpi_value = weighted_sum_kpi.get((r_id, s.id), 0)
            kvi_value = weighted_sum_kvi.get((r_id, s.id), 0)
            norm_kpi = normalized_kpi.get((r_id, s.id), 0)
            norm_kvi = normalized_kvi.get((r_id, s.id), 0)
            min_kpi_value = s.min_kpi
            min_kvi_value = s.min_kvi

            # Lista KPI/KVI del servizio
            list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
            list_s_kvi_service = [float(kvi) for kvi in s.kvi_service]

            # Lista KPI/KVI della risorsa assegnata
            resource = next((r for r in resources if r.id == r_id), None)
            list_r_kpi_resource = [float(kpi) for kpi in resource.kpi_resource] if resource else []
            list_r_kvi_resource = [float(kvi) for kvi in resource.kvi_resource] if resource else []

            results.append([
                s.id, r_id, assigned,
                norm_kpi, norm_kvi,
                kpi_value, kvi_value,
                min_kpi_value, min_kvi_value,
                list_s_kpi_service, list_s_kvi_service,
                list_r_kpi_resource, list_r_kvi_resource
            ])

    # Scrittura su CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Service_ID", "Resource_ID", "Assigned",
                         "Normalized_KPI", "Normalized_KVI",
                         "Weighted_Sum_KPI", "Weighted_Sum_KVI",
                         "Min_KPI", "Min_KVI",
                         "KPI_Service", "KVI_Service",
                         "KPI_Resource", "KVI_Resource"])
        writer.writerows(results)

        # Aggiungere una riga con i totali alla fine del file
        writer.writerow([])
        writer.writerow(["Total", "", "", "", "", total_kpi, total_kvi, "", "", "", "", "", ""])


