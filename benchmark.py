import csv
import random

from optimization import *

###### BENCHMARK APPROACHES ####

def greedy_assignment_kvi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi):
    total_kpi_sum = 0
    total_kvi_sum = 0
    final_assignment = {}

    assignment = {}
    total_kpi = 0
    total_kvi = 0
    availability = {r.id: r.availability for r in resources}

    # Rank requests by maximum KVI
    sorted_requests = sorted(range(len(service_requests)),
                             key=lambda req_id: -max(
                                 weighted_sum_kvi.get((n, service_requests[req_id]), 0)
                                 for n in range(len(resources))))

    for request_id in sorted_requests:
        service_id = service_requests[request_id]
        demand = services[service_id].demand

        # SELECT RESOURCES WITH SUFFICIENT AVAILABILITY
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


def greedy_assignment_kpi(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi):
    total_kpi_sum = 0
    total_kvi_sum = 0
    final_assignment = {}

    assignment = {}
    total_kpi = 0
    total_kvi = 0
    availability = {r.id: r.availability for r in resources}

    # Rank requests by maximum KPI
    sorted_requests = sorted(range(len(service_requests)),
                             key=lambda req_id: -max(
                                 weighted_sum_kpi.get((n, service_requests[req_id]), 0)
                                 for n in range(len(resources))))

    for request_id in sorted_requests:
        service_id = service_requests[request_id]
        demand = services[service_id].demand

        # Select resources with enough availability
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


def random_assignment(service_requests, services, resources, weighted_sum_kpi, weighted_sum_kvi, num_seeds=100,
                      max_assignments=10):
    total_kpi_sum = 0
    total_kvi_sum = 0

    for _ in range(num_seeds):
        assignment = {}
        total_kpi = 0
        total_kvi = 0

        # Map to track how many times a resource has been assigned
        resource_usage = {r.id: 0 for r in resources}

        for request_id in range(len(service_requests)):
            service_id = service_requests[request_id]
            s = services[service_id]

            # Select resources with less than max_assignments assignments only
            valid_resources = [r for r in resources if resource_usage[r.id] < max_assignments]

            if valid_resources:
                chosen_resource = random.choice(valid_resources)
                assignment[request_id] = chosen_resource.id
                resource_usage[chosen_resource.id] += 1
            else:
                print(
                    f"Attention: no available resource for the request {request_id} (service_id: {service_id})")

            total_kpi += weighted_sum_kpi.get((chosen_resource.id, service_id), 0) if chosen_resource else 0
            total_kvi += weighted_sum_kvi.get((chosen_resource.id, service_id), 0) if chosen_resource else 0

        total_kpi_sum += total_kpi
        total_kvi_sum += total_kvi
        assignment = dict(sorted(assignment.items()))

    return assignment, total_kpi_sum / num_seeds, total_kvi_sum / num_seeds


def save_assignment_results(service_requests, assignment, services, resources, weighted_sum_kpi, weighted_sum_kvi, normalized_kpi,
                            normalized_kvi, total_kpi, total_kvi, results_dir, filename):
    # Create the folder if it does not exist
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    print(f"Folder: {os.path.abspath(results_dir)}")
    print(f"Complete path of the file: {filepath}")

    results = []

    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # id service
        s = services[service_id]  # corresponding Service object
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

        print(f"File saved in: {filepath}")

    except Exception as e:
        print(f"Error while saving the file: {e}")


