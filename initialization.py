import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_single_row(kvi_service, kvi_service_req, resources, index_res, signs, kvi_values):
    # kvis_service are the 3 kvis of the i-th service requested, threshold output of the LLM
    row = np.zeros(len(kvi_service))  # row for the 3 kvi of the i-th service offered by the index_res-th resource
    maximum = np.max(kvi_values, axis=0)
    minimum = np.min(kvi_values, axis=0)

    for index, attribute in enumerate(kvi_service):
        for requested in kvi_service_req:
            exposed_kvi = resources[index_res].kvi_resource[
                index]  # vector offered by the resource
            # index_res-th
            max_val = maximum[index]  # value
            # max for that attributed evaluated on all resources for that service
            min_val = minimum[index]

            if exposed_kvi == attribute:
                row[index] = 1  # if the value is exactly the one requested

            else:
                if signs[index] == 1:  # benefit, the higher the better
                    if max_val == requested:  # no zero division
                        row[index] = 1
                    elif max_val == min_val:  # if all values are equal, set = 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (max_val - exposed_kvi) / (max_val - requested)

                else:  # Cost: the less the better
                    if min_val == requested:  # no zero division
                        row[index] = 1
                    elif max_val == min_val:  # if all values are equal, set = 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (exposed_kvi - min_val) / (requested - min_val)


    return np.abs(row)


# function computation time in h
def compute_computation_time(service, resource):
    return service.size * 1000 / resource.fpc


# function KVI environmental sustainability
def compute_energy_sustainability(resource, computation_time, CI=475, PUE=1.67):
    return  (computation_time / 3600) * resource.lambda_services_per_hour * (
            resource.availability * resource.P_c * resource.u_c + resource.availability * resource.P_m) * PUE * CI

# function KVI trustworthiness
def compute_trustworthiness(service, resource):
    cyber_risk = resource.likelihood * service.impact
    cyber_confidence = 1 - cyber_risk
    return 900 + 4100 / (1 + np.exp(- 0.6 * (cyber_confidence - 0.5)))

# function KVI inclusiveness
def compute_failure_probability(computation_time, resource):
    exponent = - 24 / resource.lambda_failure
    failure_probability = (1 - np.exp(exponent))  # p_rn
    F_rn_0 = (1 - failure_probability) ** resource.availability
    print("F_rn_0", F_rn_0)
    return F_rn_0 * computation_time * resource.lambda_services_per_hour

# function to compute indicators for each (service, resource) couple, normalization and weighted sum to get V(X)
def compute_normalized_kvi(services, resources, CI, signs):

    normalized_kvi = {}
    weighted_sum_kvi = {}
    energy_sustainability_values = {}
    trustworthiness_values = {}
    failure_probability_values = {}

    kvi_values = []  # list of future lists, length = 3

    for j, service in enumerate(services):

        # Compute all KVIs
        for n, resource in enumerate(resources):

            trustworthiness = compute_trustworthiness(service, resource)
            energy_sustainability = resource.carbon_offset - float(compute_energy_sustainability(resource, compute_computation_time(service, resource),
                                                                             CI))
            failure_probability = float(compute_failure_probability(compute_computation_time(service, resource), resource))
            energy_sustainability_values[(resource.id, service.id)] = energy_sustainability * service.weights_kvi[2]
            trustworthiness_values[(resource.id, service.id)] = trustworthiness * service.weights_kvi[0]
            failure_probability_values[(resource.id, service.id)] = failure_probability * service.weights_kvi[1]

            temp_kvi = [trustworthiness, failure_probability, energy_sustainability]
            kvi_values.append(temp_kvi)

    # Normalization
    for j, service in enumerate(services):
        for n, resource in enumerate(resources):
            v_x = np.dot(service.weights_kvi, kvi_values[j+n])
            weighted_sum_kvi[(resource.id, service.id)] = float(v_x)

    return normalized_kvi, weighted_sum_kvi, energy_sustainability_values, trustworthiness_values, failure_probability_values

def normalize_single_row_kpi(kpi_service, kpi_service_req, resources, index_res, signs, kpi_values):
    row = np.zeros(len(kpi_service))  # row for the 3 kpis of the i-th service offered by the index_res-th resource
    maximum = np.max(kpi_values, axis=0)
    minimum = np.min(kpi_values, axis=0)

    for index, attribute in enumerate(kpi_service):
        for requested in kpi_service_req:
            exposed_kpi = resources[index_res].kpi_resource[
                index]  # vector offered by the resource
            # index_res-th
            max_val = maximum[index]  # value
            min_val = minimum[index]

            if exposed_kpi == attribute:
                row[index] = 1  # if the value == the requested one

            else:
                if signs[index] == 1:  # benefit, the higher the better
                    if max_val == requested:  # no zero division
                        row[index] = 1
                    elif max_val == min_val:  # if all values are =, assign 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (max_val - exposed_kpi) / (max_val - requested)

                else:  # Cost, the lower the better
                    if min_val == requested:  # no zero division
                        row[index] = 1
                    elif max_val == min_val:  # if all values are =, assign 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (exposed_kpi - min_val) / (requested - min_val)

    return np.abs(row)


def compute_normalized_kpi(services, resources, signs):
    # function to compute indicators for each (service, resource) couple, normalization and weighted sum to get Q(X)

    normalized_kpi = {}
    weighted_sum_kpi = {}

    for j, service in enumerate(services):
        kpi_values = []  # list of future lists, length = 3

        # Indicators' computation for all resources
        for n, resource in enumerate(resources):
            kpi_values.append(resource.kpi_resource)
        # Normalization
        for n, resource in enumerate(resources):
            norm_kpi = normalize_single_row_kpi(service.kpi_service, service.kpi_service_req, resources, n, signs,
                                                     kpi_values)
            normalized_kpi[(resource.id, service.id)] = norm_kpi

            # Weighted sum
            q_x = np.dot(service.weights_kpi, norm_kpi)
            weighted_sum_kpi[(resource.id, service.id)] = float(q_x)

    return normalized_kpi, weighted_sum_kpi

# function to compute Q and V req, per the optimization problem
def q_v_big_req(services, signs_kpi, signs_kvi):
    kpi_tot = np.array([service.kpi_service_req for service in services])
    kvi_tot = np.array([service.kvi_service_req for service in services])

    max_kpi_req = np.max(kpi_tot, axis=0)
    min_kpi_req = np.min(kpi_tot, axis=0)
    max_kvi_req = np.max(kvi_tot, axis=0)
    min_kvi_req = np.min(kvi_tot, 0)

    for service in services:
        temp_kpi = np.zeros(len(service.kpi_service_req))

        for index, requested in enumerate(service.kpi_service_req):
            if max_kpi_req[index] > min_kpi_req[index]:  # no by zero division
                if signs_kpi[index] == 1:  # Benefit, the higher the better
                    temp_kpi[index] = (requested - min_kpi_req[index]) / (max_kpi_req[index] - min_kpi_req[index])
                else:  # Cost, the lowest value we can get must be 1, the highest 0
                    temp_kpi[index] = (requested - max_kpi_req[index]) / (min_kpi_req[index] - max_kpi_req[index])
            else:
                temp_kpi[index] = 1  # if all values are equal, assign 1


