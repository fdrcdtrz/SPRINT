from main import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import csv


# funzioni per la normalizzazione attributi KPI e indicatori KVI e rispettivi Q e V

# esempio per ricordarti
# Per normalizzare 2 rispetto a 0.5, ad esempio, e ottenere z = (2 - 0.5) / (4 - 0.5):
# attributi_servizio = np.array([0.5, 35, 20, 1])  # 4 attributi, valore desiderato
# attributi_risorsa_1 = np.array([2, 30, 15, 1])  # 4 attributi, valori esposti dalla risorsa 1
# attributi_risorsa_2 = np.array([4, 28, 20, 2])  # 4 attributi, valori esposti dalla risorsa 2
#
# ris = np.array([attributi_risorsa_1, attributi_risorsa_2])
#
# def normalize(attributi_servizio, ris, indice_risorsa):
#     z_jn = np.zeros(len(attributi_servizio))
#
#     for index, attributo in enumerate(attributi_servizio):
#         valore_risorsa = ris[indice_risorsa, index]
#         max_val = np.max(ris[:, index])
#
#         if max_val == attributo:
#             z_jn[index] = 0
#         else:
#             z_jn[index] = (valore_risorsa - attributo) / (max_val - attributo)
#
#     return z_jn

# def normalized_kpi(services, resources, signs):
#     # services[i].kpi_service for i in len(services)
#     normalized_kpi = {}
#     weighted_sum_kpi = {}
#
#     def normalize_single_row(kpis_service, resources, index_res):
#         row = np.zeros(len(kpis_service))  # inizializzo vettore riga con i kpi della risorsa index_res-esima
#         # normalizzati per i parametri kpis_service
#
#         for index, attribute in enumerate(kpis_service):  # faccio stessa cosa di sopra considerando una risorsa ben
#             # precisa, un servizio ben preciso, ed il valore massimo esposto
#             exposed_kpi = resources[index_res].kpi_resource[index]
#             if signs[index] == 1:
#                 max_val = np.max([resource.kpi_resource[index] for resource in resources])
#                 # check per zero
#                 if max_val == attribute:
#                     row[index] = 1
#                 else:
#                     row[index] = 1 - (max_val - exposed_kpi) / (max_val - attribute)
#             else:
#                 max_val = np.max([resource.kpi_resource[index] for resource in resources])
#                 # check per zero
#                 if max_val == attribute:
#                     row[index] = 1
#                 else:
#                     row[index] = 1 - (exposed_kpi - max_val) / (max_val - attribute)
#
#         row = np.clip(row, 0, 1)  # Forza tutti i valori tra 0 e 1
#         return np.abs(row)
#
#     for j, service in enumerate(services):
#         for n, resource in enumerate(resources):
#             norm_kpi = normalize_single_row(service.kpi_service, resources, n)
#             normalized_kpi[(resource.id, service.id)] = norm_kpi
#
#             # kpi globale, sommatoria
#             q_x = np.dot(service.weights_kpi,
#                          norm_kpi)  # somma pesata da moltiplicare alla variabile decisionale nel problema di ottimizzazione
#
#             weighted_sum_kpi[(resource.id, service.id)] = float(q_x)
#
#     return normalized_kpi, weighted_sum_kpi

# def normalized_kvi(services, resources, signs):
#     # services[i].kpi_service for i in len(services)
#     normalized_kvi = {}
#     weighted_sum_kvi = {}
#
#     def normalize_single_row(kvis_service, resources, index_res):
#         row = np.zeros(len(kvis_service))  # inizializzo vettore riga con i kpi della risorsa index_res-esima
#         # normalizzati per i parametri kpis_service
#
#         for index, attribute in enumerate(kvis_service):  # faccio stessa cosa di sopra considerando una risorsa ben
#             # precisa, un servizio ben preciso, ed il valore massimo esposto
#             exposed_kvi = resources[index_res].kvi_resource[index]
#             if signs[index] == 1:
#                 #adjusted_attribute = attribute * signs[index]
#                 max_val = np.max([resource.kvi_resource[index] for resource in resources])
#                 # check per zero
#                 if max_val == attribute:
#                     row[index] = 0
#                 else:
#                     row[index] = 1 - (max_val - exposed_kvi) / (max_val - attribute)
#             else:
#                 max_val = np.max([resource.kvi_resource[index] for resource in resources])
#                 # check per zero
#                 if max_val == attribute:
#                     row[index] = 0
#                 else:
#                     row[index] = 1 - (exposed_kvi - max_val) / (max_val - attribute)
#
#         return np.abs(row)

# for j, service in enumerate(services):
#     for n, resource in enumerate(resources):
#         norm_kvi = normalize_single_row(service.kvi_service, resources, n)
#         normalized_kvi[(resource.id, service.id)] = norm_kvi
#
#         # kpi globale, sommatoria
#         v_x = np.dot(service.weights_kvi,
#                      norm_kvi)  # somma pesata da moltiplicare alla variabile decisionale nel problema di ottimizzazione
#
#         weighted_sum_kvi[(resource.id, service.id)] = float(v_x)
#
# return normalized_kvi, weighted_sum_kvi


def normalize_single_row(kvi_service, kvi_service_req, resources, index_res, signs, kvi_values):
    # kvis_service sono i tre kvi del servizio i-esimo richiesti, cioè le soglie output di LLM
    row = np.zeros(len(kvi_service))  # creo riga per i tre kvi del servizio i-esimo offerti da risorsa index_res-esima
    maximum = np.max(kvi_values, axis=0)
    minimum = np.min(kvi_values, axis=0)
    # idea è normalizzarli in riferimento a quelli offerti dalla risorsa index_res-esima

    for index, attribute in enumerate(kvi_service):
        for requested in kvi_service_req:
            exposed_kvi = resources[index_res].kvi_resource[
                index]  # questo deve essere il vettore offerto dalla risorsa
            # index_res-esima
            max_val = maximum[index]  # questo deve essere il valore
            # massimo per quell'attributo valutato su tutte le risorse per quel servizio
            min_val = minimum[index]

            if exposed_kvi == attribute:
                row[index] = 1  # Se il valore ottenuto è esattamente quello richiesto

            else:
                if signs[index] == 1:  # Beneficio: più alto è meglio
                    if max_val == requested:  # Evita la divisione per zero
                        row[index] = 1
                    elif max_val == min_val:  # Se tutti i valori sono uguali, assegna 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (max_val - exposed_kvi) / (max_val - requested)

                else:  # Costo: più basso è meglio
                    if min_val == requested:  # Evita la divisione per zero
                        row[index] = 1
                    elif max_val == min_val:  # Se tutti i valori sono uguali, assegna 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (exposed_kvi - min_val) / (requested - min_val)

            # row[index] = np.clip(row[index], 0, 1)

    return np.abs(row)


#funzione calcolo channel gain
def compute_channel_gain_matrix(services, resources, gain_values):
    gains = np.zeros((len(services), len(resources)))
    for i, service in enumerate(services):
        for j, resource in enumerate(resources):
            gains[i, j] = gain_values[i+j]
    return gains


def compute_eavesdropper_gain(services, resources, gain_values_eavesdropper):
    gains_eavesdropper = np.zeros((len(services), len(resources)))
    for i, service in enumerate(services):
        for j, resource in enumerate(resources):
            gains_eavesdropper[i, j] = gain_values_eavesdropper[i+j]  # same
    return gains_eavesdropper


# funzione calcolo computation time in h
def compute_computation_time(service, resource):
    return service.size * 1000 / (resource.availability * resource.fpc)  # / 3600  # per passare da secondi a ore


# funzione calcolo KVI sostenibilità ambientale
def compute_energy_sustainability(resource, computation_time, CI=475, PUE=1.67):
    return resource.carbon_offset - computation_time * resource.lambda_services_per_hour * (
            resource.availability * resource.P_c * resource.u_c + resource.n_m * resource.P_m) * PUE * CI


# funzione calcolo KVI trustworthiness
def compute_secrecy_capacity(service, gain_values, gain_values_eavesdropper, resource):
    return max(0, np.log2(1 + (service.p_s * gain_values / resource.N0)) -
               np.log2(1 + (service.p_s * gain_values_eavesdropper / resource.N0)))


# funzione calcolo KVI inclusiveness
def compute_failure_probability(computation_time, resource):
    #print(f"Computation Time: {computation_time}, Lambda: {resource.lmbd}, Availability: {resource.availability}")
    exponent = - 24 / resource.lambda_failure
    failure_probability = (1 - np.exp(exponent)) ** resource.availability
    return (failure_probability * computation_time * resource.lambda_services_per_hour) / 24


def compute_normalized_kvi(services, gain_values, gain_values_eavesdropper, resources, CI, signs):
    # calcolo indicatori per ogni coppia (servizio, risorsa), normalizzo e faccio somma pesata per V(X) finale

    normalized_kvi = {}
    weighted_sum_kvi = {}
    gains = compute_channel_gain_matrix(services, resources, gain_values)
    gains_eavesdroppers = compute_eavesdropper_gain(services, resources, gain_values_eavesdropper)
    kvi_values = []  # lista di future liste di lunghezza 3, vanno tutti i kvi garantiti per il servizio s

    for j, service in enumerate(services):

        # Calcolo degli indicatori per tutte le risorse
        for n, resource in enumerate(resources):
            secrecy_capacity = float(compute_secrecy_capacity(service, gains[j, n], gains_eavesdroppers[j, n],
                                                        resource))
            energy_sustainability = float(compute_energy_sustainability(resource, compute_computation_time(service, resource),
                                                                  CI))
            failure_probability = float(compute_failure_probability(compute_computation_time(service, resource), resource))

            print(f"For ({service.id}, {resource.id}: secrecy capacity di {secrecy_capacity} bits/s/Hz, energy "
                  f"sustainability di {energy_sustainability} in gCO2e, inclusiveness di {failure_probability}")

            temp_kvi = [secrecy_capacity, failure_probability, energy_sustainability]
            kvi_values.append(temp_kvi)
            # v_x = np.dot(service.weights_kvi, temp_kvi)
            # weighted_sum_kvi[(resource.id, service.id)] = float(v_x)
            # resource.kvi_resource = [secrecy_capacity, failure_probability, energy_sustainability]
    kvi_values = MinMaxScaler().fit_transform(kvi_values)

    # Normalizzazione
    for j, service in enumerate(services):
        for n, resource in enumerate(resources):
    #         norm_kvi = kvi_values[j+n]
    #         # norm_kvi = normalize_single_row(service.kvi_service, service.kvi_service_req, resources, n, signs,
    #         #                                 kvi_values)
    #         # normalized_kvi[(resource.id, service.id)] = norm_kvi
    #
    #         # Somma pesata con i pesi del servizio
            v_x = np.dot(service.weights_kvi, kvi_values[j+n])
            #print(v_x)
            weighted_sum_kvi[(resource.id, service.id)] = float(v_x)

    return normalized_kvi, weighted_sum_kvi


def normalize_single_row_kpi(kpi_service, kpi_service_req, resources, index_res, signs, kpi_values):
    # kvis_service sono i tre kvi del servizio i-esimo richiesti, cioè le soglie output di LLM
    row = np.zeros(len(kpi_service))  # creo riga per i tre kvi del servizio i-esimo offerti da risorsa index_res-esima
    maximum = np.max(kpi_values, axis=0)
    minimum = np.min(kpi_values, axis=0)
    # idea è normalizzarli in riferimento a quelli offerti dalla risorsa index_res-esima

    for index, attribute in enumerate(kpi_service):
        for requested in kpi_service_req:
            exposed_kpi = resources[index_res].kpi_resource[
                index]  # questo deve essere il vettore offerto dalla risorsa
            # index_res-esima
            max_val = maximum[index]  # questo deve essere il valore
            min_val = minimum[index]
            # massimo per quell'attributo valutato su tutte le risorse per quel servizio

            if exposed_kpi == attribute:
                row[index] = 1  # Se il valore ottenuto è esattamente quello richiesto

            else:
                if signs[index] == 1:  # Beneficio: più alto è meglio
                    if max_val == requested:  # Evita la divisione per zero
                        row[index] = 1
                    elif max_val == min_val:  # Se tutti i valori sono uguali, assegna 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (max_val - exposed_kpi) / (max_val - requested)

                else:  # Costo: più basso è meglio
                    if min_val == requested:  # Evita la divisione per zero
                        row[index] = 1
                    elif max_val == min_val:  # Se tutti i valori sono uguali, assegna 1
                        row[index] = 1
                    else:
                        row[index] = 1 - (exposed_kpi - min_val) / (requested - min_val)

            # row[index] = np.clip(row[index], 0, 1)  # Mantieni i valori tra 0 e 1

    return np.abs(row)


def compute_normalized_kpi(services, resources, signs):
    # calcolo indicatori per ogni coppia (servizio, risorsa), normalizzo e faccio somma pesata per Q(X) finale

    normalized_kpi = {}
    weighted_sum_kpi = {}

    for j, service in enumerate(services):
        kpi_values = []  # lista di future liste di lunghezza 3, vanno tutti i kpi garantiti per il servizio s

        # Calcolo degli indicatori per tutte le risorse
        for n, resource in enumerate(resources):
            kpi_values.append(resource.kpi_resource)
        # Normalizzazione
        for n, resource in enumerate(resources):
            norm_kpi = normalize_single_row_kpi(service.kpi_service, service.kpi_service_req, resources, n, signs,
                                                kpi_values)
            normalized_kpi[(resource.id, service.id)] = norm_kpi

            # Somma pesata con i pesi del servizio
            q_x = np.dot(service.weights_kpi, norm_kpi)
            weighted_sum_kpi[(resource.id, service.id)] = float(q_x)

    return normalized_kpi, weighted_sum_kpi


def q_v_big_req(services, signs_kpi, signs_kvi):
    kpi_tot = np.array([service.kpi_service_req for service in services])
    kvi_tot = np.array([service.kvi_service_req for service in services])

    max_kpi_req = np.max(kpi_tot, axis=0)
    min_kpi_req = np.min(kpi_tot, axis=0)
    max_kvi_req = np.max(kvi_tot, axis=0)
    min_kvi_req = np.min(kvi_tot, 0)

    for service in services:
        temp_kpi = np.zeros(len(service.kpi_service_req))
        #temp_kvi = np.zeros(len(service.kvi_service_req))

        for index, requested in enumerate(service.kpi_service_req):
            if max_kpi_req[index] > min_kpi_req[index]:  # Evita divisioni per zero
                if signs_kpi[index] == 1:  # Beneficio: più alto è meglio
                    temp_kpi[index] = (requested - min_kpi_req[index]) / (max_kpi_req[index] - min_kpi_req[index])
                else:  # Costo: il valore più basso ottenibile deve essere 1, il più alto deve essere 0
                    temp_kpi[index] = (requested - max_kpi_req[index]) / (min_kpi_req[index] - max_kpi_req[index])
            else:
                temp_kpi[index] = 1  # Se tutti i valori sono uguali, assegna 1

        #service.min_kpi = np.clip(np.dot(service.weights_kpi, temp_kpi), 0, 1)

        # for index, requested in enumerate(service.kvi_service_req):
        #     if max_kvi_req[index] > min_kvi_req[index]:  # Evita divisioni per zero
        #         if signs_kvi[index] == 1:  # Beneficio
        #             temp_kvi[index] = (requested - min_kvi_req[index]) / (max_kvi_req[index] - min_kvi_req[index])
        #         else:  # Costo: il valore più basso ottenibile deve essere 1, il più alto deve essere 0
        #             temp_kvi[index] = (requested - max_kvi_req[index]) / (min_kvi_req[index] - max_kvi_req[index])
        #     else:
        #         temp_kvi[index] = 1  # Se tutti i valori sono uguali, assegna 1

        #service.min_kvi = np.clip(np.dot(service.weights_kvi, temp_kvi), 0, 1)

# def q_v_big_req(services, signs_kpi, signs_kvi):
#     kpi_tot = []
#     kvi_tot = []
#     temp_kpi = np.zeros(4)
#     temp_kvi = np.zeros(3)
#
#     for service in services:
#         kpi_tot.append(service.kpi_service_req)
#         kvi_tot.append(service.kvi_service_req)
#
#     max_kpi_req = np.max(kpi_tot, axis=0)
#     min_kpi_req = np.min(kpi_tot, axis=0)
#     max_kvi_req = np.max(kvi_tot, axis=0)
#     min_kvi_req = np.min(kvi_tot, axis=0)
#
#     for service in services:
#         for index, requested in enumerate(service.kpi_service_req):
#             if signs_kpi[index] == 1:
#                 if max_kpi_req[index] > min_kpi_req[index]: # benefit
#                     temp_kpi[index] = (requested - min_kpi_req[index]) / (max_kpi_req[index] - min_kpi_req[index]) # singolo elemento normalizzato
#                 else:
#                     temp_kpi[index] = 1
#             else:
#                 if max_kpi_req[index] > min_kpi_req[index]: # cost
#                     temp_kpi[index] = (max_kpi_req[index] - requested) / (max_kpi_req[index] - min_kpi_req[index]) # singolo elemento normalizzato
#                 else:
#                     temp_kpi[index] = 1
#
#         service.min_kpi = np.dot(service.weights_kpi, temp_kpi)
#         service.min_kpi = np.clip(service.min_kpi, 0, 1)
#
#     for service in services:
#         for index, requested in enumerate(service.kvi_service_req):
#             if signs_kvi[index] == 1:
#                 if max_kvi_req[index] > min_kvi_req[index]:  # benefit
#                     temp_kvi[index] = (requested - min_kvi_req[index]) / (
#                                 max_kvi_req[index] - min_kvi_req[index])  # singolo elemento normalizzato
#                 else:
#                     temp_kvi[index] = 1
#             else:
#                 if max_kvi_req[index] > min_kvi_req[index]:  # cost
#                     temp_kvi[index] = (max_kvi_req[index] - requested) / (
#                                 max_kvi_req[index] - min_kvi_req[index])  # singolo elemento normalizzato
#                 else:
#                     temp_kvi[index] = 1
#
#         service.min_kvi = np.dot(service.weights_kvi, temp_kvi)
#         service.min_kvi = np.clip(service.min_kvi, 0, 1)


# esempio per ricordarsi come chiamarle
# computation_time = compute_computation_time(service, resource) -> una coppia specifica
# gains_matrix = compute_channel_gain_matrix([service], [resource])
# gains_eavesdropper = compute_eavesdropper_gain([service], [resource])
# secrecy_capacity = compute_secrecy_capacity(service, gains_matrix[0, 0], gains_eavesdropper[0, 0], resource)
# failure_prob = compute_failure_probability(computation_time, resource)
# energy_sustainability = compute_energy_sustainability(resource, computation_time, CI=400)
