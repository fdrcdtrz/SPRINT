from main import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import csv


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
    return resource.carbon_offset - (computation_time / 3600) * resource.lambda_services_per_hour * (
            resource.availability * resource.P_c * resource.u_c + resource.n_m * resource.P_m) * PUE * CI


# funzione calcolo KVI trustworthiness
def compute_secrecy_capacity(service, gain_values, gain_values_eavesdropper, resource):
    return max(0, np.log2(1 + (service.p_s * gain_values / resource.N0)) -
               np.log2(1 + (service.p_s * gain_values_eavesdropper / resource.N0)))


# funzione calcolo KVI inclusiveness
def compute_failure_probability(computation_time, resource):
    exponent = - 24 / resource.lambda_failure
    failure_probability = (1 - np.exp(exponent))  # p_rn piccolina
    F_rn_0 = (1 - failure_probability) ** resource.availability
    print("F_rn_0", F_rn_0)
    time_in_hour = computation_time / 3600  # tempo di completamento da secondi ad ore perché sia coerente con lambda_24
    return (F_rn_0 * time_in_hour * resource.lambda_services_per_hour) / 24


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

