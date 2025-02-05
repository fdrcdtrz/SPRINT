from main import *
import numpy as np


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

def normalized_kpi(services, resources, signs):
    # services[i].kpi_service for i in len(services)
    normalized_kpi = {}
    weighted_sum_kpi = {}

    def normalize_single_row(kpis_service, resources, index_res):
        row = np.zeros(len(kpis_service))  # inizializzo vettore riga con i kpi della risorsa index_res-esima
        # normalizzati per i parametri kpis_service

        for index, attribute in enumerate(kpis_service):  # faccio stessa cosa di sopra considerando una risorsa ben
            # precisa, un servizio ben preciso, ed il valore massimo esposto
            adjusted_attribute = attribute * signs[index]
            exposed_kpi = resources[index_res].kpi_resource[index]
            max_val = np.max([resource.kpi_resource[index] for resource in resources])
            # check per zero
            if max_val == adjusted_attribute:
                row[index] = 0
            else:
                row[index] = (exposed_kpi - adjusted_attribute) / (max_val - adjusted_attribute)
        return np.abs(row)

    for j, service in enumerate(services):
        for n, resource in enumerate(resources):
            norm_kpi = normalize_single_row(service.kpi_service, resources, n)
            normalized_kpi[(resource.id, service.id)] = norm_kpi

            # kpi globale, sommatoria
            q_x = np.dot(service.weights_kpi,
                         norm_kpi)  # somma pesata da moltiplicare alla variabile decisionale nel problema di ottimizzazione

            weighted_sum_kpi[(resource.id, service.id)] = float(q_x)

    return normalized_kpi, weighted_sum_kpi

def normalized_kvi(services, resources, signs):
    # services[i].kpi_service for i in len(services)
    normalized_kvi = {}
    weighted_sum_kvi = {}

    def normalize_single_row(kvis_service, resources, index_res):
        row = np.zeros(len(kvis_service))  # inizializzo vettore riga con i kpi della risorsa index_res-esima
        # normalizzati per i parametri kpis_service

        for index, attribute in enumerate(kvis_service):  # faccio stessa cosa di sopra considerando una risorsa ben
            # precisa, un servizio ben preciso, ed il valore massimo esposto
            adjusted_attribute = attribute * signs[index]
            exposed_kvi = resources[index_res].kvi_resource[index]
            max_val = np.max([resource.kvi_resource[index] for resource in resources])
            # check per zero
            if max_val == adjusted_attribute:
                row[index] = 0
            else:
                row[index] = (exposed_kvi - adjusted_attribute) / (max_val - adjusted_attribute)
        return np.abs(row)

    for j, service in enumerate(services):
        for n, resource in enumerate(resources):
            norm_kvi = normalize_single_row(service.kvi_service, resources, n)
            normalized_kvi[(resource.id, service.id)] = norm_kvi

            # kpi globale, sommatoria
            v_x = np.dot(service.weights_kvi,
                         norm_kvi)  # somma pesata da moltiplicare alla variabile decisionale nel problema di ottimizzazione

            weighted_sum_kvi[(resource.id, service.id)] = float(v_x)

    return normalized_kvi, weighted_sum_kvi


# funzione calcolo KVI sostenibilità ambientale


# funzione calcolo KVI trustworthiness


# funzione calcolo KVI inclusività
