from main import *
import numpy as np

# funzioni per la normalizzazione attributi KPI e indicatori KVI e rispettivi Q e V
def normalized_kpi(services, resources):
    #services[i].kpi_service for i in len(services)
    normalized_kpi = {}

    for resource in resources:
        for service in services:
            # no divisioni per zero
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_kpi = np.where(service.kpi_service != 0, resource.kpi_resource / service.kpi_service, 0)
                norm_kpi = np.clip(norm_kpi, 0, 1)

            # kpi globale, sommatoria
            q_x = np.dot(service.weights_kpi, norm_kpi) #somma pesata da moltiplicare alla variabile decisionale nel problema di ottimizzazione

            normalized_kpi[(resource.id, service.id)] = float(q_x)

    return normalized_kpi

def normalized_kvi(services, resources):
    normalized_kvi = {}

    for resource in resources:
        for service in services:
            # no divisioni per zero
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_kvi = np.where(service.kvi_service != 0, resource.kvi_resource / service.kvi_service, 0)
                norm_kvi = np.clip(norm_kvi, 0, 1)

            # kvi globale, sommatoria
            v_x = np.dot(service.weights_kvi, norm_kvi) #somma pesata da moltiplicare alla variabile decisionale nel problema di ottimizzazione

            normalized_kvi[(resource.id, service.id)] = float(v_x)

    return normalized_kvi

# funzione calcolo KVI sostenibilità ambientale


# funzione calcolo KVI trustworthiness


# funzione calcolo KVI inclusività





