from main import *
import numpy as np

# funzioni per la normalizzazione attributi KPI e indicatori KVI
def normalized_attributes():

    return normalized_attributes


def normalized_kpi(services, resources):
    #services[i].kpi_service for i in len(services)
    normalized_kpi = {}

    for resource in resources:
        for service in services:
            # no divisioni per zero
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_kpi = np.where(service.kpi_service != 0, resource.kpi_resource / service.kpi_service, 0)

            # kpi globale, sommatoria
            q_x = np.dot(service.weights_kpi, norm_kpi)

            normalized_kpi[(resource.id, service.id)] = float(q_x)

    return normalized_kpi


def normalized_kvi(services, resources):
    normalized_kvi = {}

    for resource in resources:
        for service in services:
            # no divisioni per zero
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_kvi = np.where(service.kvi_service != 0, resource.kvi_resource / service.kvi_service, 0)

            # kvi globale, sommatoria
            v_x = np.dot(service.weights_kvi, norm_kvi)

            normalized_kvi[(resource.id, service.id)] = float(v_x)

    return normalized_kvi

# funzioni per il calcolo di KPI globale ex., Q(X)

def compute_Q():
    Q = sum()
    return Q

# funzioni per il calcolo di KVI globale ex., V(X)

def compute_V():
    V = sum()
    return V



