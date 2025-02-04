# from random import randint
# from gurobipy import Model, GRB
#
# def ideal_points_Q(services, resources):
#
#     # Definizione del modello
#     model = Model("KPI")
#     # Variabili di decisione
#
#     x = model.addVars(services, resources, vtype=GRB.BINARY, name="x")
#     # Vincoli: 1) qualità
#
#     # 2) Ogni servizio è assegnato a una sola risorsa
#     for s in services:
#         model.addConstr(sum(x[s, r] for r in resources) == 1)
#     # 2) Capacità delle risorse non deve essere superata
#     for r in resources:
#         model.addConstr(sum(x[s, r] * s. for s in services) <= Capacity[r])
#     # Definizione del primo obiettivo: Massimizzare KPI
#         model.setObjective(sum(KPI[s, r] * x[s, r] for s in services for r in resources),GRB.MAXIMIZE)
#
#     # Ottimizzazione del KPI
#     model.optimize()
#
#     # Salviamo il valore ottimale di KPI trovato
#     ptimal_KPI = model.ObjVal
#
#     # Creazione di un nuovo modello per la seconda fase
#     model2 = Model("KVI")
#     # Stesse variabili di decisione
#     #
#     x2 = model2.addVars(services, resources, vtype=GRB.BINARY, name="x")
#     # Stessi vincoli
#     for s in services:
#         model2.addConstr(sum(x2[s, r] for r in resources) == 1)
#     for r in resources:
#         model2.addConstr(sum(x2[s, r] * Demand[s] for s in services) <= Capacity[r])
#     # Aggiungiamo il vincolo che garantisce che il KPI rimanga almeno il valore ottimale trovato
#     model2.addConstr(sum(KPI[s, r] * x2[s, r] for s in services for r in resources) >= optimal_KPI)
#
#     # Definizione del secondo obiettivo: Massimizzare il Profitto
#     model2.setObjective(        sum(KVI[s, r] * x2[s, r] for s in services for r in resources),
#         GRB.MAXIMIZE    )
#     # Ottimizzazione del profitto mantenendo il KPI ottimale
#     model2.optimize()
#     # Stampa dei risultati    print("\nSoluzione finale:")
#     for s in services:
#         for r in resources:
#             if x2[s, r].x > 0.5:
#                 print(f"Servizio {s} assegnato a risorsa {r}")