import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import Model, GRB

from initialization import *


# script to define the saving method, and optimization problem to compute Q^I, V^I, Q^N, V^N

def save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                     results_dir, filename):

    filepath = os.path.join(results_dir, filename)  # Correct path
    results = []

    for request_id, service_id in enumerate(service_requests):  # Iterating on requests
        s = services[service_id]  # Get the corresponding Service object 
        for r in resources:
            assigned = round(x[request_id, r.id].x)  # 1 if assigned, 0 otherwise
            if assigned == 1:  # Only save valid assignments
                list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
                list_s_kvi_service = []
                list_r_kpi_resource = [float(kpi) for kpi in r.kpi_resource]
                list_r_kvi_resource = []

                results.append([
                    request_id, service_id, r.id, assigned,
                    normalized_kpi.get((r.id, service_id), 0),
                    0,
                    weighted_sum_kpi.get((r.id, service_id), 0),
                    weighted_sum_kvi.get((r.id, service_id), 0),
                    s.min_kpi, 0,
                    list_s_kpi_service, list_s_kvi_service,
                    list_r_kpi_resource, list_r_kvi_resource
                ])

    df = pd.DataFrame(results, columns=[
        "Request_ID", "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Weighted_Sum_KPI", "Weighted_Sum_KVI",
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource"
    ])

    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")



def save_epsilon_constraint(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                            weighted_sum_kpi, weighted_sum_kvi, results_dir, epsilon):

    filename = f"epsilon_{epsilon:.6f}.csv"
    filepath = os.path.join(results_dir, filename) 
    results = []

    for request_id, service_id in enumerate(service_requests):  
        s = services[service_id] 
        for r in resources:
            assigned = round(x[request_id, r.id].x) 
            if assigned == 1:  
                list_s_kpi_service = [float(kpi) for kpi in s.kpi_service]
                list_s_kvi_service = []
                list_r_kpi_resource = [float(kpi) for kpi in r.kpi_resource]
                list_r_kvi_resource = []

                results.append([
                    request_id, service_id, r.id, assigned,
                    normalized_kpi.get((r.id, service_id), 0),
                    0,
                    weighted_sum_kpi.get((r.id, service_id), 0),
                    weighted_sum_kvi.get((r.id, service_id), 0),
                    s.min_kpi, 0,
                    list_s_kpi_service, list_s_kvi_service,
                    list_r_kpi_resource, list_r_kvi_resource
                ])

    df = pd.DataFrame(results, columns=[
        "Request_ID", "Service_ID", "Resource_ID", "Assigned",
        "Normalized_KPI", "Normalized_KVI",
        "Weighted_Sum_KPI", "Weighted_Sum_KVI",
        "Min_KPI", "Min_KVI",
        "KPI_Service", "KVI_Service",
        "KPI_Resource", "KVI_Resource"
    ])

    df.to_csv(filepath, index=False)
    print(f"Salvato: {filepath}")

# method to filter points to only get the not dominated ones for post-processing later on

def pareto_filter_maximization(points):
    pareto = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if j != i:
                if x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i):
                    dominated = True
                    break
        if not dominated:
            pareto.append((x_i, y_i))
    return pareto

# method to save pareto solutions

def save_pareto_solutions(points, filename="pareto_solutions.csv"):
    # Salva tutti i punti originali
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["KPI_Totale", "KVI_Totale"])
        writer.writerows(points)

    # Read and filter
    df = pd.read_csv(filename)
    raw_points = df[["KPI_Totale", "KVI_Totale"]].values.tolist()
    non_dominated = pareto_filter_maximization(raw_points)

    # Rank byr KPI_Totale 
    non_dominated = sorted(non_dominated, key=lambda x: x[0])

    # Save the filtered points
    df_pareto = pd.DataFrame(non_dominated, columns=["KPI_Totale", "KVI_Totale"])
    df_pareto.to_csv(filename, index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    kpi, kvi = zip(*non_dominated)
    plt.plot(kpi, kvi, 'bo-', label="Pareto Optimal-Set")
    plt.xlabel("KPI Totale")
    plt.ylabel("KVI Totale")
    plt.title("Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# function to plot an inital pareto frontier to track the status of the simulation

def plot_pareto_front(pareto_solutions):
    pareto_solutions.sort()  # Rank solutions by KPI
    kpi_values, kvi_values = zip(*pareto_solutions)  # Separation in 2 lists

    plt.figure(figsize=(8, 6))
    plt.plot(kpi_values, kvi_values, marker='o', linestyle='-', color='b', label="Pareto Optimal-Set")
    plt.xlabel("KPI Totale")
    plt.ylabel("KVI Totale")
    plt.title("Pareto Front")
    plt.grid(True)
    plt.legend()
    plt.show()

# Optimal points

def optimize_kpi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, results_dir):
    # Creation of the model
    model = Model("Maximize_KPI")

    # Creation of decision variables x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Constraint 1: KPI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 2: KVI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 3: each service is assigned to a resource
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Constraint 4: resource capacity not surpassed
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Objective function: maximize total KPI
    model.setObjective(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.setParam("MIPFocus", 3)  # less nodes
    model.setParam("VarBranch", 2)  # aggressive branching 
    model.setParam("MIPGap", 0.03)

    model.optimize()
    if model.IsMIP == 1:
        print("The model is a MIP.")


    # Results
    if model.status == GRB.OPTIMAL:
        print("\nOptimal Solution:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Service {request_id} assigned to resource {r.id}")

        # Optimal objective value
        print(f"\nOptimal KPI value: {model.ObjVal}")

        Q_I = model.ObjVal

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_qi.csv")

    if model.Status == GRB.INFEASIBLE:
        print("Model infeasible. Analyzing the conflict...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Write the file with constraints responsible for infeasibility
        Q_I = 0

    return Q_I


def optimize_kvi(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, results_dir):

    model = Model("Maximize_KVI")

    # Creation of decision variables x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Constraint 1: KPI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 2: KVI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                (weighted_sum_kvi[(r.id, service_id)]) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 3: Each service is assigned to a resource
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Constraint 4: Resource capacity must not be surpassed
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Objective function: maximize total KVI 
    model.setObjective(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.setParam("MIPFocus", 3)  # less nodes
    model.setParam("VarBranch", 2)  # aggressive branching 
    model.setParam("MIPGap", 0.03)

    model.optimize()

    if model.IsMIP == 1:
        print("The model is a MIP.")

    # Results
    if model.status == GRB.OPTIMAL:
        print("\nOptimal Solution:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Service {request_id} assigned to resource {r.id}")

        # Optimal objective value
        print(f"\nKVI optimal value: {model.ObjVal}")

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_vi.csv")

        V_I = model.ObjVal
    if model.Status == GRB.INFEASIBLE:
        print("Model infeasible. Analyzing the conflict...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Write the file with the constraints responsible of infeasability
        V_I = 0

    return V_I


def q_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, V_I, results_dir):
    # Model creation
    model = Model("Maximize_KPI_constraining_V")

    # Creation of decision variables x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Constraint 1: objective V(X) maximum value is = V_I
    model.addConstr(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources)
        >= V_I - 0.01,
        "kvi_equals_max_kvi_value"
    )

    # Constraint 2: KPI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 3: KVI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 4: Each service is assigned to a resource
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Constraint 5: Resource capacity must not be surpassed
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Objective function: maximize total KPI 
    model.setObjective(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.setParam("MIPFocus", 3)  # less nodes
    model.setParam("VarBranch", 2)  # aggressive branching
    model.setParam("MIPGap", 0.03)

    model.optimize()
    print(f"DEBUG: Q_N computed = {model.ObjVal}")

    if model.IsMIP == 1:
        print("The model is a MIP.")

    # Results
    if model.status == GRB.OPTIMAL:
        print("\nOptimal Solution:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Service {request_id} assigned to resource {r.id}")

        # Optimal objective value
        print(f"\nKPI optimal value: {model.ObjVal}")

        Q_N = model.ObjVal

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_qn.csv")

    if model.Status == GRB.INFEASIBLE:
        print("The model is infeasible.")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Write the file with the constraints responsible of infeasability
        Q_N = 0

    return Q_N


def v_nadir(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi, Q_I, results_dir):
    # Model creation
    model = Model("Maximize_KVI_constraining_Q")

    # Creation of decision variables x[s, r] ∈ {0,1}
    x = model.addVars(
        [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
        vtype=GRB.BINARY,
        name="x"
    )

    # Constraint 1: objective Q(X) maximum value is = Q_I
    model.addConstr(
        sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources)
        >= Q_I,
        "kpi_equals_max_kpi_value"
    )

    # Constraint 2: KPI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 3: KVI offered by the resource > minimum required
    for request_id in range(len(service_requests)):
        service_id = service_requests[request_id]  # ID of the service associated to the request
        s = services[service_id]  # Corresponding service object 

        for r in resources:
            model.addConstr(
                weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                f"kpi_threshold_{request_id}_{r.id}"
            )

    # Constraint 4: Each service is assigned to a resource
    for request_id, service_id in enumerate(service_requests):
        s = services[service_id]
        model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

    # Constraint 5: Resource capacity must not be surpassed
    for r in resources:
        model.addConstr(
            sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                for request_id in range(len(service_requests))) <= r.availability,
            f"capacity_{r.id}"
        )

    # Objective function: maximize total KVI 
    model.setObjective(
        sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
            for request_id in range(len(service_requests)) for r in resources),
        GRB.MAXIMIZE
    )

    model.setParam("MIPFocus", 3)  # less nodes
    model.setParam("VarBranch", 2)  # aggressive branching
    model.setParam("MIPGap", 0.03)

    model.optimize()
    if model.IsMIP == 1:
        print("The model is a MIP.")

    # Results
    if model.status == GRB.OPTIMAL:
        print("\nOptimal Solution:")
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            for r in resources:
                if round(x[request_id, r.id].x) == 1:
                    print(f"Service {request_id} assigned to resource {r.id}")

        # Optimal objective value
        print(f"\nKVI optimal value: {model.ObjVal}")

        save_results_csv(service_requests, services, resources, x, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                         results_dir, filename="results_optimization_vn.csv")

        V_N = model.ObjVal

    if model.Status == GRB.INFEASIBLE:
        print("Model infeasible. Analyzing the conflict...")

        model.computeIIS()

        model.write("infeasible_model.ilp")  # Write the file with the constraints responsible of infeasability

        V_N = 0

    return V_N


def epsilon_constraint_exact(service_requests, services, resources, normalized_kpi, normalized_kvi, weighted_sum_kpi, weighted_sum_kvi,
                             Q_N, Q_I, delta, results_dir):
    pareto_solutions = []
    epsilon = Q_N - delta  # epsilon inital value

    while epsilon <= Q_I:
        # Model creation
        model = Model("Epsilon_Constraint_Exact")

        # Decision variable x[s, r] ∈ {0,1}
        x = model.addVars(
            [(request_id, r.id) for request_id in range(len(service_requests)) for r in resources],
            vtype=GRB.BINARY,
            name="x"
        )

        # Constraints:

        # Constraint epsilon-constraint sul KPI
        model.addConstr(
            sum(weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id]
                for request_id in range(len(service_requests)) for r in resources)
            >= epsilon,
            "epsilon_kpi"
        )

        # Constraint 2: KPI offered by the resource > minimum required
        for request_id in range(len(service_requests)):
            service_id = service_requests[request_id]  # ID of the service associated to the request
            s = services[service_id]  # Corresponding service object 

            for r in resources:
                model.addConstr(
                    (weighted_sum_kpi[(r.id, service_id)] - s.min_kpi) * x[request_id, r.id] >= 0,
                    f"kpi_threshold_{request_id}_{r.id}"
                )

        # Constraint 3: KVI offered by the resource > minimum required
        for request_id in range(len(service_requests)):
            service_id = service_requests[request_id]  # ID of the service associated to the request
            s = services[service_id]  # Corresponding service object 

            for r in resources:
                model.addConstr(
                    weighted_sum_kvi[(r.id, service_id)] * x[request_id, r.id] >= 0,
                    f"kpi_threshold_{request_id}_{r.id}"
                )

        # Constraint 4: Each service is assigned to a resource
        for request_id, service_id in enumerate(service_requests):
            s = services[service_id]
            model.addConstr(sum(x[request_id, r.id] for r in resources) == 1, f"assign_service_{request_id}")

        # Constraint 5: Resource capacity must not be surpassed
        for r in resources:
            model.addConstr(
                sum(x[request_id, r.id] * services[service_requests[request_id]].demand
                    for request_id in range(len(service_requests))) <= r.availability,
                f"capacity_{r.id}"
            )

        # Objective function: maximize total KVI 
        model.setObjective(
            sum(weighted_sum_kvi[(r.id, service_requests[request_id])] * x[request_id, r.id]
                for request_id in range(len(service_requests)) for r in resources),
            GRB.MAXIMIZE
        )

        model.setParam("MIPFocus", 3)  # less nodes
        model.setParam("VarBranch", 2)  # aggressive branching
        model.setParam("MIPGap", 0.03)  # stop search when the gap is less than 2.8%

        # Solve model
        model.optimize()
        if model.IsMIP == 1:
            print("The model is a MIP.")

        # Save solution
        if model.status == GRB.OPTIMAL:
            kpi_value = sum(
                weighted_sum_kpi[(r.id, service_requests[request_id])] * x[request_id, r.id].x
                for request_id in range(len(service_requests)) for r in resources
            )
            kvi_value = model.ObjVal
            pareto_solutions.append((kpi_value, kvi_value))
            print(f"Epsilon: {epsilon}, KPI: {kpi_value}, KVI: {kvi_value}")

            save_epsilon_constraint(service_requests, services, resources, x, normalized_kpi, normalized_kvi,
                                    weighted_sum_kpi, weighted_sum_kvi, results_dir, epsilon)

        # Increase epsilon
        epsilon += delta

    return pareto_solutions

