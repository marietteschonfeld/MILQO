from inference_model import *

# execute a query plan on data
def execute_query_plan(query, NF, data, assignment, ordering, full_excute=False, regression=False):
    classification = {}
    # mark every point as to be evaluated
    evals = [0.5]*data.shape[0]
    # assignment is pred->model dict. Reverse this.
    collapsed_assignment = collapse_assignment(assignment)
    for predicate in ordering:
        # if predicate not yet classified by other model
        if predicate not in classification.keys():
            # generate inference model
            model_name = assignment[predicate]
            model = inference_model(model_name)
            # filter all unevaluated points
            if full_excute:
                filter = [True] * len(evals)
            else:
                filter = [x == 0.5 for x in evals]
            # classify
            classi = model.classify(data, filter, regression)
            # if other predicates use same model, mark classification as well.
            for pred in collapsed_assignment[model_name]:
                classification[pred] = classi[pred]
            # for every unevaluated point see if query can be evaluated
            for point in range(data.shape[0]):
                if evals[point] == 0.5 and not regression:
                    evals[point] = evaluate_query(query, NF, classification, point)
            # if every point has been evaluated, stop.
            if 0.5 not in evals and not full_excute:
                return evals, classification
    return evals, classification

# reverse dictionary
def collapse_assignment(assignment):
    collapsed_assignment = {}
    for (predicate, model) in assignment.items():
        if model in collapsed_assignment.keys():
            collapsed_assignment[model].append(predicate)
        else:
            collapsed_assignment[model] = [predicate]
    return collapsed_assignment

# evaluate query on a single data point
def evaluate_query(query, NF, classification, point):
    # evaluate result of every group
    eval = [evaluate_group(group, NF, classification, point) for group in query]
    # separate NF cases
    if NF == 'DNF':
        # one group true
        if 1 in eval: return True
        if 1 not in eval and 0.5 not in eval: return False
        return 0.5
    else:
        # every group true
        if 0 not in eval and 0.5 not in eval: return True
        if 0 in eval: return False
        return 0.5

def evaluate_group(group, NF, classification, point):
    # separate in NFs
    if NF == 'DNF':
        for pred in group:
            # return false if one predicate False
            if pred in classification.keys():
                if classification[pred][point] == 0:
                    return False
        # return 0.5 if query cannot be evaluated yet, or 1 if total group is evaluated
        return 0.5 + 0.5 * int(set(group) <= set(classification.keys()))
    else:
        # true if one predicate true
        for pred in group:
            if pred in classification.keys():
                if classification[pred][point] == 1:
                    return True
        # return false if entire group has been evaluated
        return 0.5 - 0.5 * int(set(group) <= set(classification.keys()))