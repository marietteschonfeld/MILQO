import ast


def printer(query, NF):
    res = "$"
    for group in query:
        res += "("
        for predicate in group[:-1]:
            res += predicate + "\ "
            if NF == "CNF":
                res += "\wee" +"\ "
            else:
                res += "\wedge" + "\ "
        res += group[-1] + ")"

        if NF == "CNF":
            res += "\ " + "\wedge" + "\ "
        else:
            res += "\ " + "\wee" + "\ "

    if NF == "CNF":
        res = res[:-10]
    else:
        res = res[:-8]
    res = res.replace("_", "\_")
    return "\item "+ res + "$"

queries = []
with open('Experiment_5/queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))
for query in queries:
    for NF in ["CNF", "DNF"]:
        print(printer(query, NF))