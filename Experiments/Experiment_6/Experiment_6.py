# Experiment 6: Does splitting up model have more potential than classifying all at once?
import ast
from ExecuteQueryPlan import *
import pandas as pd
from data_loader import *
import time

# filename = "NLP_modelDB_f1_score.csv"
# sel_filename = "NLP_selectivity.csv"
# A, C, mem, sel = data_loader(filename, sel_filename)

data = pd.read_csv("train_extend.csv")
test_data = pd.read_csv("test.csv")

with open("assignment_exp6.txt", 'r') as fp:
    assignment = ast.literal_eval(fp.read())

with open("ordering_exp6.txt", 'r') as fp:
    ordering = ast.literal_eval(fp.read())

query = [["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
NF = "CNF"

start = time.time()
(eval, classification) = execute_query_plan(query, NF, test_data['comment_text'], assignment, ordering, full_excute=True, regression=True)
end = time.time()
query_time = end-start

with open("eval_exp_6.txt", 'w') as fp:
    fp.write(eval)
with open("classification_exp_6.txt", 'w') as fp:
    fp.write(classification)
with open("time_exp_6.txt", 'w') as fp:
    fp.write(query_time)

assignment_base = {"toxic": "LSTM_large_toxic_severe_toxic_obscene_threat_insult_identity_hate",
                   "severe_toxic": "LSTM_large_toxic_severe_toxic_obscene_threat_insult_identity_hate",
                   "obscene": "LSTM_large_toxic_severe_toxic_obscene_threat_insult_identity_hate",
                   "threat": "LSTM_large_toxic_severe_toxic_obscene_threat_insult_identity_hate",
                   "insult": "LSTM_large_toxic_severe_toxic_obscene_threat_insult_identity_hate",
                   "identity_hate": "LSTM_large_toxic_severe_toxic_obscene_threat_insult_identity_hate"}
ordering_base = query[0]
start = time.time()
(eval_base, classification_base) = execute_query_plan(query, NF, test_data['comment_text'], assignment_base, ordering_base, full_excute=True, regression=True)
end = time.time()
query_time_base = end-start

with open("eval_base_exp_6.txt", 'w') as fp:
    fp.write(eval_base)

with open("classification_base_exp_6.txt", 'w') as fp:
    fp.write(classification_base)

with open("time_base_exp_6.txt", 'w') as fp:
    fp.write(query_time_base)