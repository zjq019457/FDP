# Mirrors lm_eval/tasks/winogrande/preprocess_winogrande.py for local jsonl
# (partial evaluation: https://arxiv.org/abs/1806.02847)


def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[str(doc["answer"]).strip()]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]
