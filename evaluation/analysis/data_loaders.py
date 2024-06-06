import json

def load_inference_data(problem_name, prompt_name, model_name='llama2-13b'):
    filename = f"../response/{problem_name}/{problem_name}-({prompt_name})-{model_name}.json"

    with open(filename) as f:
        inferenceData = json.load(f)

    return inferenceData


def load_dataset(problem_name):
    with open(f"../../Dataset/{problem_name}.json") as f:
        dataObjects = json.load(f)

    return dataObjects


def load_data_all(problem_name):
    dataset = load_dataset(problem_name)

    inference_data = {
        model: {p: load_inference_data(
            problem_name, p, model) for p in promptTechList}
        for model in modelList
    }

    return dataset,inference_data


def accuracy(res_vec):
    return sum(res_vec) / len(res_vec)


promptTechList = ["Zero-shot", "Few-shot", "LtM", "CoT", "Few-shot-CoT"]
modelList = ['llama2-13b', 'wizardmath-13b']
problem_list = ['DirectBooleanComputation', 'IndirectBooleanComputation', 'CNF', 'DNF', 'TautologyQ', 'EquivalentQ', 'SAT', 'SAT Count']