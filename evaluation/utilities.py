from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
import json
import time


datasetpath = '../Dataset/'
modelpath = "./Path/To/Your/Model"
data_store_dir ='./'
promptTechList = ["Zero-shot", "Few-shot", "LtM", "CoT", "Few-shot-CoT"]
instruction = "Notation convention: \"&&\" stands for the \"and\" operator, \"||\" stands for the \"or\" operator and \"!\" stands for the \"not\" operator.\n"


def joinPromptTemplate(problem_name, instructive = False):
    
    with open(datasetpath + 'promptTemplate.json') as f:
        prompt_templates = json.load(f)
    

    prompt_data = prompt_templates[problem_name]
    
    
    return {
        'Zero-shot' : prompt_data['Q'],
        'Few-shot' : prompt_data['Few-shot'] + '\nQ: ' + prompt_data['Q'],
        'LtM' : prompt_data['Q'] + ' ' + prompt_data['LtM'],
        'CoT' : prompt_data['Q'] + ' ' + prompt_data['CoT'],
        'Few-shot-CoT' : prompt_data['Few-shot-CoT'] + '\nQ: ' + prompt_data['Q'] + ' ' + prompt_data['CoT'] 
    } if instructive == False else{
        'Zero-shot' : instruction + prompt_data['Q'],
        'Few-shot' : instruction + prompt_data['Few-shot'] + '\nQ: ' + prompt_data['Q'],
        'LtM' : instruction + prompt_data['Q'] + ' ' + prompt_data['LtM'],
        'CoT' : instruction + prompt_data['Q'] + ' ' + prompt_data['CoT'],
        'Few-shot-CoT' : instruction + prompt_data['Few-shot-CoT'] + '\nQ: ' + prompt_data['Q'] + ' ' + prompt_data['CoT'] 
    }

def format_time(t):
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t))