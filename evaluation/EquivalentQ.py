from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import json
import argparse
import time
import re
from utilities import modelpath, promptTechList, joinPromptTemplate, datasetpath, format_time
from inference import main as main_func




def response_evaluator(res, dataObject):
    ans = dataObject['EquivalentQ']
    lower_res = res.lower()
    
    # drop redundant response
    index = lower_res.find('Q: ')

    if index != -1:
        lower_res = lower_res[:index]
    
    if (ans == False and ('not equivalent' in lower_res or 'not equal' in lower_res)) or\
        (ans == True and (
            ('are equivalent' in lower_res and 'are equivalent or not' not in lower_res)
            or 'are equal' in lower_res)
         ):
        return 1

    return 0


def predict(chain, dataObject):

    response = chain.invoke({
        "Expression1": dataObject["Expression-1"],
        "Expression2": dataObject["Expression-2"]
    })['text']

    return response

def log_format(dataObject, promptTemplate, response):
    return {
        "prompt": promptTemplate.format(
            Expression1=dataObject["Expression-1"],
            Expression2=dataObject["Expression-2"]
            ),
        "response": response
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Equivalent Test")
    # prompt args
    parser.add_argument('--prompt', type=str, default="Zero-shot",
                        help='prompting techniques (default: Zero-shot)')
    parser.add_argument('--instructive', type=str, default='False',
                        help='Whether to use instructive prompt (default: False)')
    # model setting args
    parser.add_argument('--model', type=str, default="llama2-13b",
                        help='name of LLM (default: llama2-13b)')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='the max tokens in response (default: 200)')
    parser.add_argument('--temp', type=float, default=0.7,
                        help='temperature of the model (default : 0.7)')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='n_threads of the model (default : 8)')
    parser.add_argument('--ngl', type=int, default=128,
                        help='n_gpu_layers (default: 128)')
    parser.add_argument('--n_batch', type=int, default=1024,
                        help='n_batch of the model (default: 1024)')
    parser.add_argument('--n_ctx', type=int, default=2048,
                        help='n_ctx of the model (default: 2048)')
    # log
    parser.add_argument('--log', type=str, default="True",
                        help='whether write the log file (default : True)')

    # problem 
    parser.add_argument('--problem', type=str, default='DirectBooleanComputation',
                        help='The name of the inference problem (default : DirectBoolean Computation)')
    args = parser.parse_args()

    assert args.prompt in promptTechList
    assert args.log in ['True', 'False']
    assert args.instructive in ['True', 'False']

    args.problem = 'EquivalentQ'

    start_time = time.time()

    main_func(args)

    end_time = time.time()
    print(f"""
        The program stared at {format_time(start_time)},
        finished at {format_time(end_time)},
        and took {end_time - start_time} s\n in total.""")