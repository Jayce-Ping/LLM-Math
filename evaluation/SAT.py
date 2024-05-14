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




def response_evaluator(res, SAT_data):
    
    SAT_re = SAT_re = '|'.join(
        [
            '.{2,5}'.join(item) for item in [
                [
                    var + ' = ' + str(val) for var, val in zip(SAT_data['Variables'], combination)
                ] for combination in SAT_data['SAT']
            ]
        ] + [
            '\(' + '\W{2,5}'.join(item) + '\)' for item in [
                [str(item) for item in SAT_item]
                for SAT_item in SAT_data['SAT']
            ]
        ]
    )

    matchCase = re.search(SAT_re, res, re.I)

    if matchCase:
        return 1
    else:
        return 0


def predict(chain, dataObject):

    response = chain.invoke({
        "Expression": dataObject["Expression"],
        "Variables": ', '.join(dataObject["Variables"])
    })['text']

    return response


def log_format(dataObject, promptTemplate, response):
    return {
        "prompt": promptTemplate.format(
            Expression=dataObject["Expression"],
            Variables=', '.join(dataObject["Variables"])
            ),
        "response": response
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Satisfiability Instances Problem")
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
    args = parser.parse_args()

    assert args.prompt in promptTechList
    assert args.log in ['True', 'False']
    assert args.instructive in ['True', 'False']

    args.problem = 'SAT'

    start_time = time.time()

    main_func(args)

    end_time = time.time()
    print(f"""
        The program stared at {format_time(start_time)},
        finished at {format_time(end_time)},
        and took {end_time - start_time} s\n in total.""")