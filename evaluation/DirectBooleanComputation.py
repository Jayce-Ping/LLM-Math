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
    ans = dataObject['Answer']
    exp = dataObject['Expression']
    res_lower = res.lower()
    ans_lower = str(ans).lower()

    # answer in A: ans form
    if 'a: ' + ans_lower in res.lower():
        return 1
    
    # answer in some certain form
    if "answer is " + ans_lower in res_lower or \
        "result is " + ans_lower in res_lower or \
            "answer is `" + ans_lower + "`" in res_lower or \
            "result is `" + ans_lower + "`" in res_lower or \
            exp + ' is ' + ans_lower in res_lower or \
            exp + ' is `' + ans_lower + "`" in res_lower or \
            exp + ' = ' + ans_lower in res_lower or \
            exp + ' = `' + ans_lower + "`" in res_lower or\
            exp + ' evaluates ' + ans_lower in res_lower or\
            exp + ' evaluates `' + ans_lower + "`" in res_lower:
        return 1
    
    # allow the model to represent 0 for False and 1 for True
    if (ans == True and ' 1 ' in res_lower) or (ans == False and ' 0 ' in res_lower):
        return 1
    
    return 0
    # search the answer from the end of the response - deprecated
    rev_res = res[::-1].lower()
    false_pos = rev_res.find(' eslaf ')
    true_pos = rev_res.find(' eurt ')

    false_pos = false_pos if false_pos != -1 else len(res) + 1
    true_pos = true_pos if true_pos != -1 else len(res) + 1

    if true_pos < false_pos and ans == True:
        return 1
    elif false_pos < true_pos and ans == False:
        return 1
    else:
        return 0


def predict(chain, dataObject):

    response = chain.invoke({
        "Expression": dataObject["Expression"]
    })['text']

    return response


def log_format(dataObject, promptTemplate, response):
    return {
        "prompt": promptTemplate.format(Expression=dataObject["Expression"]),
        "response": response
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Direct Boolean Computation")
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

    args.problem = 'DirectBooleanComputation'

    start_time = time.time()

    main_func(args)

    end_time = time.time()
    print(f"""
        The program stared at {format_time(start_time)},
        finished at {format_time(end_time)},
        and took {end_time - start_time} s\n in total.""")