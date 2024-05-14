from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import json
import argparse
import time
from utilities import modelpath, promptTechList, joinPromptTemplate, datasetpath, format_time, data_store_dir


parser = argparse.ArgumentParser(description="LLM Inference")

def main(args):

    promptTech = args.prompt
    max_tokens = args.max_tokens
    problem_name = args.problem
    evaluation_result = []
    logData = []
    instructive = eval(args.instructive)

    # load predict, response evaluator and log file format functions.
    match problem_name:
        case 'Direct Boolean Computation'|'DirectBooleanComputation':
            from DirectBooleanComputation import predict, response_evaluator, log_format
        case 'Indirect Boolean Computation'|'IndirectBooleanComputation':
            from IndirectBooleanComputation import predict, response_evaluator, log_format
        case 'SAT':
            from SAT import predict, response_evaluator, log_format
        case 'SAT Count'|'SATCount':
            from SATCount import predict, response_evaluator, log_format
        case 'TautologyQ':
            from TautologyQ import predict, response_evaluator, log_format
        case 'EquivalentQ':
            from EquivalentQ import predict, response_evaluator, log_format
        case 'CNF':
            from CNF import predict, response_evaluator, log_format
        case 'DNF':
            from DNF import predict, response_evaluator, log_format

    # load dataset
    with open(datasetpath + f'{problem_name}.json') as f:
        dataset = json.load(f)
    
    # match model
    match args.model:
        case 'llama2-7b':
            targetModel = "llama-2-7b-chat.Q4_0.gguf"
        case 'llama2-13b':
            targetModel = "llama-2-13b-chat.Q4_0.gguf"
        case 'orca':
            targetModel = 'orca-mini-3b-gguf2-q4_0.gguf'
        case 'wizardmath-7b':
            targetModel = "wizardmath-7b-v1.1.Q4_0.gguf"
        case 'wizardmath-13b':
            targetModel = "wizardmath-13b-v1.0.Q4_0.gguf"
    
    # load model
    model = LlamaCpp(model_path = modelpath + targetModel,
                    max_tokens=max_tokens,
                    temperature=args.temp,
                    n_gpu_layers=args.ngl,
                    n_batch=args.n_batch,
                    n_ctx=args.n_ctx,
                    f16_kv=True,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                    verbose=False
                    )

    prompt = {
        tech: PromptTemplate.from_template(
            joinPromptTemplate(problem_name, instructive)[tech]
        ) for tech in promptTechList
    }

    llm_chain = LLMChain(prompt=prompt[promptTech],
                        llm=model, output_parser=StrOutputParser())


    for dataObject in dataset[:1]:
        
        # predict by model
        response = predict(llm_chain, dataObject)
        
        # evaluate the response
        evaluation_result.append(response_evaluator(response, dataObject))
        
        # write prompt and repsonse log file
        if eval(args.log):
            logData.append(log_format(dataObject, prompt[promptTech], response))
            with open(f"{data_store_dir}/response/{problem_name}/{problem_name}-({promptTech})-{args.model}.json", "w") as f:
                f.write(json.dumps(logData))
                f.close()
        
        # write result binary vector
        with open(f"{data_store_dir}/result/{problem_name}-({promptTech})-{args.model}.log", "w") as f:
            for element in evaluation_result:
                f.write(str(element))
                f.write('\n')


    print(f"Correct cases = {sum(evaluation_result)}")
    print(
        f"Accuracy rate = {sum(evaluation_result) / (len(dataset))}")




if __name__ == '__main__':
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
    assert args.problem in ['DirectBooleanComputation','IndirectBooleanComputation','CNF','DNF','SAT','SATCount','TautologyQ','EquivalentQ']
    
    start_time = time.time()

    main(args)
    
    end_time = time.time()
    print(f"""
        The program stared at {format_time(start_time)},
        finished at {format_time(end_time)},
        and took {end_time - start_time} s\n in total.""")
    