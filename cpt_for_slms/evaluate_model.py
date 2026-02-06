
import argparse
import json
import os

from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from loguru import logger
from transformers import AutoModelForCausalLM



TASKS = [
    'all',
    'hellaswag',
    'arc',
    #'piqa',
    'mmlu',
    'commonsenseqa',
    'triviaqa',
    'winogrande',
    'openbookqa',
    'gsm8k',
]

DTYPES = [
    "float16", 
    "bfloat16", 
    "float32", 
    "auto", 
    "4bit", 
    "8bit"
]


def check_arguments(args: argparse.Namespace) -> None:
    if args.few_shots and not isinstance(args.few_shots, int):
        raise ValueError(f"'few_shots' argument should be an integer")
    if not isinstance(args.dtype, str) or args.dtype not in DTYPES:
        raise ValueError(f"'dtype' argument should be one of the following: {DTYPES}")
    if not args.task in TASKS:
        raise ValueError(f"'task' argument should be one of the following: {TASKS}")



def evaluate(args: argparse.Namespace) -> None:
    logger.info(f"********** Model evaluation main program **********")
    logger.info(f">>> Verifying arguments")
    check_arguments(args)

    # To init before instantiating the pipeline 
    #accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])

    logger.info(f">>> Instantiating the pipeline")
    evaluation_tracker = EvaluationTracker(
        output_dir=f"{args.output_dir}",
        save_details=True,
        push_to_hub=False,
        hub_results_org=f"{args.HF_username}",  # Replace with your actual username
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory=None,  # Set to path if using custom tasks
        max_samples = 10 if args.debug else None # To disable once the configuration is tested
    )

    """
    model_config = VLLMModelConfig(
        model_name = f"{args.model_name_or_path}",
        dtype = f"{args.dtype}",
        batch_size = args.batch_size,
        accelerator = accelerator,
        model_parallel = True # Forces use of accelerate library for large models
    )
    """
    model = AutoModelForCausalLM.from_pretrained(
        f"{args.model_name_or_path}", 
        device_map="auto",
        #accelerator = accelerator,
    )
    config = TransformersModelConfig(
        model_name = f"{args.model_name_or_path}",
        dtype = f"{args.dtype}",
        batch_size = args.batch_size,
        #accelerator = accelerator,
        model_parallel = True # Forces use of accelerate library for large models
        )
    model = TransformersModel.from_model(model, config)

    task = args.task
    if task != 'all':
        if args.few_shots:
            task = f"{task}|{args.few_shots}"
    else:
        task = ','.join(f"{t}|{args.few_shots}" if args.few_shots else f"{t}" for t in TASKS if t not in {'all','gsm8k'})
        task = f"{task},gsm8k|5"
    
    print(task)

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=config,
    )

    logger.info(f">>> Evaluating")

    pipeline.evaluate()
    #pipeline.save_and_push_results()
    pipeline.show_results()
    results = pipeline.get_results()["results"]
    
      
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f'results_{task}.json')
    with open(outfile, 'w') as fp:
        json.dump(results, fp)

    logger.info(f">>> Done")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluation program')
    parser.add_argument(
        '--HF_username', 
        type = str,
        help='User identifier on HF hub.', 
        default='EmYassir'
        )
    parser.add_argument(
        '--model_name_or_path', 
        type = str,
        help='Model identifier on HF hub or path on disk.', 
        default='HuggingFaceTB/SmolLM2-135M-8K'
        )
    parser.add_argument(
        '--few_shots',
        type = int,
        help='Number of examples to show the model.', 
        default=None
        )
    parser.add_argument(
        '--dtype', 
        type = str,
        help='Precision level.', 
        default='bfloat16'
        )
    parser.add_argument(
        '--task',
        type = str,
        help='Task to evaluate on.', 
        default='mmlu'
        )
    parser.add_argument(
        '--batch_size',
        type = int,
        help='Evaluation batch size.', 
        default=32
        )
    parser.add_argument(
        '--output_dir', 
        type = str,
        help='Where to save results.', 
        default='./output'
        )
    parser.add_argument(
        '--debug', 
        help='Activate while debugging the program.', 
        action='store_true',
        default = False
        )

    args = parser.parse_args()
    evaluate(args)
