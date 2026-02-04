import os
import sys
import argparse
import lighteval

from loguru import logger



def check_arguments(args: argparse.Namespace) -> None:
    if not isinstance(args.few_shots, int):
        raise ValueError(f"'few_shots' argument should be an integer")
    if not isinstance(args.dtype, str) or args.dtype not in {'bf16', 'fp16', 'float'}:
        raise ValueError(f"'dtype' argument should be one of the following: {['bf16', 'fp16', 'float']}")






def main(args: argparse.Namespace):
    logger.info(f"********** Model evaluation main program **********")
    logger.info(f">>> Verifying arguments")
    check_arguments(args)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Main program')
    parser.add_argument(
        'model_name_or_path', 
        help='Model identifier on HF hub or path on disk.', 
        required=True
        )
    parser.add_argument(
        'few_shots', 
        help='Number of examples to show the model.', 
        default=5
        )
    parser.add_argument(
        'dtype', 
        help='Precision level.', 
        default='bf16'
        )
    parser.add_argument(
        'task', 
        help='Task to evaluate on.', 
        default='mmlu'
        )

    args = parser.parse_args()
    main(args)
