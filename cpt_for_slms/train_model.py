import argparse
import os
import torch

from datasets import load_from_disk
from loguru import logger
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def train_cpt(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f">>> Loading the model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(device)

    logger.info(f">>> Loading the tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    delattr(args, 'model_name') # No longer used

    # Ensure pad_token is set if not already present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    custom_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False 
    )

    logger.info(f">>> Loading the preprocessed dataset")
    ds = load_from_disk(f"{args.datasets_dir}")
    ds = ds.with_format("torch", device=device)


    delattr(args, 'datasets_dir') # No longer used

    logger.info(f">>> Launching the training")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**vars(args)),
        train_dataset=ds,
        #processing_class=tokenizer,
        data_collator=custom_data_collator
    )
    trainer.train()

    logger.info(f">>> Saving the final model")
    output_dir = os.path.join(args.output_dir, 'final_model')
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluation program')

    parser.add_argument(
        '--model_name', 
        type = str,
        help='Model identifier on HF hub or path on disk.', 
        default='HuggingFaceTB/SmolLM2-135M-8K'
        )
    parser.add_argument(
        '--per_device_train_batch_size',
        type = int,
        help='Training batch size.', 
        default=8
        )
    parser.add_argument(
        '--output_dir', 
        type = str,
        help='Where to save the model.', 
        default='./output/model'
        )
    parser.add_argument(
        '--datasets_dir', 
        type = str,
        help='Path to tokenized datasets.', 
        default='./data/cpt_dataset'
        )
    parser.add_argument(
        '--gradient_accumulation_steps', 
        type = int,
        default=1
        )
    parser.add_argument(
        '--learning_rate', 
        type = float,
        default=2e-5
        )
    parser.add_argument(
        '--weight_decay', 
        type = float,
        default=0.1
        )
    parser.add_argument(
        '--warmup_steps', 
        type = int,
        default=1000
        )
    parser.add_argument(
        '--max_steps', 
        type = int,
        default=8000
        )
    parser.add_argument(
        '--lr_scheduler_type', 
        type = str,
        default="cosine"
        )
    parser.add_argument(
        '--logging_steps', 
        type = int,
        default=50
        )
    parser.add_argument(
        '--save_steps', 
        type = int,
        default=1000
        )
    
    parser.add_argument(
        '--save_strategy',
        type = str,
        default='steps'
    )
    parser.add_argument(
        '--save_total_limit', 
        type = int,
        default=3
        )
    parser.add_argument(
        '--dataloader_pin_memory', 
        type = bool,
        default=True
        )
    parser.add_argument(
        '--bf16', 
        type = bool,
        default=True
        )
    parser.add_argument(
        '--deepspeed', 
        type = str,
        default=None
        )
    parser.add_argument(
        '--optim', 
        type = str,
        default='adamw_torch_fused'
        )
         
     
    """
    parser.add_argument(
        '--report_to', 
        type = str,
        default=None
        )
    """
    args = parser.parse_args()
    logger.info(f"******** Launching the training program ")
    train_cpt(args)
    logger.info(f"******** Done ")
