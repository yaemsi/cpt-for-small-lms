
import argparse
from loguru import logger

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer





def prepare_dataset(args: argparse.Namespace) -> None:

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.tokenizer_max_length,
            return_special_tokens_mask=False,
        )

    # Normalize text field
    def normalize(example):
        if "text" not in example:
            example["text"] = example[list(example.keys())[0]]
        return example
    
    def subsample(ds, ratio):
        return ds.shuffle(seed=42).select(range(int(len(ds) * ratio)))
    
    # ---- Load datasets (small but high-impact) ----
    logger.info(f">>> Loading datasets")
    logger.info(f">>> wikitext")
    #wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:80%]")
    logger.info(f">>> open-web-math")

    #math = load_dataset("open-web-math/open-web-math", split="train[:1%]")
    math = load_dataset("open-web-math/open-web-math", split="train[:40%]")
    #logger.info(f">>> cosmopedia")

    #cosmopedia = load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train[:10%]")  # Subset for PoC
    #fineweb_edu = load_dataset("HuggingFaceFW/fineweb-edu", split="train[:5%]")
    #logger.info(f">>> c4")

    #c4 = load_dataset("ola13/small-c4-repetitions", "en", split="train[:0.2%]")
    #c4 = load_dataset("ola13/small-c4-repetitions", split="train")
    #c4 = load_dataset("allenai/c4", "en", split="train[:0.2%]")

    logger.info(f">>> Normalizing datasets")
    wiki = wiki.map(normalize, num_proc = 2)
    math = math.map(normalize, num_proc = 2)
    #cosmopedia = cosmopedia.map(normalize)
    #fineweb_edu = fineweb_edu.map(normalize)
    #c4 = c4.map(normalize)

    # Tokenize
    logger.info(f">>> Tokenizing datasets")
    wiki = wiki.map(tokenize, batched=True, remove_columns=wiki.column_names, num_proc = 2)
    math = math.map(tokenize, batched=True, remove_columns=math.column_names, num_proc = 2)
    #cosmopedia = cosmopedia.map(AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer"), batched=True, remove_columns=cosmopedia.column_names)
    #fineweb_edu = fineweb_edu.map(tokenize, batched=True, remove_columns=fineweb_edu.column_names)
    #c4 = c4.map(tokenize, batched=True, remove_columns=c4.column_names)

    logger.info(f">>> Subsampling datasets")

    # ---- Dataset mixing (40/20/10/10/10/10) ----
    wiki = subsample(wiki, 0.6)
    #math = subsample(math, 0.3)
    math = subsample(math, 0.4)
    #cosmopedia = subsample(cosmopedia, 0.2)
    #fineweb_edu = subsample(fineweb_edu, 0.2)
    #c4 = subsample(c4, 0.2)

    dataset = concatenate_datasets([wiki, math]).shuffle(seed=args.seed)
    logger.info(f">>> Saving the dataset ({len(dataset)} rows)")

    dataset.save_to_disk("./data/cpt_dataset")
    logger.info(f">>> Done")


   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluation program')
    parser.add_argument(
        '--model_name_or_path', 
        type = str,
        help='Model identifier on HF hub or path on disk.', 
        default='HuggingFaceTB/SmolLM2-135M-8K'
        )
    parser.add_argument(
        '--tokenizer_max_length', 
        type = int,
        help='Max sequence length.', 
        default=1024
        )
    parser.add_argument(
        '--seed', 
        type = int,
        help='Seed for shuffling.', 
        default=42
        )
    parser.add_argument(
        '--output_dir', 
        type = str,
        help='Where to save dataset.', 
        default='./output'
        )

    args = parser.parse_args()
    prepare_dataset(args)
