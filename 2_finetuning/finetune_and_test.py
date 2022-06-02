#!/usr/bin/env python

import logging
import os
import random
import sys
import numpy as np
import pandas as pd
import transformers
import torch

from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path


# initialise logger
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
            "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },   
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv file containing the test data."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "PR custom: path to cache directory for storing datasets"}
    )
    test_results_dir: Optional[str] = field(
        default=None, metadata={"help": "PR custom: path to directory for storing test results"}
    )
    test_results_name: Optional[str] = field(default='results', metadata = {"help": "PR custom: specify name for test results .csv"}
    )
    store_prediction_logits: Optional[str] = field(default=False, metadata = {"help": "PR custom: decide whether to store logits along with discrete predictions during test"}
    )
    use_class_weights: Optional[str] = field(default=False, metadata = {"help": "PR custom: decide whether to use class weighting when calculating loss"}
    )


@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # LOADING DATASETS from specified file paths
    # Expecting "label" and "text" columns

    data_files = dict()

    # TRAIN
    if training_args.do_train:
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            logger.info(f"Loaded TRAINING set {data_args.train_file}")
        else:
            raise ValueError("Need a training file for `do_train`.")
    
    # DEV 
    if training_args.do_eval:
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            logger.info(f"Loaded DEV set {data_args.validation_file}")
        else:
            raise ValueError("Need a validation file for `do_eval`.")

    # TEST
    if training_args.do_predict:
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            logger.info(f"Loaded TEST set {data_args.test_file}")
        else:
            raise ValueError("Need a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"loaded {key}: {data_files[key]}")

    # load datafiles to dataset, expecting csv
    datasets = load_dataset("csv", data_files=data_files, cache_dir=data_args.dataset_cache_dir, lineterminator="\n")

    # count number of labels --> select dataset by index so that this works for training and testing
    # PR CUSTOM: need to set fixed label list because some splits are so small that they only have one of two labels in them
    label_list = [0, 1] #datasets[[key for key in datasets][0]].unique("label")
    label_list.sort()  # Sorting for determinism
    num_labels = len(label_list)

    # load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # set padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False # padding dynamically at batch creation, to the max sequence length in each batch

    # convert label to id.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # tokenize the texts
        args = ((examples["text"],))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # map labels to IDs
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    # log a few random samples from the training, dev or test set:
    for index in random.sample(range(len(datasets[[key for key in datasets][0]])), 3):
        logger.info(f"Sample {index} of the {[key for key in datasets][0]} set: {datasets[[key for key in datasets][0]][index]}.")

    # define custom compute_metrics function
    # takes an `EvalPrediction` object (a namedtuple with a predictions and label_ids field)
    # has to return a dictionary string to float
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        metrics_dict = dict()
        metrics_dict["macro_f1"] = f1_score(p.label_ids, preds, average="macro")
        metrics_dict["hate_precision"], metrics_dict["hate_recall"], metrics_dict["hate_f1"], _ = precision_recall_fscore_support(p.label_ids, preds, average="binary")
        return metrics_dict

    # data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = None

    # initialize the trainer
    if data_args.use_class_weights:
        
        logger.info("** USING WEIGHTED LOSS**")

        if len(pd.unique(datasets["train"]["label"]))<2: # if a small sample only has examples of one class, we cannot do weighting
            class_weights = compute_class_weight('balanced', classes = label_list, y = label_list)
        else:
            class_weights = compute_class_weight('balanced', classes = label_list, y = datasets["train"]["label"])
        
        logger.info(f"class weights: {class_weights}")

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                # compute custom loss 
                weighted_loss_fct = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
                loss = weighted_loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"] if training_args.do_train else None,
            eval_dataset=datasets["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    else:
        
        logger.info("** USING STANDARD UNWEIGHTED LOSS**")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"] if training_args.do_train else None,
            eval_dataset=datasets["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    # TRAINING
    if training_args.do_train:
        logger.info("*** TRAIN ***")
        train_result = trainer.train()
        metrics = train_result.metrics

        trainer.save_model()  # saves model + tokenizer

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info(metrics)

    # EVALUATION ON DEV SET
    if training_args.do_eval:
        logger.info("*** EVAL ***")

        eval_result = trainer.evaluate(eval_dataset=datasets["validation"])

        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)

        logger.info(eval_result)

    # TESTING ON TEST SET
    if training_args.do_predict:
        logger.info("*** TEST ***")

        # specify file path for storing test results
        test_results_path = os.path.join(data_args.test_results_dir, f"{data_args.test_results_name}")

        # get predictions on test set (includes gold label column)
        test_results=trainer.predict(test_dataset=datasets["test"]) 

        # store gold labels and discrete model predictions
        labels = pd.DataFrame(test_results.label_ids, columns = ['label']).reset_index()
        predictions = pd.DataFrame(np.argmax(test_results.predictions, axis=1), columns = ['prediction']).reset_index()
        
        export_df = labels.merge(predictions)

        # if specified, also store prediction logits
        if data_args.store_prediction_logits:
            logits = pd.DataFrame(pd.Series(tuple(map(tuple, test_results.predictions))), columns = ['logits']).reset_index()
            export_df = export_df.merge(logits)

        # save to csv in specified path
        Path(data_args.test_results_dir).mkdir(parents=True, exist_ok=True)
        export_df.to_csv(test_results_path, index = False)
    
    return 'completed finetuning'

if __name__ == "__main__":
    main()
