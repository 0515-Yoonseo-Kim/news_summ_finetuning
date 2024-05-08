from typing import Optional
import torch
import numpy as np
from transformers import(
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)
import evaluate
import wandb
from dataclasses import dataclass, field
from datasets import load_dataset


@dataclass
class ModelArgs:
    model_name_or_path : str = field(
        default = "psyche/KoT5"
    )
    model_auth_token : Optional[str] = field(
        default=None
    )
    model_type:str = field(
        default="seq2seq",
        metadata={"help":""}
        ) 

@dataclass
class DataArgs:
    data_name_or_path : str = field(
        default = "YoonDDo/aihub_summ_report"
    )
    data_auth_token : Optional[str] = field(
        default = None,
        metadata = {"help" : "데이터 토큰 받기"}
    )
    data_max_length : int = field(
        default = 512
    )
    data_max_output_length : int = field(
        default = 128
    )
    pass




def main(
        modelargs : ModelArgs,
        dataargs : DataArgs,
        trainargs : Seq2SeqTrainingArguments
):
    wandb.init(project="Summarization")
    wandb.run.name = "train"
    wandb.run.save()
    dataset = load_dataset(dataargs.data_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        modelargs.model_name_or_path,
        token = modelargs.model_auth_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        modelargs.model_name_or_path,
        token = modelargs.model_auth_token    
    )

    def tokenize_func(data):
        tokenized_input = tokenizer(
            data['passage'],
            max_length=dataargs.data_max_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            data['summaries'][0]
            ,max_length=dataargs.data_max_output_length
            ,padding = "max_length"
            ,truncation=True
        )
        tokenized_input['labels'] = labels['input_ids']
        return tokenized_input
    
    dataset = dataset.map(tokenize_func,remove_columns=dataset['train'].column_names)

    _metric = evaluate.load('rouge')
    def compute_metrics(value):
        predictions, labels = value
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = _metric.compute(predictions=decoded_preds,references=decoded_labels)


    trainer = Seq2SeqTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        compute_metrics = compute_metrics,
        args = trainargs
    )

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArgs,DataArgs,Seq2SeqTrainingArguments))
    main(*parser.parse_args_into_dataclasses())




