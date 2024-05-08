from typing import Optional
import wandb
from dataclasses import dataclass,field
from datasets import load_dataset
import numpy as np
import evaluate
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import (
    HfArgumentParser, 
    Trainer, 
    TrainingArguments,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    GenerationConfig,
    pipeline
)


@dataclass
class ModelArguments:
    model_name_or_path : str = field(
        default = "psyche/KoT5",
        metadata = {"help" : ""}
    )
    model_auth_token : Optional[str] = field(
        default = None,
        metadata = {"help" : "Put the token for model-argument"}
    )
    model_type: str = field(
        default="seq2seq",
        metadata={"help":""}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for evaluation."}
    )


@dataclass
class DataArguments:
    data_name_or_path : str = field(
        default = "YoonDDo/aihub_summ_report",
        metadata = {"help" : ""}   
    )

    data_auth_token : Optional[str] = field(
        default = None,
        metadata = {"help" : "Put the token for data-argument"}
    )

    data_max_length : int = field(
        default = 512,
        metadata = {"help" : ""}
    )

    data_max_output_length : int = 128

    eval_split: str = field(
        default="validation",
        metadata={"help": "The evaluation split."}
    )

    text_column: str = field(
        default="passage",
        metadata={"help": "The name of the column containing the main text."}
    )

    label_column: Optional[str] = field(
        default="summaries",
        metadata={"help": "The name of the column containing the labels."}
    )



@dataclass
class EvalArguments:
    max_length:int = field(
        default=512,
        metadata={"help": "The maximum length of the output text."}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."}
    )

    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty."}
    )







def main(
        model_args: ModelArguments,
        data_args: DataArguments,
        eval_args: EvalArguments,
        train_args: TrainingArguments
):
    
    wandb.init(project="Summarization")
    wandb.run.name = model_args.model_name_or_path+"Evaluation"
    wandb.run.save()

    dataset = load_dataset(
        data_args.data_name_or_path,
        token = data_args.data_auth_token
        )

    if data_args.data_auth_token is not None:
        from huggingface_hub import login
        login(token=data_args.data_auth_token)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token = model_args.model_auth_token,
        max_length= data_args.data_max_length   
        ,padding= "max_length"
        ,truncation=True

    )


    def tokenize_func(data):
        tokenized_input = tokenizer(
            data['passage'],
            max_length=data_args.data_max_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            data['summaries'][0]
            ,max_length=data_args.data_max_output_length
            ,padding = "max_length"
            ,truncation=True
        )
        tokenized_input['labels'] = labels['input_ids']
        return tokenized_input


    def compute_metric(value):
        predictions, labels = value
        decoded_preds = tokenizer.batch_decode(predictions,skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,skip_special_tokens=True)
        result = _metric.compute(predictions=decoded_preds, references=decoded_labels)
        print(result)
        
    dataset = dataset.map(tokenize_func,remove_columns=dataset['train'].column_names)


    pipe = pipeline("summarization", model_args.model_name_or_path,tokenizer=tokenizer ,device=0 if torch.cuda.is_available() else -1)

    dataset = load_dataset(
        data_args.data_name_or_path,
        token=data_args.data_auth_token,
        split=data_args.eval_split
    )

    _metric = evaluate.load("rouge")
    outputs = defaultdict(list)
    for batch in tqdm(dataset.iter(batch_size=model_args.batch_size)):
        
        predictions = pipe(
            batch[data_args.text_column],
            batch_size=model_args.batch_size, 
            max_length=eval_args.max_length,
            temperature=eval_args.temperature,
            repetition_penalty=eval_args.repetition_penalty
        )

        predictions = [pred["summary_text"] for pred in predictions]
        outputs["predictions"].extend(predictions)
        outputs["inputs"].extend(batch[data_args.text_column])
        outputs["references"].extend(batch[data_args.label_column])

    results = _metric.compute(predictions=outputs["predictions"], references=outputs["references"])
    print(results)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments,DataArguments,EvalArguments,TrainingArguments))
    main(*parser.parse_args_into_dataclasses())