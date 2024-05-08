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
    wandb.run.name = "test"
    wandb.run.save()

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
    
    input_text = "지난해 7월 고 채아무개 상병과 함께 집중호우 실종자 수색 중 급류에 휩쓸렸다 생존한 병사 2명이 윤석열 대통령에게 특검법 수용을 촉구하는 공개 편지를 썼다. 이들은 “할 수 있는 게 아무것도 없다는 미안함을 반복하고 싶지 않다”며 “거부권을 행사하지 말아달라”고 대통령에 부탁했다."
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # 모델을 사용하여 요약 생성
    summary_ids = model.generate(input_ids, num_beams=4, max_length=128, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("기사 : ",input_text)
    print("요약문 : ",summary)




if __name__ == "__main__":
    parser = HfArgumentParser((ModelArgs, DataArgs, Seq2SeqTrainingArguments))
    main(*parser.parse_args_into_dataclasses())



