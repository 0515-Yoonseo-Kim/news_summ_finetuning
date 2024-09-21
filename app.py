from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

model_name = "YoonDDo/ft_t5-base0"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    input_ids = tokenizer.encode(request.text, return_tensors="pt", max_length=1024, truncation= True)
    summary_ids = model.generate(input_ids, num_beams = 4, max_length=request.max_length, early_stopping = True)
    summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)

    return {"summary": summary}