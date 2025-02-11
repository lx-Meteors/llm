import sys
from dataclasses import dataclass
from typing import Dict

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from trl import DPOTrainer
from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

sys.path.append('/mnt/zhaorunsong/lx/My-LLM')


@dataclass
class DpoConfig:
    max_seq_len = 1024+8
    sft_model_file = "/mnt/zhaorunsong/lx/My-LLM/model_save/sft/checkpoint-1000"
    tokenizer_dir = "/mnt/zhaorunsong/lx/My-LLM/model_save/sft/checkpoint-1000"
    dpo_train_file = "/mnt/zhaorunsong/lx/My-LLM/data/dpo_train_dataset.json"
    dpo_eval_file = "/mnt/zhaorunsong/lx/My-LLM/data/dpo_eval_dataset.json"
    adapter_file = ""

    per_device_train_batch_size= 4
    num_train_epochs = 4
    gradient_accumulation_steps = 8
    learning_rate = 1e-5
    logging_first_step = True
    logging_steps = 20
    save_steps = 200
    model_save_dir = '/mnt/zhaorunsong/lx/My-LLM/model_save/dpo/'
    warmup_steps = 1000
    fp16 = True
    seed = 23333
    beta = 0.1


def get_dataset(split, file, cache_dir) -> Dataset:
    dataset = load_dataset('json', data_files=file, split=split, cache_dir=cache_dir)
    def split_prompt_responses(sample) -> Dict[str, str]:
        return {
            "prompt": f"{sample['prompt']}<|im_end|>",
            "chosen": f"{sample['chosen']}<im_end>",
            "rejected": f"{sample['rejected']}<im_end>"
        }
    return dataset.map(function=split_prompt_responses).shuffle(12345)

def train_dpo(config, peft_config):

    # step 1. 加载tokenizer
    tokenizer = QWenTokenizer.from_pretrained(config.tokenizer_dir)
    tokenizer.pad_token_id = tokenizer.im_end_id
    tokenizer.bos_token_id = tokenizer.im_end_id
    tokenizer.eos_token_id = tokenizer.im_end_id

    # step 2. 加载SFT模型
    model_train = QWenLMHeadModel.from_pretrained(config.sft_model_file)
    model_ref = QWenLMHeadModel.from_pretrained(config.sft_model_file)

    train_dataset = get_dataset("train", file=config.dpo_train_file)
    eval_dataset = get_dataset("train", file=config.dpo_eval_file)

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_first_step=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        output_dir=config.model_save_dir,
        optim="adafactor",
        log_level='info',
        warmup_steps=config.warmup_steps,
        bf16=False,
        fp16=config.fp16,
        seed=config.seed,
    )

    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,
        args=training_args,
        beta=config.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        max_target_length=config.max_seq_len,
        max_prompt_length=config.max_seq_len,
        generate_during_eval=True,
        is_encoder_decoder=True,
        # data_collator=data_collator
    )

    dpo_trainer.train()
    dpo_trainer.save_model(config.model_save_dir)


if __name__ == '__main__':

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
    )

    dpo_config = DpoConfig()

    train_dpo(dpo_config, peft_config)

















