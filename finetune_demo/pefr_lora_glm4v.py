import argparse
import gc
import json
import os
import random
import threading

import yaml
from PIL import Image
import psutil
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

from peft import get_peft_model, LoraConfig, TaskType
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Literal, Dict, Any

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils.logging import get_logger
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_conversation_input_ids(
        tokenizer: "PreTrainedTokenizer",
        *,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        images: Optional[List["PIL.Image"]] = None,
        template_version: Optional[Literal["base", "chat", "vqa"]] = None,
        answer: str = None,
        config=None
):
    image_size: int = config.vision_config['image_size']
    patch_size: int = config.vision_config['patch_size']
    # template_version = template_version or self.config.template_version
    assert images is None or len(images) <= 1, f"not support multi images by now."
    history = history or []
    text = query
    input_ids = tokenizer.get_prefix_tokens() + tokenizer.convert_tokens_to_ids(
                        ["<|begin_of_image|>", "<|endoftext|>", "<|end_of_image|>"])
    # token_type_ids = [LANGUAGE_TOKEN_TYPE]
    if images is not None and len(images) == 1:
        # vision
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        images = [transform(images[0])]
        # language
        # vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2
        # tokenizer.pad_token_id = 128002 # llama3 adapt for cogvlm
        # input_ids += [tokenizer.pad_token_id] * vision_token_num
        # token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
    text_ids = tokenizer.encode(text, add_special_tokens=False) + tokenizer.convert_tokens_to_ids(
                        ["<|assistant|>"]) # 151337代表的是Assistant这个符号
    if answer is not None:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        answer_ids += [tokenizer.eos_token_id]
        text_ids += answer_ids
    input_ids += text_ids
    # token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
    attention_mask = [1] * len(input_ids)
    if answer is not None:
        labels = [-100 for _ in range(len(input_ids) - len(answer_ids))] + answer_ids
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'images': images[0],
        'labels': labels,
    }

class ConversationDataset(Dataset):
    def __init__(self,
                img_path,
                json_path,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024
                 ):
        self.image_dir = img_path
        self.label_dir = json_path
        self.tokenizer = tokenizer
        self.model = model

        print("Loading json...")
        with open(json_path, 'r') as f:
            self.json_file = json.load(f)
        print("Finish Loading json...")

        # self.filenames = os.listdir(self.image_dir)
        self.input_length = input_length
        self.output_length = output_length
        self.device = device
        self.torch_type = torch_type
        self.padding_len = 2303
        self.max_length = self.input_length + self.output_length

    def __len__(self):
        return len(self.json_file)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")

        return batched_data

    def __getitem__(self, idx):

        label_data = self.json_file[idx]
        image = Image.open(label_data['image']).convert('RGB')

        num_rounds = len(label_data["conversations"]) // 2
        sampled_round_id = random.randint(0, num_rounds - 1)
        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None
        query = label_data["conversations"][sampled_round_id * 2]["content"]
        response = label_data["conversations"][sampled_round_id * 2 + 1]["content"]

        input_data = build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=history,
            images=[image],
            answer=response,
            config=self.model.config
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            current_length = len(unpadded_tensor)
            if current_length >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full([pad_to_length - current_length],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device)), dim=0)
        
        # print(input_data['input_ids'].shape)
        # print(self.max_length)

        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            self.max_length,
            pad_value=151329,
        )

        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            self.max_length,
            pad_value=0
        )
        # input_data['token_type_ids'] = pad_to_len(
        #     input_data['token_type_ids'],
        #     self.max_length,
        #     pad_value=0
        # )

        input_data['labels'] = pad_to_len(
            input_data['labels'],
            self.max_length,
            pad_value=-100
        )

        for data_key in input_data:
            if data_key in ['images']:
                input_data[data_key] = input_data[data_key].to(self.device).to(self.torch_type)
                                        
            else:
                input_data[data_key] = input_data[data_key].to(self.device)

        return input_data


def b2mb(x):
    return int(x / 2 ** 20)


class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


def main():
    parser = argparse.ArgumentParser(description="Finetune a CogVLM model with LoRA")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--torch_type", type=str, default="torch.bfloat16", help="Torch type")
    parser.add_argument("--save_step", type=int, default=10000000, help="Steps between checkpoints")
    parser.add_argument("--train_dataset_rate", type=float, default=1.0,
                        help="Proportion of dataset to use for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank parameter for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_target", type=str, default=["attention.query_key_value", "self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"], # attention.query_key_value
                        help="Finetune Target for LoRA")  # you can change the target to other modules such as "language_expert_query_key_value"
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA")
    parser.add_argument("--warmup_steps", type=int, default=600,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=128, help="Maximum output length")
    parser.add_argument("--model_path", type=str,
                        default="/home/wentao/project/glm-4v-9b",
                        help="Path to the pretrained model")
    parser.add_argument("--img_path", type=str,
                        default="/nas_sh/wentao/data/gui_public_ds/processed/rico_based_ds/image/",
                        help="Path to the conversation dataset")
    parser.add_argument("--dataset_path", type=str,
                        default="",
                        help="Path to the conversation dataset")
    parser.add_argument("--save_path", type=str, default="",
                        help="Path to save the finetuned model, must be a exit directory")
    parser.add_argument("--ds_config", type=str, default="ds_config.yaml",
                        help="DeepSpeed configuration file path")
    parser.add_argument("--grad_accum", type=int, default=32)

    args = parser.parse_args()
    args.torch_type = eval(args.torch_type)

    with open(args.ds_config) as f:
        ds_config = yaml.safe_load(f)
    hf_ds_config = HfDeepSpeedConfig(ds_config)

    ds_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=args.torch_type, trust_remote_code=True)

    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
    dataset = ConversationDataset(
        img_path=args.img_path,
        json_path=args.dataset_path,
        tokenizer=tokenizer,
        model=model,
        torch_type=args.torch_type,
        input_length=args.max_input_len,
        output_length=args.max_output_len
    )
    train_size = int(args.train_dataset_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,

    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        target_modules=args.lora_target,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    
    logger.info("Start getting peft model...")

    model = get_peft_model(model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )
    logger.info("Preparation done. Starting training...")
    writer = SummaryWriter(log_dir=args.save_path)
    for epoch in range(args.num_epochs):
        model.train()
        # model.gradient_checkpointing_enable()
        total_loss = 0.0
        accum_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print(batch['images'].shape)
            outputs = model(
                input_ids=batch['input_ids'],
                # token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['images'],
                labels=batch['labels']
            )
            loss = outputs.loss
            total_loss += loss.detach().float()
            # print(total_loss)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            accum_loss+=accelerator.gather(loss).mean().item()
            if step % args.grad_accum == 0:
                # if loss.device.index == 2:
                
                data = {
                        "epoch": epoch,
                        "step": step // args.grad_accum,
                        "loss": accum_loss / args.grad_accum,
                        "lr": lr_scheduler.get_last_lr()
                    }

                # 将字典转换为JSON格式的字符串
                json_str = json.dumps(data)

                # 打开或创建jsonl文件，并追加JSON字符串，每个JSON对象占一行
                with open(args.save_path + '/training_log.jsonl', 'a') as f:
                    f.write(json_str + '\n')
                print(f"Epoch {epoch}, Step {step // args.grad_accum}, Loss {accum_loss / args.grad_accum}, lr {lr_scheduler.get_last_lr()}")
                accum_loss = 0.0
                
            if (step + 1) % args.save_step == 0:
                print(f"Epoch {epoch}, Step {step + 1}, Loss {loss.item()}")
                checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}_step_{step + 1}')
                model.save_pretrained(
                    save_directory=checkpoint_path,
                    safe_serialization=True
                )
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_dataloader) + step)

        total_loss = accelerator.gather(total_loss)
        avg_loss = total_loss.mean().item() / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(avg_loss))
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Perplexity', train_ppl, epoch)
        accelerator.print(f"Epoch {epoch}: Average Loss {avg_loss:.4f}, Perplexity {train_ppl:.4f}")


        checkpoint_path = os.path.join(args.save_path, 'final_model')
        model.save_pretrained(
            save_directory=checkpoint_path,
            safe_serialization=True
        )

if __name__ == "__main__":
    main()
