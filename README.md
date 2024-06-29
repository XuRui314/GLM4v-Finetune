# GLM4v-Finetune
Support finetuning GLM4v with zero2

## 简介
使用CogVLM2代码框架，支持GLM4v微调，具体的改动有如下：
1. GLM4v模型文件forward修改、数据预处理修改等
2. 自定义数据集格式更简单

您需要从huggingface中下载的glm4v模型，把`THUDM/glm-4v-9b`中的`modeling_chatglm.py`替换为本仓库中的`modeling_chatglm.py`。

## 自定义数据集
组织格式的代码可以参考：
``` python
import os
import json
import random

folder_path = ''
# file_names = os.listdir(folder_path)
file_names = ['refexp_train.jsonl', 'widget_caption_maxlen_train.jsonl',  'taperception_train.jsonl']
step_i = 0
train_step = []
for file_name in file_names:    
    json_data = []
    with open(folder_path + file_name, 'r') as file:
        for line in file:
            json_data.append(json.loads(line))

    for sample in json_data:
        sample_json = {}
        conversations = []
        img_path = '' + '/' + sample['img_name']
        conv_user = {"role": "user", "content": ""}
        conv_user["content"] += sample['prompt']
        conv_ai = {"role": "assistant", "content": sample['text']}
        conversations.append(conv_user)
        conversations.append(conv_ai)
        sample_json['conversations'] = conversations
        sample_json['image'] = img_path
        sample_json['id'] = step_i

        train_step.append(sample_json)
        step_i += 1
random.shuffle(train_step)
print("Num of total step: " + str(len(train_step)))
json.dump(train_step, open('public_glm4_grouding_train.json', "w"), ensure_ascii=False)
```

## 运行方法
进入代码文件夹下，运行
`deepspeed --include localhost:2,3,6,7 peft_lora_glm4v.py --ds_config ds_config.yaml`
