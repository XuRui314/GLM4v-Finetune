# GLM4v-Finetune
Support finetuning GLM4v with zero2

## 简介
使用CogVLM2代码框架，支持GLM4v微调，具体的改动有如下：
1. GLM4v模型文件forward修改、数据预处理修改等
2. 自定义数据集格式更简单

## 运行方法
进入代码文件夹下，运行
`deepspeed --include localhost:2,3,6,7 peft_lora_glm4v.py --ds_config ds_config.yaml`
