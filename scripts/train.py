"""
Agent模型训练脚本
用于训练AI Agent的推理和工具选择能力
"""
import os
import json
import random
import argparse
from datetime import datetime
from typing import Any, Dict, List

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer


def set_seed(seed: int = 42) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_training_data(num_samples: int = 1000, seed: int = 42) -> List[Dict]:
    """生成Agent训练数据集"""
    random.seed(seed)
    data = []
    
    tools = {
        "weather_query": {
            "cities": ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "苏州"],
            "dates": ["今天", "明天", "后天", "本周", "下周"]
        },
        "calculator": {
            "expressions": [
                "{a} + {b}", "{a} - {b}", "{a} * {b}", "{a} / {b}",
                "({a} + {b}) * {c}", "{a} ** 2 + {b}",
            ]
        },
        "datetime": {
            "formats": ["iso", "date", "time", "timestamp"]
        },
        "search": {
            "queries": [
                "Python教程", "机器学习入门", "深度学习框架", "自然语言处理",
                "数据可视化", "API开发", "数据库优化", "前端框架", "云服务部署", "代码测试"
            ]
        },
        "text_process": {
            "operations": ["lowercase", "uppercase", "reverse", "word_count", "char_count"]
        }
    }
    
    for i in range(num_samples):
        tool_type = list(tools.keys())[i % len(tools)]
        
        if tool_type == "weather_query":
            city = random.choice(tools["weather_query"]["cities"])
            date = random.choice(tools["weather_query"]["dates"])
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"用户想查询{city}{date}的天气",
                "output": json.dumps({
                    "thought": f"用户需要{city}{date}的天气信息，调用天气查询工具",
                    "action": "weather_query",
                    "action_input": {"city": city, "date": date}
                }, ensure_ascii=False)
            })
        elif tool_type == "calculator":
            a, b, c = random.randint(1, 100), random.randint(1, 100), random.randint(1, 10)
            expr = random.choice(tools["calculator"]["expressions"]).format(a=a, b=b, c=c)
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"计算 {expr}",
                "output": json.dumps({
                    "thought": "用户需要进行数学计算，调用计算器工具",
                    "action": "calculator",
                    "action_input": {"expression": expr}
                }, ensure_ascii=False)
            })
        elif tool_type == "datetime":
            fmt = random.choice(tools["datetime"]["formats"])
            questions = ["现在几点了？", "今天日期是什么？", "给我当前时间", "显示当前时间戳"]
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": random.choice(questions),
                "output": json.dumps({
                    "thought": "用户想知道当前时间信息，调用时间工具",
                    "action": "datetime",
                    "action_input": {"format": fmt}
                }, ensure_ascii=False)
            })
        elif tool_type == "search":
            query = random.choice(tools["search"]["queries"])
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"帮我搜索{query}相关内容",
                "output": json.dumps({
                    "thought": f"用户需要搜索{query}相关信息，调用搜索工具",
                    "action": "search",
                    "action_input": {"query": query}
                }, ensure_ascii=False)
            })
        elif tool_type == "text_process":
            op = random.choice(tools["text_process"]["operations"])
            texts = ["Hello World", "Python Programming", "AI Agent Core", "Machine Learning"]
            text = random.choice(texts)
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"对文本 '{text}' 进行{op}处理",
                "output": json.dumps({
                    "thought": f"用户需要对文本进行{op}处理，调用文本处理工具",
                    "action": "text_process",
                    "action_input": {"text": text, "operation": op}
                }, ensure_ascii=False)
            })
    
    for i in range(num_samples // 5):
        data.append({
            "instruction": "根据工具执行结果生成最终回答",
            "input": f"工具返回: 操作成功完成，结果为 {random.randint(1, 1000)}",
            "output": json.dumps({
                "thought": "已获取工具执行结果，可以给出最终答案",
                "action": "final_answer",
                "action_input": {"answer": f"操作已成功完成。根据查询结果，答案是 {random.randint(1, 1000)}。"}
            }, ensure_ascii=False)
        })
    
    random.shuffle(data)
    return data


def format_example(example: Dict) -> str:
    """将数据格式化为模型输入格式"""
    return f"""<|im_start|>system
你是一个智能AI助手，能够分析用户请求并选择合适的工具执行任务。
你的输出必须是JSON格式：{{"thought": "思考过程", "action": "工具名或final_answer", "action_input": {{参数}}}}

可用工具:
- weather_query: 查询天气 (参数: city, date)
- calculator: 数学计算 (参数: expression)
- datetime: 获取时间 (参数: format)
- search: 搜索信息 (参数: query)
- text_process: 文本处理 (参数: text, operation)
<|im_end|>
<|im_start|>user
{example['instruction']}

{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""


def train(config: Dict) -> Dict:
    """执行训练"""
    set_seed(config.get("seed", 42))
    
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("生成训练数据...")
    print("=" * 60)
    
    all_data = generate_training_data(
        num_samples=config.get("num_samples", 2000),
        seed=config.get("seed", 42)
    )
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    
    print(f"训练数据: {len(train_data)} 条")
    print(f"验证数据: {len(eval_data)} 条")
    
    train_dataset = Dataset.from_list([{"text": format_example(d)} for d in train_data])
    eval_dataset = Dataset.from_list([{"text": format_example(d)} for d in eval_data])
    
    print("\n" + "=" * 60)
    print("加载模型...")
    print("=" * 60)
    
    model_name = config["model_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer加载完成，词表大小: {len(tokenizer)}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    print(f"模型加载完成: {model.num_parameters() / 1e9:.2f}B 参数")
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\n" + "=" * 60)
    print("配置训练参数...")
    print("=" * 60)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=4,
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        run_name=f"agent-core-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.get("max_length", 2048),
        packing=False,
    )
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    train_result = trainer.train()
    
    print("\n" + "=" * 60)
    print("保存模型...")
    print("=" * 60)
    
    lora_output_dir = os.path.join(output_dir, "lora_weights")
    trainer.model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"LoRA权重已保存: {lora_output_dir}")
    
    summary = {
        "model_name": model_name,
        "training_epochs": config.get("num_epochs", 3),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "learning_rate": config.get("learning_rate", 2e-4),
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"训练损失: {train_result.training_loss:.4f}")
    print(f"训练步数: {train_result.global_step}")
    
    return summary


def merge_model(config: Dict) -> str:
    """合并LoRA权重"""
    from peft import PeftModel
    
    print("=" * 60)
    print("合并LoRA权重...")
    print("=" * 60)
    
    output_dir = config["output_dir"]
    lora_output_dir = os.path.join(output_dir, "lora_weights")
    merged_output_dir = os.path.join(output_dir, "merged_model")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    merged_model = PeftModel.from_pretrained(base_model, lora_output_dir)
    merged_model = merged_model.merge_and_unload()
    
    merged_model.save_pretrained(merged_output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(lora_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    print(f"合并模型已保存: {merged_output_dir}")
    
    return merged_output_dir


def test_model(config: Dict, test_cases: List[str] = None) -> List[Dict]:
    """测试模型"""
    print("=" * 60)
    print("测试模型...")
    print("=" * 60)
    
    output_dir = config["output_dir"]
    merged_output_dir = os.path.join(output_dir, "merged_model")
    
    tokenizer = AutoTokenizer.from_pretrained(merged_output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_output_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if test_cases is None:
        test_cases = [
            "用户想查询上海今天的天气",
            "计算 123 * 456",
            "现在几点了？",
            "帮我搜索Python教程",
        ]
    
    results = []
    for test_input in test_cases:
        prompt = f"""<|im_start|>system
你是一个智能AI助手，能够分析用户请求并选择合适的工具执行任务。
你的输出必须是JSON格式：{{"thought": "思考过程", "action": "工具名或final_answer", "action_input": {{参数}}}}
<|im_end|>
<|im_start|>user
分析用户请求并决定下一步行动

{test_input}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        results.append({"input": test_input, "output": response})
        print(f"\n输入: {test_input}")
        print(f"输出: {response}")
        print("-" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Agent模型训练脚本")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "merge", "test", "all"],
                       help="运行模式: train, merge, test, all")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="基础模型名称")
    parser.add_argument("--output_dir", type=str, default="/content/outputs",
                       help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_samples", type=int, default=2000, help="训练样本数")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    config = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_samples": args.num_samples,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    
    if args.mode == "train":
        train(config)
    elif args.mode == "merge":
        merge_model(config)
    elif args.mode == "test":
        test_model(config)
    elif args.mode == "all":
        train(config)
        merge_model(config)
        test_model(config)


if __name__ == "__main__":
    main()
