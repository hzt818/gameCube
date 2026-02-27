"""
数据生成脚本
用于生成Agent训练数据
"""
import os
import json
import random
import argparse
from typing import Dict, List


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
                "output": {
                    "thought": f"用户需要{city}{date}的天气信息，调用天气查询工具",
                    "action": "weather_query",
                    "action_input": {"city": city, "date": date}
                }
            })
        elif tool_type == "calculator":
            a, b, c = random.randint(1, 100), random.randint(1, 100), random.randint(1, 10)
            expr = random.choice(tools["calculator"]["expressions"]).format(a=a, b=b, c=c)
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"计算 {expr}",
                "output": {
                    "thought": "用户需要进行数学计算，调用计算器工具",
                    "action": "calculator",
                    "action_input": {"expression": expr}
                }
            })
        elif tool_type == "datetime":
            fmt = random.choice(tools["datetime"]["formats"])
            questions = ["现在几点了？", "今天日期是什么？", "给我当前时间", "显示当前时间戳"]
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": random.choice(questions),
                "output": {
                    "thought": "用户想知道当前时间信息，调用时间工具",
                    "action": "datetime",
                    "action_input": {"format": fmt}
                }
            })
        elif tool_type == "search":
            query = random.choice(tools["search"]["queries"])
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"帮我搜索{query}相关内容",
                "output": {
                    "thought": f"用户需要搜索{query}相关信息，调用搜索工具",
                    "action": "search",
                    "action_input": {"query": query}
                }
            })
        elif tool_type == "text_process":
            op = random.choice(tools["text_process"]["operations"])
            texts = ["Hello World", "Python Programming", "AI Agent Core", "Machine Learning"]
            text = random.choice(texts)
            data.append({
                "instruction": "分析用户请求并决定下一步行动",
                "input": f"对文本 '{text}' 进行{op}处理",
                "output": {
                    "thought": f"用户需要对文本进行{op}处理，调用文本处理工具",
                    "action": "text_process",
                    "action_input": {"text": text, "operation": op}
                }
            })
    
    for i in range(num_samples // 5):
        data.append({
            "instruction": "根据工具执行结果生成最终回答",
            "input": f"工具返回: 操作成功完成，结果为 {random.randint(1, 1000)}",
            "output": {
                "thought": "已获取工具执行结果，可以给出最终答案",
                "action": "final_answer",
                "action_input": {"answer": f"操作已成功完成。根据查询结果，答案是 {random.randint(1, 1000)}。"}
            }
        })
    
    random.shuffle(data)
    return data


def main():
    parser = argparse.ArgumentParser(description="生成Agent训练数据")
    parser.add_argument("--output", type=str, default="data/train_data.json", help="输出文件路径")
    parser.add_argument("--num_samples", type=int, default=2000, help="样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--split", type=bool, default=True, help="是否划分训练/验证集")
    
    args = parser.parse_args()
    
    print(f"生成 {args.num_samples} 条训练数据...")
    data = generate_training_data(args.num_samples, args.seed)
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    if args.split:
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        train_path = args.output.replace(".json", "_train.json")
        eval_path = args.output.replace(".json", "_eval.json")
        
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        print(f"训练数据已保存: {train_path} ({len(train_data)} 条)")
        print(f"验证数据已保存: {eval_path} ({len(eval_data)} 条)")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存: {args.output} ({len(data)} 条)")


if __name__ == "__main__":
    main()
