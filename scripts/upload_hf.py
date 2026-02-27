"""
模型上传脚本
用于上传训练好的模型到Hugging Face
"""
import os
import argparse
from huggingface_hub import HfApi, login


def upload_model(model_path: str, repo_id: str, token: str = None) -> str:
    """
    上传模型到Hugging Face
    
    Args:
        model_path: 模型路径
        repo_id: Hugging Face仓库ID
        token: HF token (可选，如果已登录可省略)
    
    Returns:
        仓库URL
    """
    if token:
        login(token=token)
    
    api = HfApi()
    
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
    )
    
    url = f"https://huggingface.co/{repo_id}"
    print(f"模型已上传: {url}")
    
    return url


def main():
    parser = argparse.ArgumentParser(description="上传模型到Hugging Face")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--repo_id", type=str, required=True, help="HF仓库ID")
    parser.add_argument("--token", type=str, default=None, help="HF token")
    
    args = parser.parse_args()
    
    upload_model(args.model_path, args.repo_id, args.token)


if __name__ == "__main__":
    main()
