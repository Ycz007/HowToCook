"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any

"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any
from pathlib import Path

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "./data/cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = str(
    Path(__file__).resolve().parent
    / "huggingface"
    / "hub"
    / "models--BAAI--bge-small-zh-v1.5"
    / "snapshots"
    / "7999e1d3359715c523056ef9478215996d62a620"
)
    llm_model: str = "deepseek-chat"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
