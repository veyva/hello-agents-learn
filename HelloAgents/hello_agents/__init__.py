"""
HelloAgents - 灵活、可扩展的多智能体框架

基于OpenAI原生API构建，提供简洁高效的智能体开发体验。
"""

from .version import __version__
# 核心组件
from .core.llm import HelloAgentsLLM
from .core.config import Config
from .core.message import Message
from .core.exceptions import HelloAgentsException

# Agent实现
from .agents.simple_agent import SimpleAgent

__all__ = [
    # 核心组件
    "HelloAgentsLLM",
    "Config",
    "Message",
    "HelloAgentsException",

    # Agent范式
    "SimpleAgent"
]