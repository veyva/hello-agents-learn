"""Agent实现模块 - HelloAgents原生Agent范式"""

from .simple_agent import SimpleAgent

# 保持向后兼容性
try:
    __all__ = [
        "SimpleAgent"
    ]
except ImportError:
    __all__ = [
        "SimpleAgent"
    ]