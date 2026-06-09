# -*- coding: utf-8 -*-
"""狼人杀游戏的结构化输出模型"""
from typing import Literal, Optional, List, Protocol
from pydantic import BaseModel, Field


class PlayerLike(Protocol):
    """只要求对象拥有 name 字段，避免绑定具体 Agent 框架。"""

    name: str


class DiscussionModelCN(BaseModel):
    """中文版讨论输出格式"""
    
    reach_agreement: bool = Field(
        description="是否已达成一致意见",
    )
    confidence_level: int = Field(
        description="对当前推理的信心程度(1-10)",
        ge=1, le=10
    )
    key_evidence: Optional[str] = Field(
        description="支持你观点的关键证据",
        default=None
    )


def get_vote_model_cn(agents: list[PlayerLike | str]) -> type[BaseModel]:
    """获取中文版投票模型"""
    candidate_names = [_.name if hasattr(_, "name") else str(_) for _ in agents]
    
    class VoteModelCN(BaseModel):
        """中文版投票输出格式"""
        
        vote: Literal[tuple(candidate_names)] = Field(
            description="你要投票淘汰的玩家姓名，或在仍有弃权机会时选择'弃权'",
        )
        reason: str = Field(
            description="投票理由，简要说明为什么选择此人",
        )
        suspicion_level: int = Field(
            description="对被投票者的怀疑程度(1-10)",
            ge=1, le=10
        )
    
    return VoteModelCN


class WitchActionModelCN(BaseModel):
    """中文版女巫行动模型"""
    
    use_antidote: bool = Field(
        description="是否使用解药救人",
        default=False
    )
    use_poison: bool = Field(
        description="是否使用毒药杀人", 
        default=False
    )
    target_name: Optional[str] = Field(
        description="目标玩家姓名（救人或毒杀的对象）",
        default=None
    )
    action_reason: Optional[str] = Field(
        description="行动理由",
        default=None
    )


def get_seer_model_cn(agents: list[PlayerLike]) -> type[BaseModel]:
    """获取中文版预言家模型"""
    
    class SeerModelCN(BaseModel):
        """中文版预言家查验格式"""
        
        target: Literal[tuple(_.name for _ in agents)] = Field(
            description="要查验的玩家姓名",
        )
        check_reason: str = Field(
            description="查验此人的原因",
        )
        priority_level: int = Field(
            description="查验优先级(1-10)",
            ge=1, le=10
        )
    
    return SeerModelCN


def get_hunter_model_cn(agents: list[PlayerLike]) -> type[BaseModel]:
    """获取中文版猎人模型"""
    
    class HunterModelCN(BaseModel):
        """中文版猎人开枪格式"""
        
        shoot: bool = Field(
            description="是否使用开枪技能",
        )
        target: Optional[Literal[tuple(_.name for _ in agents)]] = Field(
            description="开枪目标玩家姓名",
            default=None
        )
        shoot_reason: Optional[str] = Field(
            description="开枪理由",
            default=None
        )
    
    return HunterModelCN


class WerewolfKillModelCN(BaseModel):
    """中文版狼人击杀模型"""
    
    target: str = Field(
        description="要击杀的玩家姓名，或选择'空刀'表示今晚不击杀任何人",
    )
    kill_strategy: str = Field(
        description="击杀策略说明",
    )
    team_coordination: Optional[str] = Field(
        description="与狼队友的配合计划",
        default=None
    )


# class GameAnalysisModelCN(BaseModel):
#     """中文版游戏分析模型"""
    
#     suspected_werewolves: List[str] = Field(
#         description="怀疑的狼人名单",
#         default_factory=list
#     )
#     trusted_players: List[str] = Field(
#         description="信任的玩家名单", 
#         default_factory=list
#     )
#     key_clues: List[str] = Field(
#         description="关键线索列表",
#         default_factory=list
#     )
#     next_strategy: str = Field(
#         description="下一步策略",
#     )


class SpeechMemorySummaryModelCN(BaseModel):
    """公开发言长期记忆摘要模型"""

    has_effective_info: bool = Field(
        description=(
            "这段发言是否包含会直接影响游戏走向的信息，例如身份声称、明确阵营判断、"
            "预言家查验、女巫用药、猎人开枪计划、明确投票建议或明确点名某人是狼人/好人。"
            "泛泛推断平安夜、信息不足、建议继续讨论不算有效信息。"
        ),
    )
    summary: str = Field(
        description=(
            "写入长期记忆的具体摘要。必须保留关键事实，不能只写'涉及身份判断'。"
            "如果是玩家自称或推断，必须写明'玩家自称/未确认'或'玩家判断/未确认'。"
            "如果没有有效信息，写'某某在第几天白天发言中没有提供有效信息。'"
            "例如：'3号在第1天自称预言家（玩家自称），"
            "并称自己查验4号为狼人，建议全票出4号。'"
        ),
    )
    identity_claims: List[str] = Field(
        description="发言者自称的身份列表，例如 ['预言家']，没有则为空列表",
        default_factory=list,
    )
    mentioned_players: List[str] = Field(
        description="摘要中涉及的玩家名或座位号",
        default_factory=list,
    )
    key_points: List[str] = Field(
        description="用于身份判断、阵营判断、查验、用药、投票建议的关键点",
        default_factory=list,
    )


class WerewolfStrategySummaryModelCN(BaseModel):
    """狼人夜聊伪装/配合策略摘要模型"""

    has_strategy: bool = Field(
        description="kill_strategy 和 team_coordination 中是否包含可执行的白天伪装、身份伪装、发言配合、投票配合或带节奏策略",
    )
    strategy_summary: str = Field(
        description=(
            "写给当前狼人的第一人称私密长期记忆摘要。只保留白天伪装/配合策略，"
            "不要重复击杀目标本身。必须写清楚'我伪装成什么/我如何配合'，"
            "以及具体队友如何伪装或配合，不能写'一名成员/另一名成员'。例如："
            "'我白天伪装成普通村民，先低调观察；2号-xxx队友负责带头怀疑4号，"
            "3号-yyy队友跟进但避免同时强冲票。'"
        ),
    )
    key_points: List[str] = Field(
        description="提取出的伪装、发言、投票、带节奏或身份伪装关键点",
        default_factory=list,
    )
