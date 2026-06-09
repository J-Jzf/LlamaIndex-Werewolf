# -*- coding: utf-8 -*-
"""
正常狼人杀 - 基于 LlamaIndex 的中文版狼人杀游戏

保留传统狼人杀角色、玩法流程、Pydantic 结构化输出和失败兜底处理，
并支持一个真人玩家加入任意角色。
"""
from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import Counter
from threading import Lock
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai_like import OpenAILike
except ImportError:  # pragma: no cover - 运行时给出友好错误
    ChatMessage = None
    OpenAILike = None

from game_roles import GameRoles
from prompt_cn import ChinesePrompts
from structured_output_cn import (
    DiscussionModelCN,
    SpeechMemorySummaryModelCN,
    WerewolfKillModelCN,
    WerewolfStrategySummaryModelCN,
    WitchActionModelCN,
    get_hunter_model_cn,
    get_seer_model_cn,
    get_vote_model_cn,
)
from utils_cn import (
    MAX_DISCUSSION_ROUND,
    MAX_GAME_ROUND,
    check_winning_cn,
    format_player_list,
    majority_vote_cn,
)


WOLVES_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_MSG_PATH = os.path.join(WOLVES_DIR, "agentmsg.md")
LONG_TERM_MEMORY_PATH = os.path.join(WOLVES_DIR, "longterm_memory.md")
AI_NAME_SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜"
AI_NAME_CHARS = "明远安宁青云星河知行若谷以子墨景辰思源怀瑾清扬婕"
if load_dotenv:
    load_dotenv(os.path.join(WOLVES_DIR, ".env"))
else:
    env_path = os.path.join(WOLVES_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class SimpleMessage:

    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WolfPlayer:
    """游戏玩家；真人玩家和 AI 玩家共用同一结构。"""

    name: str
    role: str
    is_human: bool = False
    memory: List[str] = field(default_factory=list)
    long_term_memory: List[str] = field(default_factory=list)
    abstain_remaining: int = 1

    def observe(self, content: str, round_num: int = 0) -> None:
        self.memory.append(f"[第{round_num}轮] {content}")
        self.trim_short_term_memory(round_num)

    def trim_short_term_memory(self, current_round: int, keep_rounds: int = 1) -> None:
        min_round = max(0, current_round - keep_rounds + 1)
        trimmed = []
        for item in self.memory:
            match = re.match(r"\[第(\d+)轮\]", item)
            item_round = int(match.group(1)) if match else current_round
            if item_round >= min_round:
                trimmed.append(item)
        self.memory = trimmed

    def remember_long_term(self, content: str) -> None:
        self.long_term_memory.append(content)
        if len(self.long_term_memory) > 200:
            self.long_term_memory = self.long_term_memory[-200:]


class LlamaIndexAgent:
    """使用 LlamaIndex OpenAI-like LLM 的轻量狼人杀代理。"""

    def __init__(self, player: WolfPlayer, llm: Any):
        self.player = player
        self.llm = llm

    def _build_prompt(self, task: str, model_cls: Type[BaseModel]) -> List[Any]:
        schema = model_cls.model_json_schema()
        history = "\n".join(self.player.memory) or "暂无最近三轮短期记忆。"
        long_term = "\n".join(self.player.long_term_memory[-40:]) or "暂无长期记忆。"
        system_prompt = ChinesePrompts.get_role_prompt(self.player.role, self.player.name)
        user_prompt = f"""
            当前任务：
            {task}

            长期记忆：
            {long_term}

            最近游戏记录：
            {history}

            请只输出一个 JSON 对象，必须符合这个 Pydantic schema：
            {json.dumps(schema, ensure_ascii=False)}
        """
        if ChatMessage is None:
            return [system_prompt, user_prompt]
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

    def ask_structured(
        self,
        task: str,
        model_cls: Type[BaseModel],
        fallback: Dict[str, Any],
    ) -> SimpleMessage:
        """调用大模型并做对 LLM 输出做 Pydantic 校验；失败时有容错思路。"""
        try:
            response = self.llm.chat(self._build_prompt(task, model_cls))
            text = getattr(response, "message", response).content
            data = _extract_json(text)
            parsed = model_cls.model_validate(data)
            return SimpleMessage(self.player.name, json.dumps(parsed.model_dump(), ensure_ascii=False), parsed.model_dump())
        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            print(f"⚠️ {self.player.name} 结构化输出失败，使用兜底动作：{exc}")
            parsed = model_cls.model_validate(fallback)
            return SimpleMessage(self.player.name, json.dumps(parsed.model_dump(), ensure_ascii=False), parsed.model_dump())


def _extract_json(text: str) -> Dict[str, Any]:
    """从模型回复中提取 JSON 对象。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("模型回复中没有 JSON 对象")
    return json.loads(text[start : end + 1])


def build_llm() -> Any:
    """根据 .env 创建 LlamaIndex 的 OpenAI-compatible 模型。"""
    if OpenAILike is None:
        raise RuntimeError("未安装 llama-index。请先安装 llama-index 和 llama-index-llms-openai-like。")

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
    api_base = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY 或 OPENAI_API_KEY，请在 .env 中配置。")

    return OpenAILike(
        model=model,
        api_key=api_key,
        api_base=api_base,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0.8,
    )


def generate_ai_names(count: int, reserved_names: Optional[List[str]] = None) -> List[str]:
    """生成 3 个中文字符的 AI 游戏名，并避开真人玩家名。"""
    reserved = set(reserved_names or [])
    names: List[str] = []
    attempts = 0
    while len(names) < count and attempts < count * 100:
        attempts += 1
        name = (
            random.choice(AI_NAME_SURNAMES)
            + random.choice(AI_NAME_CHARS)
            + random.choice(AI_NAME_CHARS)
        )
        if name not in reserved and name not in names:
            names.append(name)

    while len(names) < count:
        fallback = f"玩家{len(names) + 1:02d}"
        if fallback not in reserved and fallback not in names:
            names.append(fallback)
    return names


class WerewolfGame:
    """正常狼人杀游戏主类。"""

    def __init__(self, player_count: int = 6, human_name: str = "沈清安", human_role: Optional[str] = None):
        self.player_count = player_count
        self.human_name = human_name or "沈清安"
        self.human_role = human_role
        self.players: Dict[str, WolfPlayer] = {}
        self.agents: Dict[str, LlamaIndexAgent] = {}
        self.roles: Dict[str, str] = {}
        self.alive_players: List[WolfPlayer] = []
        self.werewolves: List[WolfPlayer] = []
        self.villagers: List[WolfPlayer] = []
        self.seer: List[WolfPlayer] = []
        self.witch: List[WolfPlayer] = []
        self.hunter: List[WolfPlayer] = []
        self.log: List[str] = []
        self.public_log: List[str] = []
        self.review_events: List[str] = []
        self.round_num = 0
        self.phase = "未开始"
        self.progress_status = ""
        self.winner: Optional[str] = None
        self.pending_action: Optional[Dict[str, Any]] = None
        self.pending_ai_votes: Dict[str, Optional[str]] = {}
        self.pending_ai_vote_futures: Dict[str, Future] = {}
        self.in_pk_vote = False
        self.last_killed_player: Optional[str] = None
        self.witch_has_antidote = True
        self.witch_has_poison = True
        self.llm = build_llm()
        self.vote_executor = ThreadPoolExecutor(max_workers=max(1, player_count - 1))
        self.file_lock = Lock()
        self.reset_agent_msg_file()
        self.reset_long_term_memory_file()
        self.identity_claims: Dict[str, List[Dict[str, Any]]] = {}
        self.confirmed_identities: Dict[str, str] = {}

    @property
    def human(self) -> Optional[WolfPlayer]:
        for player in self.players.values():
            if player.is_human:
                return player
        return None

    def reset_agent_msg_file(self) -> None:
        """每局新游戏开始时重置 agent 输出日志。"""
        with open(AGENT_MSG_PATH, "w", encoding="utf-8") as file:
            file.write("# Agent 输出日志\n\n")
            file.write("本文件由游戏运行时实时追加，用于复盘每个 AI/真人玩家的输出。\n\n")

    def reset_long_term_memory_file(self) -> None:
        """每局新游戏开始时重置长期记忆快照。"""
        with open(LONG_TERM_MEMORY_PATH, "w", encoding="utf-8") as file:
            file.write("# 玩家长期记忆\n\n")
            file.write("本文件由游戏运行时实时刷新，展示每个玩家当前沉淀的长期记忆。\n\n")

    def flush_long_term_memory_file(self) -> None:
        """把当前所有玩家的长期记忆刷新到文件，便于调试复盘。"""
        with open(LONG_TERM_MEMORY_PATH, "w", encoding="utf-8") as file:
            file.write("# AI 玩家长期记忆\n\n")
            for player in self.players.values():
                file.write(f"## {player.name}（{player.role}{'，真人' if player.is_human else '，AI'}）\n\n")
                if not player.long_term_memory:
                    file.write("暂无长期记忆。\n\n")
                    continue
                for item in player.long_term_memory:
                    file.write(f"- {item}\n")
                file.write("\n")

    def write_agent_msg(
        self,
        player: WolfPlayer,
        task: str,
        output: Dict[str, Any] | str,
        source: str = "AI",
    ) -> None:
        """实时追加每个玩家代理的输出。"""
        if isinstance(output, str):
            rendered_output = output
        else:
            rendered_output = json.dumps(output, ensure_ascii=False, indent=2)

        with self.file_lock:
            with open(AGENT_MSG_PATH, "a", encoding="utf-8") as file:
                file.write(f"## 第{self.round_num}轮 · {self.phase or '未开始'} · {player.name}（{player.role}，{source}）\n\n")
                file.write(f"**任务**：{task}\n\n")
                file.write("**输出**：\n\n")
                file.write("```json\n" if not isinstance(output, str) else "```text\n")
                file.write(rendered_output)
                file.write("\n```\n\n")

    def announce(self, content: str, public: bool = True) -> None:
        self.log.append(content)
        if public:
            self.public_log.append(content)
        for player in self.players.values():
            if public:
                player.observe(content, self.round_num)
        print(f"📢 {content}")

    def set_progress(self, status: str) -> None:
        self.progress_status = status

    def clear_progress(self) -> None:
        self.progress_status = ""

    def add_review_event(self, content: str) -> None:
        self.review_events.append(f"第{self.round_num}轮·{self.phase}：{content}")

    def remember_public_event(self, content: str) -> None:
        """把公开关键事实写入所有玩家长期记忆。"""
        entry = f"第{self.round_num}轮·{self.phase}：{content}"
        for player in self.players.values():
            player.remember_long_term(entry)
        self.flush_long_term_memory_file()

    def remember_private_event(self, player: Optional[WolfPlayer], content: str) -> None:
        """把私密关键事实只写入对应玩家长期记忆，避免信息泄露。"""
        if player:
            player.remember_long_term(f"第{self.round_num}轮·{self.phase}：{content}")
            self.flush_long_term_memory_file()

    def trim_all_short_term_memory(self) -> None:
        for player in self.players.values():
            player.trim_short_term_memory(self.round_num)

    def summarize_speech_for_long_term_memory(self, speaker: WolfPlayer, speech: str) -> SpeechMemorySummaryModelCN:
        """调用模型，把公开发言摘要成具体、可用的长期记忆。只能看到当前玩家的当前发言。"""
        role_info = {
            role: {
                "team": info.get("team"),
                "ability": info.get("ability"),
                "win_condition": info.get("win_condition"),
            }
            for role, info in GameRoles.ROLES.items()
        }
        prompt = f"""
            你是狼人杀游戏记录员。请把一段公开发言整理成长期记忆摘要。

            要求：
            1. 不要保存原话，不要只写“涉及身份判断”这种空泛内容。
            2. 只有会直接影响游戏走向的信息才算有效：身份声称、明确阵营判断、预言家自称查验、女巫自称用药/未用药/救人/毒人、猎人自称开枪计划、明确投票建议、明确点名某人是狼人/好人。
            3. 泛泛推断不算有效信息。例如“平安夜可能是女巫救人”“昨晚应该有人死亡”“信息有限”“建议大家继续讨论”“先听发言再判断”“需要找狼人线索”都不算有效，除非同时明确声称身份、明确技能操作、明确查验结果或明确投票对象。
            4. 如果发言只有泛泛推断或流程性建议而没有任何“左右游戏结局走向”的“阵营、身份判断或特殊身份技能操作”关键信息的，has_effective_info=false，summary写“{speaker.name}在第{self.round_num}天白天发言中没有提供有效信息。”
            5. 如果信息来自玩家自己说或玩家推断，必须标注“玩家自称”或“玩家判断”。
            6. 身份摘要要尽量具体。例如：
            - “3号在第1天自称预言家（玩家自称），并称自己查验4号为狼人，建议全票出4号。”
            - “2号在第1天自称女巫（玩家自称），并称自己没有使用任何药。”

            本局角色信息：
            {json.dumps(role_info, ensure_ascii=False)}

            本局玩家名单：
            {format_player_list(self.alive_players)}

            发言者：{speaker.name}
            当前阶段：第{self.round_num}轮·{self.phase}
            公开发言：
            {speech}
        """
        fallback = {
            "has_effective_info": False,
            "summary": f"{speaker.name}在第{self.round_num}天白天发言中没有提供有效信息。",
            "identity_claims": [],
            "mentioned_players": [],
            "key_points": [],
        }
        try:
            response = self.llm.chat([
                ChatMessage(role="system", content="你只输出符合 Pydantic schema 的 JSON。"),
                ChatMessage(role="user", content=prompt + "\n\nSchema:\n" + json.dumps(SpeechMemorySummaryModelCN.model_json_schema(), ensure_ascii=False)),
            ])
            text = getattr(response, "message", response).content
            data = _extract_json(text)
            return SpeechMemorySummaryModelCN.model_validate(data)
        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            print(f"⚠️ 公开发言长期记忆摘要失败，使用兜底摘要：{exc}")
            return SpeechMemorySummaryModelCN.model_validate(fallback)

    def record_identity_claim(self, speaker: WolfPlayer, speech: str) -> None:
        """调用模型提取公开发言摘要，并记录身份声称和冲突线索。"""
        summary = self.summarize_speech_for_long_term_memory(speaker, speech)
        if not summary.has_effective_info:
            return

        self.remember_public_event(summary.summary)

        for claimed_role in summary.identity_claims:
            if claimed_role not in GameRoles.ROLES:
                continue
            claim = {"round": self.round_num, "phase": self.phase, "role": claimed_role, "evidence": summary.summary}
            self.identity_claims.setdefault(speaker.name, []).append(claim)

        if summary.identity_claims:
            self.update_identity_conflicts()

    # def extract_speech_evidence(self, speaker: WolfPlayer, speech: str, claimed_roles: List[str]) -> str:
    #     """把原话压缩成可判断的关键证据，不把整句原文写入长期记忆。"""
    #     evidence_parts: List[str] = []
    #     names = [name for name in self.players if name != speaker.name and name in speech]

    #     if re.search(r"昨晚|昨夜|夜里|第一晚|首夜", speech):
    #         if "自救" in speech:
    #             evidence_parts.append("他说自己昨晚自救")
    #         if re.search(r"救|解药", speech):
    #             evidence_parts.append("提到夜晚救人或解药信息")
    #         if re.search(r"毒|毒药|毒死|毒杀", speech):
    #             evidence_parts.append("提到毒药或夜晚毒杀信息")
    #         if re.search(r"刀|杀|被杀|被刀", speech):
    #             evidence_parts.append("提到夜晚击杀信息")

    #     if re.search(r"查验|验了|验人|金水|查杀|好人|狼人", speech):
    #         target_text = f"涉及 {'、'.join(names)} 的身份判断" if names else "提到查验或身份判断"
    #         evidence_parts.append(target_text)

    #     if re.search(r"怀疑|可疑|狼坑|像狼|不信|矛盾", speech):
    #         target_text = f"怀疑对象包括 {'、'.join(names)}" if names else "提出怀疑或矛盾点"
    #         evidence_parts.append(target_text)

    #     if re.search(r"投|票|归票|出", speech) and names:
    #         evidence_parts.append(f"给出投票倾向，涉及 {'、'.join(names)}")

    #     if claimed_roles and not evidence_parts:
    #         evidence_parts.append("只有身份声称，缺少可验证依据")

    #     unique_parts = []
    #     for part in evidence_parts:
    #         if part not in unique_parts:
    #             unique_parts.append(part)

    #     if not unique_parts:
    #         return ""
    #     return "；".join(unique_parts) + "。还需其他更多信息以佐证。"

    def update_identity_conflicts(self) -> None:
        """根据公开身份声称生成需要重点判断的矛盾线索。"""
        latest_claims = {
            name: claims[-1]["role"]
            for name, claims in self.identity_claims.items()
            if claims
        }

        for name, claims in self.identity_claims.items():
            claimed_role_set = {claim["role"] for claim in claims}
            if len(claimed_role_set) > 1:
                self.remember_public_event(f"{name} 曾声称多个身份：{'、'.join(claimed_role_set)}，存在身份矛盾，需要重点判断。")

        role_to_claimers: Dict[str, List[str]] = {}
        for name, role in latest_claims.items():
            if role != "村民":
                role_to_claimers.setdefault(role, []).append(name)

        for role, claimers in role_to_claimers.items():
            if len(claimers) > 1:
                self.remember_public_event(f"{'、'.join(claimers)} 都声称自己是 {role}，存在对跳或身份冲突，需要重点判断。")

    def build_role_setup_summary(self) -> str:
        """生成本局角色配置、阵营、技能和胜利条件说明。"""
        role_counts: Dict[str, int] = {}
        for role in self.roles.values():
            role_counts[role] = role_counts.get(role, 0) + 1

        parts = []
        for role, count in role_counts.items():
            info = GameRoles.ROLES.get(role, {})
            parts.append(
                f"{role}×{count}，阵营：{info.get('team', '未知')}，"
                f"核心技能：{info.get('ability', '未知')}，"
                f"获胜条件：{info.get('win_condition', '未知')}"
            )
        return "本局角色配置：" + "；".join(parts) + "。规则补充：狼人每晚必须行动。"

    def build_role_count_summary(self) -> str:
        """生成给前端公开显示的角色数量摘要。"""
        role_counts: Dict[str, int] = {}
        for role in self.roles.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        parts = [f"{role}×{count}" for role, count in role_counts.items()]
        return "本局角色配置：" + "，".join(parts) + "。"

    def setup_game(self) -> None:
        roles = GameRoles.get_standard_setup(self.player_count)
        if self.human_role and self.human_role not in roles:
            replace_index = len(roles) - 1
            for i in range(len(roles) - 1, -1, -1):
                if roles[i] == "村民":
                    replace_index = i
                    break
            roles[replace_index] = self.human_role
        human_index = random.randrange(self.player_count)
        assigned_roles: List[Optional[str]] = [None] * self.player_count
        if self.human_role in roles:
            roles.remove(self.human_role)
            assigned_roles[human_index] = self.human_role
        random.shuffle(roles)
        for i in range(self.player_count):
            if assigned_roles[i] is None:
                assigned_roles[i] = roles.pop()

        ai_names = generate_ai_names(self.player_count - 1, reserved_names=[self.human_name])
        random.shuffle(ai_names)
        names: List[Optional[str]] = [None] * self.player_count
        names[human_index] = self.human_name
        for i in range(self.player_count):
            if names[i] is None:
                names[i] = ai_names.pop()

        for index, (raw_name, role) in enumerate(zip(names, assigned_roles), start=1):
            assert raw_name is not None
            assert role is not None
            name = f"{index}号-{raw_name}"
            if index - 1 == human_index:
                self.human_name = name
            assert name is not None
            player = WolfPlayer(name=name, role=role, is_human=index - 1 == human_index)
            self.players[name] = player
            self.roles[name] = role
            self.alive_players.append(player)
            if not player.is_human:
                self.agents[name] = LlamaIndexAgent(player, self.llm)
                player.observe(f"你的游戏名是：{player.name}。请在整局游戏中记住并使用这个名字。", self.round_num)
            else:
                player.observe(f"你的游戏名是：{player.name}。", self.round_num)
            self._add_to_camp(player)

        self.announce(f"狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}")
        self.announce(self.build_role_count_summary())
        self.remember_public_event(f"本局玩家名单：{format_player_list(self.alive_players)}。")
        self.remember_public_event(self.build_role_setup_summary())
        self.initialize_werewolf_private_memory()
        if self.human:
            self.human.observe(f"你的身份是：{self.human.role}。{GameRoles.get_role_ability(self.human.role)}", self.round_num)
            self.log.append(f"【私密】{self.human.name} 的身份是 {self.human.role}")

    def initialize_werewolf_private_memory(self) -> None:
        """开局把狼队友信息写入狼人私密长期记忆。"""
        for wolf in self.werewolves:
            teammates = [other.name for other in self.werewolves if other.name != wolf.name]
            teammate_text = "、".join(teammates) if teammates else "无"
            self.remember_private_event(wolf, f"我的狼队友是：{teammate_text}。")
            wolf.observe(f"你的狼队友是：{teammate_text}。", self.round_num)

    def _add_to_camp(self, player: WolfPlayer) -> None:
        if player.role == "狼人":
            self.werewolves.append(player)
        elif player.role == "预言家":
            self.seer.append(player)
        elif player.role == "女巫":
            self.witch.append(player)
        elif player.role == "猎人":
            self.hunter.append(player)
        else:
            self.villagers.append(player)

    def update_alive_players(self, dead_players: List[str]) -> None:
        for dead_name in dead_players:
            if not dead_name:
                continue
            self.alive_players = [p for p in self.alive_players if p.name != dead_name]
            self.werewolves = [p for p in self.werewolves if p.name != dead_name]
            self.villagers = [p for p in self.villagers if p.name != dead_name]
            self.seer = [p for p in self.seer if p.name != dead_name]
            self.witch = [p for p in self.witch if p.name != dead_name]
            self.hunter = [p for p in self.hunter if p.name != dead_name]

    def alive_names(self, exclude: Optional[List[str]] = None) -> List[str]:
        exclude = exclude or []
        return [p.name for p in self.alive_players if p.name not in exclude]

    def _fallback_vote(self, candidates: List[str], key: str = "vote") -> Dict[str, Any]:
        target = random.choice(candidates) if candidates else None
        base = {key: target, "reason": "信息不足，随机选择。", "suspicion_level": 5}
        if key == "target":
            base = {"target": target, "kill_strategy": "信息不足，优先选择非狼人目标。", "team_coordination": None}
        return base

    def _ask_ai(self, player: WolfPlayer, task: str, model_cls: Type[BaseModel], fallback: Dict[str, Any]) -> SimpleMessage:
        msg = self.agents[player.name].ask_structured(task, model_cls, fallback)
        self.write_agent_msg(player, task, msg.metadata, source="AI")
        return msg

    def alive_wolves(self) -> List[WolfPlayer]:
        return [p for p in self.werewolves if p in self.alive_players]

    def broadcast_wolf_chat(self, speaker: WolfPlayer, content: str) -> None:
        """狼人夜聊只进入狼队短期记忆，不进入公开日志。"""
        message = f"【狼人夜聊】{speaker.name}：{content}"
        for wolf in self.alive_wolves():
            wolf.observe(message, self.round_num)
        if self.human and self.human.role == "狼人":
            self.log.append(f"【私密】{message}")

    def ai_wolf_discussion(self, wolf: WolfPlayer, candidates: List[str], round_index: int) -> SimpleMessage:
        self.set_progress("狼人操作中")
        msg = self._ask_ai(
            wolf,
            f"第{round_index}轮狼人夜聊。请基于狼队友已有意见协商击杀目标，候选人：{candidates}。并且请与队友讨论协商白天的身份伪装、发言投票引导或其他伪装策略，伪装策略具体到每一位狼玩家。",
            DiscussionModelCN,
            {
                "reach_agreement": False,
                "confidence_level": 5,
                "key_evidence": "暂无明确证据，建议先随机选择一个非狼人玩家。",
            },
        )
        content = msg.metadata.get("key_evidence") or msg.content
        self.broadcast_wolf_chat(wolf, content)
        return msg

    def record_human_wolf_discussion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        speech = payload.get("speech", "").strip() or "我倾向于先稳妥定刀，避免暴露狼队信息。"
        target = payload.get("target")
        self.broadcast_wolf_chat(self.human, f"{speech} 建议击杀：{target}")
        self.write_agent_msg(
            self.human,
            "真人狼人夜聊协商",
            {"speech": speech, "suggested_target": target},
            source="真人",
        )
        return {"target": target, "recorded": True}

    def run_ai_wolf_discussions(self, candidates: List[str]) -> None:
        ai_wolves = [wolf for wolf in self.alive_wolves() if not wolf.is_human]
        if not ai_wolves:
            return
        for round_index in range(1, MAX_DISCUSSION_ROUND + 1):
            agreement_count = 0
            for wolf in ai_wolves:
                msg = self.ai_wolf_discussion(wolf, candidates, round_index)
                if msg.metadata.get("reach_agreement"):
                    agreement_count += 1
            if agreement_count >= max(1, len(ai_wolves) // 2 + 1):
                break

    def resolve_werewolf_kill(
        self,
        candidates: List[str],
        human_vote: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """在共享夜聊记忆基础上收集狼队最终击杀票，形成统一目标。"""
        wolves = self.alive_wolves()
        votes: Dict[str, Optional[str]] = {}
        strategy_inputs: List[Dict[str, Any]] = []

        if human_vote and self.human in wolves:
            target = human_vote.get("target")
            votes[self.human.name] = target
            if not human_vote.get("recorded"):
                speech = human_vote.get("speech", "").strip()
                self.broadcast_wolf_chat(self.human, speech)
                self.write_agent_msg(
                    self.human,
                    "真人狼人夜聊协商",
                    {"speech": speech, "suggested_target": target},
                    source="真人",
                )
            strategy_inputs.append({
                "player": self.human.name,
                "kill_strategy": human_vote.get("speech"),
                "team_coordination": human_vote.get("speech"),
            })
            self.write_agent_msg(self.human, "真人狼人最终定刀投票", {"target": target}, source="真人")
            human_kill_text = "选择空刀" if target == "空刀" else f"最终投票击杀 {target}"
            self.remember_private_event(self.human, f"狼人私密记录：第{self.round_num}夜你{human_kill_text}。")
            self.log.append(f"【私密】{self.human.name} 的最终定刀意见：{target}。")

        for wolf in wolves:
            if wolf.is_human:
                continue
            self.set_progress("狼人操作中")
            msg = self._ask_ai(
                wolf,
                f"狼队已经完成夜聊。请根据所有狼人夜聊记忆给出最终击杀目标，候选人：{candidates}",
                WerewolfKillModelCN,
                self._fallback_vote(candidates, "target"),
            )
            votes[wolf.name] = msg.metadata.get("target")
            strategy_inputs.append({
                "player": wolf.name,
                "kill_strategy": msg.metadata.get("kill_strategy"),
                "team_coordination": msg.metadata.get("team_coordination"),
            })
            wolf_target = msg.metadata.get("target")
            wolf_kill_text = "选择空刀" if wolf_target == "空刀" else f"最终投票击杀 {wolf_target}"
            self.remember_private_event(wolf, f"狼人私密记录：第{self.round_num}夜你{wolf_kill_text}。")
            if self.human and self.human.role == "狼人":
                self.log.append(f"【私密】{wolf.name} 的最终定刀意见：{msg.metadata.get('target')}。")

        killed_player, vote_count = majority_vote_cn(votes)
        if killed_player == "空刀":
            self.add_review_event("狼人选择空刀。")
            killed_player = None
        elif killed_player:
            self.add_review_event(f"狼人决定击杀 {killed_player}。")
        for wolf in wolves:
            strategy_summary = self.summarize_werewolf_strategy(strategy_inputs, wolf)
            kill_result = killed_player or "空刀"
            wolf.observe(f"【狼人夜聊】最终统一击杀目标：{kill_result}（{vote_count}票）", self.round_num)
            self.remember_private_event(
                wolf,
                f"狼人夜聊结论：第{self.round_num}夜狼队统一击杀目标为 {kill_result}（{vote_count}票）。",
            )
            if strategy_summary:
                self.remember_private_event(wolf, f"狼人夜聊结论：第{self.round_num}夜狼队伪装/配合策略：{strategy_summary}")
                if wolf.is_human:
                    self.log.append(f"【私密】狼队第{self.round_num}夜你的伪装/配合策略：{strategy_summary}")
        if self.human and self.human.role == "狼人":
            self.log.append(f"【私密】狼队第{self.round_num}夜最终决定：{killed_player or '空刀'}（{vote_count}票）。")
        return killed_player

    def summarize_werewolf_strategy(self, strategy_inputs: List[Dict[str, Any]], perspective_wolf: WolfPlayer) -> str:
        """调用模型从狼队最终意见中提取当前狼人视角的白天伪装/配合策略。"""
        useful_inputs = [
            item for item in strategy_inputs
            if item.get("kill_strategy") or item.get("team_coordination")
        ]
        if not useful_inputs:
            return ""
        teammates = [wolf.name for wolf in self.alive_wolves() if wolf.name != perspective_wolf.name]
        prompt = f"""
            你是狼人杀游戏记录员。请从狼队每个成员的 kill_strategy 和 team_coordination 中提取一条写给当前狼人的私密长期记忆。

            要求：
            1. 只总结白天伪装、身份伪装、发言配合、投票配合、带节奏策略。
            2. 不要重复“最终击杀谁”这个事实本身。
            3. 必须使用当前狼人第一人称视角，写清楚“我伪装成什么/我怎么发言或投票”，以及“xx号队友伪装成什么/如何配合”。
            4. 不要写“一名成员”“另一名成员”这种模糊说法；必须使用具体玩家名或座位号。
            5. 如果没有明确身份伪装，也要提取可执行配合策略，例如“我低调装村民，xx号队友带头怀疑某人，我跟票但避免强冲”。
            6. 如果没有可执行的伪装/配合策略，has_strategy=false。
            7. 输出必须是 JSON。

            当前狼人：{perspective_wolf.name}
            当前狼人的队友：{teammates}

            推荐摘要风格：
            “我白天伪装成普通村民，先低调观察；2号-xxx队友负责带头怀疑4号，3号-yyy队友跟进但避免同时强冲票。”

            狼队最终意见：
            {json.dumps(useful_inputs, ensure_ascii=False)}
        """
        fallback = {
            "has_strategy": False,
            "strategy_summary": "",
            "key_points": [],
        }
        try:
            response = self.llm.chat([
                ChatMessage(role="system", content="你只输出符合 Pydantic schema 的 JSON。"),
                ChatMessage(role="user", content=prompt + "\n\nSchema:\n" + json.dumps(WerewolfStrategySummaryModelCN.model_json_schema(), ensure_ascii=False)),
            ])
            text = getattr(response, "message", response).content
            data = _extract_json(text)
            parsed = WerewolfStrategySummaryModelCN.model_validate(data)
            return parsed.strategy_summary if parsed.has_strategy else ""
        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            print(f"⚠️ 狼队伪装/配合策略摘要失败，跳过记录：{exc}")
            return ""

    def step_until_input_or_end(self) -> None:
        if not self.players:
            self.setup_game()

        while not self.pending_action and not self.winner and self.round_num < MAX_GAME_ROUND:
            self.round_num += 1
            self.trim_all_short_term_memory()
            self.night_phase()
            if self.pending_action or self.winner:
                return
            self.day_phase()
            if self.pending_action or self.winner:
                return

        if not self.winner and self.round_num >= MAX_GAME_ROUND:
            self.winner = "达到最大轮数，游戏结束。"
            self.announce(self.winner)

    def night_phase(self) -> None:
        self.phase = f"第{self.round_num}夜"
        self.announce(f"🌙 第{self.round_num}夜降临，天黑请闭眼...")
        self.last_killed_player = self.werewolf_phase()
        if self.pending_action:
            return
        self.seer_phase()
        if self.pending_action:
            return
        final_killed, poisoned_player = self.witch_phase(self.last_killed_player)
        if self.pending_action:
            return
        self.finish_night(final_killed, poisoned_player)

    def finish_night(self, final_killed: Optional[str], poisoned_player: Optional[str]) -> None:
        night_deaths = [p for p in [final_killed, poisoned_player] if p]
        self.update_alive_players(night_deaths)
        self.announce("昨夜平安无事，无人死亡。" if not night_deaths else f"昨夜，{'、'.join(night_deaths)}不幸遇害。")
        if not night_deaths:
            self.remember_public_event("昨夜平安无事，无人死亡。")
        else:
            self.remember_public_event(f"夜晚死亡：{'、'.join(night_deaths)}。")
        self.check_winner()

    def werewolf_phase(self) -> Optional[str]:
        wolves = self.alive_wolves()
        if not wolves:
            return None
        candidates = self.alive_names(exclude=[w.name for w in wolves]) + ["空刀"]
        self.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")
        self.set_progress("狼人操作中")
        if any(w.is_human for w in wolves):
            self.clear_progress()
            self.pending_action = {
                "type": "werewolf_discussion",
                "candidates": candidates,
                "message": "你是狼人，请参与狼队夜聊：写下你的击杀策略，并选择一个建议击杀目标。",
            }
            return None

        self.run_ai_wolf_discussions(candidates)
        return self.resolve_werewolf_kill(candidates)

    def seer_phase(self) -> None:
        if not self.seer:
            return
        seer = self.seer[0]
        candidates = self.alive_names(exclude=[seer.name])
        self.announce("🔮 预言家请睁眼，选择要查验的玩家...")
        self.set_progress("预言家操作中")
        if seer.is_human:
            self.clear_progress()
            self.pending_action = {"type": "seer_check", "candidates": candidates, "message": "你是预言家，请选择查验目标。"}
            return
        msg = self._ask_ai(seer, f"请选择要查验的玩家，候选人：{candidates}", get_seer_model_cn(self.alive_players), {
            "target": random.choice(candidates) if candidates else seer.name,
            "check_reason": "信息不足，优先扩大身份信息。",
            "priority_level": 5,
        })
        self._record_seer_result(seer, msg.metadata.get("target"))

    def _record_seer_result(self, seer: WolfPlayer, target_name: Optional[str]) -> None:
        if not target_name:
            print("⚠️ 预言家未选择查验目标，跳过此阶段")
            return
        target_role = self.roles.get(target_name, "村民")
        result_msg = f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        seer.observe(result_msg, self.round_num)
        self.remember_private_event(seer, f"已明确确认：{target_name} 是 {'狼人' if target_role == '狼人' else '好人'}（预言家查验）。")
        if seer.is_human:
            self.log.append(f"【私密】{result_msg}")

    def witch_phase(self, killed_player: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not self.witch:
            return killed_player, None
        witch = self.witch[0]
        candidates = self.alive_names()
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        can_use_antidote = bool(self.witch_has_antidote and killed_player)
        witch.observe(death_info, self.round_num)
        self.announce("🧙‍♀️ 女巫请睁眼...")
        self.set_progress("女巫操作中")
        if witch.is_human:
            self.clear_progress()
            self.pending_action = {
                "type": "witch_action",
                "candidates": candidates,
                "killed_player": killed_player,
                "has_antidote": can_use_antidote,
                "antidote_remaining": 1 if self.witch_has_antidote else 0,
                "has_poison": self.witch_has_poison,
                "message": death_info,
            }
            return killed_player, None

        msg = self._ask_ai(witch, f"{death_info}。请选择是否使用解药或毒药，候选人：{candidates}", WitchActionModelCN, {
            "use_antidote": bool(can_use_antidote and random.random() < 0.45),
            "use_poison": bool(self.witch_has_poison and random.random() < 0.2),
            "target_name": random.choice(candidates) if candidates else None,
            "action_reason": "根据信息谨慎使用道具。",
        })
        return self._apply_witch_action(killed_player, msg.metadata)

    def _apply_witch_action(self, killed_player: Optional[str], action: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        saved_player = None
        poisoned_player = None
        if action.get("use_antidote") and self.witch_has_antidote and killed_player:
            saved_player = killed_player
            self.witch_has_antidote = False
            self.add_review_event(f"女巫对 {saved_player} 使用解药。")
        if action.get("use_poison") and self.witch_has_poison:
            poisoned_player = action.get("target_name")
            if poisoned_player:
                self.witch_has_poison = False
                self.add_review_event(f"女巫对 {poisoned_player} 使用毒药。")
        final_killed = killed_player if not saved_player else None
        if self.witch:
            witch = self.witch[0]
            if saved_player:
                saved_msg = f"女巫私密记录：第{self.round_num}夜使用解药救了 {saved_player}。解药剩余 0 瓶，毒药剩余 {1 if self.witch_has_poison else 0} 瓶。"
                self.remember_private_event(
                    witch,
                    saved_msg,
                )
                if witch.is_human:
                    self.log.append(f"【私密】{saved_msg}")
            if poisoned_player:
                poison_msg = f"女巫私密记录：第{self.round_num}夜使用毒药毒了 {poisoned_player}。解药剩余 {1 if self.witch_has_antidote else 0} 瓶，毒药剩余 0 瓶。"
                self.remember_private_event(
                    witch,
                    poison_msg,
                )
                if witch.is_human:
                    self.log.append(f"【私密】{poison_msg}")
            if not saved_player and not poisoned_player:
                no_action_msg = f"女巫私密记录：第{self.round_num}夜未使用药。解药剩余 {1 if self.witch_has_antidote else 0} 瓶，毒药剩余 {1 if self.witch_has_poison else 0} 瓶。"
                self.remember_private_event(
                    witch,
                    no_action_msg,
                )
                if witch.is_human:
                    self.log.append(f"【私密】{no_action_msg}")
        return final_killed, poisoned_player

    def day_phase(self) -> None:
        self.phase = f"第{self.round_num}天"
        self.announce(f"☀️ 第{self.round_num}天天亮了，请大家睁眼...")
        self.announce(f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}")
        for index, player in enumerate(self.alive_players):
            if player.is_human:
                next_speaker = None
                for next_player in self.alive_players[index + 1 :]:
                    if not next_player.is_human:
                        next_speaker = next_player.name
                        break
                self.pending_action = {
                    "type": "day_speech",
                    "next_index": index + 1,
                    "next_speaker": next_speaker,
                    "message": "轮到你白天发言。",
                }
                return
            self.ai_day_speech(player)
        self.start_vote()

    def ai_day_speech(self, player: WolfPlayer) -> None:
        self.set_progress(f"{player.name}正在发言")
        msg = self._ask_ai(player, "请进行一段白天公开发言，表达你的观察和推理。", DiscussionModelCN, {
            "reach_agreement": False,
            "confidence_level": 5,
            "key_evidence": "暂无明确证据",
        })
        speech = msg.metadata.get("key_evidence") or msg.content
        self.announce(f"{player.name} 发言：{speech}")
        self.record_identity_claim(player, speech)

    def start_vote(self) -> None:
        candidates = self.alive_names()
        self.start_ai_day_vote_futures(candidates)
        human_alive = bool(self.human and self.human in self.alive_players)
        if human_alive:
            human_candidates = self.vote_candidates_for_player(self.human, candidates)
            self.pending_action = {
                "type": "day_vote",
                "candidates": human_candidates,
                "abstain_remaining": self.human.abstain_remaining,
                "message": "请投票选择要淘汰的玩家。AI 玩家正在同一投票节点并发投票，提交后将统一计票。",
            }
            return
        self.finish_vote(self.collect_pending_ai_votes())

    def vote_candidates_for_player(self, player: WolfPlayer, base_candidates: List[str]) -> List[str]:
        candidates = list(base_candidates)
        if player.abstain_remaining > 0:
            candidates.append("弃权")
        return candidates

    def start_ai_day_vote_futures(self, candidates: List[str]) -> None:
        self.pending_ai_votes = {}
        self.pending_ai_vote_futures = {}
        for player in self.alive_players:
            if player.is_human:
                continue
            self.pending_ai_vote_futures[player.name] = self.vote_executor.submit(
                self.ask_ai_day_vote,
                player,
                candidates,
            )

    def ask_ai_day_vote(self, player: WolfPlayer, candidates: List[str]) -> Optional[str]:
        try:
            self.set_progress("其他玩家投票中")
            player_candidates = self.vote_candidates_for_player(player, candidates)
            fallback = self._fallback_vote(player_candidates)
            msg = self._ask_ai(
                player,
                f"请投票选择要淘汰的玩家，候选人：{player_candidates}。你本局还剩 {player.abstain_remaining} 次弃权机会，最多只能弃权 1 次。",
                get_vote_model_cn(player_candidates),
                fallback,
            )
            return msg.metadata.get("vote")
        except Exception as exc:
            print(f"⚠️ {player.name} 并发投票失败，视为弃票：{exc}")
            return None

    def collect_pending_ai_votes(self) -> Dict[str, Optional[str]]:
        votes = dict(self.pending_ai_votes)
        for player_name, future in self.pending_ai_vote_futures.items():
            if player_name in votes:
                continue
            votes[player_name] = future.result()
        self.pending_ai_votes = votes
        self.pending_ai_vote_futures = {}
        return votes

    def finish_vote(self, votes: Dict[str, Optional[str]]) -> None:
        self.clear_progress()
        self.apply_abstain_usage(votes)
        valid_votes = {voter: target for voter, target in votes.items() if target and target != "弃权"}
        self.pending_ai_votes = {}
        self.pending_ai_vote_futures = {}
        abstain_count = sum(1 for target in votes.values() if target == "弃权")
        if not valid_votes:
            voted_out = None
            vote_count = 0
            self.announce(f"投票结果：本轮共有{abstain_count}人弃权，无人被淘汰。")
            self.remember_public_event(f"白天投票结果：本轮共有 {abstain_count} 人弃权，无人被淘汰。")
        else:
            tied_targets, vote_count = self.get_top_tied_targets(valid_votes)
            if len(tied_targets) > 1:
                if self.in_pk_vote:
                    self.in_pk_vote = False
                    self.announce(f"PK重新投票仍然平票：{'、'.join(tied_targets)}各{vote_count}票，本轮无人被淘汰。")
                    self.remember_public_event(f"PK重新投票仍然平票：{'、'.join(tied_targets)}各 {vote_count} 票，本轮无人被淘汰。")
                    self.finish_day(None, None)
                    return
                self.start_pk_phase(tied_targets, vote_count, abstain_count)
                return
            voted_out = tied_targets[0]
            self.announce(f"投票结果：{voted_out}以{vote_count}票被淘汰出局，弃权{abstain_count}票。")
            self.remember_public_event(f"白天投票结果：{voted_out} 以 {vote_count} 票被淘汰出局，弃权 {abstain_count} 票。")
            self.add_review_event(f"白天投票淘汰 {voted_out}。")
        hunter_shot = self.hunter_phase(voted_out)
        if self.pending_action:
            return
        self.finish_day(voted_out, hunter_shot)

    def get_top_tied_targets(self, votes: Dict[str, str]) -> tuple[List[str], int]:
        counts = Counter(votes.values())
        if not counts:
            return [], 0
        top_count = max(counts.values())
        return [target for target, count in counts.items() if count == top_count], top_count

    def start_pk_phase(self, tied_targets: List[str], vote_count: int, abstain_count: int) -> None:
        self.in_pk_vote = True
        self.announce(f"投票出现平票：{'、'.join(tied_targets)}各{vote_count}票，弃权{abstain_count}票。进入PK发言。")
        self.remember_public_event(f"白天投票平票：{'、'.join(tied_targets)}各 {vote_count} 票，进入PK发言。")
        pk_players = [player for player in self.alive_players if player.name in tied_targets]
        for index, player in enumerate(pk_players):
            if player.is_human:
                next_speaker = None
                for next_player in pk_players[index + 1 :]:
                    if not next_player.is_human:
                        next_speaker = next_player.name
                        break
                self.pending_action = {
                    "type": "pk_speech",
                    "pk_targets": tied_targets,
                    "next_index": index + 1,
                    "next_speaker": next_speaker,
                    "message": "你进入PK，请进行PK发言。",
                }
                return
            self.ai_pk_speech(player, tied_targets)
        self.start_pk_vote(tied_targets)

    def ai_pk_speech(self, player: WolfPlayer, tied_targets: List[str]) -> None:
        self.set_progress(f"{player.name}正在发言")
        msg = self._ask_ai(player, f"你进入PK发言，PK对象：{tied_targets}。请为自己辩护并说明投票建议。", DiscussionModelCN, {
            "reach_agreement": False,
            "confidence_level": 5,
            "key_evidence": "我需要澄清自己的身份和投票逻辑。",
        })
        speech = msg.metadata.get("key_evidence") or msg.content
        self.announce(f"{player.name} PK发言：{speech}")
        self.record_identity_claim(player, speech)

    def start_pk_vote(self, tied_targets: List[str]) -> None:
        self.start_ai_day_vote_futures(tied_targets)
        human_alive = bool(self.human and self.human in self.alive_players)
        if human_alive:
            human_candidates = self.vote_candidates_for_player(self.human, tied_targets)
            self.pending_action = {
                "type": "pk_vote",
                "candidates": human_candidates,
                "abstain_remaining": self.human.abstain_remaining,
                "message": f"PK重新投票，请在平票玩家中选择：{'、'.join(tied_targets)}。",
            }
            return
        self.finish_vote(self.collect_pending_ai_votes())

    def apply_abstain_usage(self, votes: Dict[str, Optional[str]]) -> None:
        for voter, target in votes.items():
            if target != "弃权":
                continue
            player = self.players.get(voter)
            if player and player.abstain_remaining > 0:
                player.abstain_remaining -= 1

    def hunter_phase(self, voted_out: Optional[str]) -> Optional[str]:
        if not voted_out or not self.hunter:
            return None
        hunter = self.hunter[0]
        if hunter.name != voted_out:
            return None
        candidates = self.alive_names(exclude=[hunter.name])
        self.announce("🏹 猎人发动技能，可以带走一名玩家...")
        if hunter.is_human:
            self.pending_action = {"type": "hunter_shot", "candidates": candidates, "voted_out": voted_out, "message": "你是猎人，请选择是否开枪。"}
            return None
        msg = self._ask_ai(hunter, f"你被投票出局，是否开枪？候选人：{candidates}", get_hunter_model_cn(self.alive_players), {
            "shoot": bool(candidates and random.random() < 0.7),
            "target": random.choice(candidates) if candidates else None,
            "shoot_reason": "根据白天发言选择最可疑的人。",
        })
        if msg.metadata.get("shoot"):
            target = msg.metadata.get("target")
            if target:
                self.add_review_event(f"猎人 {hunter.name} 开枪带走 {target}。")
            return target
        return None

    def finish_day(self, voted_out: Optional[str], hunter_shot: Optional[str]) -> None:
        day_deaths = [p for p in [voted_out, hunter_shot] if p]
        self.update_alive_players(day_deaths)
        self.check_winner()
        if not self.winner:
            self.announce(f"第{self.round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}")
            self.remember_public_event(f"第{self.round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}。")
        self.clear_progress()

    def check_winner(self) -> None:
        winner = check_winning_cn(self.alive_players, self.roles)
        if winner:
            self.winner = winner
            self.announce(f"🎉 游戏结束！{winner}")

    def submit_human_action(self, payload: Dict[str, Any]) -> None:
        action = self.pending_action or {}
        action_type = action.get("type")
        self.pending_action = None

        if action_type in {"werewolf_discussion", "werewolf_kill"}:
            candidates = action.get("candidates") or self.alive_names(exclude=[w.name for w in self.alive_wolves()])
            human_vote = (
                self.record_human_wolf_discussion(payload)
                if action_type == "werewolf_discussion"
                else {"target": payload.get("target")}
            )
            self.run_ai_wolf_discussions(candidates)
            self.last_killed_player = self.resolve_werewolf_kill(candidates, human_vote)
            self.announce(f"狼人完成了今晚的选择。", public=True)
            self.seer_phase()
            if not self.pending_action:
                final_killed, poisoned_player = self.witch_phase(self.last_killed_player)
                if not self.pending_action:
                    self.finish_night(final_killed, poisoned_player)
        elif action_type == "seer_check":
            self.write_agent_msg(self.human, "真人预言家夜晚查验", {"target": payload.get("target")}, source="真人")
            self._record_seer_result(self.human, payload.get("target"))
            final_killed, poisoned_player = self.witch_phase(self.last_killed_player)
            if not self.pending_action:
                self.finish_night(final_killed, poisoned_player)
        elif action_type == "witch_action":
            self.write_agent_msg(self.human, "真人女巫夜晚行动", payload, source="真人")
            final_killed, poisoned_player = self._apply_witch_action(self.last_killed_player, payload)
            self.finish_night(final_killed, poisoned_player)
        elif action_type == "day_speech":
            speech = payload.get("speech", "").strip() or "我暂时没有更多信息。"
            self.write_agent_msg(self.human, "真人白天公开发言", speech, source="真人")
            self.announce(f"{self.human.name} 发言：{speech}")
            self.record_identity_claim(self.human, speech)
            for player in self.alive_players[action.get("next_index", 0) :]:
                if not player.is_human:
                    self.ai_day_speech(player)
            self.start_vote()
        elif action_type == "pk_speech":
            speech = payload.get("speech", "").strip() or "我需要澄清自己的身份和投票逻辑。"
            pk_targets = action.get("pk_targets", [])
            self.write_agent_msg(self.human, "真人PK发言", speech, source="真人")
            self.announce(f"{self.human.name} PK发言：{speech}")
            self.record_identity_claim(self.human, speech)
            pk_players = [player for player in self.alive_players if player.name in pk_targets]
            for player in pk_players[action.get("next_index", 0) :]:
                if not player.is_human:
                    self.ai_pk_speech(player, pk_targets)
            self.start_pk_vote(pk_targets)
        elif action_type == "day_vote":
            target = payload.get("target")
            if target == "弃权" and self.human.abstain_remaining <= 0:
                target = None
            self.write_agent_msg(self.human, "真人白天投票", {"target": target, "abstain_remaining_before_vote": self.human.abstain_remaining}, source="真人")
            votes = self.collect_pending_ai_votes()
            votes[self.human.name] = target
            self.finish_vote(votes)
        elif action_type == "pk_vote":
            target = payload.get("target")
            if target == "弃权" and self.human.abstain_remaining <= 0:
                target = None
            self.write_agent_msg(self.human, "真人PK重新投票", {"target": target, "abstain_remaining_before_vote": self.human.abstain_remaining}, source="真人")
            votes = self.collect_pending_ai_votes()
            votes[self.human.name] = target
            self.finish_vote(votes)
        elif action_type == "hunter_shot":
            hunter_shot = payload.get("target") if payload.get("shoot") else None
            self.write_agent_msg(self.human, "真人猎人开枪", {"shoot": payload.get("shoot"), "target": hunter_shot}, source="真人")
            if hunter_shot:
                self.add_review_event(f"猎人 {self.human.name} 开枪带走 {hunter_shot}。")
            self.finish_day(action.get("voted_out"), hunter_shot)

        if self.pending_action or self.winner:
            if self.pending_action:
                self.clear_progress()
            return

        if action_type in {"werewolf_discussion", "werewolf_kill", "seer_check", "witch_action"}:
            self.day_phase()
            if self.pending_action or self.winner:
                if self.pending_action:
                    self.clear_progress()
                return

        self.step_until_input_or_end()

    def snapshot(self, reveal_roles: bool = False) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "round": self.round_num,
            "winner": self.winner,
            "players": [
                {
                    "name": p.name,
                    "role": p.role if self.can_reveal_role_to_human(p, reveal_roles) else "未知",
                    "alive": p in self.alive_players,
                    "is_human": p.is_human,
                    "abstain_remaining": p.abstain_remaining,
                }
                for p in self.players.values()
            ],
            "human_role": self.human.role if self.human else None,
            "log": self.public_log[-80:],
            "private_log": [line for line in self.log if line.startswith("【私密】")][-20:],
            "pending_action": self.pending_action,
            "progress_status": self.progress_status,
            "review_events": self.review_events if self.winner else [],
        }

    def can_reveal_role_to_human(self, player: WolfPlayer, reveal_roles: bool = False) -> bool:
        if reveal_roles or self.winner or player.is_human:
            return True
        human = self.human
        return bool(human and human.role == "狼人" and player.role == "狼人")


def run_cli_game() -> None:
    """无前端时可用于快速跑 AI 对局。"""
    game = WerewolfGame(player_count=6, human_name="旁观者")
    game.step_until_input_or_end()
    while game.pending_action and not game.winner:
        candidates = game.pending_action.get("candidates", [])
        target = random.choice(candidates) if candidates else None
        game.submit_human_action({"target": target, "speech": "我先观察大家的发言。"})


if __name__ == "__main__":
    run_cli_game()
