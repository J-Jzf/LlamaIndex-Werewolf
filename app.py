# -*- coding: utf-8 -*-
"""狼人杀简易可视化前端。"""
from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request

from game_roles import GameRoles
from main_cn import WerewolfGame


app = Flask(__name__)
game: WerewolfGame | None = None


PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>狼人杀</title>
  <style>
    :root { color-scheme: light; font-family: Inter, "Microsoft YaHei", system-ui, sans-serif; }
    body { margin: 0; background: #f7f7f2; color: #222; }
    .shell { display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: 100vh; }
    aside { background: #1f2933; color: #f8fafc; padding: 22px; }
    main { padding: 22px; display: grid; grid-template-rows: auto 1fr auto; gap: 16px; }
    h1 { font-size: 24px; margin: 0 0 18px; letter-spacing: 0; }
    h2 { font-size: 16px; margin: 18px 0 10px; letter-spacing: 0; }
    label { display: block; font-size: 13px; margin: 12px 0 6px; color: #cbd5e1; }
    input, select, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; font: inherit; }
    textarea { min-height: 88px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; background: #2563eb; color: white; font: inherit; cursor: pointer; }
    button.secondary { background: #475569; }
    .toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .status { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
    .tile { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; min-height: 74px; }
    .tile.dead { opacity: .55; }
    .name { font-weight: 700; }
    .role { color: #64748b; font-size: 13px; margin-top: 6px; }
    .human { color: #0f766e; font-size: 12px; margin-top: 6px; }
    .log { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; overflow: auto; min-height: 360px; max-height: 58vh; }
    .line { padding: 8px 0; border-bottom: 1px solid #eef2f7; line-height: 1.55; }
    .line:last-child { border-bottom: 0; }
    .action { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 14px; }
    .meta { color: #64748b; }
    .private { color: #9333ea; }
    @media (max-width: 800px) {
      .shell { grid-template-columns: 1fr; }
      aside { min-height: auto; }
      main { padding: 14px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>狼人杀</h1>
      <label>你的名字</label>
      <input id="humanName" value="沈清安">
      <label>你的角色</label>
      <select id="humanRole">
        {% for role in roles %}
        <option value="{{ role }}">{{ role }}</option>
        {% endfor %}
      </select>
      <label>玩家人数</label>
      <select id="playerCount">
        <option value="6">6人</option>
        <option value="8">8人</option>
        <option value="9">9人</option>
      </select>
      <div style="height:14px"></div>
      <button onclick="startGame()">开始新游戏</button>
      <button class="secondary" onclick="refreshState()">刷新</button>
      <h2>当前状态</h2>
      <div id="summary" class="meta">尚未开始</div>
    </aside>
    <main>
      <section class="status" id="players"></section>
      <section class="log" id="log"></section>
      <section class="log" id="review" style="display:none"></section>
      <section class="action" id="action">需要你行动时，会在这里出现操作区。</section>
    </main>
  </div>
  <script>
    let currentState = null;
    let waitingTimer = null;
    let progressTimer = null;
    let currentWaitingMessage = '系统处理中';
    let waitingStartedAt = Date.now();

    async function api(url, payload) {
      const res = await fetch(url, {
        method: payload ? 'POST' : 'GET',
        headers: {'Content-Type': 'application/json'},
        body: payload ? JSON.stringify(payload) : undefined
      });
      return await res.json();
    }

    async function startGame() {
      const payload = {
        human_name: document.getElementById('humanName').value,
        human_role: document.getElementById('humanRole').value,
        player_count: Number(document.getElementById('playerCount').value)
      };
      startWaitingTimer('系统处理中');
      try {
        const data = await api('/api/start', payload);
        stopWaitingTimer();
        render(data);
      } finally {
        stopWaitingTimer();
      }
    }

    async function refreshState(preserveAction = false) {
      render(await api('/api/state'), {preserveAction});
    }

    async function submitAction(payload) {
      const pendingType = currentState?.pending_action?.type;
      startWaitingTimer(waitingTextAfterAction(currentState?.pending_action));
      try {
        const data = await api('/api/action', payload);
        stopWaitingTimer();
        render(data);
      } finally {
        stopWaitingTimer();
      }
    }

    function startWaitingTimer(message) {
      stopWaitingTimer();
      if (!message) return;
      currentWaitingMessage = message;
      waitingStartedAt = Date.now();
      const actionEl = document.getElementById('action');
      const tick = () => {
        const seconds = Math.max(1, Math.floor((Date.now() - waitingStartedAt) / 1000) + 1);
        actionEl.innerHTML = `<b>${currentWaitingMessage}，${seconds}秒</b><br><br><span class="meta">请稍等，系统正在处理当前阶段。</span>`;
      };
      tick();
      waitingTimer = setInterval(tick, 1000);
      progressTimer = setInterval(async () => {
        try {
          const data = await api('/api/progress');
          if (data.progress_status && data.progress_status !== currentWaitingMessage) {
            currentWaitingMessage = data.progress_status;
            waitingStartedAt = Date.now();
          }
        } catch (e) {
          // 等待中的进度轮询失败不影响主流程。
        }
      }, 800);
    }

    function stopWaitingTimer() {
      if (waitingTimer) {
        clearInterval(waitingTimer);
        waitingTimer = null;
      }
      if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
      }
    }

    function waitingTextAfterAction(action) {
      if (!action) return '';
      if (action.type === 'day_speech' || action.type === 'pk_speech') {
        return action.next_speaker ? `${action.next_speaker}正在发言` : '其他玩家发言中';
      }
      if (action.type === 'day_vote' || action.type === 'pk_vote') {
        return '其他玩家投票中';
      }
      return '系统处理中';
    }

    function render(data, options = {}) {
      currentState = data;
      document.getElementById('summary').innerHTML = `第 ${data.round || 0} 轮 · ${data.phase || '未开始'}<br>你的身份：${data.human_role || '未知'}${data.winner ? '<br>' + data.winner : ''}`;
      document.getElementById('players').innerHTML = (data.players || []).map(p => `
        <div class="tile ${p.alive ? '' : 'dead'}">
          <div class="name">${p.name}${p.alive ? '' : '（出局）'}</div>
          <div class="role">${p.role}</div>
          ${p.is_human ? '<div class="human">真人玩家</div>' : ''}
        </div>`).join('');
      const privateLines = (data.private_log || []).map(x => `<div class="line private">${x}</div>`).join('');
      const publicLines = (data.log || []).map(x => `<div class="line">${x}</div>`).join('');
      document.getElementById('log').innerHTML = privateLines + publicLines || '<div class="meta">暂无日志</div>';
      const reviewEl = document.getElementById('review');
      if (data.winner && (data.review_events || []).length) {
        reviewEl.style.display = 'block';
        reviewEl.innerHTML = '<h2 style="margin-top:0">复盘</h2>' + data.review_events.map(x => `<div class="line">${x}</div>`).join('');
      } else {
        reviewEl.style.display = 'none';
        reviewEl.innerHTML = '';
      }
      if (!options.preserveAction) {
        document.getElementById('action').innerHTML = renderAction(data.pending_action);
      }
    }

    function renderAction(action) {
      if (!action) return '当前无需你操作。';
      const candidates = action.candidates || [];
      const options = candidates.map(x => `<option value="${x}">${x}</option>`).join('');
      if (action.type === 'werewolf_discussion') {
        return `<b>${action.message}</b><br><br>
          <textarea id="speech" placeholder="写下你的狼队夜聊策略，例如为什么刀这个人、白天如何伪装"></textarea><br><br>
          <select id="target">${options}</select><br><br>
          <button onclick="submitAction({speech: document.getElementById('speech').value, target: document.getElementById('target').value})">提交夜聊意见</button>`;
      }
      if (action.type === 'day_speech' || action.type === 'pk_speech') {
        return `<b>${action.message}</b><br><br><textarea id="speech" placeholder="输入你的发言"></textarea><br><br><button onclick="submitAction({speech: document.getElementById('speech').value})">提交发言</button>`;
      }
      if (action.type === 'day_vote' || action.type === 'pk_vote') {
        const abstainRemaining = action.abstain_remaining || 0;
        return `<b>${action.message}</b><br><br>
          <div class="meta">弃权机会剩余：${abstainRemaining} 次 / 1 次</div><br>
          <select id="target">${options}</select><br><br>
          <button onclick="submitAction({target: document.getElementById('target').value})">提交投票</button>`;
      }
      if (action.type === 'witch_action') {
        const antidoteCount = action.antidote_remaining ?? (action.has_antidote ? 1 : 0);
        const poisonCount = action.has_poison ? 1 : 0;
        const antidoteHint = action.has_antidote ? '' : '<div class="meta">当前不能使用解药：解药已用完，或今晚无人死亡。</div><br>';
        return `<b>${action.message}</b><br><br>
          <div class="meta">解药剩余：${antidoteCount} 瓶 / 1 瓶；毒药剩余：${poisonCount} 瓶 / 1 瓶</div><br>
          ${antidoteHint}
          <label style="color:#475569"><input type="checkbox" id="antidote" ${action.has_antidote ? '' : 'disabled'}> 使用解药</label>
          <label style="color:#475569"><input type="checkbox" id="poison" ${action.has_poison ? '' : 'disabled'}> 使用毒药</label>
          <select id="target"><option value="">不选择目标</option>${options}</select><br><br>
          <button onclick="submitAction({use_antidote: document.getElementById('antidote').checked, use_poison: document.getElementById('poison').checked, target_name: document.getElementById('target').value})">提交行动</button>`;
      }
      if (action.type === 'hunter_shot') {
        return `<b>${action.message}</b><br><br><select id="target"><option value="">不开枪</option>${options}</select><br><br>
          <button onclick="submitAction({shoot: !!document.getElementById('target').value, target: document.getElementById('target').value})">提交</button>`;
      }
      return `<b>${action.message}</b><br><br><select id="target">${options}</select><br><br><button onclick="submitAction({target: document.getElementById('target').value})">提交</button>`;
    }
    refreshState();
  </script>
</body>
</html>
"""


@app.get("/")
def index() -> str:
    return render_template_string(PAGE, roles=list(GameRoles.ROLES.keys()))


@app.post("/api/start")
def start() -> Dict[str, Any]:
    global game
    data = request.get_json(force=True) or {}
    game = WerewolfGame(
        player_count=int(data.get("player_count", 6)),
        human_name=data.get("human_name") or "沈清安",
        human_role=data.get("human_role") or "村民",
    )
    game.step_until_input_or_end()
    return jsonify(game.snapshot())


@app.get("/api/state")
def state() -> Dict[str, Any]:
    if game is None:
        return jsonify({"phase": "未开始", "round": 0, "players": [], "log": [], "pending_action": None})
    return jsonify(game.snapshot())


@app.get("/api/progress")
def progress() -> Dict[str, Any]:
    if game is None:
        return jsonify({"progress_status": ""})
    return jsonify({"progress_status": game.progress_status})


@app.post("/api/action")
def action() -> Dict[str, Any]:
    if game is None:
        return jsonify({"error": "游戏尚未开始"}), 400
    data = request.get_json(force=True) or {}
    game.submit_human_action(data)
    return jsonify(game.snapshot())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
