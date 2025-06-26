#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
国标麻将（Chinese Standard Mahjong）随机策略示例（Botzone 协议）
改写自原 C++ 代码，保持与原实现一致的逻辑。
"""
import sys
import json
import random

# 是否使用简单交互（纯文本）而非 JSON。跟原版保持一致，默认 0=JSON 1=简单
SIMPLEIO = 1


def read_simple_io():
    """
    按简单交互格式读取输入，返回:
    turn_id, requests(list[str]), responses(list[str])
    """
    line = sys.stdin.readline()
    if not line:
        return 0, [], []
    turn_id = int(line.strip()) - 1  # 与原 C++ 行为一致（--）
    requests = []
    responses = []

    # 读取历史请求/响应
    for _ in range(turn_id):
        requests.append(sys.stdin.readline().rstrip('\n'))
        responses.append(sys.stdin.readline().rstrip('\n'))

    # 当前请求
    requests.append(sys.stdin.readline().rstrip('\n'))
    return turn_id, requests, responses


def build_initial_hand(request0: str, request1: str):
    """
    根据前两条请求构造初始手牌。
    request0: "<itmp> <playerID> <quan>"
    request1: "<...>"（包含 5 个整数 + 13 张手牌）
    返回 list[str] hand
    """
    parts = request1.split()
    # 跳过前 5 个整数
    return parts[5:5 + 13]


def update_hand_from_history(hand, request: str, response: str):
    """
    根据历史一条 request/response 更新手牌。
    当 request 类型 itmp == 2 时，会摸牌；如果随后自己打出了牌，需要从手牌中移除。
    """
    tokens = request.split()
    if not tokens:
        return
    try:
        itmp = int(tokens[0])
    except ValueError:
        return
    if itmp == 2:
        # 摸牌
        if len(tokens) > 1:
            hand.append(tokens[1])

        # 查看自己回应，如果是 PLAY 则移除打出的牌
        resp_tokens = response.split()
        if len(resp_tokens) >= 2 and resp_tokens[0] == "PLAY":
            tile = resp_tokens[1]
            if tile in hand:
                hand.remove(tile)


def decide_action(hand, current_request: str):
    """
    根据当前请求决定行动。
    如果 itmp == 2（摸牌），随机打一张牌；否则 PASS
    """
    tokens = current_request.split()
    if not tokens:
        return "PASS"
    try:
        itmp = int(tokens[0])
    except ValueError:
        return "PASS"

    if itmp == 2 and hand:
        random.shuffle(hand)
        tile = hand.pop()
        return f"PLAY {tile}"
    else:
        return "PASS"


def main():
    if SIMPLEIO:
        turn_id, requests, responses = read_simple_io()
    else:
        assert 0
        
    # 简单策略
    if turn_id < 2:
        reply = "PASS"
    else:
        hand = build_initial_hand(requests[0], requests[1])

        # 回顾历史，维护手牌
        for i in range(2, turn_id):
            if i < len(responses):
                update_hand_from_history(hand, requests[i], responses[i])

        # 决策
        reply = decide_action(hand, requests[turn_id])

    # 输出
    if SIMPLEIO:
        print(reply)
    else:
        print(json.dumps({"response": reply}))


if __name__ == "__main__":
    main()