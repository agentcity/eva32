#!/usr/bin/env python3
"""Схема слоёв EVA32. Запуск: python3 docs/diagrams/draw_layers.py (нужен Pillow)."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layers.png")
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

W, H = 1680, 1260
BG = "#FFFFFF"
INK = "#1B2330"
MUTED = "#5A6473"
ACCENT = "#2F6FEB"
REPO = "#0E7C5A"

LAYERS = [
    ("1", "Клиент", "чат orioncode: веб, мобильное приложение, каналы", "#EEF3FB", ACCENT, ""),
    ("2", "Приложение", "Symfony + RoadRunner: вход, организации, лимиты, деньги, история", "#EEF3FB", ACCENT, ""),
    ("3", "Контракт", "/v1/chat/completions · system + «Контекст [n]» + stream + usage", "#FFF4E5", "#C77700", ""),
    ("4", "Модель и промпт", "DeepSeek V4-Flash на vLLM (8×H100) · prompts/system.ru.md · config/generation.yaml",
     "#E9F7F1", REPO, "этот репозиторий"),
    ("5", "Знания и действия", "RAG: bge-m3 → pgvector (RLS) · инструменты: каталог, запись, статус", "#F2EEFB", "#6B45C7", ""),
]

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
f_title = ImageFont.truetype(BOLD, 46)
f_sub = ImageFont.truetype(REG, 24)
f_num = ImageFont.truetype(BOLD, 30)
f_name = ImageFont.truetype(BOLD, 30)
f_desc = ImageFont.truetype(REG, 21)
f_badge = ImageFont.truetype(BOLD, 18)
f_side = ImageFont.truetype(BOLD, 24)
f_side_i = ImageFont.truetype(REG, 20)

d.text((80, 52), "EVA32 — слои системы", font=f_title, fill=INK)
d.text((80, 112), "запрос идёт сверху вниз, ответ — обратно; слой 4 живёт в репозитории eva32",
       font=f_sub, fill=MUTED)

X, WID = 80, 1120
Y0, BH, GAP = 170, 148, 44
for i, (num, name, desc, fill, line, badge) in enumerate(LAYERS):
    y = Y0 + i * (BH + GAP)
    d.rounded_rectangle([X, y, X + WID, y + BH], radius=18, fill=fill, outline=line, width=3)
    d.ellipse([X + 26, y + 52, X + 82, y + 108], fill=line)
    tw = d.textlength(num, font=f_num)
    d.text((X + 54 - tw / 2, y + 62), num, font=f_num, fill="#FFFFFF")
    d.text((X + 106, y + 26), name, font=f_name, fill=INK)
    d.text((X + 106, y + 72), desc, font=f_desc, fill=MUTED)
    if badge:
        bw = d.textlength(badge, font=f_badge) + 28
        d.rounded_rectangle([X + WID - bw - 24, y + 22, X + WID - 24, y + 56],
                            radius=17, fill=REPO)
        d.text((X + WID - bw - 10, y + 30), badge, font=f_badge, fill="#FFFFFF")
    if i < len(LAYERS) - 1:
        ax = X + 54
        ay = y + BH + 8
        d.line([ax, ay, ax, ay + GAP - 16], fill=ACCENT, width=4)
        d.polygon([[ax - 9, ay + GAP - 22], [ax + 9, ay + GAP - 22], [ax, ay + GAP - 6]], fill=ACCENT)
        bx = X + 96
        d.line([bx, ay + GAP - 16, bx, ay + 2], fill="#B9C2CF", width=3)
        d.polygon([[bx - 7, ay + 12], [bx + 7, ay + 12], [bx, ay]], fill="#B9C2CF")

d.text((X + 200, Y0 + BH + 10), "запрос", font=f_badge, fill=ACCENT)
d.text((X + 200, Y0 + BH + 32), "ответ", font=f_badge, fill="#8A94A3")

SX, SW = 1250, 350
SY = Y0
SH = 5 * BH + 4 * GAP
d.rounded_rectangle([SX, SY, SX + SW, SY + SH], radius=18, fill="#F7F8FA", outline="#C9D0DA", width=2)
d.text((SX + 26, SY + 24), "Эксплуатация", font=f_side, fill=INK)
d.text((SX + 26, SY + 58), "сквозной слой", font=f_side_i, fill=MUTED)
items = [
    "версии: bundle + prompt_sha256",
    "мониторинг: латентность, отказы",
    "лимиты и стоимость диалога",
    "деградация при отказе модели",
    "безопасность: инъекции, ПД",
    "логи без персональных данных",
]
iy = SY + 110
for it in items:
    d.ellipse([SX + 26, iy + 8, SX + 34, iy + 16], fill=ACCENT)
    d.text((SX + 46, iy), it, font=f_side_i, fill=MUTED)
    iy += 44

d.text((80, H - 60), "1–3, 5 — orioncode · 4 — eva32 · эксплуатация пронизывает всё",
       font=f_side_i, fill="#8A94A3")

# самопроверка геометрии — без глаз, по числам
problems = []
bottom = Y0 + len(LAYERS) * BH + (len(LAYERS) - 1) * GAP
if bottom > H - 90:
    problems.append(f"слои упираются в низ: {bottom} > {H - 90}")
for num, name, desc, *_ in LAYERS:
    if d.textlength(desc, font=f_desc) > WID - 150:
        problems.append(f"описание слоя {num} шире блока")
    if d.textlength(name, font=f_name) > WID - 200:
        problems.append(f"имя слоя {num} не влезает")
if SY + SH > H - 90:
    problems.append(f"боковая панель упирается в низ: {SY + SH}")
print("проверка:", "ok" if not problems else problems)
img.save(OUT)
print("saved", OUT, img.size)
