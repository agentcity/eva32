#!/usr/bin/env python3
"""Подробная схема слоя 4 — что лежит в репозитории eva32.

Запуск: python3 docs/diagrams/draw_repo_layers.py (нужен Pillow).
Проверяет геометрию сама: блоки в кадре, текст не вылезает.
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo_layers.png")
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

W, H = 1800, 1240
INK, MUTED, LINE = "#1B2330", "#5A6473", "#C9D0DA"
ACCENT, REPO, WARN = "#2F6FEB", "#0E7C5A", "#C77700"

img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)
f_h = ImageFont.truetype(BOLD, 44)
f_sub = ImageFont.truetype(REG, 22)
f_t = ImageFont.truetype(BOLD, 24)
f_b = ImageFont.truetype(REG, 19)
f_s = ImageFont.truetype(REG, 17)
f_a = ImageFont.truetype(BOLD, 20)

problems = []


def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        probe = (cur + " " + w_).strip()
        if d.textlength(probe, font=font) <= maxw:
            cur = probe
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def box(x, y, w, h, title, body, fill="#FFFFFF", outline=LINE, title_color=INK, fs=f_t, fb=f_b):
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill, outline=outline, width=2)
    d.text((x + 18, y + 14), title, font=fs, fill=title_color)
    if d.textlength(title, font=fs) > w - 36:
        problems.append(f"заголовок не влезает: {title}")
    ty = y + 14 + fs.size + 10
    for line in wrap(body, fb, w - 36):
        if ty + fb.size > y + h - 10:
            problems.append(f"текст не влезает в блок «{title}»")
            break
        d.text((x + 18, ty), line, font=fb, fill=MUTED)
        ty += fb.size + 6


def arrow(x1, y1, x2, y2, color=ACCENT, width=3, dash=False):
    if dash:
        steps = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 12)
        for i in range(steps):
            if i % 2 == 0:
                t0, t1 = i / steps, (i + 0.6) / steps
                d.line([x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0,
                        x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1], fill=color, width=width)
    else:
        d.line([x1, y1, x2, y2], fill=color, width=width)
    if x1 == x2:
        d.polygon([[x2 - 9, y2 - 12], [x2 + 9, y2 - 12], [x2, y2]], fill=color)
    else:
        d.polygon([[x2 - 12, y2 - 9], [x2 - 12, y2 + 9], [x2, y2]], fill=color)


# --- рамка «этот репозиторий» ---
d.text((60, 40), "Слой 4 подробно: что лежит в eva32", font=f_h, fill=INK)
d.text((60, 96), "запрос приходит по контракту, уходит в модель; вокруг — проверка, стенд, прод и версии",
       font=f_sub, fill=MUTED)
FX, FY, FW, FH = 50, 150, 1700, 1030
d.rounded_rectangle([FX, FY, FX + FW, FY + FH], radius=20, fill="#FBFCFE", outline=REPO, width=3)
d.text((FX + 26, FY + 16), "eva32  ·  репозиторий бандла", font=f_a, fill=REPO)

# --- верхний поток: 5 блоков ---
Y1, BH1, BW, GAP = FY + 70, 150, 300, 32
X1 = FX + 26
flow = [
    ("Запрос", "из orioncode: system + «Контекст [n]» + вопрос", "#EEF3FB", ACCENT),
    ("prompts/system.ru.md", "персона, 5 правил, 3 примера", "#E9F7F1", REPO),
    ("config/generation.yaml", "temperature, top_p, max_tokens, stop", "#E9F7F1", REPO),
    ("vLLM + веса", "DeepSeek V4-Flash · 8×H100 · FP8 · 65k", "#FFF4E5", WARN),
    ("Ответ", "текст + ссылки [n]; usage для денег", "#EEF3FB", ACCENT),
]
for i, (t, b, fill, ol) in enumerate(flow):
    box(X1 + i * (BW + GAP), Y1, BW, BH1, t, b, fill=fill, outline=ol, fs=f_a, fb=f_s)
    if i:
        mid = X1 + i * (BW + GAP) - GAP + 2
        arrow(mid, Y1 + BH1 // 2, mid + GAP - 6, Y1 + BH1 // 2)

# бейдж про версию
bx = X1 + 1 * (BW + GAP)
d.rounded_rectangle([bx + 10, Y1 + BH1 + 12, bx + 2 * BW + GAP - 10, Y1 + BH1 + 52],
                    radius=12, fill="#EDF3FF", outline=ACCENT, width=2)
d.text((bx + 26, Y1 + BH1 + 22), "bundle = sha256(промпт + конфиг) — версия в каждом прогоне",
       font=f_s, fill=ACCENT)

# --- три группы: проверка / стенд / прод ---
Y2, BH2 = Y1 + BH1 + 100, 300
GW = (FW - 2 * 26 - 2 * 30) // 3
groups = [
    ("eval/  —  проверка", "golden.jsonl: 34 строки\nrun_eval.py: проверки поведения\nresults/: ответы + bundle\n--repeat: строка = все прогоны", "#F2EEFB"),
    ("dev/  —  стенд", "qwen2.5:3b (~2.5 ГБ) и 7b (~4.7 ГБ)\nmodels.lock: digest'ы моделей\nModelfile: персона в Ollama\nmake dev / make dev-stop", "#E9F7F1"),
    ("serving/  —  прод", "compose.yaml: vLLM, TP=8, FP8\nтолько 127.0.0.1 + ключ\ndocs/gpu-checklist.md: подъём на GPU\nревизия весов из .env", "#FFF4E5"),
]
for i, (t, b, fill) in enumerate(groups):
    gx = FX + 26 + i * (GW + 30)
    box(gx, Y2, GW, BH2, t, b, fill=fill, fs=f_a, fb=f_s)
    arrow(gx + GW // 2, Y1 + BH1 + 62, gx + GW // 2, Y2 - 8, color="#B9C2CF", width=3, dash=True)

# --- нижняя полоса: документы, Makefile, версии ---
Y3, BH3 = Y2 + BH2 + 40, 130
box(FX + 26, Y3, (FW - 52 - 30) // 2, BH3, "Документы",
    "MODEL_CARD.md · contract.md · roadmap.md\nREADME.md · docs/diagrams", fill="#FFFFFF", fs=f_a, fb=f_s)
box(FX + 26 + (FW - 52 - 30) // 2 + 30, Y3, (FW - 52 - 30) // 2, BH3, "Сборка и версии",
    "Makefile: status / bundle / dev / eval\nVERSION · CHANGELOG.md · .env.example", fill="#FFFFFF", fs=f_a, fb=f_s)

# --- проверки геометрии ---
for name, (x, y, w, h) in {
    "рамка": (FX, FY, FW, FH),
    "поток": (X1, Y1, 5 * BW + 4 * GAP, BH1),
    "группы": (FX + 26, Y2, 3 * GW + 2 * 30, BH2),
    "низ": (FX + 26, Y3, FW - 52, BH3),
}.items():
    if x < 0 or y < 0 or x + w > W or y + h > H:
        problems.append(f"блок «{name}» вне холста")

print("проверка:", "ok" if not problems else problems)
img.save(OUT)
print("saved", OUT, img.size)
