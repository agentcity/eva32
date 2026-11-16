from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1800, 1010
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)

REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, size)

INK, MUTED, ARROW = "#1F2937", "#6B7280", "#374151"
BOXES = []

def text_center(cx, cy, lines, size, bold=False, fill=INK, gap=6):
    font = f(size, bold)
    hs = []
    for ln in lines:
        bb = d.textbbox((0, 0), ln, font=font)
        hs.append(bb[3] - bb[1])
    total = sum(hs) + gap * (len(lines) - 1)
    y = cy - total / 2
    for ln, h in zip(lines, hs):
        bb = d.textbbox((0, 0), ln, font=font)
        d.text((cx - (bb[2] - bb[0]) / 2, y), ln, font=font, fill=fill)
        y += h + gap

def box(x1, y1, x2, y2, title, sub=None, fill="#FFFFFF", border="#94A3B8",
        tsize=21, ssize=16, tcolor=INK, scolor=MUTED, radius=12):
    d.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=border, width=2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if sub:
        text_center(cx, cy - 11, title.split("\n"), tsize, True, tcolor)
        text_center(cx, cy + 33, sub.split("\n"), ssize, False, scolor)
    else:
        text_center(cx, cy, title.split("\n"), tsize, True, tcolor)
    BOXES.append((x1, y1, x2, y2, title.split("\n")[0]))

def zone(x1, y1, x2, y2, label, fill, border, badge=None):
    d.rounded_rectangle([x1, y1, x2, y2], radius=18, fill=fill, outline=border, width=2)
    d.text((x1 + 22, y1 + 13), label, font=f(17, True), fill=border)
    if badge:
        bb = d.textbbox((0, 0), badge, font=f(13, True))
        bw = bb[2] - bb[0] + 18
        d.rounded_rectangle([x2 - bw - 14, y1 + 8, x2 - 14, y1 + 34], radius=9, fill=border)
        d.text((x2 - bw - 5, y1 + 12), badge, font=f(13, True), fill="#FFFFFF")

def draw_dashed(x1, y1, x2, y2, width, color, dash=9, gap=6):
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0: return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    t = 0
    while t < length:
        t2 = min(t + dash, length)
        d.line([x1 + dx * t, y1 + dy * t, x1 + dx * t2, y1 + dy * t2], fill=color, width=width)
        t = t2 + gap

def arrow(points, dashed=False, width=3, color=ARROW, head=13):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if dashed: draw_dashed(x1, y1, x2, y2, width, color)
        else: d.line([x1, y1, x2, y2], fill=color, width=width)
    # collision check
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        n = int(math.hypot(x2 - x1, y2 - y1) // 4)
        for i in range(n + 1):
            px = x1 + (x2 - x1) * i / max(n, 1)
            py = y1 + (y2 - y1) * i / max(n, 1)
            for bx1, by1, bx2, by2, name in BOXES:
                if bx1 + 3 < px < bx2 - 3 and by1 + 3 < py < by2 - 3:
                    print(f"WARN: arrow hits '{name}' at ({px:.0f},{py:.0f})")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    ang = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - head * math.cos(ang - 0.42), y2 - head * math.sin(ang - 0.42))
    p2 = (x2 - head * math.cos(ang + 0.42), y2 - head * math.sin(ang + 0.42))
    d.polygon([(x2, y2), p1, p2], fill=color)

def label(x, y, s, size=13, color=MUTED, anchor="mm"):
    d.text((x, y), s, font=f(size), fill=color, anchor=anchor)

# zones
zone(40, 80, 380, 470, "ИНТЕРНЕТ", "#F2F7FC", "#4B7FB5")
zone(430, 80, 1190, 790, "РОССИЯ · КОНТУР ORIONCODE", "#F0F8F0", "#5B9A5B", "152-ФЗ")
zone(1230, 80, 1760, 790, "РОССИЯ · GPU-СЕРВЕР (COLOCATION)", "#FDF3E7", "#BF8040", "152-ФЗ")

# boxes
box(70, 160, 350, 290, "Клиент\nReact-чат", "HTTPS + SSE", "#FFFFFF", "#4B7FB5")
box(470, 150, 880, 310, "Symfony 8.4", "чат · auth · промпт\ntools · лимиты · учёт",
    "#FFFFFF", "#5B9A5B", tsize=22, ssize=17)
box(930, 150, 1150, 270, "Postgres", "организации, роли\nдокументы, деньги",
    "#FFFFFF", "#5B9A5B", tsize=20, ssize=16)
box(930, 310, 1150, 420, "Redis", "rate limit\nочередь ai.doc.*",
    "#FFFFFF", "#5B9A5B", tsize=20, ssize=16)
box(470, 470, 790, 610, "Go RAG-сервис", "/v1/retrieve · ingest",
    "#FFFFFF", "#5B9A5B", tsize=22, ssize=17)
box(930, 470, 1150, 610, "Postgres + pgvector", "чанки · эмбеддинги\nRLS по org_id",
    "#FFFFFF", "#5B9A5B", tsize=20, ssize=16)
box(930, 650, 1150, 760, "bge-m3 (TEI)", "эмбеддинги по HTTP",
    "#FFFFFF", "#5B9A5B", tsize=20, ssize=16)
box(470, 660, 790, 760, "Fallback: Qwen3.5 35B", "на Ryzen AI Halo · если vLLM лёг",
    "#F7FAF7", "#9DB89D", tsize=17, ssize=14)
box(1270, 160, 1720, 300, "vLLM · OpenAI API", "DeepSeek V4-Flash · FP8 · 32–64k",
    "#FFFFFF", "#BF8040", tsize=22, ssize=17)
box(1270, 340, 1720, 470, "NVMe", "веса 167 ГБ · KV-кэш",
    "#FFFFFF", "#BF8040", tsize=20, ssize=16)
box(1270, 510, 1720, 630, "железо под инференс", "8×H100 80 ГБ · ~4–5 кВт",
    "#F7FAF7", "#9DB89D", tsize=18, ssize=15)

# arrows
arrow([(350, 225), (470, 225)])
label(410, 208, "HTTPS, SSE")

arrow([(675, 150), (675, 120), (1495, 120), (1495, 160)])
label(940, 140, "OpenAI-совместимый API · шлюз: JWT / mTLS · allowlist")

arrow([(880, 210), (930, 210)])
label(905, 194, "SQL", 12)

arrow([(880, 285), (905, 285), (905, 365), (930, 365)])
label(898, 330, "лимиты", 12, anchor="rm")

arrow([(675, 310), (675, 470)])
label(690, 390, "POST /v1/retrieve\n≤ 400 мс", 13, anchor="lm")

arrow([(790, 520), (930, 520)])
label(860, 503, "SQL + RLS", 12)

arrow([(790, 590), (860, 590), (860, 705), (930, 705)])
label(868, 645, "embed", 12, anchor="lm")

arrow([(1040, 420), (1040, 448), (630, 448), (630, 470)], dashed=True, width=2)
label(835, 434, "события ai.doc.* → ingest забирает документ", 12)


arrow([(1495, 300), (1495, 340)])
label(1512, 320, "веса / KV", 12, anchor="lm")

arrow([(1495, 470), (1495, 510)])

# title
d.text((40, 26), "EVA32 — как чат orioncode разговаривает с моделью", font=f(29, True), fill=INK)
d.text((40, 62), "RAG на Go · инференс DeepSeek V4-Flash на своём GPU · все данные в России", font=f(15), fill=MUTED)

# legend
ly = 830
d.line([60, ly, 118, ly], fill=ARROW, width=3)
d.polygon([(118, ly), (105, ly - 6), (105, ly + 6)], fill=ARROW)
d.text((130, ly - 9), "синхронный вызов: retrieve ≤ 400 мс, ответ модели — секунды", font=f(14), fill=INK)
d.line([60, ly + 38, 118, ly + 38], fill=ARROW, width=2)
d.text((130, ly + 29), "асинхронно: события ingest и внутренний API документов", font=f(14), fill=INK)
d.text((60, ly + 76), "Go не решает, кому какой документ виден: скоуп приходит из Symfony, RLS в Postgres — второй рубеж.", font=f(14), fill=MUTED)
d.text((60, ly + 104), "vLLM наружу не смотрит: только приватная сеть/VPN, публичный вход — через Symfony.", font=f(14), fill=MUTED)

img.save("/Users/a/Documents/work/my/eva32/eva/architecture.png")
print("saved", img.size, "boxes:", len(BOXES))
