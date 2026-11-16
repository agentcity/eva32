from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1900, 900
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
def f(s, b=False): return ImageFont.truetype(BOLD if b else REG, s)

INK, MUTED, ARROW = "#1F2937", "#6B7280", "#374151"
BOXES, BADGES, LABELS = [], [], []

def tc(cx, cy, lines, size, bold=False, fill=INK, gap=5):
    font=f(size,bold); hs=[]
    for ln in lines:
        bb=d.textbbox((0,0),ln,font=font); hs.append(bb[3]-bb[1])
    total=sum(hs)+gap*(len(lines)-1); y=cy-total/2
    for ln,h in zip(lines,hs):
        bb=d.textbbox((0,0),ln,font=font)
        d.text((cx-(bb[2]-bb[0])/2,y),ln,font=font,fill=fill); y+=h+gap

def box(x1,y1,x2,y2,title,sub=None,fill="#FFFFFF",border="#94A3B8",ts=18,ss=13,tc_=INK,sc=MUTED,radius=10):
    d.rounded_rectangle([x1,y1,x2,y2],radius=radius,fill=fill,outline=border,width=2)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    for ln in title.split("\n"):
        w=d.textbbox((0,0),ln,font=f(ts,True))[2]
        if w>(x2-x1)-12: print(f"WARN title overflow '{ln}'")
    if sub:
        for ln in sub.split("\n"):
            w=d.textbbox((0,0),ln,font=f(ss))[2]
            if w>(x2-x1)-12: print(f"WARN sub overflow '{ln}'")
        tc(cx,cy-9,title.split("\n"),ts,True,tc_)
        tc(cx,cy+27,sub.split("\n"),ss,False,sc)
    else:
        tc(cx,cy,title.split("\n"),ts,True,tc_)
    BOXES.append((x1,y1,x2,y2,title.split("\n")[0]))

def zone(x1,y1,x2,y2,label,fill,border,note=None,badge=None):
    d.rounded_rectangle([x1,y1,x2,y2],radius=14,fill=fill,outline=border,width=2)
    d.text((x1+18,y1+10),label,font=f(15,True),fill=border)
    if note: d.text((x1+18,y1+30),note,font=f(11),fill=MUTED)
    if badge:
        bb=d.textbbox((0,0),badge,font=f(11,True)); bw=bb[2]-bb[0]+14
        d.rounded_rectangle([x2-bw-10,y1+6,x2-10,y1+28],radius=7,fill=border)
        BADGES.append((x2-bw-10,y1+6,x2-10,y1+28,badge))
        d.text((x2-bw-3,y1+9),badge,font=f(11,True),fill="#FFFFFF")

def dashed_line(x1,y1,x2,y2,color,width=2,dash=8,gap=6):
    L=math.hypot(x2-x1,y2-y1)
    if L==0: return
    dx,dy=(x2-x1)/L,(y2-y1)/L; t=0
    while t<L:
        t2=min(t+dash,L); d.line([x1+dx*t,y1+dy*t,x1+dx*t2,y1+dy*t2],fill=color,width=width); t=t2+gap

def arrow(points,width=3,color=ARROW,head=11,dashed=False,both=False):
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        if dashed: dashed_line(x1,y1,x2,y2,color,width)
        else: d.line([x1,y1,x2,y2],fill=color,width=width)
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        n=int(math.hypot(x2-x1,y2-y1)//4)
        for i in range(n+1):
            px=x1+(x2-x1)*i/max(n,1); py=y1+(y2-y1)*i/max(n,1)
            for bx1,by1,bx2,by2,name in BOXES:
                if bx1+2<px<bx2-2 and by1+2<py<by2-2:
                    print(f"WARN arrow hits '{name}' at ({px:.0f},{py:.0f})")
    def head_at(p,q):
        ang=math.atan2(p[1]-q[1],p[0]-q[0])
        d.polygon([p,(p[0]-head*math.cos(ang-0.42),p[1]-head*math.sin(ang-0.42)),
                     (p[0]-head*math.cos(ang+0.42),p[1]-head*math.sin(ang+0.42))],fill=color)
    head_at(points[-1],points[-2])
    if both: head_at(points[0],points[1])

def label(x,y,s,size=11,color=MUTED,anchor="mm"):
    bb=d.textbbox((x,y),s,font=f(size),anchor=anchor); LABELS.append((*bb,s))
    d.text((x,y),s,font=f(size),fill=color,anchor=anchor)

# zones
zone(40,90,340,300,"ИНТЕРНЕТ","#F2F7FC","#4B7FB5")
zone(380,90,720,300,"ПУБЛИЧНЫЙ EDGE · VPS-1","#F7F9F4","#5B9A5B","2 vCPU · 4 ГБ · ~3–5 тыс. ₽/мес")
zone(760,90,1120,300,"APP · VPS-2","#F0F8F0","#4E8B4E","8 vCPU · 32 ГБ · ~10–15 тыс. ₽/мес","152-ФЗ")
zone(60,380,500,780,"ДАННЫЕ · MANAGED","#F8F6EF","#A8843C","Postgres HA · ~15–30 тыс. ₽/мес")
zone(540,380,980,780,"RAG · VPS-3","#F3F0FA","#6C5AA8","8 vCPU · 32 ГБ · ~10–20 тыс. ₽/мес","152-ФЗ")
zone(1020,380,1460,780,"GPU · COLOCATION","#FDF3E7","#BF8040","colo 5 кВт ~170 тыс. ₽/мес + железо","152-ФЗ")
zone(1500,380,1860,780,"OPS · VPS-4","#F5F5F5","#64748B","4 vCPU · 8 ГБ · ~5 тыс. ₽/мес")

# internet
box(60,140,320,210,"Пользователи","браузер + мобильные","#FFFFFF","#4B7FB5",ts=16,ss=12)
box(60,225,320,285,"Админка","загрузка документов","#FFFFFF","#4B7FB5",ts=16,ss=12)

# edge
box(400,140,700,280,"nginx / LB","публичный IP :443 · TLS 1.3\nWAF · rate limit · SSE без буфера","#FFFFFF","#5B9A5B",ts=17,ss=12)

# app
box(780,140,1100,280,"RoadRunner + Symfony","воркеры · чат · auth · промпт\ntools · лимиты · учёт · outbox relay","#FFFFFF","#4E8B4E",ts=17,ss=12)

# data
box(80,425,480,525,"Postgres · бизнес","организации, роли, документы\nденьги, outbox · бэкап ежедневно","#FFFFFF","#A8843C",ts=17,ss=12)
box(80,545,480,635,"Redis","Stream · consumer group\nrate limit","#FFFFFF","#A8843C",ts=16,ss=12)
box(80,655,480,735,"S3 / MinIO","оригиналы документов","#FFFFFF","#A8843C",ts=16,ss=12)

# rag
box(560,425,960,515,"Go · /v1/retrieve","скоуп из Symfony · ≤ 400 мс","#FFFFFF","#6C5AA8",ts=17,ss=12)
box(560,535,760,675,"Go · ingest","consumer group\nXACK · retry · DLQ","#FFFFFF","#6C5AA8",ts=16,ss=11)
box(780,535,960,675,"Postgres + pgvector","чанки · RLS · HNSW\nлокально, не по сети","#FFFFFF","#6C5AA8",ts=15,ss=11)

# gpu
box(1040,425,1440,545,"vLLM · OpenAI API","DeepSeek V4-Flash · FP8\n32–64k · :8000 mTLS","#FFFFFF","#BF8040",ts=17,ss=12)
box(1040,565,1440,635,"bge-m3 · :8081","эмбеддинги и реранк","#FFFFFF","#BF8040",ts=16,ss=12)
box(1040,655,1440,715,"NVMe","веса 167 ГБ · KV-кэш","#FFFFFF","#BF8040",ts=15,ss=12)
box(1040,735,1440,775,"8×H100 80 ГБ","~4–5 кВт","#FFFFFF","#BF8040",ts=14,ss=11)

# ops
box(1520,425,1840,515,"CI / eval","золотой набор · гейт на релиз","#FFFFFF","#64748B",ts=16,ss=11)
box(1520,535,1840,625,"monitoring","Prometheus · Grafana · Loki","#FFFFFF","#64748B",ts=16,ss=11)
box(1520,645,1840,735,"backup · VPN","WireGuard · бэкапы","#FFFFFF","#64748B",ts=16,ss=11)

# private network band
d.rounded_rectangle([60,318,1840,358],radius=10,fill="#EEF2F7",outline="#94A3B8",width=2)
d.text((80,330),"ПРИВАТНАЯ СЕТЬ 10.10.0.0/24 · WireGuard / VPC · публично доступен только :443 на VPS-1",font=f(13,True),fill="#475569")

# arrows
arrow([(320,175),(380,175)]); label(350,158,"443")
arrow([(700,190),(780,190)]); label(740,173,"8080")

arrow([(930,300),(930,318)])
label(940,309,"10.10.0.2",10,anchor="lm")

arrow([(280,358),(280,380)]); label(292,369,"5432 · 6379",10,anchor="lm")
arrow([(760,358),(760,380)]); label(772,369,"8080",10,anchor="lm")
arrow([(1240,358),(1240,380)]); label(1252,369,"8000 mTLS",10,anchor="lm")
arrow([(1680,380),(1680,358)],both=True); label(1692,369,"9100 scrape",10,anchor="lm")

arrow([(860,515),(860,535)]); label(872,525,"SQL",10,anchor="lm")
arrow([(760,600),(780,600)])
arrow([(660,675),(660,700),(1000,700),(1000,600),(1040,600)])
label(1012,588,"8081",10,anchor="lm")

# title
d.text((40,24),"EVA32: где что физически располагается",font=f(28,True),fill=INK)
d.text((40,60),"Публично — только nginx. Всё остальное — приватная сеть: VPS-2 ↔ VPS-3 ↔ GPU-сервер.",font=f(14),fill=MUTED)

# legend
d.text((60,800),"Порты: 443 — вход, 8080 — app/rag, 5432 — Postgres, 6379 — Redis, 8000 — vLLM, 8081 — эмбеддинги, 9100 — метрики.",font=f(12),fill=MUTED)
d.text((60,824),"MVP-упрощение: VPS-2 и VPS-3 можно слить в одну машину; managed Postgres заменить локальным — но границы отказов станут общими.",font=f(12),fill=MUTED)

def overlap(a,b): return not (a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1])
for lx1,ly1,lx2,ly2,ls in LABELS:
    for bx1,by1,bx2,by2,bn in BOXES:
        if overlap((lx1,ly1,lx2,ly2),(bx1+1,by1+1,bx2-1,by2-1)):
            print(f"WARN label '{ls}' overlaps box '{bn}'")
    for bx1,by1,bx2,by2,bn in BADGES:
        if overlap((lx1,ly1,lx2,ly2),(bx1,by1,bx2,by2)):
            print(f"WARN label '{ls}' overlaps badge '{bn}'")
for i in range(len(LABELS)):
    for j in range(i+1,len(LABELS)):
        if overlap(LABELS[i][:4],LABELS[j][:4]):
            print(f"WARN labels overlap: '{LABELS[i][4]}' / '{LABELS[j][4]}'")
img.save("/Users/a/Documents/work/my/eva32/eva/topology.png")
print("saved",img.size,"boxes",len(BOXES))
