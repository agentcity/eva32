from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1900, 980
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

def box(x1,y1,x2,y2,title,sub=None,fill="#FFFFFF",border="#94A3B8",ts=17,ss=12,tc_=INK,sc=MUTED):
    d.rounded_rectangle([x1,y1,x2,y2],radius=10,fill=fill,outline=border,width=2)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    for ln in title.split("\n"):
        if d.textbbox((0,0),ln,font=f(ts,True))[2]>(x2-x1)-12: print("WARN title overflow",ln)
    if sub:
        for ln in sub.split("\n"):
            if d.textbbox((0,0),ln,font=f(ss))[2]>(x2-x1)-12: print("WARN sub overflow",ln)
        tc(cx,cy-8,title.split("\n"),ts,True,tc_)
        tc(cx,cy+25,sub.split("\n"),ss,False,sc)
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

def arrow(points,width=3,color=ARROW,head=11,both=False):
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        d.line([x1,y1,x2,y2],fill=color,width=width)
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
zone(40,90,330,360,"ИНТЕРНЕТ","#F2F7FC","#4B7FB5")
zone(380,90,1050,500,"VPS-1 · EDGE + APP","#F0F8F0","#4E8B4E","8 vCPU · 32 ГБ · ~15–20 тыс. ₽/мес","152-ФЗ")
zone(1090,90,1490,500,"ДАННЫЕ · MANAGED","#F8F6EF","#A8843C","Postgres HA · Redis · S3 · ~15–30 тыс. ₽/мес")
zone(380,580,1050,900,"GPU · COLOCATION","#FDF3E7","#BF8040","colo 5 кВт ~170 тыс. ₽/мес + железо","152-ФЗ")
zone(1090,580,1490,900,"RAG · VPS-2","#F3F0FA","#6C5AA8","8 vCPU · 32 ГБ · ~10–20 тыс. ₽/мес","152-ФЗ")
zone(1530,580,1860,900,"OPS · VPS-3","#F5F5F5","#64748B","4 vCPU · 8 ГБ · ~5 тыс. ₽/мес")

# internet
box(60,140,310,215,"Пользователи","браузер + мобильные","#FFFFFF","#4B7FB5",ts=16)
box(60,235,310,300,"Админка","загрузка документов","#FFFFFF","#4B7FB5",ts=16)

# vps1
box(400,140,730,245,"nginx",":443 · TLS 1.3 · WAF\nлимиты · SSE без буфера","#FFFFFF","#4E8B4E",ts=17)
box(400,275,1030,400,"RoadRunner + Symfony","воркеры · чат · auth · промпт\ntools · лимиты · учёт токенов","#FFFFFF","#4E8B4E",ts=18,ss=13)
box(400,425,1030,485,"outbox relay","публикует события в Redis Stream","#FFFFFF","#4E8B4E",ts=15,ss=11)

# data
box(1110,140,1470,250,"Postgres · бизнес","организации, роли, документы\nденьги, outbox · бэкап","#FFFFFF","#A8843C",ts=16,ss=12)
box(1110,275,1470,355,"Redis","Stream · consumer group\nrate limit","#FFFFFF","#A8843C",ts=16,ss=12)
box(1110,380,1470,470,"S3 / MinIO","оригиналы документов","#FFFFFF","#A8843C",ts=16,ss=12)

# rag
box(1110,620,1470,710,"Go · /v1/retrieve","скоуп из Symfony · ≤ 400 мс","#FFFFFF","#6C5AA8",ts=16,ss=12)
box(1110,730,1280,850,"Go · ingest","consumer group\nXACK · retry · DLQ","#FFFFFF","#6C5AA8",ts=15,ss=10)
box(1300,730,1470,850,"pgvector","RLS · HNSW\nлокальная БД","#FFFFFF","#6C5AA8",ts=15,ss=10)

# gpu
box(400,620,1030,710,"vLLM · OpenAI API","DeepSeek V4-Flash · FP8 · 32–64k · :8000 mTLS","#FFFFFF","#BF8040",ts=17,ss=12)
box(400,730,1030,790,"bge-m3 · :8081","эмбеддинги и реранк","#FFFFFF","#BF8040",ts=16,ss=12)
box(400,805,700,870,"NVMe","веса 167 ГБ · KV-кэш","#FFFFFF","#BF8040",ts=15,ss=11)
box(720,805,1030,870,"8×H100 80 ГБ","~4–5 кВт","#FFFFFF","#BF8040",ts=15,ss=11)

# ops
box(1550,620,1840,710,"CI / eval","золотой набор · гейт на релиз","#FFFFFF","#64748B",ts=15,ss=11)
box(1550,730,1840,810,"monitoring","Prometheus · Grafana · Loki","#FFFFFF","#64748B",ts=15,ss=11)
box(1550,830,1840,890,"backup · VPN","WireGuard · бэкапы","#FFFFFF","#64748B",ts=15,ss=11)

# private network band
d.rounded_rectangle([60,520,1840,560],radius=10,fill="#EEF2F7",outline="#94A3B8",width=2)
d.text((80,532),"ПРИВАТНАЯ СЕТЬ 10.10.0.0/24 · WireGuard / VPC · публично доступен только :443 на VPS-1",font=f(13,True),fill="#475569")

# arrows
arrow([(310,175),(400,175)]); label(355,158,"443")
arrow([(565,245),(565,275)]); label(578,260,"127.0.0.1:8080",10,anchor="lm")
arrow([(565,400),(565,425)]); label(578,412,"outbox",10,anchor="lm")

arrow([(900,500),(900,520)],both=True); label(912,510,"10.10.0.1",10,anchor="lm")
arrow([(1300,500),(1300,520)],both=True); label(1312,510,"5432 · 6379",10,anchor="lm")
arrow([(1300,560),(1300,580)],both=True); label(1312,570,"8080",10,anchor="lm")
arrow([(700,560),(700,580)],both=True); label(712,570,"8000 · 8081",10,anchor="lm")
arrow([(1700,560),(1700,580)],both=True); label(1712,570,"9100",10,anchor="lm")

arrow([(1385,710),(1385,730)]); label(1397,720,"SQL",10,anchor="lm")
arrow([(1280,790),(1300,790)])
arrow([(1200,850),(1200,860),(1080,860),(1080,760),(1030,760)])
label(1074,800,"8081",10,anchor="rm")

# title
d.text((40,24),"EVA32: четыре машины — где что стоит",font=f(28,True),fill=INK)
d.text((40,60),"nginx и приложение — на одной VPS. Публично только :443, остальное — приватная сеть.",font=f(14),fill=MUTED)

# legend
d.text((60,920),"Порты: 443 вход · 8080 app/rag · 5432 Postgres · 6379 Redis · 8000 vLLM · 8081 эмбеддинги · 9100 метрики.",font=f(12),fill=MUTED)
d.text((60,944),"MVP: RAG можно держать на VPS-1, но тогда ingest конкурирует за CPU с чатом, а падение машины уносит и поиск.",font=f(12),fill=MUTED)

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

img.save("/Users/a/Documents/work/my/eva32/eva/topology_v2.png")
print("saved",img.size,"boxes",len(BOXES))
