from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1900, 1100
img = Image.new("RGB", (W, H), "#FFFFFF")
d = ImageDraw.Draw(img)
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
def f(s, b=False): return ImageFont.truetype(BOLD if b else REG, s)

INK, MUTED, ARROW = "#1F2937", "#6B7280", "#374151"
BOXES = []
BADGES = []
LABELS = []

def tc(cx, cy, lines, size, bold=False, fill=INK, gap=5):
    font = f(size, bold); hs=[]
    for ln in lines:
        bb=d.textbbox((0,0),ln,font=font); hs.append(bb[3]-bb[1])
    total=sum(hs)+gap*(len(lines)-1); y=cy-total/2
    for ln,h in zip(lines,hs):
        bb=d.textbbox((0,0),ln,font=font)
        d.text((cx-(bb[2]-bb[0])/2,y),ln,font=font,fill=fill); y+=h+gap

def box(x1,y1,x2,y2,title,sub=None,fill="#FFFFFF",border="#94A3B8",ts=19,ss=14,tc_=INK,sc=MUTED,radius=11,dashed=False):
    if dashed:
        d.rounded_rectangle([x1,y1,x2,y2],radius=radius,fill=fill,outline="#FFFFFF",width=0)
        dashed_rect(x1,y1,x2,y2,border)
    else:
        d.rounded_rectangle([x1,y1,x2,y2],radius=radius,fill=fill,outline=border,width=2)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    for ln in title.split("\n"):
        w=d.textbbox((0,0),ln,font=f(ts,True))[2]
        if w > (x2-x1)-14: print(f"WARN title overflow '{ln}' in {x1},{y1}")
    if sub:
        for ln in sub.split("\n"):
            w=d.textbbox((0,0),ln,font=f(ss))[2]
            if w > (x2-x1)-14: print(f"WARN sub overflow '{ln}' in {x1},{y1}")
        tc(cx,cy-10,title.split("\n"),ts,True,tc_)
        tc(cx,cy+30,sub.split("\n"),ss,False,sc)
    else:
        tc(cx,cy,title.split("\n"),ts,True,tc_)
    BOXES.append((x1,y1,x2,y2,title.split("\n")[0]))

def dashed_rect(x1,y1,x2,y2,color,width=2,dash=8,gap=6,r=11):
    dashed_line(x1+r,y1,x2-r,y1,color,width,dash,gap)
    dashed_line(x1+r,y2,x2-r,y2,color,width,dash,gap)
    dashed_line(x1,y1+r,x1,y2-r,color,width,dash,gap)
    dashed_line(x2,y1+r,x2,y2-r,color,width,dash,gap)

def dashed_line(x1,y1,x2,y2,color,width=2,dash=8,gap=6):
    L=math.hypot(x2-x1,y2-y1)
    if L==0: return
    dx,dy=(x2-x1)/L,(y2-y1)/L; t=0
    while t<L:
        t2=min(t+dash,L)
        d.line([x1+dx*t,y1+dy*t,x1+dx*t2,y1+dy*t2],fill=color,width=width)
        t=t2+gap

def zone(x1,y1,x2,y2,label,fill,border,badge=None):
    d.rounded_rectangle([x1,y1,x2,y2],radius=16,fill=fill,outline=border,width=2)
    d.text((x1+20,y1+12),label,font=f(16,True),fill=border)
    if badge:
        bb=d.textbbox((0,0),badge,font=f(12,True)); bw=bb[2]-bb[0]+16
        d.rounded_rectangle([x2-bw-12,y1+7,x2-12,y1+31],radius=8,fill=border)
        BADGES.append((x2-bw-12,y1+7,x2-12,y1+31,badge))
        d.text((x2-bw-4,y1+10),badge,font=f(12,True),fill="#FFFFFF")

def arrow(points,dashed=False,width=3,color=ARROW,head=12):
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        if dashed: dashed_line(x1,y1,x2,y2,color,width)
        else: d.line([x1,y1,x2,y2],fill=color,width=width)
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        n=int(math.hypot(x2-x1,y2-y1)//4)
        for i in range(n+1):
            px=x1+(x2-x1)*i/max(n,1); py=y1+(y2-y1)*i/max(n,1)
            for bx1,by1,bx2,by2,name in BOXES:
                if bx1+3<px<bx2-3 and by1+3<py<by2-3:
                    print(f"WARN arrow hits '{name}' at ({px:.0f},{py:.0f})")
    (x1,y1),(x2,y2)=points[-2],points[-1]
    ang=math.atan2(y2-y1,x2-x1)
    p1=(x2-head*math.cos(ang-0.42),y2-head*math.sin(ang-0.42))
    p2=(x2-head*math.cos(ang+0.42),y2-head*math.sin(ang+0.42))
    d.polygon([(x2,y2),p1,p2],fill=color)

def label(x,y,s,size=12,color=MUTED,anchor="mm"):
    font=f(size)
    bb=d.textbbox((x,y),s,font=font,anchor=anchor)
    LABELS.append((bb[0],bb[1],bb[2],bb[3],s))
    d.text((x,y),s,font=font,fill=color,anchor=anchor)

# zones
zone(40,80,340,470,"ИНТЕРНЕТ","#F2F7FC","#4B7FB5")
zone(380,80,1120,780,"РОССИЯ · КОНТУР ORIONCODE","#F0F8F0","#5B9A5B","152-ФЗ")
zone(1150,80,1480,780,"РОССИЯ · RAG (GO)","#F3F0FA","#6C5AA8","152-ФЗ")
zone(1510,80,1860,780,"РОССИЯ · GPU-СЕРВЕР","#FDF3E7","#BF8040","152-ФЗ")

# internet
box(65,150,315,290,"Клиент\nReact-чат","HTTPS + SSE","#FFFFFF","#4B7FB5")

# orioncode
box(410,150,700,230,"nginx","TLS · SSE без буфера","#FFFFFF","#5B9A5B",ts=18)
box(410,260,800,420,"RoadRunner + Symfony","чат · auth · промпт · tools\nлимиты · учёт токенов","#FFFFFF","#5B9A5B",ts=20,ss=15)
box(850,150,1090,290,"Postgres · бизнес","организации, роли\nдокументы, деньги\noutbox","#FFFFFF","#5B9A5B",ts=18,ss=14)
box(850,320,1090,390,"outbox-релей","публикует события","#FFFFFF","#5B9A5B",ts=17,ss=13)
box(850,440,1090,530,"Redis Stream","consumer group · XACK","#FFFFFF","#5B9A5B",ts=17,ss=13)
box(410,450,800,540,"Админка документов","загрузка · статусы · переиндексация","#FFFFFF","#5B9A5B",ts=17,ss=13)

# rag
box(1180,150,1440,250,"Go · /v1/retrieve","скоуп из Symfony · ≤ 400 мс","#FFFFFF","#6C5AA8",ts=18,ss=13)
box(1180,290,1440,430,"Postgres + pgvector","отдельная БД\nRLS по org_id · HNSW","#FFFFFF","#6C5AA8",ts=18,ss=13)
box(1180,470,1440,600,"Go · ingest","чанкинг · эмбеддинги\nидемпотентность","#FFFFFF","#6C5AA8",ts=18,ss=13)
box(1180,640,1440,740,"bge-m3 (TEI)","эмбеддинги по HTTP","#FFFFFF","#6C5AA8",ts=17,ss=13)

# gpu
box(1540,150,1830,290,"vLLM · OpenAI API","DeepSeek V4-Flash · FP8\n32–64k контекст","#FFFFFF","#BF8040",ts=19,ss=14)
box(1540,320,1830,430,"NVMe","веса 167 ГБ · KV-кэш","#FFFFFF","#BF8040",ts=17,ss=13)
box(1540,460,1830,560,"8×H100 80 ГБ","~4–5 кВт","#FFFFFF","#BF8040",ts=17,ss=13)
box(1540,600,1830,730,"Qwen 35B на Halo","опция: fallback + dev + эмбеддинги","#FBF7FF","#9B8BC4",ts=16,ss=12,dashed=True)

# arrows
arrow([(315,210),(410,190)])
label(365,174,"HTTPS/SSE")

arrow([(555,230),(555,260)])
label(565,245,"proxy",anchor="lm")

arrow([(800,210),(850,210)])
label(825,196,"SQL",11)

arrow([(970,290),(970,320)])
label(980,305,"outbox",11,anchor="lm")

arrow([(970,390),(970,440)])

arrow([(1090,485),(1140,485),(1140,535),(1180,535)],dashed=True,width=2)

arrow([(800,415),(1130,415),(1130,200),(1180,200)])
label(1005,402,"POST /v1/retrieve",11)

arrow([(1310,250),(1310,290)])
arrow([(1310,430),(1310,470)])
label(1320,450,"SQL",11,anchor="lm")
arrow([(1310,600),(1310,640)])

arrow([(800,262),(800,122),(1685,122),(1685,150)])
label(1030,138,"OpenAI-совместимый API · шлюз JWT/mTLS · приватная сеть",11)

arrow([(1685,290),(1685,320)])
arrow([(1685,430),(1685,460)])

# optional fallback link
arrow([(1685,560),(1685,600)],dashed=True,width=2)
label(1695,580,"при отказе",10,anchor="lm")

# bottom band
d.rounded_rectangle([40,800,1860,960],radius=16,fill="#FAFAFA",outline="#CBD5E1",width=2)
d.text((60,812),"ТРИ КОНТУРА ОБУЧЕНИЯ (RAG не меняет веса — он подкладывает знания и форму)",font=f(15,True),fill="#475569")

box(70,840,600,945,"1 · Контур знаний","документы → чанки → ответ с цитатой","#FFFFFF","#5B9A5B",ts=17,ss=13)
box(640,840,1170,945,"2 · Контур качества","золотой набор · recall@k · точность цитат","#FFFFFF","#6C5AA8",ts=17,ss=13)
box(1210,840,1740,945,"3 · Контур поведения","few-shot из лучших диалогов · позже LoRA","#FFFFFF","#BF8040",ts=17,ss=13)

arrow([(600,892),(640,892)])
arrow([(1170,892),(1210,892)])

arrow([(1475,945),(1475,985),(335,985),(335,945)],dashed=True,width=2)
label(905,995,"обратная связь: помогло / нет → правки документов и примеров")

# title
d.text((40,24),"EVA32: чат orioncode + RAG на Go + DeepSeek V4-Flash",font=f(28,True),fill=INK)
d.text((40,60),"RoadRunner для SSE · outbox вместо двойной записи · отдельная pgvector-БД · циклы обучения",font=f(14),fill=MUTED)

# legend
ly=1030
d.line([60,ly,115,ly],fill=ARROW,width=3); d.polygon([(115,ly),(103,ly-5),(103,ly+5)],fill=ARROW)
d.text((125,ly-8),"синхронный вызов",font=f(13),fill=INK)
d.line([240,ly,295,ly],fill=ARROW,width=2)
d.text((305,ly-8),"асинхронно / опция",font=f(13),fill=INK)
d.text((60,ly+28),"Go не решает права: скоуп приходит из Symfony, RLS — второй рубеж. vLLM слушает только приватную сеть.",font=f(13),fill=MUTED)

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
img.save("/Users/a/Documents/work/my/eva32/eva/architecture_v2.png")
print("saved",img.size,"boxes",len(BOXES))
