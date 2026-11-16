from PIL import Image, ImageDraw, ImageFont
import math
import os

W, H = 1900, 1260
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

def box(x1,y1,x2,y2,title,sub=None,fill="#FFFFFF",border="#94A3B8",ts=16,ss=11,tc_=INK,sc=MUTED):
    d.rounded_rectangle([x1,y1,x2,y2],radius=10,fill=fill,outline=border,width=2)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    for ln in title.split("\n"):
        if d.textbbox((0,0),ln,font=f(ts,True))[2]>(x2-x1)-10: print("WARN title overflow",ln)
    if sub:
        for ln in sub.split("\n"):
            if d.textbbox((0,0),ln,font=f(ss))[2]>(x2-x1)-10: print("WARN sub overflow",ln)
        tc(cx,cy-8,title.split("\n"),ts,True,tc_)
        tc(cx,cy+24,sub.split("\n"),ss,False,sc)
    else:
        tc(cx,cy,title.split("\n"),ts,True,tc_)
    BOXES.append((x1,y1,x2,y2,title.split("\n")[0]))

def zone(x1,y1,x2,y2,label,fill,border,note=None,badge=None):
    d.rounded_rectangle([x1,y1,x2,y2],radius=14,fill=fill,outline=border,width=2)
    d.text((x1+18,y1+9),label,font=f(15,True),fill=border)
    if note: d.text((x1+18,y1+28),note,font=f(11),fill=MUTED)
    if badge:
        bb=d.textbbox((0,0),badge,font=f(11,True)); bw=bb[2]-bb[0]+14
        d.rounded_rectangle([x2-bw-10,y1+6,x2-10,y1+27],radius=7,fill=border)
        BADGES.append((x2-bw-10,y1+6,x2-10,y1+27,badge))
        d.text((x2-bw-3,y1+9),badge,font=f(11,True),fill="#FFFFFF")

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

def label(x,y,s,size=10,color=MUTED,anchor="mm"):
    bb=d.textbbox((x,y),s,font=f(size),anchor=anchor); LABELS.append((*bb,s))
    d.text((x,y),s,font=f(size),fill=color,anchor=anchor)

def num(x,y,n):
    d.ellipse([x-9,y-9,x+9,y+9],fill="#374151",outline="#FFFFFF",width=1)
    d.text((x,y),str(n),font=f(10,True),fill="#FFFFFF",anchor="mm")

# zones
zone(40,90,300,360,"ИНТЕРНЕТ","#F2F7FC","#4B7FB5")
zone(340,90,900,520,"VPS-1 · EDGE + APP","#F0F8F0","#4E8B4E","8 vCPU · 32 ГБ · nginx + RoadRunner","152-ФЗ")
zone(940,90,1400,520,"ДАННЫЕ · MANAGED","#F8F6EF","#A8843C","Postgres HA · Redis · S3 · в РФ")
zone(340,560,900,900,"RAG · VPS-2","#F3F0FA","#6C5AA8","8 vCPU · 32 ГБ · pgvector локально","152-ФЗ")
zone(940,560,1400,900,"GPU · COLOCATION","#FDF3E7","#BF8040","colo 5 кВт · 8×H100","152-ФЗ")
zone(1440,560,1860,900,"OPS · VPS-3","#F5F5F5","#64748B","мониторинг · eval · бэкапы")

# internet
box(60,140,280,215,"Пользователи","браузер + мобильные","#FFFFFF","#4B7FB5")
box(60,235,280,300,"Админка","загрузка документов","#FFFFFF","#4B7FB5")

# vps1
box(360,140,640,230,"nginx",":443 · TLS · WAF · SSE без буфера","#FFFFFF","#4E8B4E")
box(360,260,880,400,"RoadRunner + Symfony","чат · auth · промпт · tools\nлимиты · учёт · outbox","#FFFFFF","#4E8B4E",ts=17,ss=12)
box(700,420,880,490,"outbox relay","публикует события","#FFFFFF","#4E8B4E",ts=14,ss=10)

# data
box(960,140,1380,250,"Postgres · бизнес","организации, роли, документы, деньги, outbox","#FFFFFF","#A8843C",ts=15,ss=11)
box(960,280,1380,370,"Redis","Stream · consumer group · rate limit","#FFFFFF","#A8843C",ts=15,ss=11)
box(960,400,1380,480,"S3 / MinIO","оригиналы документов","#FFFFFF","#A8843C",ts=15,ss=11)

# rag
box(360,610,880,700,"Go · /v1/retrieve","скоуп из Symfony · ≤ 400 мс","#FFFFFF","#6C5AA8",ts=16,ss=12)
box(360,720,700,840,"Go · ingest","consumer group\nXACK · retry · DLQ","#FFFFFF","#6C5AA8",ts=15,ss=10)
box(720,720,880,840,"pgvector","localhost\nRLS · HNSW","#FFFFFF","#6C5AA8",ts=14,ss=10)

# gpu
box(960,610,1380,700,"vLLM · OpenAI API","DeepSeek V4-Flash · FP8 · :8000 mTLS","#FFFFFF","#BF8040",ts=16,ss=11)
box(960,720,1380,790,"bge-m3 · :8081","эмбеддинги и реранк","#FFFFFF","#BF8040",ts=15,ss=11)
box(960,810,1160,880,"NVMe","веса 167 ГБ · KV","#FFFFFF","#BF8040",ts=14,ss=10)
box(1180,810,1380,880,"8×H100","~4–5 кВт","#FFFFFF","#BF8040",ts=14,ss=10)

# ops
box(1460,610,1840,700,"monitoring","Prometheus · Grafana · Loki","#FFFFFF","#64748B",ts=15,ss=11)
box(1460,720,1840,800,"CI / eval","золотой набор · гейт","#FFFFFF","#64748B",ts=15,ss=11)
box(1460,820,1840,890,"backup · VPN","WireGuard · бэкапы","#FFFFFF","#64748B",ts=15,ss=11)

# flows
arrow([(280,180),(360,180)],both=True); label(320,163,"443"); num(320,180,1)
arrow([(500,230),(500,260)],both=True); label(512,245,"8080",10,anchor="lm"); num(500,245,2)
arrow([(820,260),(820,200),(960,200)],both=True); label(895,188,"SQL",10); num(895,200,3)
arrow([(880,320),(960,320)],both=True); label(920,303,"RESP",10); num(920,337,4)
arrow([(880,455),(910,455),(910,360),(960,360)]); num(895,470,5)
arrow([(620,400),(620,610)],both=True); label(632,500,"POST /v1/retrieve",10,anchor="lm"); num(620,520,6)
arrow([(800,700),(800,720)],both=True); label(812,710,"SQL",9,anchor="lm"); num(800,710,7)
arrow([(960,340),(920,340),(920,540),(320,540),(320,760),(360,760)],width=2)
label(760,528,"XREADGROUP / XACK",10); num(320,650,8)
arrow([(620,840),(620,870),(920,870),(920,755),(960,755)]); label(930,745,"8081",9,anchor="lm"); num(920,860,9)
arrow([(700,780),(720,780)]); num(710,790,10)
arrow([(880,300),(930,300),(930,530),(1160,530),(1160,610)],both=True); label(1090,518,"8000 · stream",10); num(930,420,11)
arrow([(1460,650),(1380,650)]); label(1425,632,"9100",9); num(1420,650,12)

# title
d.text((40,24),"EVA32: размещение и потоки обращений",font=f(27,True),fill=INK)
d.text((40,58),"Два конца стрелки — запрос и ответ; один конец — в одну сторону. Номера расшифрованы в таблице.",font=f(13),fill=MUTED)

# table
tx, ty = 60, 930
cols = [tx, tx+40, tx+270, tx+800, tx+1050, tx+1240]
headers = ["#","Откуда → Куда","Что идёт","Протокол / порт","Тип"]
rows = [
 ("1","Клиент → nginx","сообщение чата; ответ — SSE-стрим","HTTPS :443","sync"),
 ("2","nginx → Symfony","проксирование запроса и стрима","HTTP :8080","sync"),
 ("3","Symfony → Postgres","auth, лимиты, outbox, usage","SQL :5432","sync"),
 ("4","Symfony → Redis","rate limit по организации","RESP :6379","sync"),
 ("5","relay → Redis","publish ai.doc.* из outbox","RESP :6379","async"),
 ("6","Symfony → Go RAG","POST /v1/retrieve → чанки с цитатами","HTTP :8080","sync ≤ 400 мс"),
 ("7","Go RAG → pgvector","векторный поиск (SQL)","localhost :5432","sync"),
 ("8","Redis → Go ingest","XREADGROUP → обработка → XACK","RESP :6379","async"),
 ("9","ingest → bge-m3","эмбеддинги чанков","HTTP :8081","sync, батчами"),
 ("10","ingest → pgvector","upsert чанков и удаление старых","localhost :5432","sync"),
 ("11","Symfony → vLLM","chat/completions; ответ — SSE","HTTP :8000 mTLS","stream"),
 ("12","Ops → vLLM","метрики, health, логи","HTTP :9100","async, 15 с"),
]
d.line([tx,ty,tx+1800,ty],fill="#CBD5E1",width=2)
for h,c in zip(headers,cols):
    d.text((c,ty+8),h,font=f(12,True),fill="#475569")
d.line([tx,ty+30,tx+1800,ty+30],fill="#CBD5E1",width=1)
y=ty+40
for r in rows:
    d.text((cols[0],y),r[0],font=f(11,True),fill=INK)
    d.text((cols[1],y),r[1],font=f(11),fill=INK)
    d.text((cols[2],y),r[2],font=f(11),fill=INK)
    d.text((cols[3],y),r[3],font=f(11),fill=MUTED)
    d.text((cols[4],y),r[4],font=f(11),fill=MUTED)
    y+=22

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

img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flows.png"))
print("saved",img.size,"boxes",len(BOXES))
