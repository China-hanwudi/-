"""ReCoMER: readable objects and transformations, revision 3.
Dialogue, waveform and portrait glyphs are illustrations, not dataset samples.
Latent representations and probabilities remain symbolic, never measured values.
"""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Arc, Ellipse, Polygon

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
                     'font.size':7.4,'pdf.fonttype':42,'svg.fonttype':'none',
                     'savefig.facecolor':'white','axes.linewidth':.6})
INK='#233642'; GREY='#76858D'; LINE='#87959C'; PALE='#F3F5F6'
T='#4B85AE'; A='#C6914F'; V='#519A91'; H='#9279B2'; NAVY='#325769'
TC='#E7F0F6'; AC='#F7EEDD'; VC='#E5F1ED'; HC='#F0EBF6'; NC='#EDF2F4'
MODS=[('T',T,TC),('A',A,AC),('V',V,VC)]
MANIFEST=[]

def figure(height,rows=1):
    fig,arr=plt.subplots(rows,1,figsize=(7.2047244094,height/25.4),squeeze=False)
    fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975,hspace=.17)
    axs=list(arr[:,0])
    for ax in axs: ax.set(xlim=(0,180),ylim=(0,height if rows==1 else 60)); ax.axis('off')
    return fig,axs

def text(ax,x,y,s,size=7.4,bold=False,ha='center',color=INK,va='center'):
    return ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
                   ha=ha,va=va,color=color,linespacing=1.25,zorder=7)

def rect(ax,x,y,w,h,fill='white',edge=LINE,lw=.65,r=1.3,dash=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                     linewidth=lw,edgecolor=edge,facecolor=fill,
                     linestyle=(0,(3,2)) if dash else '-',zorder=2)
    ax.add_patch(p); return p

def box(ax,x,y,w,h,s,fill=PALE,edge=LINE,size=7.4,bold=False,dash=False):
    p=rect(ax,x,y,w,h,fill,edge,dash=dash)
    lab=text(ax,x+w/2,y+h/2,s,size,bold)
    lab._module_patch=p
    return (x,y,w,h)

def bubble(ax,x,y,w,h,s='',colour=H,fill=HC,size=6.8):
    """A dialogue candidate, with a tail that gives it a conversational meaning."""
    p=rect(ax,x,y,w,h,fill,colour,r=1.5)
    ax.add_patch(Polygon([(x+1.5,y),(x+1.5,y-1.4),(x+3.7,y)],closed=True,
                        facecolor=fill,edgecolor=colour,lw=.65,zorder=2))
    if s:
        label=text(ax,x+w/2,y+h/2,s,size)
        label._module_patch=p
    return (x,y,w,h)

def path(ax,pts,c=LINE,dash=False,lw=.85,head=True):
    for i,(p,q) in enumerate(zip(pts,pts[1:])):
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>' if head and i==len(pts)-2 else '-',
                     shrinkA=0,shrinkB=0,mutation_scale=6.8,linewidth=lw,color=c,
                     linestyle=(0,(3,2)) if dash else '-',zorder=3))

def join(ax,a,b,**kw):
    x,y,w,h=a; X,Y,W,Hh=b; path(ax,[(x+w,y+h/2),(X,Y+Hh/2)],**kw)

def heading(ax,letter,title,y):
    text(ax,1,y,letter,9,True,ha='left')
    text(ax,8,y,title,8,True,ha='left')

def stage(ax,x,y,num,title,width):
    text(ax,x,y,num,7,True,ha='left',color=NAVY)
    text(ax,x+7,y,title,7.4,True,ha='left')
    ax.plot([x,x+width],[y-4,y-4],color='#D7E0E4',lw=.6,zorder=1)

def token(ax,x,y,s,colour=T,fill=TC,w=13,h=8,muted=False):
    # Proper modality subscripts; parent size keeps every script glyph above 5 pt.
    notation={}
    for m in ['T','A','V']:
        notation.update({'h'+m:rf'$h_{{\mathrm{{{m}}}}}$',
                         'h′'+m:rf"$h'_{{\mathrm{{{m}}}}}$",
                         'w'+m:rf'$w_{{\mathrm{{{m}}}}}$',
                         'e'+m:rf'$e_{{\mathrm{{{m}}}}}$',
                         'φ'+m:rf'$\varphi_{{\mathrm{{{m}}}}}$'})
    rendered=notation.get(s,s)
    # Explicit mathematical capsules are different objects from dialogue bubbles.
    p=rect(ax,x,y,w,h,PALE if muted else fill,LINE if muted else colour,r=min(w,h)/2)
    label=text(ax,x+w/2,y+h/2,rendered,8.4 if s in notation else 7)
    label._module_patch=p
    return (x,y,w,h)

def tokens(ax,x,y,prime=False,weighted=False,vertical=True,muted=False):
    for i,(m,c,f) in enumerate(MODS):
        xx=x if vertical else x+i*15; yy=y-i*11 if vertical else y
        s=('w'+m+' h′'+m) if weighted else ('h′'+m if prime else 'h'+m)
        token(ax,xx,yy,s,c,f,w=21 if weighted else 13,muted=muted)

def cells(ax,x,y,n=4,fill=HC,edge=H,w=5,h=7,gap=1.1,labels=None):
    for i in range(n):
        rect(ax,x+i*(w+gap),y,w,h,fill,edge,r=.55)
        if labels: text(ax,x+i*(w+gap)+w/2,y+h/2,labels[i],6.4)

def memory(ax,x,y,compact=False):
    bubble(ax,x,y,21,7,'Earlier turn',size=6.4)
    bubble(ax,x+2,y+9,21,7,'Recent turn',size=6.4)
    if not compact: text(ax,x+10,y-5,'History',7,color=H)

def probability(ax,x,y,name,colour=NAVY,w=24):
    """Emotion likelihood list; symbols carry no fabricated numerical values."""
    text(ax,x+w/2,y+12,name,7.5,True,color=colour)
    rect(ax,x,y-1.5,w,10,fill='white',edge=colour,r=1.4)
    text(ax,x+1.5,y+6.6,'Angry',6.1,ha='left',color=colour)
    text(ax,x+w-1.5,y+6.6,'p(Angry)',6.1,ha='right')
    text(ax,x+1.5,y+3.1,'Happy',6.1,ha='left',color=colour)
    text(ax,x+w-1.5,y+3.1,'p(Happy)',6.1,ha='right')
    text(ax,x+w/2,y-.2,'…',6.4,color=GREY)
    return (x,y,w,7)

def op(ax,x,y,s,r=4,colour=NAVY):
    ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=colour,lw=.9,zorder=4))
    text(ax,x,y,s,9,True,color=colour)

def modality(ax,x,y,m):
    c,f={'T':(T,TC),'A':(A,AC),'V':(V,VC)}[m]
    rect(ax,x,y,19,16,f,c)
    if m=='T':
        bubble(ax,x+2,y+4,15,9,"I'm fine.",colour=c,fill='white',size=6.8)
    elif m=='A':
        # Schematic speech waveform icon; not an experimental signal.
        u=np.linspace(0,1,80)
        wave=np.sin(u*30)*np.sin(np.pi*u)*3.5
        ax.plot(x+2+u*15,y+8+wave,color=c,lw=.9,zorder=4)
    else:
        rect(ax,x+2,y+2,15,12,'white',c,r=.5)
        ax.add_patch(Ellipse((x+9.5,y+9),6,7,facecolor=VC,edgecolor=c,lw=.65,zorder=4))
        for ex in [8.4,10.6]:ax.add_patch(Circle((x+ex,y+9.5),.3,facecolor=c,edgecolor='none',zorder=4))
        ax.plot([x+8.5,x+10.5],[y+7.3,y+7.3],color=c,lw=.6,zorder=4)
        ax.add_patch(Arc((x+9.5,y+3),10,6,theta1=0,theta2=180,color=c,lw=.7,zorder=4))
    text(ax,x+9.5,y-4,{'T':'Text','A':'Audio','V':'Visual'}[m],7,color=c)

def transformer(ax,x,y,w=25,h=24,quiet=False):
    rect(ax,x,y,w,h,PALE if quiet else NC,LINE if quiet else NAVY)
    text(ax,x+w/2,y+h-5,'Joint head',7.4,True)
    text(ax,x+w/2,y+9.5,'Attention + FFN\nMasked pooling\nClassifier',6.4,color=GREY)
    text(ax,x+w/2,y-5,'2 layers · 6 heads',6.7,color=GREY)
    return(x,y,w,h)

def export(fig,name,title,height,status='code-grounded'):
    fig.canvas.draw()
    alignment=require_matplotlib_panel_alignment(fig,json_out=ROOT/'qa'/f'{name}.alignment.json',
                   overlay_svg=ROOT/'qa'/f'{name}.alignment.svg',tolerance_pt=1.5,
                   gutter_tolerance_pt=1.5,strict=True)
    renderer=fig.canvas.get_renderer(); fits=[]
    for ax in fig.axes:
        for t in ax.texts:
            p=getattr(t,'_module_patch',None)
            if p is None:continue
            lo=ax.transData.transform((p.get_x()+.65,p.get_y()+.55))
            hi=ax.transData.transform((p.get_x()+p.get_width()-.65,p.get_y()+p.get_height()-.55))
            bb=t.get_window_extent(renderer)
            good=bb.x0>=lo[0] and bb.x1<=hi[0] and bb.y0>=lo[1] and bb.y1<=hi[1]
            fits.append({'text':t.get_text(),'fits':bool(good)})
    (ROOT/'qa'/f'{name}.box-fit.json').write_text(json.dumps(fits,indent=2),encoding='utf-8')
    fig.savefig(ROOT/f'{name}.svg')
    fig.savefig(ROOT/f'{name}.pdf')
    fig.savefig(ROOT/f'{name}.png',dpi=350)
    fig.savefig(ROOT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    MANIFEST.append({'name':name,'title':title,'status':status,'width_mm':183,
                     'height_mm':height,'alignment':alignment['verdict'],
                     'box_fit_failures':sum(not q['fits'] for q in fits)})
    plt.close(fig)

def overview():
    fig,(ax,)=figure(136)
    heading(ax,'a','ReCoMER integrates admitted history and complementary evidence',131)
    stage(ax,2,122,'01','Encode',52);stage(ax,63,122,'02','Admit & reassess',74);stage(ax,146,122,'03','Predict',31)
    for i,(m,c,f) in enumerate(MODS):
        y=91-i*24
        modality(ax,2,y,m)
        b=box(ax,29,y+3,25,10,m+' encoder',f,c,size=7)
        path(ax,[(21,y+8),(29,y+8)],c)
        token(ax,62,y+4,'h'+m,c,f,w=13)
        path(ax,[(54,y+8),(62,y+8)],c)
    # The history branch feeds a single utterance-level admitted residual.
    box(ax,81,51,25,52,'Historical\nadmission\n\n+ residual\nfeedback',HC,H,size=7.4,bold=True)
    for i,(m,c,f) in enumerate(MODS):
        y=95-i*24
        path(ax,[(75,y+4),(81,y+4)],c)
        token(ax,113,y,'h′'+m,c,f,w=13)
        path(ax,[(106,y+4),(113,y+4)],c)
    bubble(ax,82,108,24,6,'Past dialogue',size=6.4)
    text(ax,93.5,115.5,'Encoded history',6.8,color=H)
    path(ax,[(93,107),(93,103)],H)
    text(ax,113,113,'M3',7,True,ha='left',color=H)
    rect(ax,130,47,16,56,NC,NAVY)
    for m,yy in [('T',99),('A',75),('V',51)]:
        text(ax,138,yy,rf"$w_{{\mathrm{{{m}}}}} h'_{{\mathrm{{{m}}}}}$",8.4)
    text(ax,137.5,42,'M2 · Contribution router',6.8,ha='right',color=NAVY)
    for i,(m,c,f) in enumerate(MODS):
        y=99-i*24; path(ax,[(126,y),(130,y)],c)
    teacher=box(ax,130,107,47,10,'Shapley teacher · train only',AC,A,size=6.8,dash=True)
    path(ax,[(136,107),(136,103)],A,True)
    head=transformer(ax,151,66,26,26)
    for y in [99,75,51]:
        end=83 if y==99 else 79 if y==75 else 71
        path(ax,[(146,y),(148,y),(148,end),(151,end)],NAVY)
    text(ax,164,100,'M1 · closed loop',7,True,color=NAVY)
    text(ax,177,49,'p(MHnoU)',7.4,True,ha='right',color=NAVY)
    ax.plot([2,177],[37,37],color='#D7E0E4',lw=.6)
    text(ax,2,32,'Independent evidence',7.4,True,ha='left')
    box(ax,2,10,25,15,'T / A / V',PALE,size=7.4)
    expert=box(ax,35,10,49,15,'cRBEF v2\nExternal frozen predictor',AC,A,size=7.1)
    path(ax,[(27,17.5),(35,17.5)])
    pc=probability(ax,93,14,'p(cRBEF)',A,24)
    path(ax,[(84,17.5),(93,17.5)],A)
    local=box(ax,130,9,47,18,'local_current\nBounded class correction',NC,NAVY,size=7.3,bold=True)
    path(ax,[(117,17.5),(130,17.5)],A)
    path(ax,[(177,79),(179,79),(179,27),(154,27)],NAVY)
    text(ax,80,4,'Reference probabilities + relative T / A / V evidence',6.6,ha='center',color=GREY)
    path(ax,[(118,5),(124,5),(124,11),(130,11)],GREY)
    path(ax,[(161,9),(161,5)],NAVY)
    text(ax,161,2,'Final prediction ŷ',6.8,True,color=NAVY)
    export(fig,'01_ReCoMER_overview','ReCoMER system overview',136)

def closed_loop():
    fig,axs=figure(148,3)
    titles=['Direct history tokens','Feedback followed by solo-logit fusion','MHnoU: feedback → contribution reassessment → joint inference']
    for i,ax in enumerate(axs):
        heading(ax,chr(97+i),titles[i],56)
        if i<2: text(ax,178,56,'Comparison',6.8,ha='right',color=GREY)
        if i==0:
            tokens(ax,4,30,vertical=False,muted=True);text(ax,23,43,'Current tokens',7,color=GREY)
            memory(ax,10,4,True);text(ax,36,8,'History',7,ha='left',color=GREY)
            op(ax,67,30,'∥',colour=GREY)
            text(ax,67,20,'Concat',6.8,color=GREY)
            path(ax,[(47,34),(55,34),(55,30),(63,30)])
            path(ax,[(30,12),(55,12),(55,30),(63,30)])
            box(ax,80,24,38,12,'Current + past utterances',PALE,size=6.8)
            text(ax,101,43,'Current + history',7,color=GREY)
            path(ax,[(71,30),(80,30)])
            head=transformer(ax,133,18,26,24,True);path(ax,[(119,30),(133,30)])
            path(ax,[(159,30),(167,30)]);text(ax,173,30,'ŷ',9)
        else:
            tokens(ax,4,28,vertical=False);text(ax,24,43,'Current tokens',7)
            memory(ax,57,4,True)
            fb=box(ax,58,24,24,17,'Admit\n+ feedback',HC,H,size=7.2)
            path(ax,[(47,32),(58,32)],H);path(ax,[(68,20),(68,24)],H)
            tokens(ax,92,28,prime=True,vertical=False)
            path(ax,[(82,32),(92,32)],H)
            text(ax,113,43,'Corrected tokens',7)
            if i==1:
                box(ax,144,24,18,17,'wT / wA / wV\nSolo logits',PALE,size=6.7)
                path(ax,[(135,32),(144,32)]);op(ax,173,32,'Σ',colour=GREY)
                path(ax,[(162,32),(169,32)])
                text(ax,111,9,'Weights combine predictions',7,color=GREY)
            else:
                box(ax,143,23,20,20,'Bounded w\n× tokens',NC,NAVY,size=7)
                path(ax,[(135,32),(143,32)],NAVY)
                path(ax,[(163,32),(167,32)],NAVY)
                box(ax,167,23,11,20,'Joint\nhead',NC,NAVY,size=6.8)
                text(ax,112,9,'Raw history is omitted from the final joint head',7,color=NAVY)
    export(fig,'02_M1_closed_loop','Prediction-path mechanism comparison',148)

def contribution():
    fig,axs=figure(139,2);ax=axs[0]
    heading(ax,'a','Training: measured marginal contributions supervise the router',56)
    tokens(ax,3,37,prime=True);text(ax,10,7,'Current',7)
    subsets=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
    text(ax,44,46,'Modality sets',7.1,True,color=NAVY)
    for i,row in enumerate(subsets):
        members=[m for (m,c,f),on in zip(MODS,row) if on]
        text(ax,44,9.7+(6-i)*4.8,'{'+', '.join(members)+'}',7.2,color=NAVY)
    text(ax,44,3,'7 subsets',6.8,color=GREY)
    path(ax,[(16,30),(28,30)],A,True)
    utility=box(ax,67,19,32,22,'Joint-head utility\nU(S) = log p(y | S)',AC,A,size=7.1)
    path(ax,[(57,30),(67,30)],A,True)
    for j,(m,c,f) in enumerate(MODS):token(ax,112,36-j*11,'φ'+m,c,f,w=17)
    text(ax,120.5,7,'Exact Shapley',7,color=A)
    path(ax,[(99,30),(106,30),(106,29),(112,29)],A,True)
    loss=box(ax,143,19,34,22,'Standardize φ\nMSE(μ, φ)',AC,A,size=7.2,dash=True)
    path(ax,[(129,29),(143,29)],A,True)
    text(ax,83,10,'U(∅) = 0',6.8,color=GREY)
    ax=axs[1];heading(ax,'b','Deployment: local and cross-modal evidence produce bounded weights',56)
    for i,(m,c,f) in enumerate(MODS):
        yy=36-i*11
        token(ax,3,yy,'h′'+m,c,f,w=14)
        box(ax,27,yy,22,8,'Projection',f,c,size=6.9)
        path(ax,[(17,yy+4),(27,yy+4)],c)
        token(ax,60,yy,'e'+m,c,f,w=14)
        path(ax,[(49,yy+4),(60,yy+4)],c)
    text(ax,67,7,'Local evidence',7)
    context=box(ax,89,19,31,22,'Mean context\n+ difference\n+ solo statistics',PALE,size=7.0)
    for yy in [40,29,18]:path(ax,[(74,yy),(82,yy),(82,30),(89,30)])
    scorer=box(ax,132,19,22,22,'Shared\nscorer\nμ(T/A/V)',NC,NAVY,size=7.0)
    path(ax,[(120,30),(132,30)],NAVY)
    for i,(m,c,f) in enumerate(MODS):token(ax,164,36-i*11,'w'+m,c,f,w=14)
    path(ax,[(154,30),(160,30),(160,29),(164,29)],NAVY)
    text(ax,127,7,'Center → bound → normalize',7,color=NAVY)
    text(ax,90,1,'w(m) = ⅓ [1 + λ r(m) / max |r|],  r(m) = μ(m) − mean(μ)',7.1)
    export(fig,'03_M2_contribution_router','Exact contribution supervision and bounded deployment',139)

def admission():
    fig,axs=figure(142,2);ax=axs[0]
    heading(ax,'a','Historical selection and hard admission are separate operations',56)
    # History candidates and null are symbolic cells, not attention values.
    for i,s in enumerate(['t−2','t−1','…']):bubble(ax,3+i*8.1,29,7,10,s,size=6.4)
    box(ax,29,29,10,10,'None','white',H,size=6.4,dash=True)
    text(ax,21,45,'History + null',7.2,color=H)
    att=box(ax,51,25,29,18,'Initial attention\n+ pooled statistics',HC,H,size=7)
    box(ax,51,46,29,7,'Current query h',TC,T,size=7.0)
    path(ax,[(65.5,46),(65.5,43)],T)
    path(ax,[(39,34),(51,34)],H)
    cal=box(ax,91,25,32,18,'24-feature\ncalibrator',AC,A,size=7.3)
    path(ax,[(80,34),(91,34)],A)
    gate=box(ax,137,29,40,14,'k → g = 1[k ≥ τ]\nwith valid history',NC,NAVY,size=7.3)
    path(ax,[(123,34),(137,34)],NAVY)
    recom=box(ax,51,1,42,14,'Recompute attention\nwith calibrated null',HC,H,size=7)
    path(ax,[(107,25),(107,18),(72,18),(72,15)],A)
    path(ax,[(21,29),(21,8),(51,8)],H)
    path(ax,[(93,8),(111,8)],H)
    token(ax,111,4,'v',H,HC,w=13)
    text(ax,130,8,'Pooled real-history residual',7,ha='left',color=H)
    ax=axs[1];heading(ax,'b','Hard rejection removes the historical residual',56)
    text(ax,44,46,'Admit history: g = 1',7.5,True,color=H)
    text(ax,133,46,'Reject history: g = 0',7.5,True,color=NAVY)
    for x in [3,95]:
        token(ax,x,25,'h',T,TC,w=14)
        op(ax,x+37,29,'+')
        path(ax,[(x+14,29),(x+33,29)],T)
        token(ax,x+64,25,'h′' if x==3 else 'h',T,TC,w=14)
        path(ax,[(x+41,29),(x+64,29)],NAVY)
    token(ax,33,9,'γv',H,HC,w=14);path(ax,[(40,17),(40,25)],H)
    token(ax,125,9,'γv',H,HC,w=14);path(ax,[(132,17),(132,23)],GREY,True,head=False)
    ax.plot([128,136],[22,22],color=NAVY,lw=1.3)
    text(ax,44,3,'h′ = h + γv',7.5,color=H)
    text(ax,133,3,'h′ = h; history masks off when present',7.0,color=NAVY)
    export(fig,'04_M3_history_admission','Historical selection versus executable hard rejection',142)

def crbef():
    fig,(ax,)=figure(125)
    heading(ax,'a','cRBEF augments a frozen anchor with calibrated AV evidence',119)
    text(ax,2,111,'Reference-method reconstruction',7,ha='left',color=GREY)
    modality(ax,3,80,'T')
    box(ax,32,80,31,16,'Frozen text\nanchor',TC,T,size=7.5,bold=True)
    path(ax,[(22,88),(32,88)],T)
    p0=probability(ax,76,84,'p₀',T,23)
    path(ax,[(63,88),(76,88)],T)
    modality(ax,3,41,'A');modality(ax,3,15,'V')
    box(ax,32,27,31,22,'Frozen AV-only\nbranch',AC,A,size=7.4,bold=True)
    path(ax,[(22,49),(27,49),(27,38),(32,38)],A)
    path(ax,[(22,23),(27,23),(27,38),(32,38)],V)
    pav=probability(ax,76,35,'p(AV)',A,23);path(ax,[(63,38),(76,38)],A)
    box(ax,74,12,26,12,'Class prior π',PALE,size=7.2)
    evidence=box(ax,112,26,33,23,'AV log evidence\ne = log p(AV)\n− log π',AC,A,size=7.2)
    path(ax,[(99,38),(112,38)],A)
    path(ax,[(100,18),(106,18),(106,32),(112,32)])
    stats=box(ax,111,77,34,22,'6 confidence\nstatistics\nSigmoid gate g',NC,NAVY,size=7.3)
    path(ax,[(99,88),(111,88)],GREY,True)
    path(ax,[(98,43.5),(98,67),(128,67),(128,77)],GREY,True)
    op(ax,159,57,'×',colour=A)
    path(ax,[(145,38),(159,38),(159,53)],A)
    path(ax,[(145,88),(159,88),(159,61)],NAVY)
    op(ax,171,101,'+',colour=NAVY)
    path(ax,[(98,92.5),(98,101),(167,101)],T)
    text(ax,130,105,'log p₀',7,color=T)
    path(ax,[(163,57),(171,57),(171,97)],A)
    box(ax,155,12,23,20,'Softmax\np(cRBEF)',NC,NAVY,size=7.1,bold=True)
    path(ax,[(175,101),(179,101),(179,38),(166.5,38),(166.5,32)],NAVY)
    text(ax,91,4,'z = log p₀ + g e',8,True,color=NAVY)
    export(fig,'05_cRBEF_reference_structure','cRBEF reference evidence mechanism',125,
           'reference-method reconstruction; v2 expert internals not verified')

def network(ax,x,y):
    # Three columns are a structural glyph; ellipses stand for hidden units.
    for j,ys in enumerate([[y+2,y+10,y+18],[y,y+7,y+14,y+21],[y+10]]):
        for yy in ys:ax.add_patch(Circle((x+j*10,yy),1.55,facecolor='white',edgecolor=NAVY,lw=.6,zorder=4))
    for p in [2,10,18]:
        for q in [0,7,14,21]:
            path(ax,[(x+1.7,y+p),(x+8.3,y+q)],c='#C4D0D6',lw=.4,head=False)
    for q in [0,7,14,21]:path(ax,[(x+11.7,y+q),(x+18.3,y+10)],c='#C4D0D6',lw=.4,head=False)
    for j,s in enumerate(['23','32','1']):text(ax,x+j*10,y-5,s,7,color=NAVY)

def local():
    fig,(ax,)=figure(140)
    heading(ax,'a','local_current restricts correction to candidate probability mass',134)
    q=box(ax,54,111,60,12,'q = ½ [p(cRBEF) + p(MHnoU)]',NC,NAVY,size=7.1,bold=True)
    text(ax,2,122,'Probability sources',7.5,True,ha='left')
    for i,(name,c) in enumerate([('cRBEF (a)',A),('MHnoU (b)',NAVY),('TAV ref.',GREY),('No-history ref.',GREY)]):
        y=93-i*21;probability(ax,4,y,name,c,25)
        path(ax,[(29,y+3.5),(38,y+3.5)],c)
    text(ax,3,14,'Relative T / A / V\nclass evidence C',7.0,ha='left',color=GREY)
    path(ax,[(29,18),(38,18),(38,33.5)],GREY,head=False)
    # Aligned sources feed the class-wise feature bank.
    path(ax,[(38,96.5),(38,33.5)],head=False)
    path(ax,[(38,63),(44,63)])
    path(ax,[(29,96.5),(34,96.5),(34,117),(54,117)],A)
    path(ax,[(29,75.5),(32,75.5),(32,106),(80,106),(80,111)],NAVY)
    text(ax,63,102,'23 features / class',7.7,True)
    rect(ax,44,32,40,63,'white','#CAD5DB',lw=.55)
    for i,(n,s,f,c) in enumerate([(6,'Base',NC,NAVY),(4,'Confidence',PALE,LINE),(2,'Top-class flags',PALE,LINE),
                                  (8,'References',HC,H),(3,'Modality evidence',VC,V)]):
        y=82-i*12
        box(ax,46,y,36,10,f'{n}  {s}',f,c,size=7.1)
    text(ax,104,94,'Shared MLP',7.7,True,color=NAVY)
    network(ax,94,58)
    path(ax,[(84,63),(89,63)],NAVY)
    text(ax,63,26,'Standardize + clip',6.8,color=GREY)
    delta=box(ax,126,62,23,20,'δ = 1.5\ntanh(raw)',NC,NAVY,size=7.3)
    path(ax,[(116,68),(126,68)],NAVY)
    cand=box(ax,93,13,53,20,'Candidate set S\nTop-2(a) ∪ Top-2(b) ∪ Argmax(q)',HC,H,size=6.9)
    corr=box(ax,155,55,23,34,'Inside S:\nReallocate mass\nusing q exp(δ)\n\nOutside S:\nretain q',NC,NAVY,size=7.0)
    path(ax,[(149,72),(155,72)],NAVY)
    path(ax,[(146,23),(150,23),(150,47),(163,47),(163,55)],H)
    path(ax,[(114,117),(166,117),(166,89)],NAVY)
    path(ax,[(86,111),(86,9),(116,9),(116,13)],GREY)
    # The blend also receives the uncorrected baseline, independently of S.
    path(ax,[(86,9),(86,7),(148,7),(148,21),(153,21)],NAVY)
    box(ax,153,11,26,23,'Bounded blend\nq → p(final)',NC,NAVY,size=7.2,bold=True)
    path(ax,[(178,72),(179.5,72),(179.5,40),(166,40),(166,34)],NAVY)
    text(ax,90,3,'p(final) = q + ρ η [p(corrected) − q],  ρ = 0.25,  0 ≤ η ≤ 1',7.3)
    export(fig,'06_local_current_fusion','Candidate-restricted bounded outer fusion',140)

def main():
    (ROOT/'qa').mkdir(parents=True,exist_ok=True)
    overview();closed_loop();contribution();admission();crbef();local()
    (ROOT/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(MANIFEST,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
