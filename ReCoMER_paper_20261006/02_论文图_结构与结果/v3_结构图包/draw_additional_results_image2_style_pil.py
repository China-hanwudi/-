"""Additional exact-data figures: gate sensitivity, PCA views and confusion matrices."""
from pathlib import Path
import base64, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RESULT = PROJECT / "最终代码" / "实验结果" / "ReCoMER_正式测试与消融_20261004"
GATE_JSON = RESULT / "CR_GATE_FORMULA_SENSITIVITY.json"
PRED_PATHS = [RESULT / "full_test" / str(seed) / "PREDICTIONS.npz" for seed in (43,47,59)]
FEATURE_PATH = RESULT / "CR17_TEST_FEATURES.npz"
LABEL_PATH = RESULT / "full_test" / "43" / "PREDICTIONS.npz"

INK="#233642"; MUTED="#6C7A82"; GRID="#D9E1E5"; BLUE="#4E8FCB"; ORANGE="#D9924B"; TEAL="#4BA9A7"; PURPLE="#8877C9"; GREEN="#2E9B63"; RED="#CE5B5B"; GOLD="#D59B32"
CLASS_COLORS=["#4E8FCB","#7B6FB6","#4BA9A7","#D9924B","#C95B6D","#6B9D5E","#8C7CBA"]

def fnt(size,bold=False):
    for p in [("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")]:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()
def rgb(h): return tuple(int(h.lstrip('#')[i:i+2],16) for i in (0,2,4))
def text(d,xy,s,size=22,color=INK,bold=False,anchor="la"): d.text(xy,str(s),font=fnt(size,bold),fill=rgb(color),anchor=anchor)
def line(d,xy,color=GRID,width=2): d.line(xy,fill=rgb(color),width=width)
def panel(d,box,title):
    x,y,w,h=box; d.rounded_rectangle((x,y,x+w,y+h),radius=22,fill=rgb("#FFFFFF"),outline=rgb("#D5E0E5"),width=2); d.rounded_rectangle((x,y,x+w,y+68),radius=22,fill=rgb("#F4F8FA")); d.rectangle((x,y+35,x+w,y+68),fill=rgb("#F4F8FA")); text(d,(x+28,y+34),title,29,INK,True,"lm")
def save(img,stem):
    png=ROOT/f"{stem}.png"; tiff=ROOT/f"{stem}.tiff"; pdf=ROOT/f"{stem}.pdf"; svg=ROOT/f"{stem}.svg"
    img.save(png,dpi=(600,600)); img.save(tiff,dpi=(600,600),compression="tiff_lzw"); img.convert("RGB").save(pdf,"PDF",resolution=300.0)
    b=base64.b64encode(png.read_bytes()).decode("ascii"); svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}"><image width="100%" height="100%" href="data:image/png;base64,{b}"/></svg>',encoding="utf-8")

def draw_sensitivity():
    data=json.loads(GATE_JSON.read_text(encoding="utf-8"))["probes"]
    order=["TAV_only_gate0","constant_gate_half","constant_gate_one","learned_gate_without_prior_correction","TAV_AV_probability_average","original_CR17"]
    names=["TAV gate 0","constant 0.5","constant 1","learned, no prior","TAV/AV average","original CR17"]
    wf=[data[k]["weighted_f1"]*100 for k in order]; nll=[data[k]["nll"] for k in order]
    img=Image.new("RGB",(3000,1050),"white"); d=ImageDraw.Draw(img); text(d,(90,50),"Gate-formula sensitivity on the frozen M3ED test",38,INK,True); text(d,(90,95),"Post-hoc frozen probes; no refitting or test-set selection",22,MUTED)
    panel(d,(70,145,1400,790),"(a)  Weighted F1 (%)"); panel(d,(1530,145,1400,790),"(b)  NLL (lower is better)")
    for vals,box,ymin,ymax,color in [(wf,(300,800,1050,500),56,60,PURPLE),(nll,(1760,800,1050,500),1.05,1.52,BLUE)]:
        x0,y0,pw,ph=box
        ticks=np.linspace(ymin,ymax,5)
        for t in ticks:
            yy=y0-(t-ymin)/(ymax-ymin)*ph; line(d,(x0,yy,x0+pw,yy),GRID,2); text(d,(x0-18,yy),f"{t:.2f}",19,MUTED,False,"ra")
        xs=np.linspace(x0+45,x0+pw-35,len(vals)); pts=[]
        for x,v in zip(xs,vals): pts.append((x,y0-(v-ymin)/(ymax-ymin)*ph))
        for p,q in zip(pts,pts[1:]): line(d,(p[0],p[1],q[0],q[1]),color,5)
        for i,(x,y) in enumerate(pts):
            d.ellipse((x-11,y-11,x+11,y+11),fill=rgb(GOLD if i==5 else color),outline=rgb("#FFFFFF"),width=2); text(d,(x,y-25),f"{vals[i]:.2f}",18,INK,True,"mb"); text(d,(x,y0+28),names[i],17,INK,False,"ma")
    text(d,(1470,935),"Selected original CR17 is highlighted in gold; this is a formula robustness check, not a continuous hyper-parameter sweep.",19,MUTED,False,"ra")
    save(img,"09_ReCoMER_gate_sensitivity_image2_style")

def pca2(x):
    x=x.astype(np.float64); x=x-np.nanmean(x,axis=0,keepdims=True); x=np.nan_to_num(x); u,s,_=np.linalg.svd(x,full_matrices=False); z=u[:,:2]*s[:2]; z=(z-z.mean(0))/(z.std(0)+1e-8); return z

def draw_pca():
    feat=np.load(FEATURE_PATH,allow_pickle=True); pred=np.load(LABEL_PATH,allow_pickle=True); ids1=feat["ids"]; ids2=pred["ids"]; assert np.array_equal(ids1,ids2), "feature/prediction IDs do not align"; labels=pred["y_true"]
    rng=np.random.default_rng(20261005); n=min(1800,len(labels)); idx=rng.choice(len(labels),size=n,replace=False); labels=labels[idx]
    mats=[feat["h"][idx],feat["a"][idx],feat["v"][idx]]; titles=["Text feature h","Audio feature a","Visual feature v"]
    img=Image.new("RGB",(3000,1100),"white"); d=ImageDraw.Draw(img); text(d,(90,50),"Modality representation spaces on the exact M3ED test features",38,INK,True); text(d,(90,95),"PCA projections of the recorded cRBEF feature pack; points are coloured by ground-truth emotion",22,MUTED)
    boxes=[(70,145,910,820),(1045,145,910,820),(2020,145,910,820)]
    for j,(mat,title,box) in enumerate(zip(mats,titles,boxes)):
        panel(d,box,f"({chr(97+j)})  {title}"); z=pca2(mat); px,py,pw,ph=box[0]+70,box[1]+110,box[2]-120,box[3]-170; line(d,(px,py+ph,px+pw,py+ph),INK,2); line(d,(px,py,px,py+ph),INK,2)
        for k in range(n):
            xx=px+pw*(z[k,0]-z[:,0].min())/(z[:,0].max()-z[:,0].min()+1e-8); yy=py+ph*(1-(z[k,1]-z[:,1].min())/(z[:,1].max()-z[:,1].min()+1e-8)); c=CLASS_COLORS[int(labels[k])%len(CLASS_COLORS)]; d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=rgb(c))
        text(d,(px+pw/2,py+ph+28),"PC1",18,MUTED,False,"ma"); text(d,(px-30,py+ph/2),"PC2",18,MUTED,False,"mm")
    classes=["Happy","Neutral","Sad","Disgust","Anger","Fear","Surprise"]; lx,ly=110,1015
    for i,cname in enumerate(classes):
        x=lx+i*380; d.ellipse((x,ly-8,x+16,ly+8),fill=rgb(CLASS_COLORS[i])); text(d,(x+26,ly),cname,18,INK,False,"lm")
    save(img,"10_ReCoMER_modality_pca_image2_style")

def confusion(y,p):
    n=int(max(y.max(),p.max())+1); c=np.zeros((n,n),float)
    for a,b in zip(y,p): c[int(a),int(b)]+=1
    return c/(c.sum(1,keepdims=True)+1e-12)

def draw_confusion():
    mats=[]; names=["MHnoU","cRBEF","ReCoMER"]
    for key in ["MHnoU","cRBEF","probabilities"]:
        cs=[]
        for p in PRED_PATHS:
            z=np.load(p,allow_pickle=True); cs.append(confusion(z["y_true"],np.argmax(z[key],axis=1)))
        mats.append(np.mean(cs,axis=0))
    img=Image.new("RGB",(3000,1100),"white"); d=ImageDraw.Draw(img); text(d,(90,50),"Mean confusion structure across the three formal M3ED seeds",38,INK,True); text(d,(90,95),"Rows are true labels; columns are predicted labels; cells show row-normalized percentages",22,MUTED)
    boxes=[(70,145,910,820),(1045,145,910,820),(2020,145,910,820)]; classes=["H","N","S","D","A","F","Su"]
    for m,name,box in zip(mats,names,boxes):
        panel(d,box,name); x0,y0=box[0]+170,box[1]+130; cell=78
        for i in range(7):
            text(d,(x0+i*cell+cell/2,y0-30),classes[i],18,INK,True,"ms"); text(d,(x0-25,y0+i*cell+cell/2),classes[i],18,INK,True,"mm")
            for j in range(7):
                v=m[i,j]; shade=int(255-200*v); fill=(shade,shade,255); d.rectangle((x0+j*cell,y0+i*cell,x0+j*cell+cell-3,y0+i*cell+cell-3),fill=fill,outline=rgb("#FFFFFF"),width=1); text(d,(x0+j*cell+cell/2,y0+i*cell+cell/2),f"{100*v:.0f}",16,INK if v<.5 else "#FFFFFF",True,"mm")
        text(d,(x0+3*cell,y0+7*cell+28),"Predicted",18,MUTED,False,"ma"); text(d,(x0-70,y0+3*cell),"True",18,MUTED,False,"mm")
    save(img,"11_ReCoMER_confusion_matrices_image2_style")

if __name__=="__main__":
    draw_sensitivity(); draw_pca(); draw_confusion(); print("Rendered additional exact-data figures.")
