"""Render exact ReCoMER result charts with a clean Image 2-inspired layout.

This script intentionally uses only the recorded JSON values. It does not use
the Image 2-generated draft as a quantitative source.
"""
from pathlib import Path
import base64, json, math
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results_data_image2_style.json").read_text(encoding="utf-8"))

INK = "#233642"; MUTED = "#6C7A82"; GRID = "#D9E1E5"
BLUE = "#4E8FCB"; ORANGE = "#D9924B"; TEAL = "#4BA9A7"; PURPLE = "#8877C9"
GREEN = "#2E9B63"; RED = "#CE5B5B"; GOLD = "#D59B32"; PALE = "#F7FAFB"

def fnt(size, bold=False):
    paths = [
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for p in paths:
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def text(draw, xy, s, size=24, color=INK, bold=False, anchor="la"):
    draw.text(xy, str(s), font=fnt(size, bold), fill=rgb(color), anchor=anchor)

def line(draw, xy, fill=GRID, width=2): draw.line(xy, fill=rgb(fill), width=width)

def panel(draw, box, title):
    x, y, w, h = box
    draw.rounded_rectangle((x, y, x+w, y+h), radius=22, fill=rgb("#FFFFFF"), outline=rgb("#D5E0E5"), width=2)
    draw.rounded_rectangle((x, y, x+w, y+68), radius=22, fill=rgb("#F4F8FA"), outline=None)
    draw.rectangle((x, y+35, x+w, y+68), fill=rgb("#F4F8FA"), outline=None)
    text(draw, (x+28, y+34), title, 29, INK, True, "lm")

def save_image(img, stem):
    png = ROOT / f"{stem}.png"; tiff = ROOT / f"{stem}.tiff"; pdf = ROOT / f"{stem}.pdf"; svg = ROOT / f"{stem}.svg"
    img.save(png, dpi=(600,600)); img.save(tiff, dpi=(600,600), compression="tiff_lzw")
    img.convert("RGB").save(pdf, "PDF", resolution=300.0)
    # Editable-friendly SVG wrapper retaining the exact high-resolution preview.
    data = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}"><image width="100%" height="100%" href="data:image/png;base64,{data}"/></svg>', encoding="utf-8")

def interp(c1, c2, t):
    a, b = rgb(c1), rgb(c2); t = max(0, min(1, t))
    return tuple(round(a[i] + (b[i]-a[i])*t) for i in range(3))

def diverging(v, lim=0.75):
    if math.isnan(v): return rgb("#F1F4F5")
    if v < 0: return interp("#F3B1B1", "#FFFFFF", (v+lim)/lim)
    return interp("#FFFFFF", "#9AD8B5", v/lim)

def draw_overview():
    W, H = 3000, 1600
    img = Image.new("RGB", (W,H), "white"); d = ImageDraw.Draw(img)
    text(d, (95, 54), "ReCoMER results: main comparison, conditional module gains and ablations", 38, INK, True)
    panel(d, (70, 125, 1390, 1350), "(a)  Frozen M3ED test comparison")
    panel(d, (1510, 125, 1420, 650), "(b)  Module effects across datasets (pp)")
    panel(d, (1510, 835, 1420, 640), "(c)  Integrated ablation deltas")

    # Main bars.
    main = DATA["main_m3ed_test"]; x0, y0, pw, ph = 310, 1300, 1020, 1030
    for tick in range(40, 63, 5):
        yy = y0 - (tick-40)/(62-40)*ph; line(d, (x0,yy,x0+pw,yy), GRID, 2); text(d,(x0-26,yy),tick,22,MUTED,False,"ra")
    colors = ["#A4B1B8", BLUE, ORANGE, TEAL, PURPLE, GOLD]
    bw = 105; gap = 52
    for i, row in enumerate(main):
        x = x0 + 45 + i*(bw+gap); v = row["wf1"]; e=row["sd"]
        top = y0 - (v-40)/(62-40)*ph; bottom=y0
        d.rounded_rectangle((x,top,x+bw,bottom), radius=8, fill=rgb(colors[i]), outline=rgb("#FFFFFF"), width=2)
        if e:
            ey = y0 - (v+e-40)/(62-40)*ph; line(d,(x+bw/2,ey,x+bw/2,top),INK,3); line(d,(x+bw/2-14,ey,x+bw/2+14,ey),INK,3)
        text(d,(x+bw/2,top-18),f"{v:.2f}",22,INK,True,"ms")
        # two-line labels
        lab = row["model"].replace("-", "-\n")
        text(d,(x+bw/2,y0+34),lab,20,INK,False,"ma")
    text(d,(x0-85, 765), "WF1 (%)", 24, INK, True, "mm")
    text(d,(720, 1374), "+0.607 pp vs equal-weight", 22, PURPLE, True, "ma")

    # Heatmap.
    modules = ["History\nfeedback", "Evidence\nRouter", "History\nabstention"]
    datasets = ["M3ED", "MELD", "IEMOCAP", "MOSEI"]
    lookup = {(r["module"],r["dataset"]): r["effect_pp"] for r in DATA["module_effects_pp"]}
    hx, hy, cw, ch = 1920, 300, 230, 105
    for j, ds in enumerate(datasets): text(d,(hx+j*cw+cw/2,hy-32),ds,22,INK,True,"ms")
    for i, mod in enumerate(["History feedback","EvidenceRouter","History abstention"]):
        text(d,(hx-36,hy+i*ch+ch/2),modules[i],20,INK,False,"rm")
        for j, ds in enumerate(datasets):
            v = lookup.get((mod,ds), float("nan")); box=(hx+j*cw,hy+i*ch,hx+j*cw+cw-8,hy+i*ch+ch-8)
            d.rounded_rectangle(box, radius=8, fill=diverging(v), outline=rgb("#FFFFFF"), width=2)
            text(d,((box[0]+box[2])/2,(box[1]+box[3])/2),"—" if math.isnan(v) else f"{v:+.3f}",22,INK,True,"mm")
    text(d,(2220,700),"positive = WF1 ↑ or MAE ↓",19,MUTED,False,"ma")

    # Integrated deltas.
    rows = list(reversed(DATA["integrated_deltas_pp"])); ix, iy, iw = 1920, 980, 830; scale=270
    line(d,(ix+iw/2,iy-10,ix+iw/2,iy+len(rows)*65+20),INK,2)
    for i,row in enumerate(rows):
        yy=iy+i*65+24; val=row["delta_pp"]; lo=row["ci_low"]; hi=row["ci_high"]
        text(d,(ix-25,yy),row["control"],20,INK,False,"ra")
        x=ix+iw/2+val*scale; xl=ix+iw/2+lo*scale; xh=ix+iw/2+hi*scale
        line(d,(xl,yy,xh,yy),"#7D8A90",3); line(d,(xl,yy-8,xl,yy+8),"#7D8A90",3); line(d,(xh,yy-8,xh,yy+8),"#7D8A90",3)
        d.ellipse((x-9,yy-9,x+9,yy+9), fill=rgb(GREEN if val>=0 else RED), outline=rgb("#FFFFFF"), width=2)
        text(d,(ix+iw+18,yy),f"{val:+.3f}",20,GREEN if val>=0 else RED,True,"lm")
    text(d,(ix+iw/2,1415),"Full model − control (WF1 pp); bars show 95% bootstrap CI",18,MUTED,False,"ms")
    save_image(img, "07_ReCoMER_results_overview_image2_style")

def draw_exact():
    W,H=2800,1050; img=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(img)
    text(d,(90,50),"Exact-feature protocol: supplementary reproducibility checks",38,INK,True)
    panel(d,(70,125,1280,820),"(a)  Exact-feature M3ED controls")
    panel(d,(1450,125,1280,820),"(b)  Exact-feature cRBEF replication")
    for px, data, ymin, ymax, colors, ylabel in [
        (280, DATA["exact_feature_m3ed"], 52.4, 54.9, ["#A4B1B8",TEAL,BLUE,PURPLE,ORANGE], "WF1 (%)"),
        (1660, DATA["exact_feature_crbef"], 38, 57, [BLUE,TEAL,ORANGE,PURPLE,GOLD], "WF1 (%)")]:
        pw=930; y0=850; ph=590
        ticks=[52.5,53,53.5,54,54.5] if ymin>50 else [40,45,50,55]
        for t in ticks:
            yy=y0-(t-ymin)/(ymax-ymin)*ph; line(d,(px,yy,px+pw,yy),GRID,2); text(d,(px-20,yy),t,20,MUTED,False,"ra")
        bw=100; gap=(pw-bw*len(data))/(len(data)+1)
        for i,row in enumerate(data):
            x=px+gap+i*(bw+gap); v=row["wf1"]; e=row["sd"]; top=y0-(v-ymin)/(ymax-ymin)*ph
            d.rounded_rectangle((x,top,x+bw,y0),radius=8,fill=rgb(colors[i]),outline=rgb("#FFFFFF"),width=2)
            ey=y0-(v+e-ymin)/(ymax-ymin)*ph; line(d,(x+bw/2,ey,x+bw/2,top),INK,3); line(d,(x+bw/2-12,ey,x+bw/2+12,ey),INK,3)
            text(d,(x+bw/2,top-15),f"{v:.2f}",19,INK,True,"ms")
            text(d,(x+bw/2,y0+30),row["arm"],17,INK,False,"ma")
        text(d,(px-75,520),ylabel,22,INK,True,"mm")
    save_image(img, "08_ReCoMER_exact_feature_checks_image2_style")

if __name__ == "__main__":
    draw_overview(); draw_exact(); print("Rendered exact-data Image 2-style figures.")
