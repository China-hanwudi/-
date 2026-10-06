"""Inspect vector outputs, audit PDF typography/collisions and build review assets."""
from pathlib import Path
import sys
import subprocess
import json
import pymupdf
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
SCRIPTS=Path.home()/'.codex/skills/nature-figure/scripts'
manifest=json.loads((ROOT/'figure_manifest.json').read_text(encoding='utf-8'))
reports=[]
merged=pymupdf.open()
for rec in manifest:
    name=rec['name']
    pdf=ROOT/f'{name}.pdf'
    for tool,flags in [
        ('audit_pdf_text.py',['--min-pt','5','--json']),
        ('audit_figure_collisions.py',['--json-out',str(ROOT/'qa'/f'{name}.collision.json'),
                                      '--overlay-pdf',str(ROOT/'qa'/f'{name}.collision.pdf')])]:
        run=subprocess.run([sys.executable,str(SCRIPTS/tool),str(pdf),*flags],
                           text=True,capture_output=True,encoding='utf-8',errors='replace')
        (ROOT/'qa'/f'{name}.{tool}.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
        reports.append({'figure':name,'check':tool,'exit_code':run.returncode})
    d=pymupdf.open(pdf)
    merged.insert_pdf(d)
    d.close()
merged.save(ROOT/'ReCoMER_structure_bundle.pdf')
merged.close()
# Review contact sheet; submission assets remain the six original vector PDFs/SVGs.
canvas=Image.new('RGB',(2200,2400),'white')
draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
for i,rec in enumerate(manifest):
    tile=Image.open(ROOT/f"{rec['name']}.png").convert('RGB')
    tile=ImageOps.contain(tile,(1040,710))
    col,row=i%2,i//2
    x,y=col*1100+(1100-tile.width)//2,row*800+65
    canvas.paste(tile,(x,y))
    draw.text((col*1100+35,row*800+20),rec['name'].replace('_',' '),font=font,fill='#26343E')
canvas.save(ROOT/'structure_contact_sheet.png')
source=subprocess.run([sys.executable,str(SCRIPTS/'validate_figure.py'),str(ROOT/'draw_structures.py'),'--json'],
                      text=True,capture_output=True,encoding='utf-8',errors='replace')
(ROOT/'qa'/'source-validation.json').write_text(source.stdout,encoding='utf-8')
(ROOT/'qa'/'verification_summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
checks=[]
for rec in manifest:
    name=rec['name']
    font=json.loads((ROOT/'qa'/f'{name}.audit_pdf_text.py.txt').read_text(encoding='utf-8'))
    collision=json.loads((ROOT/'qa'/f'{name}.collision.json').read_text(encoding='utf-8'))
    checks.append({'figure':name,'minimum_font_pt':font['minimum_found_pt'],
                   'collision_verdict':collision['verdict'],
                   'box_fit_failures':rec['box_fit_failures'],
                   'alignment':rec['alignment']})
(ROOT/'qa'/'delivery_preflight.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(reports,indent=2))
if any(r['exit_code'] for r in reports) or any(r['box_fit_failures'] for r in checks):
    raise SystemExit('Fix failed checks before delivery.')
