"""Render responses.html as a standalone PDF using ReportLab."""
from html import escape
from pathlib import Path
from bs4 import BeautifulSoup
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, Image, KeepTogether, PageBreak
import matplotlib

P=Path(__file__).resolve().parent
font_dir=Path('/System/Library/Fonts/Supplemental')
if (font_dir/'Arial.ttf').exists():
    normal_font=font_dir/'Arial.ttf'
    bold_font=font_dir/'Arial Bold.ttf'
else:
    font_dir=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    normal_font=font_dir/'DejaVuSans.ttf'
    bold_font=font_dir/'DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('ArialLocal',str(normal_font)))
pdfmetrics.registerFont(TTFont('ArialLocalBold',str(bold_font)))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyLocal',fontName='ArialLocal',fontSize=8.5,leading=11,spaceAfter=6))
styles.add(ParagraphStyle(name='TitleLocal',fontName='ArialLocalBold',fontSize=19,leading=23,spaceAfter=10,textColor=colors.HexColor('#263d5b')))
styles.add(ParagraphStyle(name='Head2Local',fontName='ArialLocalBold',fontSize=14,leading=17,spaceAfter=9,spaceBefore=7,textColor=colors.HexColor('#263d5b')))
styles.add(ParagraphStyle(name='Head3Local',fontName='ArialLocalBold',fontSize=11,leading=14,spaceAfter=5,spaceBefore=8,textColor=colors.HexColor('#263d5b')))
styles.add(ParagraphStyle(name='Head4Local',fontName='ArialLocalBold',fontSize=9,leading=12,spaceAfter=4,spaceBefore=6))
styles.add(ParagraphStyle(name='TableLocal',fontName='ArialLocal',fontSize=6.6,leading=8))
styles.add(ParagraphStyle(name='TableHeadLocal',fontName='ArialLocalBold',fontSize=6.6,leading=8))
styles.add(ParagraphStyle(name='CaptionLocal',fontName='ArialLocal',fontSize=7.5,leading=10,alignment=TA_CENTER,textColor=colors.HexColor('#556677')))

def markup(node):
    # Preserve limited inline styling and escape literal angle brackets.
    out=[]
    for c in node.children:
        if isinstance(c,str):out.append(escape(c))
        elif c.name=='code':out.append('<font name="ArialLocal">'+markup(c)+'</font>')
        elif c.name in ('b','strong','i','em','sup','sub'):out.append(f'<{c.name}>'+markup(c)+f'</{c.name}>')
        else:out.append(markup(c))
    return ''.join(out)

def para(node,style):return Paragraph(markup(node),styles[style])

soup=BeautifulSoup((P/'responses.html').read_text(),'html.parser')
story=[]
seen_section=False
for node in soup.body.children:
    if not getattr(node,'name',None):continue
    if node.name=='h1':story.append(para(node,'TitleLocal'))
    elif node.name=='h2':
        if seen_section and node.get_text(strip=True)!='Methods and reproducibility':story.append(PageBreak())
        story.append(para(node,'Head2Local'))
        seen_section=True
    elif node.name=='h3':story.append(para(node,'Head3Local'))
    elif node.name=='h4':story.append(para(node,'Head4Local'))
    elif node.name=='p':story.append(para(node,'BodyLocal'))
    elif node.name=='table':
        rows=[]
        for tr in node.find_all('tr'):
            rows.append([para(cell,'TableHeadLocal' if cell.name=='th' else 'TableLocal') for cell in tr.find_all(['td','th'])])
        ncol=max(map(len,rows));width=(A4[0]-64)/ncol
        t=LongTable(rows,colWidths=[width]*ncol,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.4,colors.HexColor('#8da2b7')),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d5dde5')),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.extend([t,Spacer(1,8)])
    elif node.name=='figure':
        src=P/node.img['src'];im=Image(str(src));scale=min((A4[0]-64)/im.imageWidth,330/im.imageHeight);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
        story.append(KeepTogether([im,Spacer(1,4),para(node.figcaption,'CaptionLocal'),Spacer(1,8)]))

def page_num(canvas,doc):
    canvas.setFont('ArialLocal',8);canvas.setFillColor(colors.HexColor('#667788'))
    canvas.drawRightString(A4[0]-32,22,str(doc.page))

doc=SimpleDocTemplate(str(P/'responses.pdf'),pagesize=A4,leftMargin=32,rightMargin=32,topMargin=32,bottomMargin=35,title='Assignment 3 responses')
doc.build(story,onFirstPage=page_num,onLaterPages=page_num)
print('Wrote',P/'responses.pdf')
