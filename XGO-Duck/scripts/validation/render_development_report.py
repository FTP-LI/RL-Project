"""Render the main report and check all local references, using installed Markdown.

Run with the existing Linux training Python; no additional package install.
Media stays in its evidence directory and loads on demand. Embedding every
historical clip made the old HTML about 70 MB and slow to open in VS Code.
Keep the project folder together for offline playback; no server is needed.
"""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
import markdown

root=Path(__file__).resolve().parents[2]
source=root/'docs/reports/development_report.md'
text=source.read_text(encoding='utf-8')
links=[]
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
    if '://' in target or target.startswith('#'):
        continue
    path=(source.parent/target.split('#')[0]).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise SystemExit(f'Missing or out-of-project report reference: {target}')
    links.append({'target':target,'size':path.stat().st_size})
if re.search(r'\bV\d+\b|修订稿|版本号',text):
    raise SystemExit('Internal development label in report')
body=markdown.markdown(text,extensions=['tables','fenced_code','toc'],extension_configs={'toc':{'toc_depth':'2'}})
body=body.replace('<img ', '<img loading="lazy" ')
body=re.sub(r'<video\b([^>]*)>', lambda m:'<video'+re.sub(r'\s+preload="[^"]*"','',m[1])+' preload="none">',body)
body=body.replace('href="development_history.md', 'href="development_history.html')
style='''body{font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#243547;background:#edf2f5;margin:0}main{max-width:1000px;margin:auto;background:white;padding:44px 55px}h1{font-size:32px}h2{margin-top:52px;border-bottom:2px solid #dce8ed;padding-bottom:10px}h3{margin-top:30px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #cedae1;padding:9px 12px;text-align:left}th{background:#eaf2f6}pre{background:#142434;color:#eff6fa;padding:20px;overflow:auto;border-radius:8px;font-size:13px;line-height:1.6}code{overflow-wrap:anywhere}img,video{max-width:100%;height:auto}a{color:#007e9e}p,td{overflow-wrap:anywhere}@media print{body{background:white}main{padding:0}h2{break-before:auto}pre{white-space:pre-wrap}tr,img{break-inside:avoid}}'''
out=source.with_suffix('.html')
out.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>XGO-Duck 开发与部署报告</title><style>'+style+'</style><main>'+body+'</main></html>',encoding='utf-8')
record=root/'records/reports'
record.mkdir(parents=True,exist_ok=True)
result={'checked_at':datetime.now(timezone.utc).isoformat(),'markdown':str(source),'html':str(out),'local_links_checked':len(links),'links':links,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'html_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'status':'passed','media_mode':'relative-files-on-demand','offline_requirement':'Keep the XGO-Duck folder structure together'}
(record/'development-report-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({key:result[key] for key in ['status','local_links_checked','html','html_sha256']},indent=2))

# Keep all historical commands and media accessible without copying video data.
history=source.with_name('development_history.md')
if history.exists():
    history_text=history.read_text(encoding='utf-8')
    history_body=markdown.markdown(history_text,extensions=['tables','fenced_code','toc'])
    history_body=history_body.replace('<img ', '<img loading="lazy" ')
    history_body=re.sub(r'<video\b([^>]*)>',lambda m:'<video'+re.sub(r'\s+preload="[^"]*"','',m[1])+' preload="none">',history_body)
    history_body=history_body.replace('href="development_report.md','href="development_report.html')
    history.with_suffix('.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>XGO-Duck 完整过程记录</title><style>'+style+'</style><main>'+history_body+'</main></html>',encoding='utf-8')
