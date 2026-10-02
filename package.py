from pathlib import Path
import json,gzip,base64,zipfile
p=Path(__file__).parent
common=json.loads((p/'corpus-common.json').read_text());assert len(common['rows'])==5000 and len(common['trainIndices'])==4286 and len(common['testIndices'])==714
blob=base64.b64encode(gzip.compress(json.dumps(common,ensure_ascii=False,separators=(',',':')).encode(),compresslevel=9,mtime=0)).decode()
loader=(p/'live-loader.js').read_text().replace("JSON.parse(document.getElementById('training-data').textContent)","JSON.parse(await new Response(new Blob([Uint8Array.from(atob(document.getElementById('training-data').textContent.trim()),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())")
js="(async()=>{\n"+loader+'\n'+(p/'decoder.js').read_text()+'\n'+(p/'live-app.js').read_text()+'\n'+(p/'details.js').read_text()+"\nawait initializeModel();\n})().catch(error=>{document.getElementById('status').textContent='開始できませんでした：'+error.message});"
training=(p/'engine.js').read_text()+'\n'+(p/'train-worker.js').read_text();inference=(p/'decoder.js').read_text()+'\n'+(p/'beam.js').read_text()+'\n'+(p/'infer-worker.js').read_text()
for code in [training,inference,js]:assert '</script' not in code.lower()
h=(p/'live-page.html').read_text().replace('__CSS__',(p/'style.css').read_text()).replace('__DATA__',blob).replace('__WASM__',base64.b64encode((p/'engine.wasm').read_bytes()).decode()).replace('__TRAIN_WORKER__',training).replace('__INFER_WORKER__',inference).replace('__JS__',js)
assert '__' not in h.replace('__proto__','')
assert '<script src=' not in h and 'weightsPayload' not in h
output_dir=p/'dist';output_dir.mkdir(exist_ok=True)
out=output_dir/'index.html';archive=output_dir/'transformer-lab.zip';out.write_text(h);out.chmod(0o644)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:z.write(out,out.name)
(p/'combined.js').write_text(js);(p/'combined-train-worker.js').write_text(training);(p/'combined-infer-worker.js').write_text(inference)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
Path(archive).chmod(0o644)
print('Single offline HTML:',out.stat().st_size,'bytes; ZIP:',Path(archive).stat().st_size,'bytes; embedded Wasm:',(p/'engine.wasm').stat().st_size,'bytes')
