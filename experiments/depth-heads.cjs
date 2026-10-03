// Trains one configuration on the SF stories and records what the comparison needs:
// loss, attention distance per block and head, and stories generated from evaluation titles.
// Usage: node experiments/depth-heads.cjs <layers> <dimension> <heads> [epochs]
const fs=require('fs'),path=require('path');
const root=path.join(__dirname,'..');
const {WasmTransformer}=require(root+'/engine.js');
const {decoderTrace,decoderCache,decoderStep}=require(root+'/decoder.js');
const [layers,dimension,heads,epochs=4]=process.argv.slice(2).map(Number);
const data=JSON.parse(fs.readFileSync(root+'/corpus-sf.json')),context=Math.max(...data.rows.map(r=>r.length))-1;
const BATCH=16,LR=.003,EVAL_ATTENTION=64,GENERATE=120;

function shuffle(items,state){const copy=items.slice();for(let i=copy.length-1;i>0;i--){state^=state<<13;state^=state>>>17;state^=state<<5;const j=(state>>>0)%(i+1);[copy[i],copy[j]]=[copy[j],copy[i]]}return [copy,state]}

// Mean distance (in characters) from each query position to what it attends to,
// and the share of attention reaching 20 or more characters back.
function attentionStats(rows){let distance=0,far=0,count=0;rows.forEach((row,i)=>{if(!i)return;let d=0,f=0;for(let j=0;j<=i;j++){d+=row[j]*(i-j);if(i-j>=20)f+=row[j]}distance+=d;far+=f;count++});return {distance:distance/count,far:far/count}}

(async()=>{
 const started=Date.now(),engine=await WasmTransformer.create(fs.readFileSync(root+'/engine.wasm'),{dimension,layers,heads,vocabSize:data.vocab.length,context,seed:42});
 let state=2026,curve=[];const steps=Math.ceil(data.trainIndices.length/BATCH)*epochs,probe=data.testIndices.slice(0,128);
 for(let epoch=0;epoch<epochs;epoch++){let order;[order,state]=shuffle(data.trainIndices,state);for(let i=0;i<order.length;i+=BATCH){engine.train(data.rows,order.slice(i,i+BATCH),LR);if(engine.step%100===0||engine.step===steps){curve.push({step:engine.step,evalLoss:engine.evaluate(data.rows,probe).loss});process.stderr.write(`${layers}x${dimension}h${heads} step ${engine.step}/${steps} eval ${curve.at(-1).evalLoss.toFixed(3)}\n`)}}}
 const trainMs=Date.now()-started,test=engine.evaluate(data.rows,data.testIndices),train=engine.evaluate(data.rows,data.trainIndices.slice(0,754)),weights=engine.arrays();

 // Attention distance per block and head on evaluation stories.
 const attention=Array.from({length:layers},()=>Array.from({length:heads},()=>({distance:0,far:0})));
 for(const index of data.testIndices.slice(0,EVAL_ATTENTION)){const trace=decoderTrace(weights,data.rows[index].slice(0,-1));trace.blocks.forEach((block,b)=>{for(let h=0;h<heads;h++){const s=attentionStats(block['attention'+h]);attention[b][h].distance+=s.distance/EVAL_ATTENTION;attention[b][h].far+=s.far/EVAL_ATTENTION}})}

 // Greedy generation from each evaluation title (〈始〉 through 』) with the KV cache.
 const close=data.vocab.indexOf('』'),generations=[];
 for(const index of data.testIndices.slice(0,GENERATE)){const row=data.rows[index],prompt=row.slice(0,row.indexOf(close)+1),cache=decoderCache(weights);let probs;for(const id of prompt)probs=decoderStep(weights,cache,id);const out=[];
  while(cache.length<context){let best=2;for(let i=3;i<probs.length;i++)if(probs[i]>probs[best])best=i;if(best===2)break;out.push(best);probs=decoderStep(weights,cache,best)}
  generations.push({index,reference:data.corpusRecords[index].text,text:prompt.slice(1).concat(out).map(id=>data.vocab[id]).join('')})}

 const result={layers,dimension,heads,epochs,steps,parameters:engine.count,trainMs,msPerStep:trainMs/steps,trainLoss:train.loss,testLoss:test.loss,testAccuracy:test.accuracy,curve,attention,generations};
 fs.writeFileSync(path.join(__dirname,'results',`${layers}x${dimension}h${heads}.json`),JSON.stringify(result));
 console.log(`${layers}x${dimension}h${heads}`,'test loss',test.loss.toFixed(3),'minutes',(trainMs/60000).toFixed(1));
})().catch(e=>{console.error(e);process.exitCode=1});
