'use strict';
// In the browser decoder.js is concatenated before this file; Node loads it here.
if(typeof module!=='undefined'&&typeof decoderForward==='undefined')var {decoderForward,decoderModel}=require('./decoder.js');
// Masked-diffusion sampling. The sequence starts as 〈始〉 + prompt + masks up to the context length.
// Each step predicts every masked position at once and reveals the most confident predictions
// (low-confidence remasking, as in LLaDA); the rest stay masked for the next step. Every intermediate state is returned.
async function diffusionGenerate(weights,seedIds,steps=32,cancelled=()=>false,yieldStep=()=>new Promise(resolve=>setTimeout(resolve,0))){
 const context=decoderModel(weights).context,seq=seedIds.slice(0,context),start=seq.length;while(seq.length<context)seq.push(0);
 const masked=new Set(Array.from({length:context-start},(_,i)=>start+i)),total=masked.size,frames=[{tokens:seq.slice(),revealed:[]}];steps=Math.max(1,Math.min(steps,total));
 for(let s=0;s<steps&&masked.size;s++){
  if(cancelled())return null;
  const probs=decoderForward(weights,seq,'all'),candidates=[];
  // Padding (= mask) and the start token are never valid predictions; 〈終〉 is.
  for(const pos of masked){let best=2;for(let id=3;id<probs[pos].length;id++)if(probs[pos][id]>probs[pos][best])best=id;candidates.push({pos,id:best,p:probs[pos][best]})}
  const count=Math.round(total*(s+1)/steps)-Math.round(total*s/steps),chosen=candidates.sort((a,b)=>b.p-a.p).slice(0,count);
  for(const c of chosen){seq[c.pos]=c.id;masked.delete(c.pos)}
  frames.push({tokens:seq.slice(),revealed:chosen.map(c=>c.pos)});
  await yieldStep();
 }
 if(cancelled())return null;
 const end=seq.indexOf(2,start);
 return {tokens:seq.slice(start,end<0?context:end),reason:end<0?'limit':'eos',frames,start,steps};
}
if(typeof module!=='undefined')module.exports={diffusionGenerate};
