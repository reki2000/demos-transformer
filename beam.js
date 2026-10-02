'use strict';
// Each hypothesis feeds only its own predicted characters back into the decoder.
async function decoderBeamGenerate(weights,seedIds,width=5,limit=40,cancelled=()=>false,yieldStep=()=>new Promise(resolve=>setTimeout(resolve,0))){
 let active=[{tokens:[],logProbability:0,reason:null}],finished=[];
 const compare=(a,b)=>b.logProbability-a.logProbability;
 for(let step=0;step<limit&&active.length;step++){
  if(cancelled())return null;const candidates=[];
  for(const beam of active){
   const probs=decoderPredict(weights,seedIds.concat(beam.tokens));
   // The start token and padding are excluded; EOS remains a valid outcome.
   const ranked=probs.map((p,id)=>({id,p})).filter(x=>x.id>=2).sort((a,b)=>b.p-a.p).slice(0,width);
   for(const next of ranked){const logProbability=beam.logProbability+Math.log(Math.max(next.p,1e-300));if(next.id===2)finished.push({tokens:beam.tokens.slice(),logProbability,reason:'eos',scoredLength:beam.tokens.length+1});else candidates.push({tokens:beam.tokens.concat(next.id),logProbability,reason:null})}
  }
  active=candidates.sort(compare).slice(0,width);
  // Bound the finished pool while preserving the final length-normalized ranking.
  finished=finished.sort((a,b)=>b.logProbability/b.scoredLength-a.logProbability/a.scoredLength).slice(0,50);
  await yieldStep();
 }
 if(cancelled())return null;
 const all=finished.concat(active.map(b=>({...b,reason:'limit',scoredLength:b.tokens.length}))),seen=new Set();
 return all.map(b=>({...b,score:b.logProbability/Math.max(1,b.scoredLength)})).sort((a,b)=>b.score-a.score).filter(b=>{const key=b.tokens.join(',');if(seen.has(key))return false;seen.add(key);return true}).slice(0,width);
}
if(typeof module!=='undefined')module.exports={decoderBeamGenerate};
