'use strict';
let generationRun=0;
onmessage=async event=>{const m=event.data;if(m.type==='cancel'){generationRun++;return}if(m.type!=='generate')return;const run=++generationRun;try{const results=await decoderBeamGenerate(m.weights,m.seed,5,40,()=>run!==generationRun);if(results&&run===generationRun)postMessage({type:'generated',requestId:m.requestId,scope:m.scope,results})}catch(error){if(run===generationRun)postMessage({type:'generationError',requestId:m.requestId,message:error.message})}};
