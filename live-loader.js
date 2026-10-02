'use strict';
const archiveData=JSON.parse(document.getElementById('training-data').textContent);
const wasmBytes=Uint8Array.from(atob(document.getElementById('wasm-data').textContent.trim()),c=>c.charCodeAt(0));
if(typeof WebAssembly==='undefined'||!WebAssembly.validate(wasmBytes))throw Error('WebAssembly SIMDに対応する現行のブラウザで開いてください。');
let activeModelId='2-16',data={...archiveData,config:{dimension:16,layers:2,ff:32,heads:1,maxContext:Math.max(...archiveData.rows.map(r=>r.length))-1,parameters:0,trainCount:archiveData.trainIndices.length,testCount:archiveData.testIndices.length,batchSize:16,learningRate:.003,epochs:4,totalSteps:Math.ceil(archiveData.trainIndices.length/16)*4},snapshots:[]};
function makeWorker(scriptId){const url=URL.createObjectURL(new Blob([document.getElementById(scriptId).textContent],{type:'text/javascript'}));const worker=new Worker(url);URL.revokeObjectURL(url);return worker}
let trainWorker=makeWorker('train-worker'),inferWorker=makeWorker('infer-worker');
