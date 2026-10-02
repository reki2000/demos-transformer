'use strict';
// Actual autoregressive inference with the weights of the selected training state.
function decoderForward(weights, tokenIds, traceMode=false) {
  const dim = weights.E[0].length;
  const ids = tokenIds.slice(-weights.P.length);
  const linear = (row, matrix) => {
    const out = Array(matrix[0].length).fill(0);
    for (let i = 0; i < row.length; i++) {
      const scalar = row[i], wr = matrix[i];
      for (let j = 0; j < out.length; j++) out[j] += scalar * wr[j];
    }
    return out;
  };
  const norm = row => {
    let mean = 0; for (const x of row) mean += x; mean /= row.length;
    let variance = 0; for (const x of row) variance += (x - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / row.length + 1e-5);
    return row.map(x => (x - mean) * scale);
  };
  const softmax = row => {
    const max = Math.max(...row), exp = row.map(x => Math.exp(x - max));
    const total = exp.reduce((a,b) => a+b,0); return exp.map(x => x/total);
  };
  let x = ids.map((id,t) => weights.E[id].map((v,j) => v + weights.P[t][j]));
  const trace={tokenEmbedding:ids.map(id=>weights.E[id].slice()),positionEmbedding:ids.map((_,t)=>weights.P[t].slice()),embedding:x.map(row=>row.slice()),blocks:[]};
  for (let block = 0; weights[block+'q']; block++) {
    const normalized = x.map(norm), q = normalized.map(r=>linear(r,weights[block+'q'])),
      k = normalized.map(r=>linear(r,weights[block+'k'])), v = normalized.map(r=>linear(r,weights[block+'v']));
    const attentionRows=[],mixedRows=[];
    const attentionOut = q.map((query,t) => {
      const scores = k.slice(0,t+1).map(key => {
        let dot=0;for(let j=0;j<dim;j++)dot+=query[j]*key[j];return dot/Math.sqrt(dim);
      });
      const attention = softmax(scores), mixed=Array(dim).fill(0);
      for(let i=0;i<=t;i++)for(let j=0;j<dim;j++)mixed[j]+=attention[i]*v[i][j];
      if(traceMode){attentionRows.push([...attention,...Array(ids.length-t-1).fill(0)]);mixedRows.push(mixed)}
      return linear(mixed,weights[block+'o']);
    });
    const residual=x.map((row,t)=>row.map((a,j)=>a+attentionOut[t][j]));
    const normFF=residual.map(norm),hiddenPre=normFF.map(row=>linear(row,weights[block+'f'])),hidden=hiddenPre.map(row=>row.map(Math.tanh)),mlp=hidden.map(row=>linear(row,weights[block+'b']));
    x=residual.map((row,t)=>row.map((a,j)=>a+mlp[t][j]));
    if(traceMode)trace.blocks.push({normAttention:normalized,q,k,v,attention:attentionRows,attentionMixed:mixedRows,attentionOutput:attentionOut,attentionResidual:residual,normFF,hiddenPre,hidden,ffOutput:mlp,output:x});
  }
  if(traceMode){trace.finalNorm=x.map(norm);trace.logits=trace.finalNorm.map(row=>linear(row,weights.head));trace.probs=trace.logits.map(softmax);return trace}
  return softmax(linear(norm(x.at(-1)),weights.head));
}
function decoderPredict(weights,tokenIds){return decoderForward(weights,tokenIds,false)}
function decoderTrace(weights,tokenIds){return decoderForward(weights,tokenIds,true)}
function decoderGenerate(weights, seedIds, limit=40) {
  const input=seedIds.slice(), generated=[];let reason='limit';
  for(let t=0;t<limit;t++){
    const probs=decoderPredict(weights,input);let next=2;
    // Padding and the start token are not valid generated text.
    for(let i=3;i<probs.length;i++)if(probs[i]>probs[next])next=i;
    if(next===2){reason='eos';break}
    generated.push(next);input.push(next);
  }
  return {tokens:generated,reason,contextWindow:weights.P.length};
}
if(typeof module!=='undefined')module.exports={decoderPredict,decoderTrace,decoderGenerate};
