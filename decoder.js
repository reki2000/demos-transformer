'use strict';
// Actual autoregressive inference with the weights of the selected training state.
const decoderLinear = (row, matrix) => {
  const out = Array(matrix[0].length).fill(0);
  for (let i = 0; i < row.length; i++) {
    const scalar = row[i], wr = matrix[i];
    for (let j = 0; j < out.length; j++) out[j] += scalar * wr[j];
  }
  return out;
};
const decoderNorm = row => {
  let mean = 0; for (const x of row) mean += x; mean /= row.length;
  let variance = 0; for (const x of row) variance += (x - mean) ** 2;
  const scale = 1 / Math.sqrt(variance / row.length + 1e-5);
  return row.map(x => (x - mean) * scale);
};
const decoderSoftmax = row => {
  const max = Math.max(...row), exp = row.map(x => Math.exp(x - max));
  const total = exp.reduce((a,b) => a+b,0); return exp.map(x => x/total);
};
// Causal attention for one query position. With several heads, head h uses
// columns h·dh … (h+1)·dh−1 of Q, K and V; the mixed values are concatenated.
function decoderAttend(query, keys, values, heads) {
  const dim = query.length, dh = dim / heads, mixed = Array(dim).fill(0), attention = [];
  for (let head = 0; head < heads; head++) {
    const o = head * dh;
    const scores = keys.map(key => {
      let dot=0;for(let j=o;j<o+dh;j++)dot+=query[j]*key[j];return dot/Math.sqrt(dh);
    });
    const weights = decoderSoftmax(scores);
    for(let i=0;i<keys.length;i++)for(let j=o;j<o+dh;j++)mixed[j]+=weights[i]*values[i][j];
    attention.push(weights);
  }
  return {mixed, attention};
}
function decoderForward(weights, tokenIds, traceMode=false) {
  const dim = weights.E[0].length;
  const ids = tokenIds.slice(-weights.P.length);
  const linear = decoderLinear, norm = decoderNorm, softmax = decoderSoftmax;
  let x = ids.map((id,t) => weights.E[id].map((v,j) => v + weights.P[t][j]));
  const trace={tokenEmbedding:ids.map(id=>weights.E[id].slice()),positionEmbedding:ids.map((_,t)=>weights.P[t].slice()),embedding:x.map(row=>row.slice()),blocks:[]};
  for (let block = 0; weights[block+'q']; block++) {
    const normalized = x.map(norm), q = normalized.map(r=>linear(r,weights[block+'q'])),
      k = normalized.map(r=>linear(r,weights[block+'k'])), v = normalized.map(r=>linear(r,weights[block+'v']));
    const heads = weights.heads || 1, attentionRows=[], headRows=Array.from({length:heads},()=>[]), mixedRows=[];
    const pad = row => [...row, ...Array(ids.length-row.length).fill(0)];
    const attentionOut = q.map((query,t) => {
      const {mixed, attention} = decoderAttend(query, k.slice(0,t+1), v, heads);
      if(traceMode){
        // The single map shown for a block is the mean over heads.
        attentionRows.push(pad(attention[0].map((_,i)=>attention.reduce((a,h)=>a+h[i],0)/heads)));
        attention.forEach((h,i)=>headRows[i].push(pad(h)));mixedRows.push(mixed);
      }
      return linear(mixed,weights[block+'o']);
    });
    const residual=x.map((row,t)=>row.map((a,j)=>a+attentionOut[t][j]));
    const normFF=residual.map(norm),hiddenPre=normFF.map(row=>linear(row,weights[block+'f'])),hidden=hiddenPre.map(row=>row.map(Math.tanh)),mlp=hidden.map(row=>linear(row,weights[block+'b']));
    x=residual.map((row,t)=>row.map((a,j)=>a+mlp[t][j]));
    if(traceMode)trace.blocks.push({normAttention:normalized,q,k,v,attention:attentionRows,...Object.fromEntries(headRows.map((rows,h)=>['attention'+h,rows])),attentionMixed:mixedRows,attentionOutput:attentionOut,attentionResidual:residual,normFF,hiddenPre,hidden,ffOutput:mlp,output:x});
  }
  if(traceMode){trace.finalNorm=x.map(norm);trace.logits=trace.finalNorm.map(row=>linear(row,weights.head));trace.probs=trace.logits.map(softmax);return trace}
  return softmax(linear(norm(x.at(-1)),weights.head));
}
// Incremental decoding with a key/value cache: each block keeps the K and V
// rows of earlier positions, so a new token costs one position, not the prefix.
// Rows are never modified, so a beam can share its parent's rows.
function decoderCache(weights){const blocks=[];for(let b=0;weights[b+'q'];b++)blocks.push({k:[],v:[]});return {length:0,blocks}}
function decoderCacheClone(cache){return {length:cache.length,blocks:cache.blocks.map(b=>({k:b.k.slice(),v:b.v.slice()}))}}
// Appends one token and returns the next-token probabilities, matching decoderPredict.
function decoderStep(weights, cache, id) {
  const t = cache.length;
  if (t >= weights.P.length) throw Error('文脈長を超えました');
  let x = weights.E[id].map((v,j) => v + weights.P[t][j]);
  cache.blocks.forEach((c, block) => {
    const normalized = decoderNorm(x), q = decoderLinear(normalized, weights[block+'q']);
    c.k.push(decoderLinear(normalized, weights[block+'k'])); c.v.push(decoderLinear(normalized, weights[block+'v']));
    const {mixed} = decoderAttend(q, c.k, c.v, weights.heads || 1);
    const attentionOut = decoderLinear(mixed, weights[block+'o']), residual = x.map((a,j)=>a+attentionOut[j]);
    const hidden = decoderLinear(decoderNorm(residual), weights[block+'f']).map(Math.tanh), mlp = decoderLinear(hidden, weights[block+'b']);
    x = residual.map((a,j)=>a+mlp[j]);
  });
  cache.length++;
  return decoderSoftmax(decoderLinear(decoderNorm(x), weights.head));
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
if(typeof module!=='undefined')module.exports={decoderPredict,decoderTrace,decoderGenerate,decoderCache,decoderCacheClone,decoderStep};
