'use strict';
// Actual inference with the weights of the selected training state, for every model family in engine.js.
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
const decoderRms = row => {
  let square = 0; for (const x of row) square += x * x;
  const scale = 1 / Math.sqrt(square / row.length + 1e-5);
  return row.map(x => x * scale);
};
const decoderSoftmax = row => {
  const max = Math.max(...row), exp = row.map(x => Math.exp(x - max));
  const total = exp.reduce((a,b) => a+b,0); return exp.map(x => x/total);
};
const decoderSilu = x => x / (1 + Math.exp(-x));
const decoderSoftplus = x => x > 20 ? x : Math.log1p(Math.exp(x));
const decoderRopeFrequency = (i, dh) => 10000 ** (-2 * i / dh);
// Rotates column pairs (2i, 2i+1) by angles[i]; angles has one entry per pair.
function decoderRotate(row, angles) {
  const out = row.slice();
  for (let i = 0; i < angles.length; i++) {
    const c = Math.cos(angles[i]), s = Math.sin(angles[i]), a = row[2*i], b = row[2*i+1];
    out[2*i] = a*c - b*s; out[2*i+1] = a*s + b*c;
  }
  return out;
}
// Shape of the model stored with the weights (older snapshots only carry heads).
function decoderModel(weights) {
  const arch = weights.arch || 'classic', dim = weights.E[0].length, heads = weights.heads || 1;
  let blocks = weights.blocks; if (!blocks) { blocks = 0; while (weights[blocks + 'o']) blocks++; }
  return {arch, dim, heads, dh: dim / heads, blocks, context: weights.context || weights.P.length,
    rms: arch !== 'classic', rope: ['modern','diffusion','looped'].includes(arch), causal: arch !== 'diffusion', ssm: arch === 'mamba',
    key: b => arch === 'looped' ? '0' : String(b)};
}
function decoderRopeAngles(model, position) {
  const angles = [];
  for (let h = 0; h < model.heads; h++) for (let i = 0; i < model.dh / 2; i++) angles.push(position * decoderRopeFrequency(i, model.dh));
  return angles;
}
// Attention for one query over keys[0..]. With several heads, head h uses
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
// Per-position Mamba-3 quantities shared by the full pass and the recurrent step.
function decoderSsmInputs(weights, key, model, normalized, runBefore) {
  const X = decoderLinear(normalized, weights[key+'x']), B = decoderLinear(normalized, weights[key+'B']), C = decoderLinear(normalized, weights[key+'C']), Z = decoderLinear(normalized, weights[key+'z']);
  const dtp = decoderLinear(normalized, weights[key+'dt']), lp = decoderLinear(normalized, weights[key+'lam']);
  const delta = dtp.map((v,h) => decoderSoftplus(v + weights[key+'dtb'][0][h])), lambda = lp.map(v => 1 / (1 + Math.exp(-v)));
  const run = delta.map((v,h) => v + runBefore[h]), half = model.dh / 2, w = weights[key+'w'][0];
  const angles = []; for (let h = 0; h < model.heads; h++) for (let i = 0; i < half; i++) angles.push(run[h] * w[h*half+i]);
  return {X, B, C, Z, delta, lambda, run, Br: decoderRotate(B, angles), Cr: decoderRotate(C, angles), angles};
}
// Full Mamba-3 mixer in its quadratic form; M[i][j] is the weight of X[j] in y[i] (see ssm in build-wasm.py).
function decoderSsmMix(weights, key, model, normalized) {
  const n = normalized.length, H = model.heads, dh = model.dh, A = weights[key+'A'][0].map(a => -Math.exp(a)), D = weights[key+'D'][0];
  const rows = []; let run = Array(H).fill(0);
  for (const row of normalized) { const s = decoderSsmInputs(weights, key, model, row, run); rows.push(s); run = s.run; }
  const mix = Array.from({length: H}, () => []), y = rows.map(s => s.X.map((x,j) => D[Math.floor(j/dh)] * x));
  for (let h = 0; h < H; h++) for (let i = 0; i < n; i++) {
    const out = Array(n).fill(0);
    for (let j = 0; j <= i; j++) {
      let dot = 0; for (let c = h*dh; c < (h+1)*dh; c++) dot += rows[i].Cr[c] * rows[j].Br[c];
      const w = rows[j].lambda[h] * rows[j].delta[h] + (j < i ? (1 - rows[j+1].lambda[h]) * rows[j+1].delta[h] : 0);
      out[j] = Math.exp(A[h] * (rows[i].run[h] - rows[j].run[h])) * w * dot;
      for (let c = h*dh; c < (h+1)*dh; c++) y[i][c] += out[j] * rows[j].X[c];
    }
    mix[h].push(out);
  }
  const gate = rows.map(s => s.Z.map(decoderSilu)), gated = y.map((row,i) => row.map((v,j) => v * gate[i][j]));
  return {rows, mix, y, gate, gated};
}
function decoderMlp(weights, key, model, row) {
  if (!model.rms) { const pre = decoderLinear(row, weights[key+'f']), hidden = pre.map(Math.tanh); return {pre, hidden, out: decoderLinear(hidden, weights[key+'b'])}; }
  const gatePre = decoderLinear(row, weights[key+'g']), up = decoderLinear(row, weights[key+'f']), hidden = gatePre.map((g,j) => decoderSilu(g) * up[j]);
  return {gatePre, up, hidden, out: decoderLinear(hidden, weights[key+'b'])};
}
// mode: 'last' → next-character probabilities after the last position, 'all' → probabilities at every position, 'trace' → every intermediate value.
function decoderForward(weights, tokenIds, mode='last') {
  const model = decoderModel(weights), traceMode = mode === 'trace', norm = model.rms ? decoderRms : decoderNorm, linear = decoderLinear, softmax = decoderSoftmax;
  const ids = tokenIds.slice(-model.context);
  let x = ids.map((id,t) => weights.E[id].map((v,j) => v + (weights.P ? weights.P[t][j] : 0)));
  const trace={tokenEmbedding:ids.map(id=>weights.E[id].slice()),embedding:x.map(row=>row.slice()),blocks:[]};
  if(weights.P)trace.positionEmbedding=ids.map((_,t)=>weights.P[t].slice());
  if(model.rope)trace.ropeAngle=ids.map((_,t)=>decoderRopeAngles(model,t).map(Math.cos));
  for (let block = 0; block < model.blocks; block++) {
    const key = model.key(block), normalized = x.map(norm), bt = {normAttention: normalized};
    let mixedRows;
    if (model.ssm) {
      const s = decoderSsmMix(weights, key, model, normalized);mixedRows = s.gated;
      if (traceMode) {
        Object.assign(bt, {ssmX: s.rows.map(r=>r.X), ssmB: s.rows.map(r=>r.B), ssmC: s.rows.map(r=>r.C), ssmZ: s.rows.map(r=>r.Z), delta: s.rows.map(r=>r.delta), lambda: s.rows.map(r=>r.lambda), run: s.rows.map(r=>r.run), ssmBr: s.rows.map(r=>r.Br), ssmCr: s.rows.map(r=>r.Cr), ssmY: s.y, gate: s.gate, mixRaw: s.mix});
        // The single map shown for a block is the mean over heads.
        bt.mix = s.mix[0].map((row,i) => row.map((_,j) => s.mix.reduce((a,h) => a + h[i][j], 0) / model.heads));
        if (model.heads > 1) s.mix.forEach((rows,h) => { bt['mix'+h] = rows; });
      }
    } else {
      const qRaw = normalized.map(r=>linear(r,weights[key+'q'])), kRaw = normalized.map(r=>linear(r,weights[key+'k'])), v = normalized.map(r=>linear(r,weights[key+'v']));
      const q = model.rope ? qRaw.map((r,t) => decoderRotate(r, decoderRopeAngles(model,t))) : qRaw, k = model.rope ? kRaw.map((r,t) => decoderRotate(r, decoderRopeAngles(model,t))) : kRaw;
      const heads = model.heads, attentionRows=[], headRows=Array.from({length:heads},()=>[]);
      const pad = row => [...row, ...Array(ids.length-row.length).fill(0)];
      mixedRows = q.map((query,t) => {
        const end = model.causal ? t+1 : ids.length, {mixed, attention} = decoderAttend(query, k.slice(0,end), v, heads);
        if(traceMode){
          // The single map shown for a block is the mean over heads.
          attentionRows.push(pad(attention[0].map((_,i)=>attention.reduce((a,h)=>a+h[i],0)/heads)));
          attention.forEach((h,i)=>headRows[i].push(pad(h)));
        }
        return mixed;
      });
      if (traceMode) Object.assign(bt, {q, k, v, qRaw, kRaw, attention: attentionRows, ...Object.fromEntries(headRows.map((rows,h)=>['attention'+h,rows]))});
    }
    const attentionOut = mixedRows.map(row => linear(row, weights[key+'o']));
    const residual=x.map((row,t)=>row.map((a,j)=>a+attentionOut[t][j]));
    const normFF=residual.map(norm),mlp=normFF.map(row=>decoderMlp(weights,key,model,row));
    x=residual.map((row,t)=>row.map((a,j)=>a+mlp[t].out[j]));
    if(traceMode){Object.assign(bt,{attentionMixed:mixedRows,attentionOutput:attentionOut,attentionResidual:residual,normFF,hidden:mlp.map(m=>m.hidden),ffOutput:mlp.map(m=>m.out),output:x});if(model.rms){bt.gatePre=mlp.map(m=>m.gatePre);bt.up=mlp.map(m=>m.up)}else bt.hiddenPre=mlp.map(m=>m.pre);trace.blocks.push(bt)}
  }
  if(traceMode){trace.finalNorm=x.map(norm);trace.logits=trace.finalNorm.map(row=>linear(row,weights.head));trace.probs=trace.logits.map(softmax);return trace}
  if(mode==='all')return x.map(row=>softmax(linear(norm(row),weights.head)));
  return softmax(linear(norm(x.at(-1)),weights.head));
}
// Incremental decoding. Attention blocks keep the K and V rows of earlier positions,
// so a new token costs one position, not the prefix; rows are never modified, so a beam can share its parent's rows.
// Mamba-3 blocks keep a fixed-size state instead (one dh × dh matrix per head), the recurrent form of the same mixer.
function decoderCache(weights){const model=decoderModel(weights),blocks=[];for(let b=0;b<model.blocks;b++)blocks.push(model.ssm?{state:Array.from({length:model.heads},()=>new Float64Array(model.dh*model.dh)),run:Array(model.heads).fill(0),prev:null}:{k:[],v:[]});return {length:0,blocks}}
function decoderCacheClone(cache){return {length:cache.length,blocks:cache.blocks.map(b=>b.state?{state:b.state.map(s=>s.slice()),run:b.run,prev:b.prev}:{k:b.k.slice(),v:b.v.slice()})}}
// Mamba-3 recurrence (exponential-trapezoidal): S_t = α_t S_{t−1} + (1−λ_t)Δ_t α_t B̃_{t−1} x_{t−1}ᵀ + λ_t Δ_t B̃_t x_tᵀ, y_t = S_tᵀ C̃_t,
// with α_t = exp(Δ_t A). Unrolled, it gives exactly the weights M[i][j] of decoderSsmMix.
function decoderSsmStep(weights, key, model, c, normalized) {
  const s = decoderSsmInputs(weights, key, model, normalized, c.run), H = model.heads, dh = model.dh, A = weights[key+'A'][0], D = weights[key+'D'][0], y = Array(model.dim).fill(0);
  for (let h = 0; h < H; h++) {
    const S = c.state[h], o = h*dh, alpha = Math.exp(-Math.exp(A[h]) * s.delta[h]), p = s.lambda[h] * s.delta[h], q = (1 - s.lambda[h]) * s.delta[h] * alpha;
    for (let i = 0; i < dh; i++) for (let j = 0; j < dh; j++) {
      let v = alpha * S[i*dh+j] + p * s.Br[o+i] * s.X[o+j];
      if (c.prev) v += q * c.prev.Br[o+i] * c.prev.X[o+j];
      S[i*dh+j] = v;
    }
    for (let j = 0; j < dh; j++) { let v = D[h] * s.X[o+j]; for (let i = 0; i < dh; i++) v += S[i*dh+j] * s.Cr[o+i]; y[o+j] = v; }
  }
  c.run = s.run; c.prev = {Br: s.Br, X: s.X};
  return y.map((v,j) => v * decoderSilu(s.Z[j]));
}
// Appends one token and returns the next-token probabilities, matching decoderPredict.
function decoderStep(weights, cache, id) {
  const model = decoderModel(weights), t = cache.length, norm = model.rms ? decoderRms : decoderNorm;
  if (t >= model.context) throw Error('文脈長を超えました');
  let x = weights.E[id].map((v,j) => v + (weights.P ? weights.P[t][j] : 0));
  cache.blocks.forEach((c, block) => {
    const key = model.key(block), normalized = norm(x);let mixed;
    if (model.ssm) mixed = decoderSsmStep(weights, key, model, c, normalized);
    else {
      const angles = model.rope ? decoderRopeAngles(model, t) : null, rot = r => angles ? decoderRotate(r, angles) : r, q = rot(decoderLinear(normalized, weights[key+'q']));
      c.k.push(rot(decoderLinear(normalized, weights[key+'k']))); c.v.push(decoderLinear(normalized, weights[key+'v']));
      mixed = decoderAttend(q, c.k, c.v, model.heads).mixed;
    }
    const attentionOut = decoderLinear(mixed, weights[key+'o']), residual = x.map((a,j)=>a+attentionOut[j]);
    const mlp = decoderMlp(weights, key, model, norm(residual)).out;
    x = residual.map((a,j)=>a+mlp[j]);
  });
  cache.length++;
  return decoderSoftmax(decoderLinear(norm(x), weights.head));
}
function decoderPredict(weights,tokenIds){return decoderForward(weights,tokenIds,'last')}
function decoderTrace(weights,tokenIds){return decoderForward(weights,tokenIds,'trace')}
function decoderGenerate(weights, seedIds, limit=40) {
  const input=seedIds.slice(), generated=[];let reason='limit';
  for(let t=0;t<limit;t++){
    const probs=decoderPredict(weights,input);let next=2;
    // Padding and the start token are not valid generated text.
    for(let i=3;i<probs.length;i++)if(probs[i]>probs[next])next=i;
    if(next===2){reason='eos';break}
    generated.push(next);input.push(next);
  }
  return {tokens:generated,reason,contextWindow:decoderModel(weights).context};
}
if(typeof module!=='undefined')module.exports={decoderPredict,decoderTrace,decoderGenerate,decoderCache,decoderCacheClone,decoderStep,decoderForward,decoderModel};
