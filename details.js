'use strict';
let detailTraceKey='',detailTraceValue;
const detailNumber=v=>Number.isFinite(v)?v.toFixed(6):'—';
const detailEscape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const baselineTraceCache=new Map();
function getDetailTrace(){return current()}
function getBaselineTrace(){const key=activeModelId+':'+example+(isDiffusion()?':'+position:'');if(!baselineTraceCache.has(key))baselineTraceCache.set(key,decoderTrace(data.snapshots[0].weights,traceIds()));return baselineTraceCache.get(key)}
function selectDetailCell(map,row,col){
 const m=matrix(map);selectedCell={map,row:Math.min(row,m.length-1),col:Math.min(col,m[0].length-1)};inspected={...selectedCell};stop();$('cell-details').hidden=false;document.body.classList.add('details-open');render();
}
function closeDetailPanel(){selectedCell=null;$('cell-details').hidden=true;document.body.classList.remove('details-open');requestAnimationFrame(()=>{if(!data.snapshots.length)return;maps.forEach(drawMap);drawLoss()})}
function renderCellDetails(){
 if(!selectedCell)return;
 const {map}=selectedCell,m=matrix(map),row=Math.min(selectedCell.row,m.length-1),col=Math.min(selectedCell.col,m[0].length-1),value=m[row][col],key=map.key,block=map.block;
 selectedCell.row=row;selectedCell.col=col;
 const dim=data.config.dimension,ff=data.config.ff,trace=getDetailTrace(),w=data.snapshots[frame].weights,bt=block===undefined?null:trace.blocks[block],tokens=exTokens(),token=tokens[row],arch=modelArch(),rms=arch!=='classic',wk=arch==='looped'?'0':String(block),label=labelFor(key);
 const isProbability=isAttentionKey(key)||key==='probs',isVocabulary=key==='logits'||key==='probs',isAttention=isAttentionKey(key),isMix=isMixKey(key),isPairs=isAttention||isMix;
 const rowLabel=`位置 ${row}「${visibleToken(token)}」${isDiffusion()&&row>position?'（〈マスク〉で入力）':''}`,colLabel=isAttention?`参照先の位置 ${col}「${visibleToken(tokens[col])}」`:isMix?`入力位置 ${col}「${visibleToken(tokens[col])}」`:isVocabulary?`候補文字「${data.vocab[col]}」`:key==='delta'?`ヘッド ${col+1}`:key==='ropeAngle'?`成分の組 ${col+1}（d${2*col+1}, d${2*col+2}）`:`特徴成分 d${col+1}`;
 $('cell-detail-title').textContent=(block===undefined?'':`${blockTitle(block)} · `)+label[0];
 $('cell-detail-state').textContent=`更新 ${data.snapshots[frame].step} / ${data.config.totalSteps} · 例文「${data.texts[example]}」`;
 $('cell-detail-axes').innerHTML=`<div><span>行</span><strong>${detailEscape(rowLabel)}</strong></div><div><span>列</span><strong>${detailEscape(colLabel)}</strong></div>`;
 $('cell-detail-value').textContent=isProbability?(value*100).toFixed(2)+'%':value.toFixed(4);
 const baselineExample=getBaselineTrace(),base=block===undefined?baselineExample[key][row][col]:baselineExample.blocks[block][key][row][col],difference=value-base;
 $('cell-detail-change').textContent=`学習前 ${isProbability?(base*100).toFixed(2)+'%':base.toFixed(4)} → 現在 ／ 差 ${difference>=0?'+':''}${isProbability?(difference*100).toFixed(2)+'ポイント':difference.toFixed(4)}`;
 let meaning=label[1],formula='',calculation='',products=[];
 const addFeatures='dは学習された特徴の番号です。d1が「喜び」のように、成分ごとの意味をあらかじめ決めているわけではありません。';
 const blockInput=block===undefined?null:(block===0?trace.embedding:trace.blocks[block-1].output),ropeDetail=(raw,name,prefix)=>{const pair=col>>1,dh=dim/(data.config.heads||1),local=pair%(dh/2),theta=row*10000**(-2*local/dh),a=raw[row][pair*2],b=raw[row][pair*2+1],c=Math.cos(theta),s=Math.sin(theta),out=col%2?a*s+b*c:a*c-b*s;formula=`${name} = RoPE(LN出力 × W${prefix})　成分の組 (d${pair*2+1}, d${pair*2+2}) を角度 θ で回転`;calculation=`回転前：d${pair*2+1} = ${detailNumber(a)}、d${pair*2+2} = ${detailNumber(b)}\nθ = 位置 ${row} × 10000^(−2·${local}/${dh}) = ${detailNumber(theta)}\n${col%2?`a·sinθ + b·cosθ`:`a·cosθ − b·sinθ`} = ${detailNumber(out)}`;return out};
 function linearDetail(input,weights,name){
  products=input.map((x,i)=>({label:'d'+(i+1),input:x,weight:weights[i][col],product:x*weights[i][col]}));
  const sum=products.reduce((a,x)=>a+x.product,0);formula=`${name}[${col+1}] = Σ 入力[d] × 重み[d, ${col+1}]`;
  calculation=`${products.length}成分の掛け算を合計 = ${detailNumber(sum)}`;return sum;
 }
 if(key==='tokenEmbedding'){
  const id=data.vocab.indexOf(token);formula=`E[文字ID ${id}, d${col+1}]`;
  calculation=`文字「${token}」の埋め込み表から読み出す値 = ${detailNumber(w.E[id][col])}`;meaning+=` ${addFeatures} 同じ文字であれば、文中の位置にかかわらずこの文字埋め込みは同じです。`;
 }else if(key==='positionEmbedding'){
  formula=`P[位置 ${row}, d${col+1}]`;calculation=`学習可能な位置埋め込み表から読み出す値 = ${detailNumber(w.P[row][col])}`;meaning+=` 位置0は開始記号〈始〉です。文字の種類ではなく、その文字がある位置を表します。`;
 }else if(key==='ropeAngle'){
  const dh=dim/(data.config.heads||1),local=col%(dh/2),theta=row*10000**(-2*local/dh);formula=`cos θ、θ = 位置 × 10000^(−2k/${dh})（k = ヘッド内の組の番号）`;calculation=`位置 ${row}、k = ${local}\nθ = ${detailNumber(theta)} ／ cos θ = ${detailNumber(Math.cos(theta))}`;meaning+=' 学習しない固定の表です。ヘッドごとに同じ角度の組を使います。';
 }else if(key==='embedding'){
  const a=trace.tokenEmbedding[row][col],b=trace.positionEmbedding[row][col];formula='入力 = 文字埋め込み + 位置埋め込み';calculation=`${detailNumber(a)} + ${detailNumber(b)} = ${detailNumber(a+b)}`;meaning+=` ${addFeatures}`;
 }else if(['normAttention','normFF','finalNorm'].includes(key)){
  const input=key==='finalNorm'?trace.blocks.at(-1).output[row]:key==='normFF'?bt.attentionResidual[row]:blockInput[row];const mean=input.reduce((a,b)=>a+b,0)/input.length,variance=input.reduce((a,b)=>a+(b-mean)**2,0)/input.length;
  if(rms){const square=input.reduce((a,b)=>a+b*b,0)/input.length;formula='正規化[d] = 入力[d] / √(行の二乗平均 + 0.00001)';calculation=`入力[d${col+1}] = ${detailNumber(input[col])}\n二乗平均 = ${detailNumber(square)}\n${detailNumber(input[col])} / √(${detailNumber(square)} + 0.00001) = ${detailNumber(input[col]/Math.sqrt(square+1e-5))}`;meaning+=' RMSNormは平均を引かず、大きさだけをそろえます。このモデルは正規化後の追加の倍率を使いません。'}
  else{formula='正規化[d] = (入力[d] − 行の平均) / √(行の分散 + 0.00001)';
  calculation=`入力[d${col+1}] = ${detailNumber(input[col])}\n平均 = ${detailNumber(mean)} ／ 分散 = ${detailNumber(variance)}\n(${detailNumber(input[col])} − ${detailNumber(mean)}) / √(${detailNumber(variance)} + 0.00001) = ${detailNumber((input[col]-mean)/Math.sqrt(variance+1e-5))}`;
  meaning+=' このモデルは正規化後の追加の倍率・バイアスを使いません。負の値は誤りではなく、正規化された成分が行平均より小さいことを表します。'}
 }else if(['q','k','v'].includes(key)){
  if(rms&&key!=='v'){linearDetail(bt.normAttention[row],w[wk+key],key.toUpperCase());const f=formula,c=calculation;ropeDetail(bt[key+'Raw'],key.toUpperCase(),key);calculation=`${f}（回転前、掛け算の内訳は下表）\n${c}\n`+calculation}else linearDetail(bt.normAttention[row],w[wk+key],key.toUpperCase());meaning+=` ${addFeatures} Q・K・Vのセルは確率ではありません。`;
 }else if(isAttentionKey(key)){
  meaning=`${rowLabel}が、${colLabel}から情報を受け取る割合です。${(value*100).toFixed(2)}%という値は、Vを混ぜるときの重みを表します。文章全体での「重要度」の絶対評価ではありません。`;
  if(maskedCell(key,row,col)){formula='未来の位置 → 因果マスク → 重み 0';calculation=`参照先 ${col} は現在位置 ${row} より先です。\nこの列はSoftmaxの分母にも含めず、注意の値を0に固定します。`;meaning+=' 斜線は未来の文字を見ないための制限であり、学習によってゼロになったわけではありません。'}
  else{
   const heads=data.config.heads||1,dh=dim/heads,o=attentionHead(key)*dh,slice=r=>r.slice(o,o+dh);products=slice(bt.q[row]).map((v,j)=>({label:'d'+(o+j+1),input:v,weight:bt.k[col][o+j],product:v*bt.k[col][o+j]}));const scale=Math.sqrt(dh),dot=products.reduce((a,b)=>a+b.product,0),scores=bt.k.slice(0,isDiffusion()?bt.k.length:row+1).map(r=>slice(r).reduce((a,k,j)=>a+bt.q[row][o+j]*k,0)/scale),max=Math.max(...scores),denom=scores.reduce((a,s)=>a+Math.exp(s-max),0),prob=Math.exp(dot/scale-max)/denom;
   const range=isDiffusion()?'全位置 j（双方向）':'j ≤ i のみ';formula=heads>1?`注意[i,j] = softmax(Q[i] · K[j] / √${dh})　※ ヘッド${attentionHead(key)+1}の成分 d${o+1}〜d${o+dh}、${range}`:`注意[i,j] = softmax(Q[i] · K[j] / √${dh})　※ ${range}`;calculation=`QとKの${dh}成分の内積 = ${detailNumber(dot)}\n√${dh} = ${detailNumber(scale)} で割ったスコア = ${detailNumber(dot/scale)}\nexp(スコア − 最大値) / Σ exp(各スコア − 最大値)\n= ${detailNumber(prob)} = ${(prob*100).toFixed(4)}%`;
  }
 }else if(key==='attentionOutput'){
  linearDetail(bt.attentionMixed[row],w[wk+'o'],'Attention出力');calculation=`先に各参照先のVを、注意の割合で混ぜます。\n混ぜた${dim}成分をWoで線形変換。\n`+calculation;
 }else if(['ssmX','ssmB','ssmC'].includes(key)){
  linearDetail(bt.normAttention[row],w[wk+key.slice(3).replace('X','x')],key.slice(3));meaning+=` ${addFeatures}${key==='ssmX'?'':' 混合行列では、この値を Δ の累積に比例した角度で回してから内積をとります（複素数の状態を回転で表したもの）。'}`;
 }else if(key==='delta'){
  const pre=linearDetail(bt.normAttention[row],w[wk+'dt'],'Δの前段'),bias=w[wk+'dtb'][0][col],value=Math.log1p(Math.exp(pre+bias));formula='Δ = softplus(LN出力 × Wdt + b)　softplus(u) = log(1 + eᵘ)';calculation=`線形変換 = ${detailNumber(pre)} ／ b = ${detailNumber(bias)}\nsoftplus(${detailNumber(pre+bias)}) = ${detailNumber(value)}\n1文字ぶんの減衰 exp(Δ·A) = ${detailNumber(Math.exp(-Math.exp(w[wk+'A'][0][col])*value))}`;meaning+=' 大きい Δ は、状態を速く忘れさせ、この位置の文字を強く書き込みます。';
 }else if(isMix){
  const heads=data.config.heads||1,dh=dim/heads,list=heads>1&&key==='mix'?[...Array(heads).keys()]:[key==='mix'?0:+key.slice(3)],A=w[wk+'A'][0];
  if(maskedCell(key,row,col)){formula='未来の位置 → 0';calculation=`入力位置 ${col} は出力位置 ${row} より先です。状態は過去から順に書き込まれるので、この重みは常に0です。`}
  else{const parts=list.map(h=>{let dot=0;for(let c=h*dh;c<(h+1)*dh;c++)dot+=bt.ssmCr[row][c]*bt.ssmBr[col][c];const decay=Math.exp(-Math.exp(A[h])*(bt.run[row][h]-bt.run[col][h])),wt=bt.lambda[col][h]*bt.delta[col][h]+(col<row?(1-bt.lambda[col+1][h])*bt.delta[col+1][h]:0);return {h,dot,decay,wt,m:decay*wt*dot}});
   formula='M[i,j] = exp(A·(Δ累積[i] − Δ累積[j])) × 台形重み[j] × (C̃[i] · B̃[j])　※ j ≤ i';calculation=parts.map(p=>`${heads>1?`ヘッド${p.h+1}：`:''}減衰 = ${detailNumber(p.decay)}（${row-col}文字ぶん）\n台形重み = λ[j]Δ[j]${col<row?' + (1−λ[j+1])Δ[j+1]':''} = ${detailNumber(p.wt)}\nC̃·B̃ = ${detailNumber(p.dot)}\nM = ${detailNumber(p.m)}`).join('\n')+(list.length>1?`\n全ヘッドの平均 = ${detailNumber(parts.reduce((a,p)=>a+p.m,0)/parts.length)}`:'');
   meaning=`出力位置 ${row} の y が、${colLabel}の X をどれだけ受け取るかの重みです。減衰は離れるほど小さくなり、C̃·B̃ が「何を書いて何を読むか」を決めます。生成時はこの行列を作らず、同じ結果を固定サイズの状態の更新で計算します。`}
 }else if(key==='attentionResidual'){
  const a=blockInput[row][col],b=bt.attentionOutput[row][col];formula='残差1 = ブロック入力 + Attention出力';calculation=`${detailNumber(a)} + ${detailNumber(b)} = ${detailNumber(a+b)}`;meaning+=' 元の表現を残したまま、Attentionで集めた情報を加える値です。';
 }else if(key==='hidden'){
  if(rms){const up=linearDetail(bt.normFF[row],w[wk+'f'],'線形側'),gate=bt.gatePre[row][col],silu=gate/(1+Math.exp(-gate));formula='隠れ層[d] = SiLU(LN2 × Wg)[d] × (LN2 × Wf)[d]　SiLU(u) = u·σ(u)';calculation=`ゲート側 LN2 × Wg = ${detailNumber(gate)} → SiLU = ${detailNumber(silu)}\n線形側 LN2 × Wf = ${detailNumber(up)}（掛け算の内訳は下表）\n${detailNumber(silu)} × ${detailNumber(up)} = ${detailNumber(silu*up)}`;meaning+=` 各セルは${ff}個の隠れ成分の1つです。値の範囲に上限はありません。`}
  else{const pre=linearDetail(bt.normFF[row],w[wk+'f'],'MLPの線形変換');formula='隠れ層[d] = tanh(Σ LN2成分 × Wf)';calculation=`線形変換の合計 = ${detailNumber(pre)}\ntanh(${detailNumber(pre)}) = ${detailNumber(Math.tanh(pre))}`;meaning+=` 各セルは${ff}個の隠れ成分の1つです。tanhにより出力の範囲は−1〜1になります。`}
 }else if(key==='ffOutput'){
  linearDetail(bt.hidden[row],w[wk+'b'],'MLP出力');meaning+=` ${ff}成分の隠れ表現を、残差に足せる${dim}成分に戻しています。`;
 }else if(key==='output'){
  const a=bt.attentionResidual[row][col],b=bt.ffOutput[row][col];formula='ブロック出力 = 残差1 + MLP出力';calculation=`${detailNumber(a)} + ${detailNumber(b)} = ${detailNumber(a+b)}`;
 }else if(key==='logits'){
  linearDetail(trace.finalNorm[row],w.head,'語彙スコア');meaning=isDiffusion()?`${rowLabel}に入る文字の候補「${data.vocab[col]}」に与えるスコアです。確率ではなく、0〜1の制限もありません。Softmaxで、他の候補のスコアと比較して確率に変えます。`:`${rowLabel}までを読んだとき、次の文字の候補「${data.vocab[col]}」に与えるスコアです。確率ではなく、0〜1の制限もありません。Softmaxで、他の候補のスコアと比較して確率に変えます。`;
 }else if(key==='probs'){
  const logits=trace.logits[row],max=Math.max(...logits),denom=logits.reduce((a,b)=>a+Math.exp(b-max),0),prob=Math.exp(logits[col]-max)/denom,rank=1+trace.probs[row].filter(v=>v>prob).length;
  formula='次文字の確率 = exp(候補のLogit − 最大Logit) / Σ exp(各Logit − 最大Logit)';calculation=`候補「${data.vocab[col]}」のLogit = ${detailNumber(logits[col])}\n最大Logit = ${detailNumber(max)} ／ 分母 = ${detailNumber(denom)}\n確率 = ${detailNumber(prob)} = ${(prob*100).toFixed(4)}%\nこの行の予測順位：${rank}位 / ${data.vocab.length}語彙`;
  meaning=isDiffusion()?`${rowLabel}に「${data.vocab[col]}」が入るとモデルが予測する確率です。拡散モデルは各位置そのものの文字を予測します。${row>position?'この位置は〈マスク〉で入力したので、隠した文字を当てる予測です。':'この位置は文字を見せているので、ほぼ確実に入力どおりの文字を予測します。'}正解は「${tokens[row]}」。全候補の確率の合計は約100%です。`:`${rowLabel}までを読んだとき、次の位置 ${row+1} に「${data.vocab[col]}」が来るとモデルが予測する確率です。現在の行に書かれている文字の確率ではありません。正解は「${exTargets()[row]}」。全候補の確率の合計は約100%です。`;
 }
 if(!isProbability&&!['logits','positionEmbedding','tokenEmbedding','embedding','normAttention','normFF','finalNorm','ropeAngle','delta'].includes(key)&&!isMix)meaning+=` このセルはベクトルの1成分であり、単独の値を正解率として読むことはできません。`;
 if(!isProbability&&!isMix&&Math.abs(value)>2)meaning+=' 色は±2で飽和していますが、表示している数値は飽和前の実値です。';
 $('cell-detail-meaning').textContent=meaning;$('cell-detail-formula').textContent=formula;$('cell-detail-calculation').textContent=calculation;
 const values=m[row];$('cell-detail-vector-title').textContent=isAttention?'この位置が参照する全位置':isMix?'この位置が受け取る全位置':isVocabulary?'この位置の全語彙候補':'同じ行の全成分';
 $('cell-detail-vector').innerHTML=values.map((v,j)=>{
  const label=isPairs?`${j}「${visibleToken(tokens[j])}」`:isVocabulary?data.vocab[j]:key==='delta'?'H'+(j+1):'d'+(j+1),masked=maskedCell(key,row,j);
  return `<button class="detail-component ${j===col?'active':''} ${masked?'masked':''}" data-detail-col="${j}" aria-pressed="${j===col}"><span>${detailEscape(label)}</span><b>${isProbability?(v*100).toFixed(2)+'%':v.toFixed(isMix?5:4)}</b>${masked?'<small>未来・禁止</small>':''}</button>`
 }).join('');$('cell-detail-vector').querySelectorAll('[data-detail-col]').forEach(button=>button.onclick=()=>selectDetailCell(map,row,+button.dataset.detailCol));
 const mean=values.reduce((a,b)=>a+b,0)/values.length;$('cell-detail-stats').textContent=isProbability?`この行の合計 ${(values.reduce((a,b)=>a+b,0)*100).toFixed(2)}%（保存値の丸め差を含む）`:`この行の平均 ${mean.toFixed(4)} ／ 最小 ${Math.min(...values).toFixed(4)} ／ 最大 ${Math.max(...values).toFixed(4)}`;
 $('cell-detail-products').hidden=!products.length;
 $('cell-detail-products-content').innerHTML=products.length?`<table><thead><tr><th>成分</th><th>${isAttentionKey(key)?'Q':'入力'}</th><th>${isAttentionKey(key)?'K':'重み'}</th><th>積</th></tr></thead><tbody>${products.map(p=>`<tr><th>${p.label}</th><td>${p.input.toFixed(6)}</td><td>${p.weight.toFixed(6)}</td><td>${p.product.toFixed(6)}</td></tr>`).join('')}</tbody></table>`:'';
}
$('close-cell-details').onclick=closeDetailPanel;
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!$('info').open&&selectedCell)closeDetailPanel()});
