'use strict';
// Model families shown in the UI, and the structure diagram of the active configuration.
const ARCH_INFO={
 classic:{label:'標準 Transformer',short:'標準',base:null,
  note:'2017年の原型に近い構成。位置は学習する表 P を足して入れ、LayerNorm・tanh MLP・未来を見ない因果Attention を使う。',
  details:cfg=>`${cfg.layers}ブロック・各${cfg.heads}ヘッド・${cfg.dimension}次元・MLP隠れ${cfg.ff}次元。Pre-LayerNorm、学習する位置埋め込み、因果Attention、残差接続、tanh MLP。`},
 modern:{label:'現代版 Transformer',short:'現代版',base:'classic',
  note:'Llama などと同じ部品に置き換えた構成。位置は RoPE（Q・K を位置に比例した角度で回す）で入れ、RMSNorm と SwiGLU を使う。パラメータ数は標準とほぼ同じ。',
  details:cfg=>`${cfg.layers}ブロック・各${cfg.heads}ヘッド・${cfg.dimension}次元・SwiGLU隠れ${cfg.ff}次元（3行列で標準の2行列とほぼ同数）。Pre-RMSNorm、RoPE、因果Attention、残差接続。`},
 diffusion:{label:'拡散LM（マスク拡散）',short:'拡散',base:'modern',
  note:'現代版の Attention を双方向にし、ランダムに隠した文字（〈マスク〉）を当てる学習に変えた。生成は全体を〈マスク〉から始め、確信度の高い位置から並列に少しずつ確定していく。',
  details:cfg=>`現代版と同じ${cfg.layers}ブロック・各${cfg.heads}ヘッド・${cfg.dimension}次元で、Attentionは双方向。各文をマスク率 r〜U(0,1) で隠し、隠した文字の交差エントロピーを学習（重み1/rは省略）。文は文脈長まで〈終〉で埋め、長さもモデルが決める。損失は隠した文字だけの平均で、次文字予測の損失とは直接比べられない。`},
 looped:{label:'ループ型 Transformer',short:'ループ',base:'modern',
  note:'現代版のブロックを1つだけ持ち、同じ重みのまま L 回繰り返す。パラメータは1ブロック分で、深さ（計算量）だけが L 倍になる。推論時のループ回数も変えられる。',
  details:cfg=>`現代版のブロック1つ（各${cfg.heads}ヘッド・${cfg.dimension}次元）を重み共有で${cfg.layers}回適用。勾配は${cfg.layers}回分を合計して同じ重みを更新する。`},
 mamba:{label:'Mamba-3（状態空間モデル）',short:'Mamba-3',base:'modern',
  note:'現代版の Attention を Mamba-3 の SSM に置き換えた。過去は Δ で減衰しながら固定サイズの状態に書き込まれ、生成時の1文字あたりの計算は文脈長によらない。学習は同じ計算を注意と同じ形の行列（混合行列 M）で行う。',
  details:cfg=>`${cfg.layers}ブロック・各${cfg.heads}ヘッド・${cfg.dimension}次元。ミキサーは Mamba-3（SISO）：台形則の離散化、Δの累積に比例した回転（複素状態）、出力ゲート SiLU(z)、スキップ D。短い畳み込みと位置埋め込みはなし。MLPは SwiGLU 隠れ${cfg.ff}次元。`}};
function archInfo(arch=modelArch()){return ARCH_INFO[arch]||ARCH_INFO.classic}
function mlpWidth(arch,d){return arch==='classic'?d*2:Math.round(d/3)*4}
// Labels and the inference-loop control follow the selected (not yet trained) family.
function syncArchControls(){const arch=$('model-arch').value||'classic',layers=+$('layer-count').value;$('layer-label').textContent=arch==='looped'?'ループ回数':'ブロック';$('head-label').textContent=arch==='mamba'?'SSMヘッド':'ヘッド';$('loop-control').hidden=arch!=='looped';$('loop-count').innerHTML=Array.from({length:12},(_,i)=>`<option value="${i+1}" ${i+1===layers?'selected':''}>${i+1}${i+1===layers?'（学習時）':''}</option>`).join('');$('loop-count').value=String(layers)}
function renderArchitecture(){
 const cfg=data.config,arch=modelArch(),info=archInfo(),rms=arch!=='classic',H=cfg.heads,d=cfg.dimension,hot=part=>arch===part||(arch==='modern'&&['norm','rope','mlp'].includes(part)),parts=[];let x=8;
 const box=(label,sub,w,changed=false)=>{parts.push({type:'box',x,w,label,sub,changed});x+=w+20},add=()=>{parts.push({type:'add',x});x+=38};
 box('文字埋め込み E',arch==='diffusion'?'〈マスク〉も1文字':`語彙${data.vocab.length} → ${d}次元`,110,arch==='diffusion');
 if(arch==='classic')box('＋ 位置 P','学習する表',74);
 const blockStart=x-10,blockInput=x-20;
 box(rms?'RMSNorm':'LayerNorm','',74,hot('norm'));
 if(arch==='mamba')box('Mamba-3 SSM',`${H}ヘッド · Δ減衰 · 回転 · ゲート`,150,true);
 else box(arch==='classic'?'因果 Attention':arch==='diffusion'?'RoPE + 双方向 Attn':'RoPE + 因果 Attention',`${H}ヘッド`+(arch==='diffusion'?' · 未来も見る':''),150,hot('rope')||arch==='diffusion');
 const add1=x;add();const mid=x-20;
 box(rms?'RMSNorm':'LayerNorm','',74,hot('norm'));box(rms?'SwiGLU MLP':'tanh MLP',`${d} → ${cfg.ff} → ${d}`,104,hot('mlp'));
 const add2=x;add();const blockEnd=x-10;
 box(rms?'RMSNorm':'LayerNorm','',74,hot('norm'));box(arch==='diffusion'?'各位置の文字':'次の文字',arch==='diffusion'?'確定して再入力 ↺':`Softmax · ${data.vocab.length}語彙`,112,arch==='diffusion');
 const W=x-12,cy=60,boxTop=cy-15,boxH=30,esc=detailEscape;
 let svg=`<defs><marker id="arch-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8z" fill="#6f8b99"/></marker></defs>`;
 const group=arch==='looped'?`共有ブロック1つを ${cfg.layers}回ループ（重みは1組）`:`ブロック × ${cfg.layers}`;
 svg+=`<rect x="${blockStart}" y="20" width="${blockEnd-blockStart}" height="62" rx="7" class="arch-group ${arch==='looped'?'changed':''}"/><text x="${blockStart+8}" y="15" class="arch-group-label ${arch==='looped'?'changed':''}">${esc(group)}</text>`;
 if(arch==='looped')svg+=`<path d="M${blockEnd-6} 20 C${blockEnd-6} 6 ${blockEnd-40} 4 ${(blockStart+blockEnd)/2+60} 6" class="arch-loop" marker-end="url(#arch-arrow)"/>`;
 // Residual connections skip over each sub-layer to the ⊕ nodes.
 for(const [from,to] of [[blockInput,add1+9],[mid,add2+9]])svg+=`<path d="M${from} ${cy} V31 H${to} V${cy-9}" class="arch-residual" marker-end="url(#arch-arrow)"/>`;
 parts.forEach((part,i)=>{const next=parts[i+1];if(next){const from=part.type==='add'?part.x+18:part.x+part.w,to=next.x;svg+=`<line x1="${from}" y1="${cy}" x2="${to-1}" y2="${cy}" class="arch-flow" marker-end="url(#arch-arrow)"/>`}
  if(part.type==='add')svg+=`<circle cx="${part.x+9}" cy="${cy}" r="9" class="arch-add"/><text x="${part.x+9}" y="${cy+4}" class="arch-add-label">+</text>`;
  else svg+=`<rect x="${part.x}" y="${boxTop}" width="${part.w}" height="${boxH}" rx="5" class="arch-box ${part.changed?'changed':''}"/><text x="${part.x+part.w/2}" y="${part.sub?cy-2:cy+4}" class="arch-label">${esc(part.label)}</text>${part.sub?`<text x="${part.x+part.w/2}" y="${cy+10}" class="arch-sub">${esc(part.sub)}</text>`:''}`});
 const svgEl=$('arch-svg');svgEl.setAttribute('viewBox',`0 0 ${W} 92`);svgEl.style.maxWidth=W+'px';svgEl.innerHTML=svg;
 svgEl.setAttribute('aria-label',`${info.label}の構造：${parts.filter(p=>p.type==='box').map(p=>p.label).join('、')}。${group}。`);
 $('arch-note').innerHTML=`<strong>${esc(info.label)}</strong>${info.base?`<span class="arch-base">橙＝${esc(archInfo(info.base).label)}からの変更点</span>`:''}${esc(info.note)}`;
}
