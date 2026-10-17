'use strict';
// Model families. Every family except 'classic' starts from the modern backbone
// (RMSNorm, RoPE, SwiGLU) and changes exactly one thing:
//   diffusion — bidirectional attention, trained to fill in masked characters
//   looped    — one block whose weights are reused for every loop
//   mamba     — the attention mixer is replaced by a Mamba-3 state-space mixer (no RoPE)
const MODEL_ARCHS=['classic','modern','diffusion','looped','mamba'];
// Diffusion models read PAD (id 0) as the mask token and never see real padding.
const MASK_ID=0,EOS_ID=2;
class WasmTransformer {
 static async create(bytes,config){const memory=new WebAssembly.Memory({initial:64,maximum:WasmTransformer.MAX_PAGES});const {instance}=await WebAssembly.instantiate(bytes,{env:{memory,exp:Math.exp,log:Math.log,tanh:Math.tanh,cos:Math.cos,sin:Math.sin}});return new WasmTransformer(instance.exports,memory,config)}
 constructor(k,memory,{arch='classic',dimension=16,layers=2,heads=1,vocabSize=340,context=21,seed=42}={}){
  if(!MODEL_ARCHS.includes(arch)||![8,16,32,48,64].includes(dimension)||![2,4,6].includes(layers)||![1,2,4].includes(heads)||dimension/heads%4||vocabSize%4)throw Error('未対応のモデル構成です');
  this.k=k;this.memory=memory;this.arch=arch;this.d=dimension;this.l=layers;this.h=heads;this.dh=dimension/heads;this.v=vocabSize;this.t=context;this.step=0;
  this.rms=arch!=='classic';this.rope=['modern','diffusion','looped'].includes(arch);this.causal=arch!=='diffusion';this.ssm=arch==='mamba';this.unique=arch==='looped'?1:layers;
  // SwiGLU has three matrices, so its hidden width is about 4d/3 (a multiple of 4) to keep the MLP near the classic 2 × d × 2d parameters.
  this.f=arch==='classic'?dimension*2:Math.round(dimension/3)*4;
  this.shapes={E:[vocabSize,dimension]};if(arch==='classic')this.shapes.P=[context,dimension];
  for(let b=0;b<this.unique;b++){
   if(this.ssm){for(const key of ['x','B','C','z','o'])this.shapes[b+key]=[dimension,dimension];this.shapes[b+'dt']=[dimension,heads];this.shapes[b+'lam']=[dimension,heads];for(const key of ['dtb','A','D'])this.shapes[b+key]=[1,heads];this.shapes[b+'w']=[1,dimension/2]}
   else for(const key of ['q','k','v','o'])this.shapes[b+key]=[dimension,dimension];
   if(arch==='classic'){this.shapes[b+'f']=[dimension,this.f];this.shapes[b+'b']=[this.f,dimension]}else{this.shapes[b+'g']=[dimension,this.f];this.shapes[b+'f']=[dimension,this.f];this.shapes[b+'b']=[this.f,dimension]}
  }
  this.shapes.head=[dimension,vocabSize];this.count=Object.values(this.shapes).reduce((n,[r,c])=>n+r*c,0);this.cursor=4096;this.p=this.alloc(this.count);this.g=this.alloc(this.count);this.m=this.alloc(this.count);this.vv=this.alloc(this.count);
  // Families without a learned position table pass zeros to the embedding kernel.
  if(arch!=='classic'){this.zeroP=this.alloc(context*dimension);this.zeroGP=this.alloc(context*dimension)}
  this.permanent=this.cursor;this.weights={};this.grads={};let offset=0;
  for(const [key,[r,c]] of Object.entries(this.shapes)){this.weights[key]={ptr:this.p+offset*4,r,c};this.grads[key]={ptr:this.g+offset*4,r,c};offset+=r*c}
  let randomState=seed>>>0;const uniform=()=>{randomState^=randomState<<13;randomState^=randomState>>>17;randomState^=randomState<<5;return ((randomState>>>0)+.5)/4294967296};
  for(const [key,tensor] of Object.entries(this.weights)){const out=this.view(tensor.ptr,tensor.r*tensor.c),scale=key==='P'?.05:/[qkvoxBCz]$/.test(key)?.16:/(dt|lam)$/.test(key)?.02:.15;for(let i=0;i<out.length;i+=2){const mag=Math.sqrt(-2*Math.log(uniform())),angle=2*Math.PI*uniform();out[i]=mag*Math.cos(angle)*scale;if(i+1<out.length)out[i+1]=mag*Math.sin(angle)*scale}}
  // Mamba-3 per-head parameters: step sizes Δ from 0.01 to 0.1 (memory of roughly 100 to 10 characters), A = −1, skip D = 1,
  // and rotation speeds ω chosen so the initial rotation per character equals RoPE's frequencies.
  if(this.ssm)for(let b=0;b<this.unique;b++){const dtb=this.view(this.weights[b+'dtb'].ptr,heads),A=this.view(this.weights[b+'A'].ptr,heads),D=this.view(this.weights[b+'D'].ptr,heads),w=this.view(this.weights[b+'w'].ptr,dimension/2);for(let h=0;h<heads;h++){const dt0=.01*10**((h+.5)/heads);dtb[h]=Math.log(Math.expm1(dt0));A[h]=0;D[h]=1;for(let i=0;i<this.dh/2;i++)w[h*this.dh/2+i]=ropeFrequency(i,this.dh)/dt0}}
 }
 alloc(count,clear=true){const ptr=this.cursor;this.cursor=(ptr+count*4+15)&~15;if(this.cursor>this.memory.buffer.byteLength){const pages=Math.ceil((this.cursor-this.memory.buffer.byteLength)/65536);this.memory.grow(Math.max(pages,Math.min(this.memory.buffer.byteLength/65536,WasmTransformer.MAX_PAGES-this.memory.buffer.byteLength/65536)))}if(clear)this.view(ptr,count).fill(0);return ptr}
 view(ptr,count){return new Float32Array(this.memory.buffer,ptr,count)}
 tensor(r,c,clear=true){return {ptr:this.alloc(r*c,clear),r,c}}
 values(x){return this.view(x.ptr,x.r*x.c)}
 // A looped model applies its single block (key '0') at every loop.
 key(b){return this.arch==='looped'?'0':String(b)}
 norm(x){const out=this.tensor(x.r,x.c,false),inv=this.alloc(x.r,false);this.k[this.rms?'rmsnorm':'ln'](x.ptr,out.ptr,inv,x.r,x.c);return {out,inv}}
 normback(dy,norm){const out=this.tensor(dy.r,dy.c,false);this.k[this.rms?'rmsback':'lnback'](dy.ptr,norm.out.ptr,norm.inv,out.ptr,dy.r,dy.c);return out}
 mat(x,key){const w=this.weights[key],out=this.tensor(x.r,w.c,false);this.k[w.c%4?'mats':'mat'](x.ptr,w.ptr,out.ptr,x.r,w.r,w.c);return out}
 matback(x,key,dy){const w=this.weights[key],dx=this.tensor(x.r,x.c,false);this.k[w.c%4?'matsback':'matback'](x.ptr,w.ptr,dy.ptr,dx.ptr,this.grads[key].ptr,x.r,w.r,w.c);return dx}
 rotate(x,ang,sign=1){const out=this.tensor(x.r,x.c,false);this.k.rope(x.ptr,ang,out.ptr,x.r,x.c,sign);return out}
 // RoPE angles for packed rows: row r is position r mod t; pair i of a head turns by position × 10000^(−2i/dh).
 ropeAngles(n){const ptr=this.alloc(n*this.d/2,false),ang=this.view(ptr,n*this.d/2),half=this.dh/2;for(let r=0,o=0;r<n;r++){const pos=r%this.t;for(let h=0;h<this.h;h++)for(let i=0;i<half;i++)ang[o++]=pos*ropeFrequency(i,this.dh)}return ptr}
 // Multi-head attention splits Q, K and V by columns: head h owns columns h·dh … (h+1)·dh−1.
 gather(x,head){if(this.h===1)return x;const out=this.tensor(x.r,this.dh,false),src=this.view(x.ptr,x.r*x.c),dst=this.view(out.ptr,x.r*this.dh),o=head*this.dh,dh=this.dh;for(let r=0,i=0;r<x.r;r++){const base=r*x.c+o;for(let j=0;j<dh;j++)dst[i++]=src[base+j]}return out}
 scatter(part,x,head){if(part===x)return;const src=this.view(part.ptr,part.r*part.c),dst=this.view(x.ptr,x.r*x.c),o=head*this.dh,dh=this.dh;for(let r=0,i=0;r<x.r;r++){const base=r*x.c+o;for(let j=0;j<dh;j++)dst[base+j]=src[i++]}}
 add(a,b){const out=this.tensor(a.r,a.c,false);this.k.add(a.ptr,b.ptr,out.ptr,a.r*a.c);return out}
 attentionForward(x,key,batch,ang){
  const n=x.r;let q=this.mat(x,key+'q'),k=this.mat(x,key+'k');const v=this.mat(x,key+'v'),h=this.tensor(n,this.d,false),heads=[];if(this.rope){q=this.rotate(q,ang);k=this.rotate(k,ang)}
  for(let head=0;head<this.h;head++){const qh=this.gather(q,head),kh=this.gather(k,head),vh=this.gather(v,head),a=this.tensor(batch*this.t,this.t,false),out=this.h===1?h:this.tensor(n,this.dh,false);this.k.attention(qh.ptr,kh.ptr,vh.ptr,a.ptr,out.ptr,batch,this.t,this.dh,this.causal?1:0);this.scatter(out,h,head);heads.push({q:qh,k:kh,v:vh,a})}
  return {h,cache:{heads}};
 }
 attentionBackward(x,dh,c,key,batch,ang){
  const n=x.r,dq=this.tensor(n,this.d,this.h===1),dk=this.tensor(n,this.d,this.h===1),dv=this.tensor(n,this.d,this.h===1),da=this.alloc(this.t);
  c.heads.forEach((s,head)=>{const dho=this.gather(dh,head),[dqh,dkh,dvh]=this.h===1?[dq,dk,dv]:[this.tensor(n,this.dh),this.tensor(n,this.dh),this.tensor(n,this.dh)];this.k.attentionback(s.q.ptr,s.k.ptr,s.v.ptr,s.a.ptr,dho.ptr,dqh.ptr,dkh.ptr,dvh.ptr,da,batch,this.t,this.dh,this.causal?1:0);this.scatter(dqh,dq,head);this.scatter(dkh,dk,head);this.scatter(dvh,dv,head)});
  // RoPE is a rotation, so its gradient is the inverse rotation.
  const dqr=this.rope?this.rotate(dq,ang,-1):dq,dkr=this.rope?this.rotate(dk,ang,-1):dk;
  return this.add(this.add(this.matback(x,key+'q',dqr),this.matback(x,key+'k',dkr)),this.matback(x,key+'v',dv));
 }
 // Mamba-3 mixer (single-input single-output, no short convolution):
 //   X = x·Wx, B = x·WB, C = x·WC, Z = x·Wz, Δ = softplus(x·Wdt + bias), λ = σ(x·Wλ), A = −exp(A_log)
 //   B and C turn by the angle (running sum of Δ) × ω — the complex-valued state written as a rotation.
 //   y = M·X + D·X per head (the ssm kernel), then the output gate y·SiLU(Z).
 // Memory can grow at any alloc, which detaches existing views, so a view is taken right before it is used.
 ssmForward(x,key,batch){
  const n=x.r,t=this.t,H=this.h,dh=this.dh,half=dh/2,X=this.mat(x,key+'x'),B=this.mat(x,key+'B'),C=this.mat(x,key+'C'),Z=this.mat(x,key+'z'),dtp=this.mat(x,key+'dt'),lp=this.mat(x,key+'lam');
  const vec=name=>Array.from(this.view(this.weights[key+name].ptr,this.weights[key+name].c)),dtb=vec('dtb'),D=vec('D'),w=vec('w'),A=vec('A').map(a=>-Math.exp(a)),dtpv=this.values(dtp).slice(),lpv=this.values(lp).slice();
  const delta=new Float32Array(n*H),lam=new Float32Array(n*H),run=new Float32Array(n*H);
  for(let r=0;r<n;r++)for(let h=0;h<H;h++){const i=r*H+h;delta[i]=softplus(dtpv[i]+dtb[h]);lam[i]=1/(1+Math.exp(-lpv[i]));run[i]=delta[i]+(r%t?run[i-H]:0)}
  const angPtr=this.alloc(n*this.d/2,false),ang=this.view(angPtr,n*this.d/2);for(let r=0,o=0;r<n;r++)for(let h=0;h<H;h++)for(let i=0;i<half;i++,o++)ang[o]=run[r*H+h]*w[h*half+i];
  const Cr=this.rotate(C,angPtr),Br=this.rotate(B,angPtr),Y=this.tensor(n,this.d,false),heads=[];
  for(let h=0;h<H;h++){const cum=this.alloc(n,false),p=this.alloc(n,false),q=this.alloc(n,false),cv=this.view(cum,n),pv=this.view(p,n),qv=this.view(q,n);for(let r=0;r<n;r++){const i=r*H+h;cv[r]=A[h]*run[i];pv[r]=lam[i]*delta[i];qv[r]=(1-lam[i])*delta[i]}
   const cr=this.gather(Cr,h),br=this.gather(Br,h),xh=this.gather(X,h),m=this.tensor(batch*t,t,false),out=H===1?Y:this.tensor(n,dh,false);this.k.ssm(cr.ptr,br.ptr,xh.ptr,cum,p,q,m.ptr,out.ptr,batch,t,dh);this.scatter(out,Y,h);heads.push({cr,br,x:xh,cum,p,q,m})}
  const yv=this.values(Y),xv=this.values(X);for(let r=0;r<n;r++)for(let j=0;j<this.d;j++)yv[r*this.d+j]+=D[Math.floor(j/dh)]*xv[r*this.d+j];
  const gated=this.tensor(n,this.d,false);this.k.swiglu(Z.ptr,Y.ptr,gated.ptr,n*this.d);
  return {h:gated,cache:{X,B,C,Z,Y,Cr,Br,dtpv,delta,lam,run,A,D,w,dtb,angPtr,heads}};
 }
 ssmBackward(x,dgated,c,key,batch){
  const n=x.r,t=this.t,H=this.h,dh=this.dh,half=dh/2,d=this.d,gD=new Float64Array(H),gA=new Float64Array(H),gdtb=new Float64Array(H),gw=new Float64Array(d/2);
  const dZ=this.tensor(n,d,false),dY=this.tensor(n,d,false);this.k.swigluback(dgated.ptr,c.Z.ptr,c.Y.ptr,dZ.ptr,dY.ptr,n*d);
  const dX=this.tensor(n,d),dCr=this.tensor(n,d,H===1),dBr=this.tensor(n,d,H===1);{const dYv=this.values(dY),dXv=this.values(dX),Xv=this.values(c.X);for(let r=0;r<n;r++)for(let j=0;j<d;j++){const h=Math.floor(j/dh),i=r*d+j;dXv[i]=c.D[h]*dYv[i];gD[h]+=dYv[i]*Xv[i]}}
  const dRun=new Float64Array(n*H),dDelta=new Float64Array(n*H),dLam=new Float64Array(n*H);
  c.heads.forEach((s,h)=>{const dyh=this.gather(dY,h),[dcr,dbr]=H===1?[dCr,dBr]:[this.tensor(n,dh),this.tensor(n,dh)],dxh=this.tensor(n,dh),dcum=this.alloc(n),dp=this.alloc(n),dq=this.alloc(n);
   this.k.ssmback(s.cr.ptr,s.br.ptr,s.x.ptr,s.cum,s.p,s.q,s.m.ptr,dyh.ptr,dcr.ptr,dbr.ptr,dxh.ptr,dcum,dp,dq,batch,t,dh);this.scatter(dcr,dCr,h);this.scatter(dbr,dBr,h);
   const dXv=this.values(dX),dxv=this.values(dxh),dcv=this.view(dcum,n),dpv=this.view(dp,n),dqv=this.view(dq,n);for(let r=0;r<n;r++){for(let j=0;j<dh;j++)dXv[r*d+h*dh+j]+=dxv[r*dh+j];const i=r*H+h;
    // cum = A·run, p = λΔ, q = (1−λ)Δ; A = −exp(A_log), so ∂A/∂A_log = A.
    dRun[i]+=dcv[r]*c.A[h];gA[h]+=dcv[r]*c.run[i]*c.A[h];dDelta[i]+=dpv[r]*c.lam[i]+dqv[r]*(1-c.lam[i]);dLam[i]+=(dpv[r]-dqv[r])*c.delta[i]}});
  // Rotation angle = run × ω: ∂out/∂θ for a pair is (−out_b, out_a).
  const dC=this.rotate(dCr,c.angPtr,-1),dB=this.rotate(dBr,c.angPtr,-1);{const Crv=this.values(c.Cr),Brv=this.values(c.Br),dCrv=this.values(dCr),dBrv=this.values(dBr);
  for(let r=0;r<n;r++)for(let h=0;h<H;h++)for(let i=0;i<half;i++){const a=r*d+h*dh+2*i,dTheta=-dCrv[a]*Crv[a+1]+dCrv[a+1]*Crv[a]-dBrv[a]*Brv[a+1]+dBrv[a+1]*Brv[a];dRun[r*H+h]+=dTheta*c.w[h*half+i];gw[h*half+i]+=dTheta*c.run[r*H+h]}}
  // run is a running sum within each sequence, so Δ at row r receives the gradient of every later row.
  const ddt=this.tensor(n,H,false),dlp=this.tensor(n,H,false),ddtv=this.values(ddt),dlpv=this.values(dlp);
  for(let r=n-1;r>=0;r--)for(let h=0;h<H;h++){const i=r*H+h;if(r%t!==t-1)dRun[i]+=dRun[i+H];dDelta[i]+=dRun[i];ddtv[i]=dDelta[i]/(1+Math.exp(-(c.dtpv[i]+c.dtb[h])));gdtb[h]+=ddtv[i];dlpv[i]=dLam[i]*c.lam[i]*(1-c.lam[i])}
  let dx=this.matback(x,key+'x',dX);for(const [name,g] of [['B',dB],['C',dC],['z',dZ],['dt',ddt],['lam',dlp]])dx=this.add(dx,this.matback(x,key+name,g));
  for(const [name,g] of [['D',gD],['A',gA],['dtb',gdtb],['w',gw]]){const view=this.view(this.grads[key+name].ptr,g.length);for(let i=0;i<g.length;i++)view[i]+=g[i]}
  return dx;
 }
 mlpForward(x,key){
  const n=x.r,hidden=this.tensor(n,this.f,false);
  if(this.arch==='classic'){const pre=this.mat(x,key+'f');this.k.tanh(pre.ptr,hidden.ptr,n*this.f);return {out:this.mat(hidden,key+'b'),cache:{hidden}}}
  const gate=this.mat(x,key+'g'),up=this.mat(x,key+'f');this.k.swiglu(gate.ptr,up.ptr,hidden.ptr,n*this.f);return {out:this.mat(hidden,key+'b'),cache:{hidden,gate,up}};
 }
 mlpBackward(x,dy,c,key){
  const n=x.r,dHidden=this.matback(c.hidden,key+'b',dy);
  if(this.arch==='classic'){const df=this.tensor(n,this.f,false);this.k.tanhback(dHidden.ptr,c.hidden.ptr,df.ptr,n*this.f);return this.matback(x,key+'f',df)}
  const dg=this.tensor(n,this.f,false),du=this.tensor(n,this.f,false);this.k.swigluback(dHidden.ptr,c.gate.ptr,c.up.ptr,dg.ptr,du.ptr,n*this.f);return this.add(this.matback(x,key+'g',dg),this.matback(x,key+'f',du));
 }
 batch(tokens,targets,backprop=false,weights=null){
  this.cursor=this.permanent;const n=tokens.length;if(n%this.t)throw Error('入力の長さが不正です');const batch=n/this.t,xPtr=this.alloc(n,false),targetPtr=this.alloc(n,false),weightPtr=weights?this.alloc(n,false):0;new Int32Array(this.memory.buffer,xPtr,n).set(tokens);new Int32Array(this.memory.buffer,targetPtr,n).set(targets);if(weights)this.view(weightPtr,n).set(weights);
  if(backprop)this.view(this.g,this.count).fill(0);
  const P=this.weights.P?this.weights.P.ptr:this.zeroP,ang=this.rope?this.ropeAngles(n):0;
  let z=this.tensor(n,this.d,false);this.k.embedding(xPtr,this.weights.E.ptr,P,z.ptr,n,this.t,this.d);const caches=[];
  for(let b=0;b<this.l;b++){
   const key=this.key(b),norm=this.norm(z),mixer=this.ssm?this.ssmForward(norm.out,key,batch):this.attentionForward(norm.out,key,batch,ang);
   const res=this.add(z,this.mat(mixer.h,key+'o')),nf=this.norm(res),mlp=this.mlpForward(nf.out,key);z=this.add(res,mlp.out);if(backprop)caches.push({norm,mixer:mixer.cache,h:mixer.h,nf,mlp:mlp.cache});
  }
  const finalNorm=this.norm(z),logits=this.mat(finalNorm.out,'head'),probs=this.tensor(n,this.v,false),dg=backprop?this.tensor(n,this.v,false):null;const loss=this.k.crossentropy(logits.ptr,targetPtr,probs.ptr,dg?.ptr||0,n,this.v,weightPtr);let valid=0,correct=0;const probabilities=this.view(probs.ptr,n*this.v);
  for(let i=0;i<n;i++){if(!targets[i])continue;valid++;let best=0;for(let j=1;j<this.v;j++)if(probabilities[i*this.v+j]>probabilities[i*this.v+best])best=j;if(best===targets[i])correct++}
  if(backprop){
   let dz=this.normback(this.matback(finalNorm.out,'head',dg),finalNorm);
   for(let b=this.l-1;b>=0;b--){const key=this.key(b),c=caches[b],dy=this.add(dz,this.normback(this.mlpBackward(c.nf.out,dz,c.mlp,key),c.nf)),dh=this.matback(c.h,key+'o',dy);
    const dn=this.ssm?this.ssmBackward(c.norm.out,dh,c.mixer,key,batch):this.attentionBackward(c.norm.out,dh,c.mixer,key,batch,ang);dz=this.add(dy,this.normback(dn,c.norm));
   }
   if(!this.weights.P)this.view(this.zeroGP,this.t*this.d).fill(0);
   this.k.embeddingback(xPtr,dz.ptr,this.grads.E.ptr,this.weights.P?this.grads.P.ptr:this.zeroGP,n,this.t,this.d);
  }
  if(!Number.isFinite(loss))throw Error('学習値が数値範囲を超えました。初期化して学習率を下げてください。');
  return {loss,accuracy:correct/valid,valid,probsPtr:probs.ptr,rows:n};
 }
 // salt picks the masks of a diffusion batch: the training step for updates, a constant for evaluation.
 packRows(rows,indices,salt=0){if(this.arch==='diffusion')return this.packMasked(rows,indices,salt);const tokens=new Int32Array(indices.length*this.t),targets=new Int32Array(tokens.length);for(let b=0;b<indices.length;b++){const row=rows[indices[b]];for(let t=0;t<this.t;t++){tokens[b*this.t+t]=row[t]||0;targets[b*this.t+t]=row[t+1]||0}}return {tokens,targets}}
 // Masked diffusion: each sequence draws a mask rate r ~ U(0,1) and hides each character (never 〈始〉) with probability r.
 // The loss is the cross entropy of the hidden characters, averaged within each sequence and then over sequences
 // (MDLM's 1/r weight, normalized), so lightly masked sequences count as much as heavily masked ones.
 // Rows are padded with 〈終〉 so the model also decides the length.
 packMasked(rows,indices,salt){const t=this.t,tokens=new Int32Array(indices.length*t),targets=new Int32Array(tokens.length),weights=new Float32Array(tokens.length);
  for(let b=0;b<indices.length;b++){const row=rows[indices[b]],random=maskRandom(salt,indices[b]),rate=random();let hidden=0;
   for(let p=0;p<t;p++){const id=p<row.length?row[p]:EOS_ID,i=b*t+p;if(p>0&&random()<rate){tokens[i]=MASK_ID;targets[i]=id;hidden++}else tokens[i]=id}
   if(!hidden){const p=1+Math.floor(random()*(t-1)),i=b*t+p;targets[i]=tokens[i];tokens[i]=MASK_ID;hidden=1}
   for(let p=0;p<t;p++)if(targets[b*t+p])weights[b*t+p]=1/hidden}
  return {tokens,targets,weights}}
 train(rows,indices,lr=.003){const packed=this.packRows(rows,indices,this.step+1),stats=this.batch(packed.tokens,packed.targets,true,packed.weights);this.step++;stats.grad=this.k.adam(this.p,this.g,this.m,this.vv,this.count,lr,1-.9**this.step,1-.999**this.step);return stats}
 evaluate(rows,indices,chunk=16){let loss=0,correct=0,count=0;for(let start=0;start<indices.length;start+=chunk){const packed=this.packRows(rows,indices.slice(start,start+chunk),0x5eed),s=this.batch(packed.tokens,packed.targets,false,packed.weights);loss+=s.loss*s.valid;correct+=s.accuracy*s.valid;count+=s.valid}return {loss:loss/count,accuracy:correct/count,count}}
 copyWeights(){return this.view(this.p,this.count).slice()}
 meta(){return {arch:this.arch,heads:this.h,context:this.t,blocks:this.l}}
 arrays(){const out={};for(const [key,{ptr,r,c}] of Object.entries(this.weights)){const flat=this.view(ptr,r*c);out[key]=Array.from({length:r},(_,i)=>Array.from(flat.subarray(i*c,(i+1)*c)))}return Object.assign(out,this.meta())}
}
function ropeFrequency(i,dh){return 10000**(-2*i/dh)}
function softplus(x){return x>20?x:Math.log1p(Math.exp(x))}
// Deterministic per (salt, row) generator, so a restored checkpoint repeats the same masks.
function maskRandom(salt,row){let s=(Math.imul(salt+1,0x9e3779b1)^Math.imul(row+1,0x85ebca77))>>>0||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return ((s>>>0)+.5)/4294967296}}
// Must match MAX_PAGES in build-wasm.py.
WasmTransformer.MAX_PAGES=16384;
if(typeof module!=='undefined')module.exports={WasmTransformer,MODEL_ARCHS,ropeFrequency,softplus};
