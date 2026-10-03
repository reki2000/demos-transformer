'use strict';
class WasmTransformer {
 static async create(bytes,config){const memory=new WebAssembly.Memory({initial:64,maximum:WasmTransformer.MAX_PAGES});const {instance}=await WebAssembly.instantiate(bytes,{env:{memory,exp:Math.exp,log:Math.log,tanh:Math.tanh}});return new WasmTransformer(instance.exports,memory,config)}
 constructor(k,memory,{dimension=16,layers=2,heads=1,vocabSize=340,context=21,seed=42}={}){
  if(![8,16,32,48,64].includes(dimension)||![2,4,6].includes(layers)||![1,2,4].includes(heads)||dimension/heads%4||vocabSize%4)throw Error('未対応のモデル構成です');
  this.k=k;this.memory=memory;this.d=dimension;this.f=dimension*2;this.l=layers;this.h=heads;this.dh=dimension/heads;this.v=vocabSize;this.t=context;this.step=0;this.shapes={E:[vocabSize,dimension],P:[context,dimension]};
  for(let b=0;b<layers;b++){for(const key of ['q','k','v','o'])this.shapes[b+key]=[dimension,dimension];this.shapes[b+'f']=[dimension,this.f];this.shapes[b+'b']=[this.f,dimension]}
  this.shapes.head=[dimension,vocabSize];this.count=Object.values(this.shapes).reduce((n,[r,c])=>n+r*c,0);this.cursor=4096;this.p=this.alloc(this.count);this.g=this.alloc(this.count);this.m=this.alloc(this.count);this.vv=this.alloc(this.count);this.permanent=this.cursor;this.weights={};this.grads={};let offset=0;
  for(const [key,[r,c]] of Object.entries(this.shapes)){this.weights[key]={ptr:this.p+offset*4,r,c};this.grads[key]={ptr:this.g+offset*4,r,c};offset+=r*c}
  let randomState=seed>>>0;const uniform=()=>{randomState^=randomState<<13;randomState^=randomState>>>17;randomState^=randomState<<5;return ((randomState>>>0)+.5)/4294967296};
  for(const [key,tensor] of Object.entries(this.weights)){const out=this.view(tensor.ptr,tensor.r*tensor.c),scale=key==='P'?.05:/[qkvo]$/.test(key)?.16:.15;for(let i=0;i<out.length;i+=2){const mag=Math.sqrt(-2*Math.log(uniform())),angle=2*Math.PI*uniform();out[i]=mag*Math.cos(angle)*scale;if(i+1<out.length)out[i+1]=mag*Math.sin(angle)*scale}}
 }
 alloc(count,clear=true){const ptr=this.cursor;this.cursor=(ptr+count*4+15)&~15;if(this.cursor>this.memory.buffer.byteLength){const pages=Math.ceil((this.cursor-this.memory.buffer.byteLength)/65536);this.memory.grow(Math.max(pages,Math.min(this.memory.buffer.byteLength/65536,WasmTransformer.MAX_PAGES-this.memory.buffer.byteLength/65536)))}if(clear)this.view(ptr,count).fill(0);return ptr}
 view(ptr,count){return new Float32Array(this.memory.buffer,ptr,count)}
 tensor(r,c,clear=true){return {ptr:this.alloc(r*c,clear),r,c}}
 norm(x){const out=this.tensor(x.r,x.c,false),inv=this.alloc(x.r,false);this.k.ln(x.ptr,out.ptr,inv,x.r,x.c);return {out,inv}}
 normback(dy,norm){const out=this.tensor(dy.r,dy.c,false);this.k.lnback(dy.ptr,norm.out.ptr,norm.inv,out.ptr,dy.r,dy.c);return out}
 mat(x,key){const w=this.weights[key],out=this.tensor(x.r,w.c,false);this.k.mat(x.ptr,w.ptr,out.ptr,x.r,w.r,w.c);return out}
 matback(x,key,dy){const w=this.weights[key],dx=this.tensor(x.r,x.c,false);this.k.matback(x.ptr,w.ptr,dy.ptr,dx.ptr,this.grads[key].ptr,x.r,w.r,w.c);return dx}
 // Multi-head attention splits Q, K and V by columns: head h owns columns h·dh … (h+1)·dh−1.
 gather(x,head){if(this.h===1)return x;const out=this.tensor(x.r,this.dh,false),src=this.view(x.ptr,x.r*x.c),dst=this.view(out.ptr,x.r*this.dh),o=head*this.dh,dh=this.dh;for(let r=0,i=0;r<x.r;r++){const base=r*x.c+o;for(let j=0;j<dh;j++)dst[i++]=src[base+j]}return out}
 scatter(part,x,head){if(part===x)return;const src=this.view(part.ptr,part.r*part.c),dst=this.view(x.ptr,x.r*x.c),o=head*this.dh,dh=this.dh;for(let r=0,i=0;r<x.r;r++){const base=r*x.c+o;for(let j=0;j<dh;j++)dst[base+j]=src[i++]}}
 add(a,b){const out=this.tensor(a.r,a.c,false);this.k.add(a.ptr,b.ptr,out.ptr,a.r*a.c);return out}
 batch(tokens,targets,backprop=false){
  this.cursor=this.permanent;const n=tokens.length;if(n%this.t)throw Error('入力の長さが不正です');const batch=n/this.t,xPtr=this.alloc(n,false),targetPtr=this.alloc(n,false);new Int32Array(this.memory.buffer,xPtr,n).set(tokens);new Int32Array(this.memory.buffer,targetPtr,n).set(targets);
  if(backprop)this.view(this.g,this.count).fill(0);
  let z=this.tensor(n,this.d,false);this.k.embedding(xPtr,this.weights.E.ptr,this.weights.P.ptr,z.ptr,n,this.t,this.d);const caches=[];
  for(let b=0;b<this.l;b++){
   const norm=this.norm(z),q=this.mat(norm.out,b+'q'),k=this.mat(norm.out,b+'k'),v=this.mat(norm.out,b+'v'),h=this.tensor(n,this.d,false),heads=[];for(let head=0;head<this.h;head++){const qh=this.gather(q,head),kh=this.gather(k,head),vh=this.gather(v,head),a=this.tensor(batch*this.t,this.t,false),out=this.h===1?h:this.tensor(n,this.dh,false);this.k.attention(qh.ptr,kh.ptr,vh.ptr,a.ptr,out.ptr,batch,this.t,this.dh);this.scatter(out,h,head);heads.push({q:qh,k:kh,v:vh,a})}
   const res=this.add(z,this.mat(h,b+'o')),nf=this.norm(res),hiddenPre=this.mat(nf.out,b+'f'),hidden=this.tensor(n,this.f,false);this.k.tanh(hiddenPre.ptr,hidden.ptr,n*this.f);z=this.add(res,this.mat(hidden,b+'b'));if(backprop)caches.push({norm,heads,h,nf,hidden});
  }
  const finalNorm=this.norm(z),logits=this.mat(finalNorm.out,'head'),probs=this.tensor(n,this.v,false),dg=backprop?this.tensor(n,this.v,false):null;const loss=this.k.crossentropy(logits.ptr,targetPtr,probs.ptr,dg?.ptr||0,n,this.v);let valid=0,correct=0;const probabilities=this.view(probs.ptr,n*this.v);
  for(let i=0;i<n;i++){if(!targets[i])continue;valid++;let best=0;for(let j=1;j<this.v;j++)if(probabilities[i*this.v+j]>probabilities[i*this.v+best])best=j;if(best===targets[i])correct++}
  if(backprop){
   let dz=this.normback(this.matback(finalNorm.out,'head',dg),finalNorm);
   for(let b=this.l-1;b>=0;b--){const c=caches[b],dHidden=this.matback(c.hidden,b+'b',dz),df=this.tensor(n,this.f,false);this.k.tanhback(dHidden.ptr,c.hidden.ptr,df.ptr,n*this.f);const dy=this.add(dz,this.normback(this.matback(c.nf.out,b+'f',df),c.nf)),dh=this.matback(c.h,b+'o',dy),dq=this.tensor(n,this.d,this.h===1),dk=this.tensor(n,this.d,this.h===1),dv=this.tensor(n,this.d,this.h===1),da=this.alloc(this.t);
    c.heads.forEach((s,head)=>{const dho=this.gather(dh,head),[dqh,dkh,dvh]=this.h===1?[dq,dk,dv]:[this.tensor(n,this.dh),this.tensor(n,this.dh),this.tensor(n,this.dh)];this.k.attentionback(s.q.ptr,s.k.ptr,s.v.ptr,s.a.ptr,dho.ptr,dqh.ptr,dkh.ptr,dvh.ptr,da,batch,this.t,this.dh);this.scatter(dqh,dq,head);this.scatter(dkh,dk,head);this.scatter(dvh,dv,head)});const dn=this.add(this.add(this.matback(c.norm.out,b+'q',dq),this.matback(c.norm.out,b+'k',dk)),this.matback(c.norm.out,b+'v',dv));dz=this.add(dy,this.normback(dn,c.norm));
   }
   this.k.embeddingback(xPtr,dz.ptr,this.grads.E.ptr,this.grads.P.ptr,n,this.t,this.d);
  }
  if(!Number.isFinite(loss))throw Error('学習値が数値範囲を超えました。初期化して学習率を下げてください。');
  return {loss,accuracy:correct/valid,valid,probsPtr:probs.ptr,rows:n};
 }
 packRows(rows,indices){const tokens=new Int32Array(indices.length*this.t),targets=new Int32Array(tokens.length);for(let b=0;b<indices.length;b++){const row=rows[indices[b]];for(let t=0;t<this.t;t++){tokens[b*this.t+t]=row[t]||0;targets[b*this.t+t]=row[t+1]||0}}return {tokens,targets}}
 train(rows,indices,lr=.003){const packed=this.packRows(rows,indices),stats=this.batch(packed.tokens,packed.targets,true);this.step++;stats.grad=this.k.adam(this.p,this.g,this.m,this.vv,this.count,lr,1-.9**this.step,1-.999**this.step);return stats}
 evaluate(rows,indices,chunk=16){let loss=0,correct=0,count=0;for(let start=0;start<indices.length;start+=chunk){const packed=this.packRows(rows,indices.slice(start,start+chunk)),s=this.batch(packed.tokens,packed.targets);loss+=s.loss*s.valid;correct+=s.accuracy*s.valid;count+=s.valid}return {loss:loss/count,accuracy:correct/count,count}}
 copyWeights(){return this.view(this.p,this.count).slice()}
 arrays(){const out={};for(const [key,{ptr,r,c}] of Object.entries(this.weights)){const flat=this.view(ptr,r*c);out[key]=Array.from({length:r},(_,i)=>Array.from(flat.subarray(i*c,(i+1)*c)))}out.heads=this.h;return out}
}
// Must match MAX_PAGES in build-wasm.py.
WasmTransformer.MAX_PAGES=16384;
if(typeof module!=='undefined')module.exports={WasmTransformer};
