# Self-contained WebAssembly binary builder. No compiler downloads are needed.
# Actual tensor arithmetic and optimizer run in Wasm; worker JS schedules operations.
from pathlib import Path
import struct
I,F,V=0x7f,0x7d,0x7b

def u(n):
 out=bytearray()
 while True:
  b=n&127;n>>=7;out.append(b|(128 if n else 0))
  if not n:return bytes(out)
def si(n):
 out=bytearray()
 while True:
  b=n&127;n>>=7;done=(n==0 and not b&64) or (n==-1 and b&64);out.append(b|(0 if done else 128))
  if done:return bytes(out)
def name(s):return u(len(s))+s.encode()
def vec(xs):return u(len(xs))+b''.join(xs)
def sim(op):return b'\xfd'+u(op)
class E:
 def __init__(self,code,t=I):self.c=code;self.t=t
 def op(self,other,op):return E(self.c+val(other,self.t).c+bytes([op]),self.t)
 def __add__(self,o):return self.op(o,0x6a if self.t==I else 0x92)
 def __sub__(self,o):return self.op(o,0x6b if self.t==I else 0x93)
 def __mul__(self,o):return self.op(o,0x6c if self.t==I else 0x94)
 def __truediv__(self,o):return self.op(o,0x6e if self.t==I else 0x95)
 def lt(self,o):return E(self.c+val(o,self.t).c+bytes([0x49 if self.t==I else 0x5d]),I)
 def gt(self,o):return E(self.c+val(o,self.t).c+bytes([0x4b if self.t==I else 0x5e]),I)
 def eq(self,o):return E(self.c+val(o,self.t).c+bytes([0x46 if self.t==I else 0x5b]),I)
 def rem(self,o):return self.op(o,0x70)
 def float(self):return E(self.c+b'\xb3',F)
 def sqrt(self):return E(self.c+b'\x91',F)
 def neg(self):return E(self.c+b'\x8c',F)
 def max(self,o):return self.op(o,0x97)
 def min(self,o):return self.op(o,0x96)
def val(x,t=I):
 if isinstance(x,E):return x
 return E(b'\x41'+si(int(x)),I) if t==I else E(b'\x43'+struct.pack('<f',x),F)
def fl(x):return val(x,F)
def load(p):return E(val(p).c+b'\x2a'+u(2)+u(0),F)
def iload(p):return E(val(p).c+b'\x28'+u(2)+u(0),I)
def store(p,x):return val(p).c+val(x,F).c+b'\x38'+u(2)+u(0)
def vload(p):return E(val(p).c+sim(0)+u(2)+u(0),V)
def vstore(p,x):return val(p).c+x.c+sim(11)+u(2)+u(0)
def splat(x):return E(val(x,F).c+sim(19),V)
def vadd(a,b):return E(a.c+b.c+sim(0xe4),V)
def vmul(a,b):return E(a.c+b.c+sim(0xe6),V)
def lane(a,n):return E(a.c+sim(31)+bytes([n]),F)
def sum4(a):return lane(a,0)+lane(a,1)+lane(a,2)+lane(a,3)
class Fn:
 def __init__(self,params,locals_,result=None):
  self.ps=params;self.ls=locals_;self.result=result;self.names={};self.vars={}
  for i,(k,t) in enumerate(params+locals_):self.names[k]=i;self.vars[k]=E(b'\x20'+u(i),t)
 def __getitem__(self,k):return self.vars[k]
 def set(self,k,x):return val(x,self.vars[k].t).c+b'\x21'+u(self.names[k])
 def loop(self,k,start,end,body,step=1):
  v=self[k]
  return self.set(k,start)+b'\x02\x40\x03\x40'+v.lt(end).c+b'\x45\x0d\x01'+body+self.set(k,v+step)+b'\x0c\x00\x0b\x0b'
 def condition(self,test,yes,no=b''):return test.c+b'\x04\x40'+yes+(b'\x05'+no if no else b'')+b'\x0b'
imports=[('exp',[F],F),('log',[F],F),('tanh',[F],F)]
functions=[]
def call(which,*args):return E(b''.join(val(a,F).c for a in args)+b'\x10'+u(next(i for i,x in enumerate(imports) if x[0]==which)),F)
def add(n,f,code):functions.append((n,f,code+b'\x0b'))
def ptr(base,index):return base+index*4
# SIMD matrix product: Y[N,O] = X[N,I] @ W[I,O]. Output columns are multiples of 4.
f=Fn([(k,I) for k in ['x','w','y','n','di','do']],[(k,I) for k in ['r','j','k']]+[('acc',V)])
x,w,y,n,di,do,r,j,k,a=[f[z] for z in ['x','w','y','n','di','do','r','j','k','acc']]
inner=f.set('acc',vadd(a,vmul(splat(load(ptr(x,r*di+k))),vload(ptr(w,k*do+j)))))
code=f.loop('r',0,n,f.loop('j',0,do,f.set('acc',splat(0))+f.loop('k',0,di,inner)+vstore(ptr(y,r*do+j),a),4));add('mat',f,code)
# Both reverse-mode products, calculated without a transpose allocation.
f=Fn([(k,I) for k in ['x','w','dy','dx','dw','n','di','do']],[(k,I) for k in ['r','i','j']]+[('acc',V)])
x,w,dy,dx,dw,n,di,do,r,i,j,a=[f[z] for z in ['x','w','dy','dx','dw','n','di','do','r','i','j','acc']]
code=f.loop('r',0,n,f.loop('i',0,di,f.set('acc',splat(0))+f.loop('j',0,do,f.set('acc',vadd(a,vmul(vload(ptr(dy,r*do+j)),vload(ptr(w,i*do+j))))),4)+store(ptr(dx,r*di+i),sum4(a))))
code+=f.loop('i',0,di,f.loop('j',0,do,f.set('acc',splat(0))+f.loop('r',0,n,f.set('acc',vadd(a,vmul(splat(load(ptr(x,r*di+i))),vload(ptr(dy,r*do+j))))))+vstore(ptr(dw,i*do+j),a),4));add('matback',f,code)
# LayerNorm without affine scale/bias, matching the educational model.
f=Fn([(k,I) for k in ['x','out','inv','n','d']],[(k,I) for k in ['r','j']]+[(k,F) for k in ['mean','variance','z','s']])
x,out,inv,n,d,r,j,mean,var,z,s=[f[k] for k in ['x','out','inv','n','d','r','j','mean','variance','z','s']]
body=f.set('mean',0)+f.loop('j',0,d,f.set('mean',mean+load(ptr(x,r*d+j))))+f.set('mean',mean/d.float())+f.set('variance',0)+f.loop('j',0,d,f.set('z',load(ptr(x,r*d+j))-mean)+f.set('variance',var+z*z))+f.set('s',fl(1)/(var/d.float()+1e-5).sqrt())+store(ptr(inv,r),s)+f.loop('j',0,d,store(ptr(out,r*d+j),(load(ptr(x,r*d+j))-mean)*s));add('ln',f,f.loop('r',0,n,body))
f=Fn([(k,I) for k in ['dy','normalized','inv','dx','n','d']],[(k,I) for k in ['r','j']]+[(k,F) for k in ['mean','dot','a','b']])
dy,nor,inv,dx,n,d,r,j,mean,dot,a,b=[f[k] for k in ['dy','normalized','inv','dx','n','d','r','j','mean','dot','a','b']]
body=f.set('mean',0)+f.set('dot',0)+f.loop('j',0,d,f.set('a',load(ptr(dy,r*d+j)))+f.set('b',load(ptr(nor,r*d+j)))+f.set('mean',mean+a)+f.set('dot',dot+a*b))+f.set('mean',mean/d.float())+f.set('dot',dot/d.float())+f.loop('j',0,d,store(ptr(dx,r*d+j),load(ptr(inv,r))*(load(ptr(dy,r*d+j))-mean-load(ptr(nor,r*d+j))*dot)));add('lnback',f,f.loop('r',0,n,body))
# Generic residual operations and tanh MLP derivative.
f=Fn([(k,I) for k in ['a','b','out','n']],[('j',I)])
a,b,out,n,j=[f[k] for k in ['a','b','out','n','j']];add('add',f,f.loop('j',0,n,vstore(ptr(out,j),vadd(vload(ptr(a,j)),vload(ptr(b,j)))),4))
f=Fn([(k,I) for k in ['x','out','n']],[('j',I)])
x,out,n,j=[f[k] for k in ['x','out','n','j']];add('tanh',f,f.loop('j',0,n,store(ptr(out,j),call('tanh',load(ptr(x,j))))))
f=Fn([(k,I) for k in ['dy','hidden','out','n']],[('j',I),('h',F)])
dy,hid,out,n,j,h=[f[k] for k in ['dy','hidden','out','n','j','h']];add('tanhback',f,f.loop('j',0,n,f.set('h',load(ptr(hid,j)))+store(ptr(out,j),load(ptr(dy,j))*(fl(1)-h*h))))
# Causal attention. Batch members never communicate.
f=Fn([(k,I) for k in ['q','k','v','a','h','batch','t','d']],[(k,I) for k in ['b','i','j','c','row','other']]+[(k,F) for k in ['score','maximum','total','prob','scale']]+[('acc',V)])
q,k,v,a,h,batch,t,d,b,i,j,c,row,other,score,maximum,total,prob,scale,acc=[f[z] for z in ['q','k','v','a','h','batch','t','d','b','i','j','c','row','other','score','maximum','total','prob','scale','acc']]
base=(b*t+i)*t
scores=f.set('other',(b*t+j)*d)+f.set('acc',splat(0))+f.loop('c',0,d,f.set('acc',vadd(acc,vmul(vload(ptr(q,row+c)),vload(ptr(k,other+c))))),4)+f.set('score',sum4(acc)*scale)+store(ptr(a,base+j),score)+f.set('maximum',maximum.max(score))
body=f.set('row',(b*t+i)*d)+f.set('maximum',-1e30)+f.loop('j',0,i+1,scores)+f.set('total',0)+f.loop('j',0,i+1,f.set('prob',call('exp',load(ptr(a,base+j))-maximum))+store(ptr(a,base+j),prob)+f.set('total',total+prob))+f.loop('j',0,i+1,store(ptr(a,base+j),load(ptr(a,base+j))/total))+f.loop('j',i+1,t,store(ptr(a,base+j),0))
body+=f.loop('c',0,d,f.set('acc',splat(0))+f.loop('j',0,i+1,f.set('acc',vadd(acc,vmul(splat(load(ptr(a,base+j))),vload(ptr(v,(b*t+j)*d+c))))))+vstore(ptr(h,row+c),acc),4)
add('attention',f,f.set('scale',fl(1)/d.float().sqrt())+f.loop('b',0,batch,f.loop('i',0,t,body)))
f=Fn([(k,I) for k in ['q','k','v','a','dh','dq','dk','dv','da','batch','t','d']],[(k,I) for k in ['b','i','j','c','row','other']]+[(k,F) for k in ['dot','score','prob','ds','scale']]+[('acc',V)])
q,k,v,a,dh,dq,dk,dv,da,batch,t,d,b,i,j,c,row,other,dot,score,prob,ds,scale,acc=[f[z] for z in ['q','k','v','a','dh','dq','dk','dv','da','batch','t','d','b','i','j','c','row','other','dot','score','prob','ds','scale','acc']];base=(b*t+i)*t
backj=f.set('other',(b*t+j)*d)+f.set('prob',load(ptr(a,base+j)))+f.set('ds',prob*(load(ptr(da,j))-dot)*scale)
backj+=f.loop('c',0,d,vstore(ptr(dq,row+c),vadd(vload(ptr(dq,row+c)),vmul(splat(ds),vload(ptr(k,other+c)))))+vstore(ptr(dk,other+c),vadd(vload(ptr(dk,other+c)),vmul(splat(ds),vload(ptr(q,row+c)))))+vstore(ptr(dv,other+c),vadd(vload(ptr(dv,other+c)),vmul(splat(prob),vload(ptr(dh,row+c))))),4)
body=f.set('row',(b*t+i)*d)+f.set('dot',0)+f.loop('j',0,i+1,f.set('other',(b*t+j)*d)+f.set('acc',splat(0))+f.loop('c',0,d,f.set('acc',vadd(acc,vmul(vload(ptr(dh,row+c)),vload(ptr(v,other+c))))),4)+f.set('score',sum4(acc))+store(ptr(da,j),score)+f.set('dot',dot+score*load(ptr(a,base+j))))+f.loop('j',0,i+1,backj)
add('attentionback',f,f.set('scale',fl(1)/d.float().sqrt())+f.loop('b',0,batch,f.loop('i',0,t,body)))
# Embedding lookup and accumulation (repeated character IDs sum their gradients).
f=Fn([(k,I) for k in ['tokens','e','p','z','n','t','d']],[(k,I) for k in ['r','j','id','position']])
tok,e,p,z,n,t,d,r,j,idx,pos=[f[k] for k in ['tokens','e','p','z','n','t','d','r','j','id','position']]
body=f.set('id',iload(ptr(tok,r)))+f.set('position',r.rem(t))+f.loop('j',0,d,vstore(ptr(z,r*d+j),vadd(vload(ptr(e,idx*d+j)),vload(ptr(p,pos*d+j)))),4);add('embedding',f,f.loop('r',0,n,body))
f=Fn([(k,I) for k in ['tokens','dz','ge','gp','n','t','d']],[(k,I) for k in ['r','j','id','position']])
tok,dz,ge,gp,n,t,d,r,j,idx,pos=[f[k] for k in ['tokens','dz','ge','gp','n','t','d','r','j','id','position']]
body=f.set('id',iload(ptr(tok,r)))+f.set('position',r.rem(t))+f.loop('j',0,d,vstore(ptr(ge,idx*d+j),vadd(vload(ptr(ge,idx*d+j)),vload(ptr(dz,r*d+j))))+vstore(ptr(gp,pos*d+j),vadd(vload(ptr(gp,pos*d+j)),vload(ptr(dz,r*d+j)))),4);add('embeddingback',f,f.loop('r',0,n,body))
# Mean cross entropy, PAD ignored; optionally stores output gradient.
f=Fn([(k,I) for k in ['logits','targets','probs','grad','n','v']],[(k,I) for k in ['r','j','target','count']]+[(k,F) for k in ['maximum','total','prob','loss','truth'] ],F)
logits,targets,probs,grad,n,v,r,j,target,count,maximum,total,prob,loss,truth=[f[k] for k in ['logits','targets','probs','grad','n','v','r','j','target','count','maximum','total','prob','loss','truth']]
body=f.set('maximum',-1e30)+f.loop('j',0,v,f.set('maximum',maximum.max(load(ptr(logits,r*v+j)))))+f.set('total',0)+f.loop('j',0,v,f.set('prob',call('exp',load(ptr(logits,r*v+j))-maximum))+store(ptr(probs,r*v+j),prob)+f.set('total',total+prob))+f.loop('j',0,v,store(ptr(probs,r*v+j),load(ptr(probs,r*v+j))/total))+f.set('target',iload(ptr(targets,r)))+f.condition(target.eq(0),b'',f.set('loss',loss-call('log',load(ptr(probs,r*v+target)).max(1e-12))))
code=f.set('count',0)+f.set('loss',0)+f.loop('r',0,n,f.condition(iload(ptr(targets,r)).eq(0),b'',f.set('count',count+1)))+f.loop('r',0,n,body)
gradbody=f.set('target',iload(ptr(targets,r)))+f.loop('j',0,v,f.set('truth',0)+f.condition(j.eq(target),f.set('truth',1))+f.condition(target.eq(0),store(ptr(grad,r*v+j),0),store(ptr(grad,r*v+j),(load(ptr(probs,r*v+j))-truth)/count.float())))
code+=f.condition(grad.eq(0),b'',f.loop('r',0,n,gradbody))+(loss/count.float()).c;add('crossentropy',f,code)
# Adam, global gradient norm clipped to 1; all state is float32 in Wasm memory.
f=Fn([(k,I) for k in ['p','g','m','v','n']]+[(k,F) for k in ['lr','cm','cv']], [('j',I)]+[(k,F) for k in ['norm','clip','gg','mm','vv']],F)
p,g,m,v,n,lr,cm,cv,j,norm,clip,gg,mm,vv=[f[k] for k in ['p','g','m','v','n','lr','cm','cv','j','norm','clip','gg','mm','vv']]
body=f.set('gg',load(ptr(g,j))*clip)+f.set('mm',load(ptr(m,j))*.9+gg*.1)+f.set('vv',load(ptr(v,j))*.999+gg*gg*.001)+store(ptr(m,j),mm)+store(ptr(v,j),vv)+store(ptr(p,j),load(ptr(p,j))-lr*(mm/cm)/((vv/cv).sqrt()+1e-8))
code=f.set('norm',0)+f.loop('j',0,n,f.set('gg',load(ptr(g,j)))+f.set('norm',norm+gg*gg))+f.set('norm',norm.sqrt())+f.set('clip',(fl(1)/norm.max(1e-9)).min(1))+f.loop('j',0,n,body)+norm.c;add('adam',f,code)
# Construct module and include memory as an import so the worker can grow it.
# 16384 pages = 1 GiB: six blocks of 64 dimensions with four heads over 128 positions
# and a batch of 64 need about 600 MiB; memory only grows as far as a configuration needs.
MAX_PAGES=16384
types=[]
def tid(params,result):
 sig=(tuple(params),result)
 if sig not in types:types.append(sig)
 return types.index(sig)
for _,params,result in imports:tid(params,result)
for _,f,_ in functions:tid([t for _,t in f.ps],f.result)
def section(n,b):return bytes([n])+u(len(b))+b
module=b'\0asm\1\0\0\0'
module+=section(1,vec([b'\x60'+vec([bytes([x]) for x in ps])+vec([] if rt is None else [bytes([rt])]) for ps,rt in types]))
module+=section(2,vec([name('env')+name(n)+b'\x00'+u(tid(ps,rt)) for n,ps,rt in imports]+[name('env')+name('memory')+b'\x02\x01'+u(64)+u(MAX_PAGES)]))
module+=section(3,vec([u(tid([t for _,t in f.ps],f.result)) for _,f,_ in functions]))
module+=section(7,vec([name(n)+b'\x00'+u(i+len(imports)) for i,(n,_,_) in enumerate(functions)]))
bodies=[]
for _,f,code in functions:
 locals_=vec([u(1)+bytes([t]) for _,t in f.ls]);body=locals_+code;bodies.append(u(len(body))+body)
module+=section(10,vec(bodies))
path=Path(__file__).parent/'engine.wasm';path.write_bytes(module);print('Built Wasm SIMD engine:',len(module),'bytes,',len(functions),'numerical kernels')
