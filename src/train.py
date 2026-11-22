"""Recording-separated retrospective augmentation experiment."""
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import sklearn
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from models import primitive
from compare import compare_methods

ROOT=Path(__file__).resolve().parents[1]
class CVAE(nn.Module):
    def __init__(self,d):
        super().__init__(); self.enc=nn.Sequential(nn.Linear(d+1,96),nn.ReLU(),nn.Linear(96,24)); self.dec=nn.Sequential(nn.Linear(13,96),nn.ReLU(),nn.Linear(96,d))
    def encode(self,x,y): return self.enc(torch.cat([x,y],1)).chunk(2,1)

def augment(x,y,seed,epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); m=CVAE(x.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=.001)
    xx=torch.tensor(x,dtype=torch.float32); yy=torch.tensor(y[:,None],dtype=torch.float32)
    for epoch in range(epochs):
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            mu,lv=m.encode(xx[ix],yy[ix]); lv=lv.clamp(-10,10); z=mu+torch.randn_like(mu)*torch.exp(lv/2)
            pred=m.dec(torch.cat([z,yy[ix]],1)); loss=(pred-xx[ix]).square().mean()+.01*(-.5*(1+lv-mu.square()-lv.exp()).mean())
            opt.zero_grad();loss.backward();opt.step()
    minority=int(np.argmin(np.bincount(y))); ids=np.flatnonzero(y==minority); n=abs(int((y==0).sum()-(y==1).sum()))
    if len(ids)<2 or n==0: raise ValueError('Augmentation requires an imbalanced training set and >=2 minority cuts')
    a=rng.choice(ids,n); labels=np.full(n,minority); condition=torch.tensor(labels[:,None],dtype=torch.float32)
    k=min(6,len(ids)); neighbours=NearestNeighbors(n_neighbors=k).fit(x[ids]).kneighbors(x[a],return_distance=False)
    # Explicitly remove the anchor even when equal-distance ties change its rank.
    b=np.array([rng.choice(ids[row][ids[row]!=anchor]) for anchor,row in zip(a,neighbours)])
    with torch.no_grad():
        mu,_=m.encode(xx[a],condition)
        reconstruction=m.dec(torch.cat([mu,condition],1)).numpy()
        generated=m.dec(torch.cat([torch.randn(n,12),condition],1)).numpy()
    return {'real_only':None,'cvae':generated,'reconstruction':reconstruction,'jitter':x[a]+rng.normal(0,.05,(n,x.shape[1])),'smote':x[a]+rng.uniform(0,1,(n,1))*(x[b]-x[a]),'neighbour_interpolation':(x[a]+x[b])/2,'random_duplication':x[a].copy()},labels,m

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from xgboost import XGBClassifier, __version__ as xgboost_version
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    seeds=[11] if args.smoke else [11,29,47]
    protocol=dict(seeds=seeds,epochs=3 if args.smoke else 100,latent=12,hidden=96,kl_weight=.01,adam_lr=.001,
                  tree_counts=40 if args.smoke else 200,depths=[3,6],xgb_lr=.05,calibration_far=.05,jitter_sd=.05,
                  python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,xgboost=xgboost_version,
                  training_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features_sha256=hashlib.sha256((ROOT/'src/models.py').read_bytes()).hexdigest())
    protocol.update(
        arms=['real_only','cvae','reconstruction','jitter','smote','neighbour_interpolation','random_duplication'],
        duplication='Reuse the sampled minority anchors without perturbation; same added count as every augmented arm.',
        interpretation='Post-hoc descriptive rule added after the original six-arm results were inspected. Observed improvement requires mean paired recall delta > 0 and mean paired FAR delta <= 0. Higher recall with higher FAR is a trade-off.',
        comparison_sha256=hashlib.sha256((ROOT/'src/compare.py').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    df=pd.read_csv(ROOT/'data/segments.csv');sp=pd.read_csv(ROOT/'data/split_manifest.csv');raw=np.load(ROOT/'data/sequences.npz')['X']
    assert len(raw)==len(df) and set(df.machine_id)=={'imi_vm20i'}
    assert set(df.label)<= {0,1,2} and np.array_equal((df.label==1).astype(int),df.abnormal_cut)
    assert df.row_id.is_unique and np.isfinite(raw).all()
    # One independent prediction per annotated cut; average chronological feature windows.
    features=primitive(raw).mean(axis=1);y=(df.label==1).to_numpy(dtype=int)
    audit={'cuts':len(df),'recordings':df.run_id.nunique(),'labels':df.label.value_counts().to_dict(),'representation':'mean of eight windows in 72 primitive sensor coordinates','source':'cnc_all_sensors prepared data; retrospective reuse','files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'data').iterdir() if f.is_file()}}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    results=[];quality=[];predictions=[];partition_counts=[]
    for fold in sorted(sp.fold.unique()):
        s=sp[sp.fold==fold];assert s.array_index.is_unique and len(s)==len(df)
        assert s.groupby('run_id').split.nunique().max()==1
        assert np.array_equal(df.row_id.iloc[s.array_index].values,s.row_id.values)
        assert np.array_equal(df.run_id.iloc[s.array_index].values,s.run_id.values)
        parts={name:g.array_index.to_numpy() for name,g in s.groupby('split')};tr=parts['train'];va=parts['val'] if 'val' in parts else parts['validation'];ca=parts['calibration']
        assert set(y[tr])=={0,1} and set(y[va])=={0,1}
        scaler=StandardScaler().fit(features[tr]);x=scaler.transform(features).astype('float32')
        joblib.dump(scaler,out/f'scaler_{fold}.joblib')
        for name,indices in parts.items():
            partition_counts.append(dict(fold=fold,partition=name,recordings=df.run_id.iloc[indices].nunique(),cuts=len(indices),positive=y[indices].sum(),negative=len(indices)-y[indices].sum()))
        for seed in seeds:
            arms,sy,generator=augment(x[tr],y[tr],seed,3 if args.smoke else 100)
            torch.save(generator.state_dict(),out/f'cvae_{fold}_{seed}.pt')
            for arm,syn in arms.items():
                tx=x[tr] if syn is None else np.concatenate([x[tr],syn]);ty=y[tr] if syn is None else np.r_[y[tr],sy]
                if syn is not None:
                    reference=x[va][y[va]==sy[0]]
                    bank=x[tr][y[tr]==sy[0]].astype('float64')
                    nearest=NearestNeighbors(n_neighbors=1).fit(bank)
                    # Recompute residual norms directly to avoid cancellation for exact copies.
                    syn_neighbours=nearest.kneighbors(syn.astype('float64'),return_distance=False)[:,0]
                    real_neighbours=nearest.kneighbors(reference.astype('float64'),return_distance=False)[:,0]
                    dist=np.linalg.norm(syn.astype('float64')-bank[syn_neighbours],axis=1)
                    real_dist=np.linalg.norm(reference.astype('float64')-bank[real_neighbours],axis=1)
                    def dependence(a):
                        centered=a-a.mean(axis=0);norm=np.linalg.norm(centered,axis=0)
                        unit=centered/np.where(norm>0,norm,1)
                        return (unit.T@unit)[np.triu_indices(a.shape[1],k=1)]
                    quality.append(dict(fold=fold,seed=seed,arm=arm,generated=len(syn),validation_reference_cuts=len(reference),nearest_train_distance=float(np.median(dist)),validation_nearest_train_distance=float(np.median(real_dist)),duplicate_fraction=float((dist<1e-6).mean()),unique_fraction=len(np.unique(syn,axis=0))/len(syn),mean_shift=float(np.abs(syn.mean(0)-reference.mean(0)).mean()),correlation_error=float(np.abs(dependence(syn)-dependence(reference)).mean()),variance=float(syn.var(0).mean()),validation_variance=float(reference.var(0).mean())))
                for family in ['rf','xgb']:
                    candidates=[]
                    for depth in [3,6]:
                        model=RandomForestClassifier(n_estimators=40 if args.smoke else 200,max_depth=depth,random_state=seed,n_jobs=2) if family=='rf' else XGBClassifier(n_estimators=40 if args.smoke else 200,max_depth=depth,learning_rate=.05,random_state=seed,n_jobs=2,eval_metric='logloss')
                        model.fit(tx,ty);score=average_precision_score(y[va],model.predict_proba(x[va])[:,1]);candidates.append((score,model,depth))
                    validation_ap,model,depth=max(candidates,key=lambda a:a[0]);stable=model.predict_proba(x[ca][y[ca]==0])[:,1].astype(float)
                    assert len(stable)>0
                    threshold=next(t for t in np.r_[np.unique(stable),np.nextafter(float(stable.max()),np.inf)] if (stable>=t).mean()<=.05)
                    joblib.dump(model,out/f'{family}_{arm}_{fold}_{seed}.joblib')
                    for partition,ix in parts.items():
                        if 'test' not in partition:continue
                        prob=model.predict_proba(x[ix])[:,1].astype(float);flag=prob>=threshold;truth=y[ix];pos=truth==1;neg=~pos
                        results.append(dict(fold=int(fold),seed=seed,arm=arm,detector=family,partition=partition,depth=depth,threshold=float(threshold),positive=int(pos.sum()),negative=int(neg.sum()),recall=float(flag[pos].mean()) if pos.any() else None,false_alarm_rate=float(flag[neg].mean()) if neg.any() else None,average_precision=float(average_precision_score(truth,prob)) if pos.any() else None))
                        results[-1].update(validation_ap=validation_ap,calibration_far=float((stable>=threshold).mean()),tp=int((pos&flag).sum()),fp=int((neg&flag).sum()))
                        predictions.extend(dict(fold=int(fold),seed=seed,arm=arm,detector=family,partition=partition,row_id=df.row_id.iloc[i],run_id=df.run_id.iloc[i],label=int(y[i]),score=float(v),alarm=bool(v>=threshold)) for i,v in zip(ix,prob))
            pd.DataFrame(results).to_csv(out/'metrics.csv',index=False);pd.DataFrame(quality).to_csv(out/'feature_quality.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
            print(f'Completed fold {fold}, seed {seed}',flush=True)
    pd.DataFrame(results).groupby(['arm','detector','partition'])[['recall','false_alarm_rate','average_precision']].mean().to_csv(out/'summary.csv')
    pd.DataFrame(partition_counts).to_csv(out/'partition_counts.csv',index=False)
    compare_methods(pd.DataFrame(results),out)
    (out/'completion.json').write_text(json.dumps({'complete':True,'smoke_only':args.smoke,'metric_rows':len(results),'seeds':seeds},indent=2))
    print(out)
if __name__=='__main__':main()
