from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent/'recovered_data'
RTC=ROOT/'planetary_computer_rtc'; MODEL=ROOT/'model_inputs'; OUT=ROOT/'candidate_model'
OUT.mkdir(parents=True,exist_ok=True)
PHASES=[('baseline','2017-11-01'),('rising','2017-11-13'),('peak','2017-11-25'),('early_recession','2017-12-07'),('late_recession','2017-12-19')]
THRESHOLDS=[0.35,0.50,0.65]; NODATA=-9999.0

def read(path):
    with rasterio.open(path) as src: return src.read(1).astype('float32'),src.profile.copy(),src.transform

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def make_features(vv,vh,dem,slope):
    a=10*np.log10(np.clip(vv,1e-8,None)); b=10*np.log10(np.clip(vh,1e-8,None))
    return np.column_stack([a.ravel(),b.ravel(),(a-b).ravel(),dem.ravel(),slope.ravel()])

def write_tif(path,arr,profile,dtype,nodata):
    p=profile.copy(); p.update(driver='GTiff',count=1,dtype=dtype,nodata=nodata,compress='DEFLATE',tiled=True,blockxsize=256,blockysize=256)
    with rasterio.open(path,'w',**p) as dst: dst.write(arr.astype(dtype),1)

def main():
    occ,profile,transform=read(MODEL/'jrc_gsw_v1_5_occurrence_30m.tif')
    seas,_,_=read(MODEL/'jrc_gsw_v1_5_seasonality_30m.tif'); ext,_,_=read(MODEL/'jrc_gsw_v1_5_extent_30m.tif')
    dem,_,_=read(MODEL/'cop_dem_glo30_30m.tif'); common,_,_=read(RTC/'common_valid_mask_30m.tif')
    vv0,_,_=read(RTC/'baseline/baseline_vv_rtc_30m.tif'); vh0,_,_=read(RTC/'baseline/baseline_vh_rtc_30m.tif')
    gy,gx=np.gradient(dem.astype('float64'),30,30); slope=np.degrees(np.arctan(np.hypot(gx,gy))).astype('float32')
    water=(seas==12)&(occ>=95)&(ext==1)&(common==1); land=(occ==0)&(ext==0)&(common==1)
    rows=np.arange(0,occ.shape[0],3); cols=np.arange(0,occ.shape[1],3); rr,cc=np.meshgrid(rows,cols,indexing='ij'); rr=rr.ravel(); cc=cc.ravel()
    keep=water[rr,cc]|land[rr,cc]; rr=rr[keep]; cc=cc[keep]; y=water[rr,cc].astype('uint8')
    xs,ys=xy(transform,rr,cc,offset='center'); groups=np.array([f'{int(x//10000)}_{int(z//10000)}' for x,z in zip(xs,ys)])
    cube=make_features(vv0,vh0,dem,slope).reshape(*occ.shape,5); X=cube[rr,cc]
    ok=np.all(np.isfinite(X),axis=1); X,y,groups=X[ok],y[ok],groups[ok]
    rng=np.random.default_rng(20260728); selected=[]
    for g in np.unique(groups):
        idx=np.flatnonzero(groups==g); wi=idx[y[idx]==1]; li=idx[y[idx]==0]
        if len(wi)==0 or len(li)==0: continue
        selected.extend(wi); selected.extend(rng.choice(li,size=min(len(li),max(100,4*len(wi))),replace=False))
    selected=np.array(sorted(set(selected))); Xs,ysamp,gs=X[selected],y[selected],groups[selected]
    if len(np.unique(gs))<5: raise RuntimeError('fewer than five dual-class spatial groups')
    cv=GroupKFold(5); oof=np.full(len(ysamp),np.nan); folds=[]
    for fold,(tr,te) in enumerate(cv.split(Xs,ysamp,gs),1):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',random_state=20260728)); m.fit(Xs[tr],ysamp[tr]); p=m.predict_proba(Xs[te])[:,1]; oof[te]=p
        folds.append({'fold':fold,'train_samples':len(tr),'test_samples':len(te),'test_groups':len(np.unique(gs[te])),'water_test':int(ysamp[te].sum()),'land_test':int((ysamp[te]==0).sum()),'auc':roc_auc_score(ysamp[te],p),'brier':brier_score_loss(ysamp[te],p)})
    pd.DataFrame(folds).to_csv(OUT/'candidate_spatial_cv_folds.csv',index=False)
    pd.DataFrame({'label':ysamp,'probability':oof,'block_id':gs}).to_csv(OUT/'candidate_oof_predictions.csv',index=False)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',random_state=20260728)); model.fit(Xs,ysamp)
    area=[]; seed=water; structure=ndimage.generate_binary_structure(2,2); valid=common.ravel()==1
    for phase,date in PHASES:
        vv,_,_=read(RTC/phase/f'{phase}_vv_rtc_30m.tif'); vh,_,_=read(RTC/phase/f'{phase}_vh_rtc_30m.tif'); F=make_features(vv,vh,dem,slope); ok=valid&np.all(np.isfinite(F),axis=1)
        prob=np.full(F.shape[0],NODATA,dtype='float32'); prob[ok]=model.predict_proba(F[ok])[:,1]; prob=prob.reshape(occ.shape)
        write_tif(OUT/f'{phase}_candidate_water_probability_30m.tif',prob,profile,'float32',NODATA)
        for t in THRESHOLDS:
            binary=(prob>=t)&(common==1); labels,n=ndimage.label(binary,structure=structure); touching=np.unique(labels[seed&(labels>0)]); connected=np.isin(labels,touching) if touching.size else np.zeros_like(binary)
            write_tif(OUT/f'{phase}_connected_water_t{int(t*100):02d}_30m.tif',connected,profile,'uint8',0)
            area.append({'phase':phase,'date':date,'threshold':t,'connected_pixels':int(connected.sum()),'connected_area_km2':connected.sum()*0.0009,'components_before_seed_filter':int(n),'seeded_components':int(len(touching))})
    pd.DataFrame(area).to_csv(OUT/'candidate_connected_water_area.csv',index=False)
    report={'scope':'transparent public-source candidate reconstruction; not frozen HPC model','features':['VV_dB','VH_dB','VV_minus_VH_dB','DEM_m','slope_deg'],'label_rule':'water: JRC seasonality=12 occurrence>=95 extent=1; land: occurrence=0 extent=0','sampling_m':90,'block_m':10000,'cv_folds':5,'training_samples':len(ysamp),'water_samples':int(ysamp.sum()),'land_samples':int((ysamp==0).sum()),'spatial_groups':len(np.unique(gs)),'oof_auc':roc_auc_score(ysamp,oof),'oof_brier':brier_score_loss(ysamp,oof),'thresholds':THRESHOLDS,'qa_passed':bool(len(area)==15 and not np.isnan(oof).any()),'prohibited_interpretation':'Do not report as reproducing the original nine-feature HPC model or its metrics/areas.'}
    (OUT/'candidate_model_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='sha256_manifest.csv']; pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False)
    if not report['qa_passed']: raise RuntimeError(json.dumps(report))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
