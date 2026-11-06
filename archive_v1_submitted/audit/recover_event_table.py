#!/usr/bin/env python3
"""Recover the event table (agency fire record per export window) - added 2026-10-12 after the folder search.
Sources (read-only, the archived agency-record folder <AGENCY_RECORDS_DIR>):
  Excel data used in Modelling/*.xlsx  (AB rows in the Alberta Wildfire historical format; BC/SK rows in the CNFDB point format)
  Historical Wildfire Data/fp-historical-wildfire-data-2006-2024.xlsx (full Alberta database, used when no small-file record is within 3 km)
Matching: for each export window (<AUDIT_DIR>/splits/event_manifest.csv, centre of the chip grid), the nearest agency record in the same
province. The chip-grid centre is offset ~0.6-1.0 km east and south of the GEE point because the last chip row/column overhangs the raster,
so offsets of ~1-1.8 km are expected. Output: <AUDIT_DIR>/temporal/event_table_recovered.csv"""
import os, math, glob, pandas as pd
SRC=os.environ.get('AGENCY_RECORDS_DIR', 'agency_records')
REV=os.environ.get('AUDIT_DIR', 'audit')
def hav(a,b,c,d):
    p=math.pi/180; x=math.sin((c-a)*p/2)**2+math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2
    return 2*6371*math.asin(math.sqrt(x))
def ts(v):
    s=str(v); return '' if s in ('nan','NaT','None') else s
recs=[]
for f in glob.glob(os.path.join(SRC,'Excel data used in Modelling','*.xlsx')):
    df=pd.read_excel(f,header=None); fn=os.path.basename(f)
    for _,r in df.iterrows():
        if fn.startswith('AB'):
            try: y=int(float(r[0]))
            except Exception: continue
            recs.append(dict(source=fn,prov='AB',agency='Alberta Wildfire',fire_id=str(r[1]),name=str(r[2]).strip(),year=y,lat=float(r[5]),lon=float(r[6]),
                             size_ha=r[3],date_field='FIRE_START_DATE',event_date=ts(r[13]),discovered=ts(r[16]),out_date=ts(r[48])))
        elif str(r[0]).strip() in ('BC','SK'):
            recs.append(dict(source=fn,prov=str(r[0]).strip(),agency='CNFDB point record ('+str(r[0]).strip()+')',fire_id=str(r[1]),name=str(r[2]).strip() or str(r[17]).strip(),
                             year=int(float(r[5])),lat=float(r[3]),lon=float(r[4]),size_ha=r[11],date_field='REP_DATE',event_date=ts(r[8]),discovered='',out_date=ts(r[10])))
R=pd.DataFrame(recs).drop_duplicates(subset=['prov','fire_id'])
abdb=pd.read_excel(os.path.join(SRC,'Historical Wildfire Data','fp-historical-wildfire-data-2006-2024.xlsx'),
                   usecols=['YEAR','FIRE_NUMBER','FIRE_NAME','CURRENT_SIZE','LATITUDE','LONGITUDE','FIRE_START_DATE','DISCOVERED_DATE','FIRST_EX_DATE'])
ev=pd.read_csv(os.path.join(REV,'splits/event_manifest.csv')); ev=ev[ev.status=='OK']
out=[]
for _,e in ev.iterrows():
    c=R[R.prov==e.province_folder].copy(); c['d']=[hav(e.centroid_lat,e.centroid_lon,a,b) for a,b in zip(c.lat,c.lon)]
    b=c.sort_values('d').iloc[0]; basis='nearest record in the modelling spreadsheets'
    if b.d>3 and e.province_folder=='AB':
        a=abdb.copy(); a['d']=[hav(e.centroid_lat,e.centroid_lon,x,y) for x,y in zip(a.LATITUDE,a.LONGITUDE)]
        a=a[a.d<3].sort_values(['CURRENT_SIZE'],ascending=False)
        if len(a):
            x=a.iloc[0]
            b=pd.Series(dict(source='fp-historical-wildfire-data-2006-2024.xlsx',agency='Alberta Wildfire',fire_id=x.FIRE_NUMBER,name=str(x.FIRE_NAME),year=int(x.YEAR),lat=x.LATITUDE,lon=x.LONGITUDE,
                             size_ha=x.CURRENT_SIZE,date_field='FIRE_START_DATE',event_date=ts(x.FIRE_START_DATE),discovered=ts(x.DISCOVERED_DATE),out_date=ts(x.FIRST_EX_DATE),d=x.d))
            basis='largest record within 3 km in the full Alberta database (no spreadsheet record within 3 km)'
    out.append(dict(event=e.event,province=e.province_folder,export_date=e.export_date,in_original_training_pool=e.in_training_pool,
        agency=b.agency,agency_fire_id=b.fire_id,agency_fire_name=b['name'],fire_year=int(b.year),final_size_ha=b.size_ha,
        event_date=str(b.event_date)[:10],event_date_field=b.date_field,discovered_date=b.discovered,out_date=b.out_date,
        agency_lat=round(float(b.lat),6),agency_lon=round(float(b.lon),6),window_centre_lat=e.centroid_lat,window_centre_lon=e.centroid_lon,offset_km=round(float(b.d),2),
        match_basis=basis,source_file=b.source,
        date_status='VERIFIED: equals the literal date in the GEE script (Code for all bands data of wildfires.txt)' if e.event=='Algar Lake Complex'
                    else 'INFERRED: date the GEE search most likely started from (same rule as Algar); confirm'))
O=pd.DataFrame(out).sort_values(['province','event'])
O.to_csv(os.path.join(REV,'temporal/event_table_recovered.csv'),index=False)
pd.set_option('display.width',250)
print(O[['event','province','agency_fire_id','agency_fire_name','fire_year','event_date','final_size_ha','offset_km','source_file']].to_string())
