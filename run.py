import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibrate','evaluate'],required=True);p.add_argument('--dataset',default='all');a=p.parse_args()
names=['CitiBike_count','CitiBike_duration_seconds','Taxi_pickup','USGS_all_types'] if a.dataset=='all' else [a.dataset]
for name in names:subprocess.run([sys.executable,'-m','experiments.'+a.stage,name],check=True)
