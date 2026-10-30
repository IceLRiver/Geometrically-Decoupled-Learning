from pathlib import Path
import argparse,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'data/legacy_simulation/manifest.json').read_text())
for key,shape in [('X',(manifest['samples'],240)),('y',(manifest['samples'],6))]:
 dst=np.lib.format.open_memmap(args.output/f'dataset_{key}_block_0000.npy',mode='w+',dtype='float32',shape=shape)
 for item in manifest['chunks']:
  z=np.load(ROOT/'data/legacy_simulation'/item['file']);dst[item['start']:item['stop']]=z[key]
 dst.flush();del dst
print('Exported original-shape NPY arrays to',args.output)
