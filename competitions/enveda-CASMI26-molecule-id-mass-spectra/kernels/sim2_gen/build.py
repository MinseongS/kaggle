"""Embed selection.json into the kernel script. usage: python kernels/sim2_gen/build.py"""
import base64, json, zlib, pathlib
d = pathlib.Path(__file__).parent
sel = json.load(open(d / 'selection.json'))
blob = base64.b64encode(zlib.compress(json.dumps(sel).encode(), 9)).decode()
src = (d / 'sim2_gen_src.py').read_text().replace('__SELECTION__', blob)
(d / 'casmi26-sim2-gen.py').write_text(src)
print('wrote', len(src), 'bytes,', len(sel), 'molecules')
