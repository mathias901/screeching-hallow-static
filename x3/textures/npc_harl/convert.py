import struct, json, sys, base64, io
import numpy as np
from PIL import Image
SRC = '/mnt/user-data/attachments/39f581a8-af5b-5508-9c72-f1486222055e/12214931-npc_dock_worker.glb'
d = open(SRC, 'rb').read()
off = 12; chunks = []
while off < len(d):
    cl, ct = struct.unpack('<II', d[off:off+8]); chunks.append((ct, d[off+8:off+8+cl])); off += 8+cl
j = json.loads(chunks[0][1]); BIN = chunks[1][1]
def acc(i):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    o = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    nc = {'SCALAR':1,'VEC2':2,'VEC3':3}[a['type']]
    dt = {5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']]
    arr = np.frombuffer(BIN, dtype=dt, count=a['count']*nc, offset=o)
    return arr.reshape(a['count'], nc) if nc > 1 else arr
prim = j['meshes'][0]['primitives'][0]
P = acc(prim['attributes']['POSITION']).astype(np.float64)
N = acc(prim['attributes']['NORMAL']).astype(np.float64)
UV = acc(prim['attributes']['TEXCOORD_0']).astype(np.float64)
I = acc(prim['indices']).astype(np.int64).reshape(-1, 3)
# scale to 1.458 m tall, feet at y = 0, centred on x and z
ymin, ymax = P[:,1].min(), P[:,1].max()
H_TARGET = 1.458
k = H_TARGET / (ymax - ymin)
P = P * k
P[:,1] -= P[:,1].min()
P[:,0] -= (P[:,0].max() + P[:,0].min())/2
P[:,2] -= (P[:,2].max() + P[:,2].min())/2
print('scaled bounds', P.min(0).round(3), P.max(0).round(3), 'tris', len(I))
def cluster(s):
    keys = np.floor(P / s).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    M = inv.max() + 1
    cnt = np.bincount(inv, minlength=M).astype(np.float64)
    def avg(A):
        return np.stack([np.bincount(inv, weights=A[:,c], minlength=M) / cnt for c in range(A.shape[1])], 1)
    Pc = avg(P); Nc = avg(N); UVc = avg(UV)
    Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-9)
    tri = inv[I]
    keep = (tri[:,0] != tri[:,1]) & (tri[:,1] != tri[:,2]) & (tri[:,0] != tri[:,2])
    return Pc, Nc, UVc, tri[keep]
TARGET = 60000
lo, hi = 0.0005, 0.05
best = None
for _ in range(18):
    mid = (lo*hi)**0.5
    Pc, Nc, UVc, T = cluster(mid)
    if len(T) > TARGET: lo = mid
    else: hi = mid; best = (mid, Pc, Nc, UVc, T)
s, Pc, Nc, UVc, T = best
print('cell', round(s,5), 'verts', len(Pc), 'tris', len(T))
# binary layout: positions, normals, uvs (float32) then indices (uint32)
n = len(Pc)
bin_ = Pc.astype('<f4').tobytes() + Nc.astype('<f4').tobytes() + UVc.astype('<f4').tobytes() + T.astype('<u4').tobytes()
assert len(Pc.astype('<f4').tobytes()) == n*12
H = H_TARGET
part = {'o': 0, 'n': n, 'oi': n*32, 'ni': len(T)*3, 'u32': True, 'mat': 'm'}
objects = {
  'mdl': {'parts': [part], 'w': float(Pc[:,0].max()-Pc[:,0].min()), 'h': H, 'd': float(Pc[:,2].max()-Pc[:,2].min()), 'px': 0, 'py': 0, 'pz': 0},
  'legA': {'parts': [], 'w': 0, 'h': 0, 'd': 0, 'px': 0, 'py': round(0.42*H,4), 'pz': 0},
  'legB': {'parts': [], 'w': 0, 'h': 0, 'd': 0, 'px': 0, 'py': round(0.42*H,4), 'pz': 0},
}
out = {'data': base64.b64encode(bin_).decode(), 'objects': objects}
open('npc.json', 'w').write(json.dumps(out, separators=(',', ':')))
print('npc.json bytes', len(json.dumps(out, separators=(',', ':'))))
# textures: find the three images in the file
def img(bvi):
    bv = j['bufferViews'][bvi]; o = bv.get('byteOffset', 0)
    return Image.open(io.BytesIO(BIN[o:o+bv['byteLength']]))
imgs = {im['name']: im for im in j['images']}
tex = lambda name: img(imgs[name]['bufferView'])
color = tex('texture_diffuse').convert('RGB').resize((1024,1024), Image.LANCZOS)
normal = tex('texture_normal').convert('RGB').resize((1024,1024), Image.LANCZOS)
mr = tex('texture_metallic-texture_roughness').convert('RGB').resize((1024,1024), Image.LANCZOS)
color.save('m_color.jpg', quality=90)
normal.save('m_normal.jpg', quality=90)
r, g, b = mr.split()
orm = Image.merge('RGB', (Image.new('L', (1024,1024), 255), g, b))
orm.save('m_orm.jpg', quality=90)
print('textures written')
