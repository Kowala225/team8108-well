import sys
from pathlib import Path

def normalize(bbox):
    c = max(0, int(bbox[0]))
    x = max(0.0, min(1.0, bbox[1]))
    y = max(0.0, min(1.0, bbox[2]))
    w = max(0.001, min(1.0, bbox[3]))
    h = max(0.001, min(1.0, bbox[4]))
    hw, hh = w/2, h/2
    x = max(hw, min(1-hw, x))
    y = max(hh, min(1-hh, y))
    return [c, x, y, w, h]

label_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'labels')
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

if output_dir:
    output_dir.mkdir(parents=True, exist_ok=True)

for f in label_dir.glob('*.txt'):
    lines = []
    for line in open(f):
        parts = line.split()
        if len(parts) == 5:
            try:
                lines.append(normalize([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]))
            except:
                pass
    out = output_dir / f.name if output_dir else f
    with open(out, 'w') as o:
        for bbox in lines:
            o.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
