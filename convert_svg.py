import matplotlib.colors as colors
import matplotlib.cm as cmx
from pathlib import Path
import imageio
import shutil
import subprocess
import tempfile
import matplotlib.pyplot as plt
import numpy as np

def render_svg(image, bins, node_num, pth):
    cmap = plt.get_cmap('jet')
    cnorm = colors.Normalize(vmin=0, vmax=len(bins) - 1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    first_svg = True
    tgt_svg_path = Path(pth)
#     _, tgt_svg_path = tempfile.mkstemp(suffix='.svg')
#     print(_, tgt_svg_path)
#     tgt_svg_path = Path(tgt_svg_path)
#     print(tgt_svg_path)

    _, tmp_bmp_path = tempfile.mkstemp(suffix='.bmp')
    tmp_bmp_path = Path(tmp_bmp_path)
    cond = (bins == True) & (image.mean(axis=2) < 0.5)
    #cond = (bins == True) & (image.mean(axis=1) < 0.5)
    tmp_bmp_img = np.where(cond, 255, 0)
    imageio.imsave(str(tmp_bmp_path), tmp_bmp_img.astype(np.uint8))
    color = np.asarray(cscalarmap.to_rgba(node_num))
    color *= 255
    color_hex = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in color])
    exe = "/tmp/mozilla_rhythm0/potrace-1.16.linux-x86_64/potrace"
    subprocess.call([exe, '-s', '-i', '-C' + color_hex, tmp_bmp_path])
    tmp_bmp_path = Path(tmp_bmp_path)
    tmp_svg_path = tmp_bmp_path.with_suffix(".svg")
    if first_svg:
        shutil.move(str(tmp_svg_path), str(tgt_svg_path))
        first_svg = False
    else:
        with tgt_svg_path.open("r") as f_tgt:
            tgt_svg = f_tgt.read()
        with tmp_svg_path.open("r") as f_src:
            src_svg = f_src.read()

        path_start = src_svg.find('<g')
        path_end = src_svg.find('</svg>')

        insert_pos = tgt_svg.find('</svg>')
        tgt_svg = tgt_svg[:insert_pos] + \
                  src_svg[path_start:path_end] + tgt_svg[insert_pos:]
        with tgt_svg_path.open("w") as f_tgt:
            f_tgt.write(tgt_svg)
        tmp_svg_path.unlink()
    tmp_bmp_path.unlink()

    # set opacity 0.5 to see overlaps
    with tgt_svg_path.open("r") as f_tgt:
        tgt_svg = f_tgt.read()
    insert_pos = tgt_svg.find('<g')
    tgt_svg = tgt_svg[:insert_pos] + \
              '<g fill-opacity="0.5">' + tgt_svg[insert_pos:]
    insert_pos = tgt_svg.find('</svg>')
    tgt_svg = tgt_svg[:insert_pos] + '</g>' + tgt_svg[insert_pos:]
    with tgt_svg_path.open("w") as f_tgt:
        f_tgt.write(tgt_svg)
    return tgt_svg_path