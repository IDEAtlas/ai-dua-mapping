import os
import subprocess
import argparse
from glob import glob
from pathlib import Path
from tqdm import tqdm

from osgeo import gdal

def extract_bands(input_path, output_path, bands_to_keep):
    gdal.Translate(
        output_path,
        input_path,
        bandList=bands_to_keep
    )


def get_msi_pan_pairs(msi_dirs, pan_dirs):
    """
    Match MSI and PAN images by sorted order (assumes both lists are aligned).
    """
    def sorted_tifs(dirs):
        return sorted([
            f for d in dirs for f in glob(os.path.join(d, '**', '*.TIF'), recursive=True)
        ])

    msi_files = sorted_tifs(msi_dirs)
    pan_files = sorted_tifs(pan_dirs)

    if len(msi_files) != len(pan_files):
        raise ValueError(f"Mismatch: {len(msi_files)} MSI files vs {len(pan_files)} PAN files")

    # Optional: sanity check filenames
    for i, (m, p) in enumerate(zip(msi_files, pan_files)):
        m_base = os.path.basename(m).split('.')[0]
        p_base = os.path.basename(p).split('.')[0]
        if m_base[:8] != p_base[:8]:
            print(f"⚠️ Warning: Possible mismatch at pair {i}: {m_base} vs {p_base}")

    print(f"Matched {len(msi_files)} MSI–PAN pairs by sorted order")
    return list(zip(msi_files, pan_files))


def pansharpen_tiles(msi_pan_pairs, out_dir, bands_to_keep=(5, 3, 2, 7)):
    """
    Pansharpen MSI-PAN pairs and extract RGBNIR bands.

    - For **WorldView-3 8-band MSI** (0.3m PAN, 1.2m MSI):
        RGBNIR = bands 5 (Red), 3 (Green), 2 (Blue), 7 (NIR1)

    - For **Pleiades 0.5m** imagery (4-band MSI):
        RGBNIR = bands 3 (Red), 2 (Green), 1 (Blue), 4 (NIR)
        --> Set bands_to_keep=(3, 2, 1, 4)
    """
    out_paths = []
    os.makedirs(out_dir, exist_ok=True)

    for msi_path, pan_path in tqdm(msi_pan_pairs, desc="Pansharpening tiles"):
        base_name = os.path.basename(msi_path).replace('M', 'PSHARP', 1)
        full_path = os.path.join(out_dir, base_name)
        tmp_path = full_path.replace('.TIF', '_FULL.TIF')  # temporary full band output

        cmd = [
            'gdal_pansharpen.py',
            pan_path, msi_path, tmp_path,
            '-of', 'GTiff',
            '-r', 'bilinear'
        ]

        try:
            subprocess.run(cmd, check=True)

            # Extract only RGBNIR bands
            extract_bands(tmp_path, full_path, bands_to_keep)

            os.remove(tmp_path)
            out_paths.append(full_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to process {base_name}: {e}")

    return out_paths


def main():
    parser = argparse.ArgumentParser(description="Pansharpen and extract RGBNIR bands from PAN+MSI images")
    parser.add_argument("--msi_dirs", nargs='+', required=True, help="MSI directories")
    parser.add_argument("--pan_dirs", nargs='+', required=True, help="PAN directories")
    parser.add_argument("--out_dir", required=True, help="Output directory for pansharpened tiles")

    # Optional: band config for Pleiades or other sensors
    parser.add_argument("--bands", nargs=4, type=int, default=[5, 3, 2, 7],
                        help="Band indices to extract as RGBNIR (1-based GDAL index)")

    args = parser.parse_args()

    pairs = get_msi_pan_pairs(args.msi_dirs, args.pan_dirs)
    print(f"Found {len(pairs)} MSI–PAN pairs")

    out_paths = pansharpen_tiles(pairs, args.out_dir, bands_to_keep=tuple(args.bands))
    print(f"Pansharpened {len(out_paths)} tiles to {args.out_dir}")


if __name__ == "__main__":
    main()
