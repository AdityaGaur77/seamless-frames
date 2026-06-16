# Satellite Frame Interpolation Prototype

AI/ML optical-flow-based temporal interpolation for geostationary satellite imagery.

## Goal

Generate intermediate satellite frames between two consecutive thermal-infrared frames, improving temporal resolution such as:

- 30 min → 15 min
- 15 min → 7.5 min
- 20 min → 10 min

The prototype focuses on `.nc` NetCDF inputs and outputs, optical-flow-style motion estimation, deep-interpolation-style frame synthesis, validation metrics, and a visualization dashboard.

## Project Structure

```text
satellite-frame-interpolation/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
├── demo_output/
├── dashboard/
│   ├── index.html
│   └── app.js
├── notebooks/
│   └── demo_notebook.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── demo_data.py
    ├── io_nc.py
    ├── metrics.py
    ├── interpolation.py
    ├── pipeline.py
    └── report.py
```

## Quick Start

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Generate synthetic demo NetCDF frames:

```bash
python -m src.demo_data --out-dir data/demo --num-frames 4 --height 128 --width 128
```

Run interpolation and validation:

```bash
python -m src.pipeline --config config.yaml
```

Outputs will be written to `demo_output/`.

Launch the dashboard:

```bash
cd dashboard
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Input and Output Format

Expected input files are NetCDF `.nc` files containing a 2D thermal-infrared variable.

Default variable names supported:

- `TIR1`
- `thermal_infrared`
- `brightness_temperature`
- `radiance`
- `data`

Example input pair:

```text
data/demo/frame_0000.nc
data/demo/frame_0001.nc
```

Example output:

```text
demo_output/interpolated/frame_0000_0001_t050.nc
demo_output/interpolated/frame_0000_0001_t050.png
demo_output/report.json
demo_output/report.html
```

## Model Approach

This prototype implements a practical baseline that can later be replaced or extended with RIFE/Super-SloMo-style deep networks.

Current pipeline:

1. Load two consecutive satellite frames from NetCDF.
2. Normalize thermal-infrared values.
3. Estimate optical flow using Farnebäck dense optical flow.
4. Warp both frames toward the target time.
5. Blend warped frames using time-weighted masks.
6. Optionally apply edge-preserving sharpening.
7. Save the interpolated frame as NetCDF.
8. Validate against available ground-truth frames using:
   - MSE
   - RMSE
   - PSNR
   - SSIM
   - FSIM-lite
   - gradient difference
   - temporal consistency

## Configuration

Edit `config.yaml` to change paths, variable names, interpolation factors, and metric settings.

Example:

```yaml
input_dir: data/demo
output_dir: demo_output
variable_name: TIR1
interpolation_factor: 2
```

## Dashboard

The dashboard displays:

- Original frame animation
- Interpolated frame animation
- Ground truth vs predicted comparison
- Metric plots
- Per-frame metric table

The dashboard expects generated files in:

```text
demo_output/images/original/*.png
demo_output/images/interpolated/*.png
demo_output/metrics.json
```

## Next Development Steps

1. Replace Farnebäck flow with trainable RIFE/Super-SloMo-style model.
2. Add training/fine-tuning on GOES-19 ABI Channel 13 or Himawari-8 TIR data.
3. Add AWS/NOAA and MOSDAC data download scripts.
4. Add cloud-motion-specific evaluation metrics.
5. Add batch processing for long time series.
6. Add deployment packaging for a web app.
