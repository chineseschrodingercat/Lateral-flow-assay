=========================================================
LFA Automatic Analyzer - Streamlit Web App
=========================================================

DESCRIPTION
A Python-based Streamlit application for the automated quantification 
of Lateral Flow Assay (LFA) strips. It extracts Test (T) and Control (C) 
line intensities, applies baseline flattening, and calculates area-based 
T/C ratios.

KEY FEATURES
- Assay Type Selector: Toggle between Traditional (hCG) and Competitive 
  (Xylazine) modes to auto-set peak detection sensitivity (Prominence).
- Dual Input Modes: Batch upload pre-cropped strips or upload a single 
  board photo for auto-segmentation.
- Signal Processing: Savitzky-Golay smoothing and linear background 
  detrending (flattening).
- Global Baseline Integration: Both T and C peaks are integrated using 
  the same horizontal floor for high-precision T/C ratios.
- Automated Reporting: Generates a ZIP file containing profile plots, 
  comparison graphs, and a master Excel summary.

HOW TO RUN
1. Install dependencies:
   pip install streamlit opencv-python-headless numpy pandas matplotlib scipy XlsxWriter
2. Run the app:
   streamlit run app.py

INTERNAL CONFIGURATION (app.py)
Edit these hardcoded values at the top of the script for custom tuning:
- FIXED_CROP_MARGIN: 0.1 (10% edge removal)
- MAX_PEAK_WIDTH: 30 (Clamps integration width to prevent area inflation)
- T_DIST_NEAR / T_DIST_FAR: Defines the search window for the T-line 
  relative to the C-line.
- BASELINE_METHOD: 'lower' (Anchors global baseline to the lowest 
  detected peak boundary).

OUTPUT FILES
- Summary_Analysis.xlsx: Master report with pivot tables of T/C ratios.
- plot_adjusted.png: Cleaned profiles showing the flat global baseline.
- plot_comparison.png: Overlay of raw vs. processed signal.
=========================================================
