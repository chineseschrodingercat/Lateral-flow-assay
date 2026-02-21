LFA Automatic Analyzer - Streamlit Web App

DESCRIPTION
This is a Python-based web application built with Streamlit to automatically process, analyze, and quantify Lateral Flow Assay (LFA) strips. It extracts the Test (T) and Control (C) line intensities, applies baseline corrections, calculates the T/C area ratios, and generates visual plots and a summary Excel report.

FEATURES
- Dual Input Modes: 
  1. Batch Upload: Process a folder of individually pre-cropped strip images.
  2. Single Photo: Upload one image containing multiple strips; use sliders to define the active area, and the app will auto-segment them.
- Auto-Orientation: Automatically rotates horizontal strips to vertical.
- Signal Processing: Uses Savitzky-Golay smoothing and linear background detrending (flattening) to isolate peaks.
- Smart Peak Detection: Finds the Control line first, then searches for the Test line within a dynamically defined distance.
- Global Baseline Integration: Uses a shared horizontal baseline across both peaks for consistent Area Under Curve (AUC) calculation.
- Automated Reporting: Bundles individual strip plots, comparison plots, and a formatted Excel summary into a single downloadable ZIP file.

REQUIREMENTS
Ensure you have Python installed along with the following packages. You can save these to a `requirements.txt` file:
streamlit
opencv-python-headless
numpy
pandas
matplotlib
scipy
XlsxWriter

HOW TO RUN
1. Save the code as `app.py`.
2. Open your terminal or command prompt.
3. Run the following command:
   streamlit run app.py
4. The application will open in your default web browser.

CONFIGURATION (HARDCODED KNOBS)
If you need to tune the algorithm for different strip types or image resolutions, adjust these constants at the top of `app.py`:

- FIXED_CROP_MARGIN (0.1): Crops 10% off the left and right edges to remove plastic housing shadows.
- SMOOTH_WINDOW (15): Window size for the Savitzky-Golay filter. Increase for noisier images.
- MAX_PEAK_WIDTH (30): Maximum allowed pixel width for peak integration to prevent runaway baselines.
- T_DIST_NEAR (20): The minimum distance (in pixels) upstream from the Control line to stop searching for the Test line.
- T_DIST_FAR (120): The maximum distance (in pixels) upstream from the Control line to start searching for the Test line.
- BASELINE_METHOD ('lower'): Determines the shared global baseline. Options are 'lower' (safest/max area), 'higher', or 'average'.

OUTPUT
The app generates a ZIP file containing:
- `Summary_Analysis.xlsx`: An Excel file with pivot tables of the calculated T/C ratios.
- `plot_adjusted.png`: Profile plots with the background flattening applied.
- `plot_unadjusted.png`: Raw profile plots for reference.
- `plot_comparison.png`: Overlay of adjusted vs. unadjusted profiles.
=========================================================
