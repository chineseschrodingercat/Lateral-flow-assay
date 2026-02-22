# 🧬 LFA Automatic Analyzer

The **LFA Automatic Analyzer** is a Python-based Streamlit web application designed for researchers and scientists to automatically process, quantify, and analyze Lateral Flow Assay (LFA) strips. 


By applying robust computer vision and signal processing techniques, this tool extracts **Test (T)** and **Control (C)** line intensities, applies flat global baselines, and calculates highly accurate **Area Under the Curve (AUC)** T/C ratios.

---

## ✨ Key Features

### Dual Input Workflows
* **📂 Batch Upload:** Process entire folders of individually pre-cropped LFA strip images simultaneously.
* **✂️ Single Photo Mode:** Upload a single image of a board containing multiple strips. Use intuitive UI sliders to define the active test window, and the app will automatically segment, extract, and analyze every strip.

### Assay Type Selector (Dynamic Sensitivity)
* **Traditional (e.g., hCG):** High sensitivity (`prominence = 0.5`) to catch faint positive test lines.
* **Competitive (e.g., Xylazine):** Lower sensitivity (`prominence = 1.8`) to ignore background noise and correctly identify negative/faint competitive lines.

### Advanced Signal Processing
* **Auto-Orientation:** Automatically rotates horizontal strip images to a vertical processing format.
* **Background Detrending:** Uses linear polynomial fitting to remove membrane background gradients ("smiling" or lighting effects), creating a perfectly flat profile.

### Smart Peak Integration
* **Relative T-Line Search:** Anchors to the reliable Control line first, then searches for the Test line within a defined spatial window (ignoring sample pad artifacts).
* **Global Baseline Integration:** Calculates both T and C areas using a shared horizontal baseline floor, ensuring unbiased T/C ratio comparisons.

---

## 🚀 Installation & Setup

You do not need any web development experience to run this app. It runs locally in your browser via Python.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Open your terminal (or Anaconda command prompt) and install the required packages:

pip install streamlit opencv-python-headless numpy pandas matplotlib scipy XlsxWriter


### 3. Run the Application
Navigate to the directory containing the code and run:

streamlit run app.py

The application will automatically open in your default web browser (usually at `http://localhost:8501`).

---

## ⚙️ Advanced Configuration (Under the Hood)

For developers or researchers needing to adapt the algorithm to different physical strip dimensions or noisy membranes, you can adjust the hardcoded parameters at the top of `app.py`:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `FIXED_CROP_MARGIN` | `0.10` | Crops 10% off the left and right edges of the strip to remove shadows from the plastic cassette housing. |
| `SMOOTH_WINDOW` | `15` | The window size for the Savitzky-Golay filter. Increase this number for highly noisy or granular images. |
| `MAX_PEAK_WIDTH` | `30` | The maximum allowed width (in pixels) for integrating a peak. Prevents the baseline from extending infinitely on noisy backgrounds. |
| `T_DIST_NEAR` | `30` | The minimum distance (in pixels) upstream from the Control line to stop searching for the Test line (prevents merging). |
| `T_DIST_FAR` | `100` | The maximum distance (in pixels) upstream from the Control line to start searching (prevents picking up the sample pad). |
| `BASELINE_METHOD` | `'lower'` | Determines how the global baseline is drawn. `'lower'` anchors to the lowest detected peak boundary (safest for max area). |

---

## 📂 Output Files & Data

Once the analysis is complete, the application generates a single, downloadable ZIP file containing structured data for your records:

### 1. Summary_Analysis.xlsx
A master Excel report containing automatically generated Pivot Tables.
* **Adjusted Sheet:** Contains the final T/C ratios using the smoothed, detrended, and baseline-corrected data.
* **Unadjusted Sheet:** Contains raw T/C ratios for algorithm verification and comparison.

### 2. Individual Strip Visualizations
Inside the ZIP, each strip gets its own dedicated folder (`VideoID/strip_ID/`) containing:
* `plot_adjusted.png`: The cleaned signal profile showing the flat global baseline (red dashed line) and shaded integration areas.
* `plot_unadjusted.png`: The raw, unprocessed signal profile.
* `plot_comparison.png`: An overlay graph showing the raw vs. adjusted profiles and the gray "Search Zone" where the algorithm looked for the Test line.

---

## 📝 File Naming Convention (Batch Mode)

When using the Batch Upload mode, ensure your pre-cropped images follow this naming structure so the app can correctly map them to the Excel spreadsheet:

`[VideoID]_strip_[StripID].jpg`

> **Example:** `00158_strip_1.jpg`, `00158_strip_2.jpg`
