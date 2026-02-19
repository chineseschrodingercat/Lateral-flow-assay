import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, savgol_filter
import io
import zipfile
import re

# ==========================================
# ANALYSIS CONFIGURATION
# ==========================================
CROP_MARGIN = 0.1
SMOOTH_WINDOW = 15
SMOOTH_POLY = 3
PEAK_PROMINENCE = 0.5

# --- NEW: RELATIVE POSITIONING ---
# Instead of fixed pixels (30-60), we define where T is relative to C.
# "The Test line is usually found at 30% to 60% of the distance from the top,
# relative to the Control line's position."
# BUT simpler: T is always "upstream" of C.
# We will search for T in a window defined by the C position.
# E.g., T is usually 100-200 pixels away? No, that depends on resolution.
# Better: We assume T is roughly at 1/3 to 2/3 of the strip length relative to C.
# Let's stick to the physical logic: T is always ABOVE C.

# --- NEW: BASELINE CLAMPING ---
MAX_PEAK_WIDTH = 35  # Max width (in pixels) allowed for a single peak's base.
                     # If the algorithm finds a base width of 80, we clip it to +/- 25 from peak.

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def flatten_profile(profile):
    """Removes linear tilt from profile."""
    x = np.arange(len(profile))
    margin = int(len(profile) * 0.10)
    bg_mask = np.concatenate([np.arange(margin), np.arange(len(profile)-margin, len(profile))])
    if len(bg_mask) > 0:
        poly = np.polyfit(x[bg_mask], profile[bg_mask], 1)
        background_trend = np.polyval(poly, x)
        flattened = profile - background_trend
        return flattened - np.min(flattened)
    return profile

def analyze_single_strip(img_array, crop_margin=0.0):
    # Ensure input is a valid numpy array
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array

    # Auto-rotate if horizontal
    h, w = img.shape
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape
    
    # Cropping
    if crop_margin > 0:
        px_cut = int(w * crop_margin)
        img_roi = img[:, px_cut : w - px_cut]
    else:
        img_roi = img 

    if img_roi.shape[1] < 5: return None

    # Processing
    img_inverted = 255 - img_roi
    raw_profile = np.mean(img_inverted, axis=1)
    
    try:
        smooth_profile = savgol_filter(raw_profile, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLY)
    except:
        smooth_profile = raw_profile

    smooth_profile = flatten_profile(smooth_profile)
    
    # Find ALL peaks first
    peaks, properties = find_peaks(smooth_profile, prominence=PEAK_PROMINENCE, distance=10, width=2)
    
    height = len(smooth_profile)
    # Estimate Split Point (roughly middle) to distinguish T vs C zone initially
    split_point = int(height * 0.6)
    
    t_area, c_area = 0.0, 0.0
    t_pos, c_pos = np.nan, np.nan
    t_baseline_x, t_baseline_y = [], []
    c_baseline_x, c_baseline_y = [], []

    # 1. FIND CONTROL LINE FIRST (It's the strongest/most reliable anchor)
    # It usually sits in the bottom half of the strip
    c_candidates = [p for p in peaks if p > split_point]
    
    if c_candidates:
        c_indices = [np.where(peaks == p)[0][0] for p in c_candidates]
        # Pick the strongest peak in the bottom half
        best_c_idx = c_indices[np.argmax(properties['prominences'][c_indices])]
        c_pos = peaks[best_c_idx]
        
        # --- BASELINE CLAMPING (Fix #2) ---
        results = peak_widths(smooth_profile, [c_pos], rel_height=0.95)
        raw_width = results[0][0]
        
        # Center of the peak
        center = c_pos
        
        # Calculate boundaries
        # If width is excessive, clamp it
        half_width = min(raw_width / 2, MAX_PEAK_WIDTH / 2)
        x_s = max(0, int(center - half_width))
        x_e = min(height - 1, int(center + half_width))
        
        # Integration
        if x_e > x_s:
            base_val = min(smooth_profile[x_s], smooth_profile[x_e])
            c_area = np.sum(smooth_profile[x_s:x_e+1] - base_val)
            c_baseline_x = [x_s, x_e]
            c_baseline_y = [base_val, base_val]

    # 2. FIND TEST LINE (Relative to Control)
    # We look for a peak that is "upstream" (lower pixel index) from C
    # Expected Distance: usually 1/3 to 1/2 of the total strip length away
    # Search Window: From 0 to (C_Pos - Buffer)
    if not np.isnan(c_pos):
        # Search for T in the region BEFORE C
        # e.g., T must be at least 20 pixels away from C to avoid merging
        search_end = int(c_pos - 20)
        t_candidates = [p for p in peaks if p < search_end]
        
        if t_candidates:
            t_indices = [np.where(peaks == p)[0][0] for p in t_candidates]
            # Pick strongest in that region
            best_t_idx = t_indices[np.argmax(properties['prominences'][t_indices])]
            t_pos = peaks[best_t_idx]
            
            # --- BASELINE CLAMPING (Fix #2) ---
            results = peak_widths(smooth_profile, [t_pos], rel_height=0.95)
            raw_width = results[0][0]
            
            center = t_pos
            half_width = min(raw_width / 2, MAX_PEAK_WIDTH / 2)
            x_s = max(0, int(center - half_width))
            x_e = min(height - 1, int(center + half_width))
            
            if x_e > x_s:
                base_val = min(smooth_profile[x_s], smooth_profile[x_e])
                t_area = np.sum(smooth_profile[x_s:x_e+1] - base_val)
                t_baseline_x = [x_s, x_e]
                t_baseline_y = [base_val, base_val]

    ratio = t_area / c_area if c_area > 0 else 0
    
    return {
        "profile": smooth_profile,
        "t_area": t_area, "c_area": c_area, "ratio": ratio,
        "t_pos": t_pos, "c_pos": c_pos,
        "t_base_x": t_baseline_x, "t_base_y": t_baseline_y,
        "c_base_x": c_baseline_x, "c_base_y": c_baseline_y
    }

def create_plot(res, title, color='blue'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['profile'], color=color, linewidth=2, label='Profile')
    
    # Mark Test Line Baseline
    if len(res['t_base_x']) > 0:
        t_val = res['t_base_y'][0]
        ax.plot(res['t_base_x'], res['t_base_y'], 'r--', linewidth=2)
        ax.fill_between(range(res['t_base_x'][0], res['t_base_x'][1]+1), 
                         res['profile'][res['t_base_x'][0]:res['t_base_x'][1]+1], 
                         t_val, alpha=0.3, color='green')
        ax.text(res['t_pos'], res['profile'][int(res['t_pos'])]+5, f"T:{res['t_area']:.1f}", color="green", ha='center')

    # Mark Control Line Baseline
    if len(res['c_base_x']) > 0:
        c_val = res['c_base_y'][0]
        ax.plot(res['c_base_x'], res['c_base_y'], 'r--', linewidth=2)
        ax.fill_between(range(res['c_base_x'][0], res['c_base_x'][1]+1), 
                         res['profile'][res['c_base_x'][0]:res['c_base_x'][1]+1], 
                         c_val, alpha=0.3, color='orange')
        ax.text(res['c_pos'], res['profile'][int(res['c_pos'])]+5, f"C:{res['c_area']:.1f}", color="red", ha='center')

    # Remove the fixed gray span, since T-region is now dynamic
    # ax.axvspan(...) 
    
    ax.set_title(f"{title}\nR={res['ratio']:.4f}")
    plt.tight_layout()
    return fig

# ==========================================
# NEW MODULE: AUTO-SEGMENTATION
# ==========================================
def detect_and_slice_strips(full_img, top_crop, bottom_crop):
    """
    Takes the full image and the manual Y-coordinates.
    detects white vertical strips and returns a list of cropped images.
    """
    # 1. Apply Manual Crop
    roi = full_img[top_crop:bottom_crop, :]
    
    # 2. Convert to Grayscale & Threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Otsu's thresholding to find white parts
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Vertical Projection (Sum pixels down columns)
    # White strips will have HIGH sum, Black gaps will have LOW sum
    vertical_sum = np.sum(thresh, axis=0)
    
    # Normalize
    vertical_sum = vertical_sum / np.max(vertical_sum)
    
    # 4. Find where the strips are (columns with signal > 0.5)
    # We look for continuous regions of high intensity
    strip_locs = []
    in_strip = False
    start_x = 0
    
    # Threshold for "Is this a strip?"
    col_threshold = 0.2 
    
    for x, val in enumerate(vertical_sum):
        if val > col_threshold and not in_strip:
            in_strip = True
            start_x = x
        elif val < col_threshold and in_strip:
            in_strip = False
            end_x = x
            # Filter tiny noise (width must be > 10px)
            if (end_x - start_x) > 20:
                # Add margin to capture full width
                margin = 5
                s = max(0, start_x - margin)
                e = min(roi.shape[1], end_x + margin)
                strip_locs.append((s, e))
                
    # 5. Extract Images
    strip_images = []
    for (s, e) in strip_locs:
        strip_img = roi[:, s:e]
        strip_images.append(strip_img)
        
    return strip_images, strip_locs, roi

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="LFA Auto-Analyzer", layout="wide")
st.title("🧬 LFA Automatic Analyzer")

# Sidebar for Mode Selection
analysis_mode = st.sidebar.radio("Select Input Mode", 
    ["📂 Batch Upload (Already Cropped)", "✂️ Single Photo (Crop & Extract)"])

if analysis_mode == "📂 Batch Upload (Already Cropped)":
    st.info("Upload a folder of pre-cropped strip images (e.g. `00158_strip_1.jpg`).")
    uploaded_files = st.file_uploader("Select Images", accept_multiple_files=True, type=['jpg', 'png', 'tif'])
    
    if uploaded_files and st.button("Start Batch Analysis"):
        # ... (Same Batch Logic as before) ...
        progress_bar = st.progress(0)
        zip_buffer = io.BytesIO()
        summary_rows = []
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Load as color for consistency
                
                filename = uploaded_file.name
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Logic to parse VideoID/StripID
                video_id, strip_id = 0, i+1
                match = re.match(r"(\d+)_strip_(\d+)", filename)
                if match:
                    video_id = int(match.group(1))
                    strip_id = int(match.group(2))
                
                res_adj = analyze_single_strip(img, crop_margin=CROP_MARGIN)
                res_unadj = analyze_single_strip(img, crop_margin=0.0)
                
                if res_adj and res_unadj:
                    folder_path = f"{video_id}/strip_{strip_id}/"
                    
                    # Save Graphs
                    fig_adj = create_plot(res_adj, f"Adjusted: {filename}")
                    img_buf = io.BytesIO()
                    fig_adj.savefig(img_buf, format='png')
                    zf.writestr(f"{folder_path}plot_adjusted.png", img_buf.getvalue())
                    plt.close(fig_adj)
                    
                    fig_unadj = create_plot(res_unadj, f"Unadjusted: {filename}", color='gray')
                    img_buf = io.BytesIO()
                    fig_unadj.savefig(img_buf, format='png')
                    zf.writestr(f"{folder_path}plot_unadjusted.png", img_buf.getvalue())
                    plt.close(fig_unadj)
                    
                    fig_comp, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(res_unadj['profile'], color='gray', linestyle='--', alpha=0.6, label='Unadjusted')
                    ax.plot(res_adj['profile'], color='blue', linewidth=2, label='Adjusted')
                    ax.set_title(f"Comparison: {filename}")
                    img_buf = io.BytesIO()
                    fig_comp.savefig(img_buf, format='png')
                    zf.writestr(f"{folder_path}plot_comparison.png", img_buf.getvalue())
                    plt.close(fig_comp)
                    
                    summary_rows.append({
                        "Video_ID": video_id, "Strip_ID": strip_id,
                        "Adj_Ratio": res_adj['ratio'], "Unadj_Ratio": res_unadj['ratio']
                    })

            if summary_rows:
                df = pd.DataFrame(summary_rows)
                # Pivot and Save Excel
                pivot_adj = df.pivot_table(index="Video_ID", columns="Strip_ID", values="Adj_Ratio")
                pivot_unadj = df.pivot_table(index="Video_ID", columns="Strip_ID", values="Unadj_Ratio")
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    pivot_adj.to_excel(writer, sheet_name='Adjusted')
                    pivot_unadj.to_excel(writer, sheet_name='Unadjusted')
                zf.writestr("Summary_Analysis.xlsx", excel_buffer.getvalue())

        st.success("Processing Complete!")
        st.download_button("📥 Download Results (ZIP)", data=zip_buffer.getvalue(), file_name="Batch_Results.zip", mime="application/zip")


elif analysis_mode == "✂️ Single Photo (Crop & Extract)":
    st.info("Upload a photo containing multiple strips on a black board.")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        full_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB) # Fix colors for display
        
        height, width, _ = full_img.shape
        
        st.write("### Step 1: Adjust Crop Region")
        st.markdown("Use the sliders to define the **Top** and **Bottom** boundaries of the white windows.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_crop = st.slider("Top Cut", 0, height, int(height*0.3))
            bottom_crop = st.slider("Bottom Cut", 0, height, int(height*0.7))
        
        with col2:
            # Visualize the crop lines
            display_img = full_img.copy()
            cv2.line(display_img, (0, top_crop), (width, top_crop), (255, 0, 0), 5) # Red Line Top
            cv2.line(display_img, (0, bottom_crop), (width, bottom_crop), (255, 0, 0), 5) # Red Line Bottom
            st.image(display_img, use_container_width=True)

        if bottom_crop > top_crop:
            if st.button("Detect & Analyze Strips"):
                
                # Perform Auto-Segmentation
                strips, locs, roi_img = detect_and_slice_strips(full_img, top_crop, bottom_crop)
                
                st.write(f"### Step 2: Found {len(strips)} Strips")
                
                # Visualize detection
                for (s, e) in locs:
                    cv2.rectangle(roi_img, (s, 0), (e, roi_img.shape[0]), (0, 255, 0), 2)
                st.image(roi_img, caption="Detected Boundaries (Green)", use_container_width=True)
                
                # Process extracted strips
                zip_buffer = io.BytesIO()
                summary_rows = []
                
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # Save the "Master Crop" image for reference
                    img_buf = io.BytesIO()
                    plt.imsave(img_buf, roi_img, format='png')
                    zf.writestr("Detection_Reference.png", img_buf.getvalue())
                    
                    cols = st.columns(min(len(strips), 5)) # Display preview of first 5
                    
                    for i, strip_img in enumerate(strips):
                        strip_id = i + 1
                        
                        # Analyze
                        res_adj = analyze_single_strip(strip_img, crop_margin=CROP_MARGIN)
                        res_unadj = analyze_single_strip(strip_img, crop_margin=0.0)
                        
                        if res_adj and res_unadj:
                            # Display mini preview
                            if i < 5:
                                cols[i].image(strip_img, caption=f"Strip {strip_id}")
                            
                            folder_path = f"Extracted_Strips/strip_{strip_id}/"
                            
                            # Save Graphs
                            fig_adj = create_plot(res_adj, f"Strip {strip_id} (Adj)")
                            img_buf = io.BytesIO()
                            fig_adj.savefig(img_buf, format='png')
                            zf.writestr(f"{folder_path}plot_adjusted.png", img_buf.getvalue())
                            plt.close(fig_adj)
                            
                            fig_unadj = create_plot(res_unadj, f"Strip {strip_id} (Unadj)", color='gray')
                            img_buf = io.BytesIO()
                            fig_unadj.savefig(img_buf, format='png')
                            zf.writestr(f"{folder_path}plot_unadjusted.png", img_buf.getvalue())
                            plt.close(fig_unadj)
                            
                            fig_comp, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(res_unadj['profile'], color='gray', linestyle='--', alpha=0.6)
                            ax.plot(res_adj['profile'], color='blue', linewidth=2)
                            ax.set_title(f"Comparison Strip {strip_id}")
                            img_buf = io.BytesIO()
                            fig_comp.savefig(img_buf, format='png')
                            zf.writestr(f"{folder_path}plot_comparison.png", img_buf.getvalue())
                            plt.close(fig_comp)
                            
                            summary_rows.append({
                                "Video_ID": 1, "Strip_ID": strip_id, # Default Video ID 1 for single photo
                                "Adj_Ratio": res_adj['ratio'], "Unadj_Ratio": res_unadj['ratio']
                            })

                    # Excel Summary
                    if summary_rows:
                        df = pd.DataFrame(summary_rows)
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name='Summary', index=False)
                        zf.writestr("Summary_Analysis.xlsx", excel_buffer.getvalue())
                
                st.success("Analysis Complete!")
                st.download_button("📥 Download Extracted Results", data=zip_buffer.getvalue(), file_name="Extracted_Analysis.zip", mime="application/zip")
