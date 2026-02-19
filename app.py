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
# ⚙️ HARDCODED KNOBS
# ==========================================
FIXED_CROP_MARGIN = 0.1
SMOOTH_WINDOW = 15
MAX_PEAK_WIDTH = 30 # Upper limit: If a peak is wider than this, we clip it.

# --- Search Parameters ---
T_DIST_NEAR = 20
T_DIST_FAR = 120
BASELINE_METHOD = 'lower'

# ==========================================
# CORE ALGORITHMS
# ==========================================
def flatten_profile(profile):
    x = np.arange(len(profile))
    margin = int(len(profile) * 0.10)
    bg_mask = np.concatenate([np.arange(margin), np.arange(len(profile)-margin, len(profile))])
    if len(bg_mask) > 0:
        poly = np.polyfit(x[bg_mask], profile[bg_mask], 1)
        background_trend = np.polyval(poly, x)
        flattened = profile - background_trend
        return flattened - np.min(flattened)
    return profile

def analyze_single_strip(img_array, crop_margin=FIXED_CROP_MARGIN):
    if len(img_array.shape) == 3: img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else: img = img_array

    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape
    px_cut = int(w * crop_margin)
    if px_cut * 2 >= w: return None 
    img_roi = img[:, px_cut : w - px_cut]

    raw_profile = np.mean(255 - img_roi, axis=1)
    try:
        smooth_profile = savgol_filter(raw_profile, window_length=SMOOTH_WINDOW, polyorder=3)
    except:
        smooth_profile = raw_profile
    
    smooth_profile = flatten_profile(smooth_profile)
    peaks, properties = find_peaks(smooth_profile, prominence=0.5, distance=10, width=2)
    
    res = {
        "profile": smooth_profile, "ratio": 0.0,
        "t_area": 0.0, "c_area": 0.0,
        "t_pos": np.nan, "c_pos": np.nan,
        "t_base_x": [], "t_base_y": [], "c_base_x": [], "c_base_y": [],
        "search_window": []
    }
    
    # Calculate widths for ALL peaks found
    # rel_height=0.95 measures width near the base of the peak
    widths_results = peak_widths(smooth_profile, peaks, rel_height=0.95)
    
    # 3. Find CONTROL Line
    height = len(smooth_profile)
    split_point = int(height * 0.5)
    c_candidates = [p for p in peaks if p > split_point]
    
    if c_candidates:
        c_indices = [np.where(peaks == p)[0][0] for p in c_candidates]
        best_c_idx = c_indices[np.argmax(properties['prominences'][c_indices])]
        best_c = peaks[best_c_idx]
        res['c_pos'] = best_c
        
        # Get the detected width for this specific peak
        # widths_results[2] is Left Intersection, widths_results[3] is Right Intersection
        c_left = widths_results[2][best_c_idx]
        c_right = widths_results[3][best_c_idx]
        
        integrate_peak_smart(res, best_c, c_left, c_right, height, 'c')

        # 4. Find TEST Line
        search_end = int(best_c - T_DIST_NEAR)
        search_start = max(0, int(best_c - T_DIST_FAR))
        res['search_window'] = [search_start, search_end]
        
        t_candidates = [p for p in peaks if search_start <= p < search_end]
        
        if t_candidates:
            t_indices = [np.where(peaks == p)[0][0] for p in t_candidates]
            best_t_idx = t_indices[np.argmax(properties['prominences'][t_indices])]
            best_t = peaks[best_t_idx]
            res['t_pos'] = best_t
            
            t_left = widths_results[2][best_t_idx]
            t_right = widths_results[3][best_t_idx]
            
            integrate_peak_smart(res, best_t, t_left, t_right, height, 't')

    res['ratio'] = res['t_area'] / res['c_area'] if res['c_area'] > 0 else 0
    return res

def integrate_peak_smart(res, peak_pos, left_idx, right_idx, height, prefix):
    """
    Uses detected peak width BUT clamps it if it exceeds MAX_PEAK_WIDTH.
    """
    # 1. Get detected start/end
    x_s = int(left_idx)
    x_e = int(right_idx)
    
    # 2. Check width
    current_width = x_e - x_s
    
    # 3. Clamp if too wide (Center +/- Max/2)
    if current_width > MAX_PEAK_WIDTH:
        half_max = MAX_PEAK_WIDTH // 2
        x_s = max(0, int(peak_pos - half_max))
        x_e = min(height - 1, int(peak_pos + half_max))
    else:
        # Safety bounds
        x_s = max(0, x_s)
        x_e = min(height - 1, x_e)

    if x_e > x_s:
        curve = res['profile']
        y_s = curve[x_s]
        y_e = curve[x_e]
        
        if BASELINE_METHOD == 'lower':
            base_val = min(y_s, y_e)
        elif BASELINE_METHOD == 'higher':
            base_val = max(y_s, y_e)
        else: # average
            base_val = (y_s + y_e) / 2
            
        area = np.sum(curve[x_s:x_e+1] - base_val)
        
        res[f'{prefix}_area'] = area
        res[f'{prefix}_base_x'] = [x_s, x_e]
        res[f'{prefix}_base_y'] = [base_val, base_val]

def create_plot(res, title, color='blue'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['profile'], color=color, linewidth=2)
    
    for p_type, col in [('t', 'green'), ('c', 'orange')]:
        if len(res[f'{p_type}_base_x']) > 0:
            bx, by = res[f'{p_type}_base_x'], res[f'{p_type}_base_y']
            ax.plot(bx, by, 'r--', linewidth=2)
            ax.fill_between(range(bx[0], bx[1]+1), res['profile'][bx[0]:bx[1]+1], by[0], alpha=0.3, color=col)
            if not np.isnan(res[f'{p_type}_pos']):
                ax.text(res[f'{p_type}_pos'], res['profile'][int(res[f'{p_type}_pos'])]+5, 
                        f"{p_type.upper()}:{res[f'{p_type}_area']:.1f}", color='red', ha='center')

    if res['search_window']:
        s, e = res['search_window']
        ax.axvspan(s, e, color='gray', alpha=0.15, label='Search Zone')

    ax.set_title(f"{title}\nR={res['ratio']:.4f}")
    plt.tight_layout()
    return fig

# ... (The rest of detect_and_slice_strips and UI code remains exactly the same) ...
# ... (Copy the detect_and_slice_strips, process_and_download, and UI handlers from the previous response) ...
# ==========================================
# (For completeness, simply pasting the previous UI Logic below will work perfectly with these new functions)
# ==========================================

def detect_and_slice_strips(full_img, top, bottom):
    roi = full_img[top:bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    v_sum = np.sum(thresh, axis=0)
    v_sum = v_sum / np.max(v_sum)
    
    locs = []
    in_strip = False
    start_x = 0
    for x, val in enumerate(v_sum):
        if val > 0.2 and not in_strip:
            in_strip, start_x = True, x
        elif val < 0.2 and in_strip:
            in_strip = False
            if (x - start_x) > 20:
                locs.append((max(0, start_x-5), min(roi.shape[1], x+5)))
    
    return [roi[:, s:e] for s, e in locs], locs, roi

st.set_page_config(page_title="LFA Auto-Analyzer", layout="wide")
st.title("🧬 LFA Automatic Analyzer")
mode = st.sidebar.radio("Input Mode", ["📂 Batch Upload", "✂️ Single Photo"])

def process_and_download(image_list, filenames, video_ids, strip_ids):
    zip_buf = io.BytesIO()
    summary = []
    prog = st.progress(0)
    
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, (img, fname, vid, sid) in enumerate(zip(image_list, filenames, video_ids, strip_ids)):
            prog.progress((i+1)/len(image_list))
            
            res_adj = analyze_single_strip(img, crop_margin=FIXED_CROP_MARGIN)
            res_unadj = analyze_single_strip(img, crop_margin=0.0)
            
            if res_adj and res_unadj:
                path = f"{vid}/strip_{sid}/"
                for r, name, col in [(res_adj, 'adjusted', 'blue'), (res_unadj, 'unadjusted', 'gray')]:
                    fig = create_plot(r, f"{name.title()}: {fname}", col)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    zf.writestr(f"{path}plot_{name}.png", buf.getvalue())
                    plt.close(fig)

                fig_c, ax = plt.subplots(figsize=(10,6))
                ax.plot(res_unadj['profile'], 'gray', linestyle='--', alpha=0.6, label='Unadjusted')
                ax.plot(res_adj['profile'], 'blue', linewidth=2, label='Adjusted')
                if res_adj['search_window']:
                    ax.axvspan(*res_adj['search_window'], color='gray', alpha=0.1)
                ax.legend()
                ax.set_title(f"Comparison: {fname}")
                buf = io.BytesIO()
                fig_c.savefig(buf, format='png')
                zf.writestr(f"{path}plot_comparison.png", buf.getvalue())
                plt.close(fig_c)

                summary.append({"Video_ID": vid, "Strip_ID": sid, 
                                "Adj_Ratio": res_adj['ratio'], "Unadj_Ratio": res_unadj['ratio']})
        
        if summary:
            df = pd.DataFrame(summary)
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
                df.pivot_table(index="Video_ID", columns="Strip_ID", values="Adj_Ratio").to_excel(writer, sheet_name='Adjusted')
                df.pivot_table(index="Video_ID", columns="Strip_ID", values="Unadj_Ratio").to_excel(writer, sheet_name='Unadjusted')
            zf.writestr("Summary_Analysis.xlsx", excel_buf.getvalue())

    st.success("Done!")
    st.download_button("📥 Download ZIP", zip_buf.getvalue(), "LFA_Results.zip", "application/zip")

if mode == "📂 Batch Upload":
    files = st.file_uploader("Upload Strips", accept_multiple_files=True, type=['jpg','png','tif'])
    if files and st.button("Analyze Batch"):
        imgs, names, vids, sids = [], [], [], []
        for f in files:
            imgs.append(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR))
            names.append(f.name)
            m = re.match(r"(\d+)_strip_(\d+)", f.name)
            vids.append(int(m.group(1)) if m else 0)
            sids.append(int(m.group(2)) if m else 0)
        process_and_download(imgs, names, vids, sids)

elif mode == "✂️ Single Photo":
    f = st.file_uploader("Upload Board Photo", type=['jpg','png'])
    if f:
        full_img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        h, w, _ = full_img.shape
        c1, c2 = st.columns([1, 3])
        top = c1.slider("Top Crop", 0, h, int(h*0.3))
        bot = c1.slider("Bottom Crop", 0, h, int(h*0.7))
        prev = full_img.copy()
        cv2.line(prev, (0, top), (w, top), (255,0,0), 5)
        cv2.line(prev, (0, bot), (w, bot), (255,0,0), 5)
        c2.image(prev, use_container_width=True)
        if bot > top and st.button("Extract & Analyze"):
            strips, locs, _ = detect_and_slice_strips(full_img, top, bot)
            st.write(f"Found {len(strips)} strips")
            process_and_download(strips, [f"Strip_{i+1}" for i in range(len(strips))], 
                                 [1]*len(strips), list(range(1, len(strips)+1)))
