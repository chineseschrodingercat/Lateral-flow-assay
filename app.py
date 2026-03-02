# LFA Automatic Analyzer
# Copyright (C) 2026 Minhao Liu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, savgol_filter
import io
import zipfile
import re

FIXED_CROP_MARGIN = 0.1
SMOOTH_WINDOW = 15
MAX_PEAK_WIDTH = 30

# --- Search Parameters ---
T_DIST_NEAR = 30
T_DIST_FAR = 100
BASELINE_METHOD = 'lower'

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

def create_plot(res, title, color='blue'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['profile'], color=color, linewidth=2)
    
    global_base = None
    if len(res['c_base_y']) > 0:
        global_base = res['c_base_y'][0]
    elif len(res['t_base_y']) > 0:
        global_base = res['t_base_y'][0]
        
    if global_base is not None:
        ax.axhline(y=global_base, color='r', linestyle='--', linewidth=2, label='Global Baseline')

    for p_type, col in [('t', 'green'), ('c', 'orange')]:
        if len(res[f'{p_type}_base_x']) > 0:
            bx = res[f'{p_type}_base_x']
            ax.fill_between(range(bx[0], bx[1]+1), 
                            res['profile'][bx[0]:bx[1]+1], 
                            global_base, 
                            alpha=0.3, color=col)
            if not np.isnan(res[f'{p_type}_pos']):
                ax.text(res[f'{p_type}_pos'], res['profile'][int(res[f'{p_type}_pos'])]+5, 
                        f"{p_type.upper()}:{res[f'{p_type}_area']:.1f}", color='red', ha='center')

    if res['search_window']:
        s, e = res['search_window']
        ax.axvspan(s, e, color='gray', alpha=0.15, label='Search Zone')

    ax.set_title(f"{title}\nR={res['ratio']:.4f}")
    plt.tight_layout()
    return fig

def get_peak_bounds(peak_pos, left_idx, right_idx, height):
    x_s = int(left_idx)
    x_e = int(right_idx)
    
    current_width = x_e - x_s
    
    if current_width > MAX_PEAK_WIDTH:
        half_max = MAX_PEAK_WIDTH // 2
        x_s = max(0, int(peak_pos - half_max))
        x_e = min(height - 1, int(peak_pos + half_max))
    else:
        x_s = max(0, x_s)
        x_e = min(height - 1, x_e)
        
    return x_s, x_e

def analyze_single_strip(img_array, t_dist_near, t_dist_far, dynamic_prominence, crop_margin=FIXED_CROP_MARGIN):
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
    
    peaks, properties = find_peaks(smooth_profile, prominence=dynamic_prominence, distance=10, width=2)
    
    res = {
        "profile": smooth_profile, "ratio": 0.0,
        "t_area": 0.0, "c_area": 0.0,
        "t_pos": np.nan, "c_pos": np.nan,
        "t_base_x": [], "t_base_y": [], "c_base_x": [], "c_base_y": [],
        "search_window": []
    }
    
    widths_results = peak_widths(smooth_profile, peaks, rel_height=0.95)
    height = len(smooth_profile)
    split_point = int(height * 0.5)
    
    c_bounds = None
    t_bounds = None
    
    c_candidates = [p for p in peaks if p > split_point]
    if c_candidates:
        c_indices = [np.where(peaks == p)[0][0] for p in c_candidates]
        best_c_idx = c_indices[np.argmax(properties['prominences'][c_indices])]
        res['c_pos'] = peaks[best_c_idx]
        
        c_left = widths_results[2][best_c_idx]
        c_right = widths_results[3][best_c_idx]
        c_bounds = get_peak_bounds(res['c_pos'], c_left, c_right, height)

    if not np.isnan(res['c_pos']):
        search_end = int(res['c_pos'] - t_dist_near)
        search_start = max(0, int(res['c_pos'] - t_dist_far))
        res['search_window'] = [search_start, search_end]
        
        t_candidates = [p for p in peaks if search_start <= p < search_end]
        if t_candidates:
            t_indices = [np.where(peaks == p)[0][0] for p in t_candidates]
            best_t_idx = t_indices[np.argmax(properties['prominences'][t_indices])]
            res['t_pos'] = peaks[best_t_idx]
            
            t_left = widths_results[2][best_t_idx]
            t_right = widths_results[3][best_t_idx]
            t_bounds = get_peak_bounds(res['t_pos'], t_left, t_right, height)

    y_vals_for_baseline = []
    if c_bounds and c_bounds[1] > c_bounds[0]:
        y_vals_for_baseline.extend([smooth_profile[c_bounds[0]], smooth_profile[c_bounds[1]]])
    if t_bounds and t_bounds[1] > t_bounds[0]:
        y_vals_for_baseline.extend([smooth_profile[t_bounds[0]], smooth_profile[t_bounds[1]]])

    if y_vals_for_baseline:
        if BASELINE_METHOD == 'lower':
            global_base_val = min(y_vals_for_baseline)
        elif BASELINE_METHOD == 'higher':
            global_base_val = max(y_vals_for_baseline)
        else:
            global_base_val = sum(y_vals_for_baseline) / len(y_vals_for_baseline)
    else:
        global_base_val = 0

    if c_bounds and c_bounds[1] > c_bounds[0]:
        x_s, x_e = c_bounds
        curve_segment = np.maximum(smooth_profile[x_s:x_e+1] - global_base_val, 0)
        res['c_area'] = np.sum(curve_segment)
        res['c_base_x'] = [x_s, x_e]
        res['c_base_y'] = [global_base_val, global_base_val]

    if t_bounds and t_bounds[1] > t_bounds[0]:
        x_s, x_e = t_bounds
        curve_segment = np.maximum(smooth_profile[x_s:x_e+1] - global_base_val, 0)
        res['t_area'] = np.sum(curve_segment)
        res['t_base_x'] = [x_s, x_e]
        res['t_base_y'] = [global_base_val, global_base_val]

    res['ratio'] = res['t_area'] / res['c_area'] if res['c_area'] > 0 else 0
    return res

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

# --- UI FOR STRIP TYPE SELECTION ---
strip_type = st.sidebar.radio(
    "Select Strip Type",
    ("Traditional (e.g., hCG)", "Competitive (e.g., Xylazine)"),
    help="Traditional uses higher sensitivity to find faint lines. Competitive uses lower sensitivity to ignore noise."
)

if strip_type == "Traditional (e.g., hCG)":
    current_prominence = 0.5
else:
    current_prominence = 1.8

mode = st.sidebar.radio("Input Mode", ["📂 Batch Upload", "✂️ Single Photo"])

# --- NEW FUNCTION FOR BATCH PROCESSING (Returns Data to Session State) ---
def process_batch_data(image_list, filenames, video_ids, strip_ids, active_prominence):
    zip_buf = io.BytesIO()
    summary = []
    individual_data = [] # Stores data for the UI dropdown
    prog = st.progress(0)
    
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, (img, fname, vid, sid) in enumerate(zip(image_list, filenames, video_ids, strip_ids)):
            prog.progress((i+1)/len(image_list))
            
            res_adj = analyze_single_strip(img, T_DIST_NEAR, T_DIST_FAR, active_prominence, crop_margin=FIXED_CROP_MARGIN)
            res_unadj = analyze_single_strip(img, T_DIST_NEAR, T_DIST_FAR, active_prominence, crop_margin=0.0)
            
            if res_adj and res_unadj:
                path = f"{vid}/strip_{sid}/"
                adj_plot_buf = None
                
                for r, name, col in [(res_adj, 'adjusted', 'blue'), (res_unadj, 'unadjusted', 'gray')]:
                    fig = create_plot(r, f"{name.title()}: {fname}", col)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    img_bytes = buf.getvalue()
                    zf.writestr(f"{path}plot_{name}.png", img_bytes)
                    if name == 'adjusted':
                        adj_plot_buf = img_bytes
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
                
                # Save into array for the interactive UI dropdown
                individual_data.append({
                    "filename": fname,
                    "adj_ratio": res_adj['ratio'],
                    "adj_plot_buf": adj_plot_buf
                })
        
        if summary:
            df = pd.DataFrame(summary)
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
                df.pivot_table(index="Video_ID", columns="Strip_ID", values="Adj_Ratio").to_excel(writer, sheet_name='Adjusted')
                df.pivot_table(index="Video_ID", columns="Strip_ID", values="Unadj_Ratio").to_excel(writer, sheet_name='Unadjusted')
            zf.writestr("Summary_Analysis.xlsx", excel_buf.getvalue())

    prog.empty()
    return zip_buf, individual_data


# --- ORIGINAL FUNCTION FOR SINGLE PHOTO MODE ---
def process_and_download(image_list, filenames, video_ids, strip_ids, active_prominence):
    zip_buf, _ = process_batch_data(image_list, filenames, video_ids, strip_ids, active_prominence)
    st.success("Done!")
    st.download_button("📥 Download ZIP", zip_buf.getvalue(), "LFA_Results.zip", "application/zip")


# --- UI LOGIC ---
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
        
        # Process and store results in Streamlit Session State
        zip_buf, individual_data = process_batch_data(imgs, names, vids, sids, current_prominence)
        
        st.session_state['batch_zip'] = zip_buf.getvalue()
        st.session_state['batch_data'] = individual_data
        st.session_state['batch_names'] = names
        st.success("Batch Analysis Complete! Select results from the sidebar.")

    # If processing is complete, display the dynamic UI
    if 'batch_data' in st.session_state and 'batch_names' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Browse Results")
        
        # 1. Drop-down Menu Search
        selected_file = st.sidebar.selectbox("Select an image to view:", st.session_state['batch_names'])
        
        # Retrieve the selected item
        idx = st.session_state['batch_names'].index(selected_file)
        data = st.session_state['batch_data'][idx]
        
        st.markdown("---")
        # 2. Big Text Display for T/C Ratio
        st.markdown(f"<h3 style='text-align: center;'>Results for: {selected_file}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #E03C31; font-size: 3.5rem; margin-bottom: 20px;'>T/C Ratio: {data['adj_ratio']:.4f}</h1>", unsafe_allow_html=True)
        
        # 3. Display the Adjusted AUC Graph
        col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
        with col_img2:
            st.image(data['adj_plot_buf'], caption=f"Adjusted AUC Profile: {selected_file}", use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4. Independent Download Options
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="🖼️ Download Current Plot", 
                data=data['adj_plot_buf'], 
                file_name=f"adjusted_{selected_file}.png", 
                mime="image/png",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📥 Download ALL (ZIP & Excel Summary)", 
                data=st.session_state['batch_zip'], 
                file_name="LFA_Batch_Results.zip", 
                mime="application/zip",
                use_container_width=True
            )

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
                                 [1]*len(strips), list(range(1, len(strips)+1)), current_prominence)

