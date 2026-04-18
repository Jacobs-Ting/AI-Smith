import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from scipy.optimize import minimize
import plotly.graph_objects as go
import tempfile
from plotly.subplots import make_subplots

# ==========================================
# 🌟 Streamlit 頁面與全域暗黑設定
# ==========================================
st.set_page_config(layout="wide", page_title="RF Matching Master - Web Edition")
plt.style.use('dark_background')

# 儀器級配色調色盤
BG_MAIN = '#1E1E1E'
FG_TEXT = '#CCCCCC'
C_START = '#1E90FF'  # 天際藍
C_TARGET = '#FF3333' # 警戒紅
C_FINAL = '#00FF00'  # 亮綠 (Pass)
C_IND = '#FFD700'    # 琥珀金 (L)
C_CAP = '#00BFFF'    # 深天空藍 (C)
C_TRACE = '#B266FF'  # 柔和紫 (Trace/Stub)
C_VSWR = '#CC7722'   # 暗橘 (VSWR圓)

# 針對 Tab2 (Freq Response) 的 Matplotlib 設定
plt.rcParams.update({
    'figure.facecolor': BG_MAIN, 'axes.facecolor': BG_MAIN, 
    'savefig.facecolor': BG_MAIN, 'axes.edgecolor': '#555555', 
    'grid.color': '#333333', 'text.color': FG_TEXT,
    'axes.labelcolor': FG_TEXT, 'xtick.color': FG_TEXT, 'ytick.color': FG_TEXT
})

# ==========================================
# 核心模組 1：虛擬 PLM 資料庫
# ==========================================
def get_plm_data():
    return [
        {"vendor": "Murata", "pn": "LQP03HQ1N0W02", "type": "L", "val": 1.0, "desc": "0201, ±0.05nH, High Q"},
        {"vendor": "Murata", "pn": "LQP03HQ1N2W02", "type": "L", "val": 1.2, "desc": "0201, ±0.05nH, High Q"},
        {"vendor": "Murata", "pn": "LQP03HQ5N6H02", "type": "L", "val": 5.6, "desc": "0201, ±3%, High Q"},
        {"vendor": "TDK",    "pn": "MLG0603P2N0S",  "type": "L", "val": 2.0, "desc": "0201, ±0.3nH, Multilayer"},
        {"vendor": "Murata", "pn": "GJM0335C1E0R5B", "type": "C", "val": 0.5, "desc": "0201, ±0.1pF, C0G"},
        {"vendor": "Murata", "pn": "GJM0335C1E1R0B", "type": "C", "val": 1.0, "desc": "0201, ±0.1pF, C0G"},
        {"vendor": "Samsung","pn": "CL03C1R5BA3GNN", "type": "C", "val": 1.5, "desc": "0201, ±0.1pF, C0G"}
    ]

# ==========================================
# Session State 初始化
# ==========================================
if 'netlist' not in st.session_state: st.session_state.netlist = []
if 'z_curr' not in st.session_state: st.session_state.z_curr = 50.0 + 0j
if 'gamma_points' not in st.session_state: st.session_state.gamma_points = [] 

# ==========================================
# 核心數學與物理模型
# ==========================================
def calc_microstrip(w, h, er, f):
    if w <= 0: w = 0.01
    if h <= 0: h = 0.01
    u = w / h
    e_eff = (er + 1) / 2 + ((er - 1) / 2) * (1 + 10/u)**(-0.555)
    z0 = (60 / np.sqrt(e_eff)) * np.log(8/u + u/4) if u <= 1 else (120 * np.pi) / (np.sqrt(e_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
    return z0, 2 * np.pi * f * np.sqrt(e_eff) / 3e8

def get_z_branch(f, val, esr, par, comp_type):
    w = 2 * np.pi * f
    if 'L' in comp_type: return 1 / (1/(esr + 1j * w * val * 1e-9) + 1j * w * par * 1e-12) if par > 0 else (esr + 1j * w * val * 1e-9)
    else: return esr + 1 / (1j * w * val * 1e-12) + 1j * w * par * 1e-9

def z2g(z): return (z - 50) / (z + 50)

# ==========================================
# UI 佈局區塊
# ==========================================
st.title("RF Matching Master - Web Edition")
st.markdown("___")

# --- 左側控制列 (Sidebar) ---
with st.sidebar:
    st.header("1. System & Load")
    fc = st.number_input("Center Freq (Hz)", value=5e9, format="%e")
    span = st.number_input("Span (Hz)", value=2e9, format="%e")
    colA, colB = st.columns(2)
    pcb_er = colA.number_input("Substrate Dk", value=4.4, step=0.1)
    pcb_h = colB.number_input("Height (mm)", value=0.1, step=0.01)
    colC, colD = st.columns(2)
    zl_real = colC.number_input("ZL Real (Ω)", value=25.0)
    zl_imag = colD.number_input("ZL Imag (Ω)", value=-60.0)
    
    st.markdown("___")

    st.header("2. Source & Target")
    colE, colF = st.columns(2)
    zs_real = colE.number_input("Zs Real (Ω)", value=30.0)
    zs_imag = colF.number_input("Zs Imag (Ω)", value=-10.0)
    conj_mode = st.checkbox("🔥 Conjugate Match (Target = Zs*)", value=False)
    show_vswr = st.checkbox("⭕ Target VSWR Circle", value=True)
    tgt_vswr = st.number_input("VSWR Value", value=1.4, step=0.1) if show_vswr else 1.5

    st.markdown("___")

    st.header("3. Smart Match & Optimization")
    col_p1, col_p2, col_p3 = st.columns(3)
    g_esr = col_p1.number_input("Global ESR", value=0.5, step=0.1)
    g_cp = col_p2.number_input("L Cp(pF)", value=0.05, step=0.01)
    g_ls = col_p3.number_input("C Ls(nH)", value=0.5, step=0.1)

    # 🌟 滿血版 Nodal Q 演算法
    if st.button("🚀 Run Smart Match (Early Stopping)", use_container_width=True, type="primary"):
        target_gamma = (tgt_vswr - 1) / (tgt_vswr + 1)
        sweep_grid = np.concatenate([np.arange(0.1, 3.0, 0.1), np.arange(3.0, 10.0, 0.5), np.arange(10.0, 31.0, 1.0)])
        
        zl_cmplx = complex(zl_real, zl_imag)
        zs_cmplx = complex(zs_real, zs_imag)
        zt = zs_cmplx.conjugate() if conj_mode else 50.0 + 0j
        
        def obj_smart(vals, topology):
            z_c = zl_cmplx; w = 2 * np.pi * fc
            for i, t in enumerate(topology):
                v = vals[i]
                if t == 'shn_C': z_c = 1 / (1/z_c + 1j*w*v*1e-12)
                elif t == 'ser_L': z_c = z_c + 1j*w*v*1e-9
                elif t == 'ser_C': z_c = z_c + 1/(1j*w*v*1e-12)
                elif t == 'shn_L': z_c = 1 / (1/z_c + 1/(1j*w*v*1e-9))
            return abs((z_c - zt) / (z_c + zt.conjugate()))

        single_topologies = [('ser_L',), ('ser_C',), ('shn_L',), ('shn_C',)]
        single_found = False
        for top in single_topologies:
            for v in sweep_grid:
                if obj_smart([v], top) <= target_gamma:
                    st.session_state.netlist = [{'type': f"{'Shunt' if 'shn' in top[0] else 'Series'} {'C' if 'C' in top[0] else 'L'}", 'comp': top[0][-1], 'mode': top[0][:3], 'val': v, 'esr': 0, 'par': 0}]
                    st.success(f"AI Found Single Component Match! Topology: {top}")
                    single_found = True
                    break
            if single_found: break

        if not single_found:
            double_topologies = [('shn_C', 'ser_L'), ('ser_L', 'shn_C'), ('ser_C', 'shn_L'), ('shn_L', 'ser_C'), ('ser_C', 'shn_C'), ('shn_L', 'ser_L')]
            valid_solutions = []

            for top in double_topologies:
                found_for_this_top = False
                for v1 in sweep_grid:
                    if found_for_this_top: break
                    for v2 in sweep_grid:
                        g = obj_smart([v1, v2], top)
                        if g <= target_gamma:
                            z_mid = zl_cmplx; w = 2 * np.pi * fc; t1 = top[0]
                            if t1 == 'shn_C': z_mid = 1 / (1/z_mid + 1j * w * v1 * 1e-12)
                            elif t1 == 'ser_L': z_mid = z_mid + 1j * w * v1 * 1e-9
                            elif t1 == 'ser_C': z_mid = z_mid + 1 / (1j * w * v1 * 1e-12)
                            elif t1 == 'shn_L': z_mid = 1 / (1/z_mid + 1 / (1j * w * v1 * 1e-9))

                            gamma_mid = abs((z_mid - 50) / (z_mid + 50))
                            valid_solutions.append({'topology': top, 'vals': [v1, v2], 'gamma': g, 'cost': gamma_mid})
                            found_for_this_top = True
                            break 

            if valid_solutions:
                best = min(valid_solutions, key=lambda x: x['cost'])
                st.session_state.netlist = [
                    {'type': f"{'Shunt' if 'shn' in best['topology'][0] else 'Series'} {'C' if 'C' in best['topology'][0] else 'L'}", 'comp': best['topology'][0][-1], 'mode': best['topology'][0][:3], 'val': best['vals'][0], 'esr': 0, 'par': 0},
                    {'type': f"{'Shunt' if 'shn' in best['topology'][1] else 'Series'} {'C' if 'C' in best['topology'][1] else 'L'}", 'comp': best['topology'][1][-1], 'mode': best['topology'][1][:3], 'val': best['vals'][1], 'esr': 0, 'par': 0}
                ]
                vswr = (1+best['gamma'])/(1-best['gamma'])
                st.success(f"AI Found Shortest Path (Nodal Q)! Topology: {best['topology']} | VSWR: {vswr:.3f}")
            else: 
                st.warning(f"Failed to find a match within VSWR {tgt_vswr} circle.")
        st.rerun()

    col_opt1, col_opt2 = st.columns(2)
    if col_opt1.button("👁️ Apply Parasitics", use_container_width=True):
        for it in st.session_state.netlist:
            if 'val' in it:
                it['esr'] = g_esr; it['par'] = g_cp if it['comp'] == 'L' else g_ls
        st.rerun()
        
    if col_opt2.button("✨ Run Optimization", use_container_width=True):
        for it in st.session_state.netlist:
            if 'val' in it: it['esr'] = g_esr; it['par'] = g_cp if it['comp'] == 'L' else g_ls
            
        def obj_opt(vals):
            z_c = complex(zl_real, zl_imag)
            zt = complex(zs_real, zs_imag).conjugate() if conj_mode else 50.0 + 0j
            v_idx = 0
            for item in st.session_state.netlist:
                if 'val' in item:
                    z_br = get_z_branch(fc, vals[v_idx], item['esr'], item['par'], item['comp'])
                    z_c = (z_c + z_br) if item['mode'] == 'ser' else (1 / (1/z_c + 1/z_br))
                    v_idx += 1
            return abs((z_c - zt)/(z_c + zt.conjugate()))
            
        opt_vals = [it['val'] for it in st.session_state.netlist if 'val' in it]
        if opt_vals:
            res = minimize(obj_opt, opt_vals, bounds=[(0.01, 100)]*len(opt_vals), method='Nelder-Mead')
            v_idx = 0
            for item in st.session_state.netlist:
                if 'val' in item: item['val'] = round(res.x[v_idx], 3); v_idx += 1
        st.rerun()

    st.markdown("___")

    st.header("4. Manual Add Component")
    comp_types = ["Series L", "Series C", "Shunt L", "Shunt C", "Trace", "Shunt Open Stub", "Shunt Short Stub", "Import .s2p"]
    ctype = st.selectbox("Component Type", comp_types)
    
    with st.expander("🛒 Select from Virtual PLM"):
        plm_db = get_plm_data()
        plm_options = [f"[{p['vendor']}] {p['pn']} - {p['val']}{'nH' if p['type']=='L' else 'pF'}" for p in plm_db]
        selected_plm_str = st.selectbox("Company Inventory", ["None"] + plm_options)

    if "s2p" in ctype:
        uploaded_file = st.file_uploader("Upload Touchstone file", type=["s2p", "s1p"])
    elif "Trace" in ctype or "Stub" in ctype:
        colW, colL = st.columns(2)
        v_w = colW.number_input("Width W (mm)", value=0.4, step=0.1)
        v_l = colL.number_input("Length L (mm)", value=5.0, step=0.1)
    else:
        colV, colE = st.columns(2)
        def_val = 2.0
        if selected_plm_str != "None": def_val = float(selected_plm_str.split("-")[1].replace("nH","").replace("pF","").strip())
        v_val = colV.number_input("Value (nH/pF)", value=def_val, step=0.1)
        v_esr = colE.number_input("ESR (Ω)", value=0.5, step=0.1)

    if st.button("⬇️ Add to Netlist", type="primary", use_container_width=True):
        info = {'type': ctype}
        if "s2p" in ctype and uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".s2p") as tmp:
                tmp.write(uploaded_file.getvalue())
                info['file'] = tmp.name
        elif "Trace" in ctype or "Stub" in ctype:
            info['w'] = v_w; info['l'] = v_l
        else:
            info['val'] = v_val; info['esr'] = v_esr; info['par'] = 0.0
            info['comp'] = 'L' if "L" in ctype else 'C'
            info['mode'] = 'ser' if "Series" in ctype else 'shn'
            if selected_plm_str != "None": info['plm'] = selected_plm_str.split("]")[1].split("-")[0].strip()
        st.session_state.netlist.append(info)
        st.rerun()

    st.markdown("___")

    st.subheader("Netlist")
    for i, item in enumerate(st.session_state.netlist):
        if 'val' in item: st.markdown(f"`{item['type']}`: {item['val']} | ESR:{item['esr']} | Par:{item['par']}")
        elif 'w' in item: st.markdown(f"`{item['type']}`: W={item['w']}mm, L={item['l']}mm")
        else: st.markdown(f"`{item['type']}`: {item.get('file', 'File')}")
        
    col_d1, col_d2 = st.columns(2)
    if col_d1.button("🗑️ Delete Last", use_container_width=True) and st.session_state.netlist:
        st.session_state.netlist.pop(); st.rerun()
    if col_d2.button("❌ Clear All", use_container_width=True):
        st.session_state.netlist = []; st.session_state.gamma_points = []; st.rerun()

# 獨立定義起點與終點給圖表使用
zl = complex(zl_real, zl_imag)
zs = complex(zs_real, zs_imag)
z_target = zs.conjugate() if conj_mode else 50.0 + 0j

# --- 主畫面區 ---
tab1, tab2, tab3 = st.tabs(["🎯 Smith Chart (Plotly)", "📈 Freq Response (Gain/Delay)", "🎲 Yield Analysis"])

with tab1:
    # 幫助計算即時 VSWR 的小函式
    def calc_vswr(z_val):
        g_val = (z_val - z_target) / (z_val + z_target.conjugate())
        g_mag = min(abs(g_val), 0.999) 
        return (1 + g_mag) / (1 - g_mag)

    fig = go.Figure()

    # 1. 畫 VSWR 圓
    if show_vswr and tgt_vswr > 1.0:
        rho = (tgt_vswr - 1) / (tgt_vswr + 1)
        theta = np.linspace(0, 2*np.pi, 150)
        gamma_circle = rho * np.exp(1j * theta)
        z_circle = (z_target + gamma_circle * z_target.conjugate()) / (1 - gamma_circle)
        fig.add_trace(go.Scattersmith(
            real=np.real(z_circle)/50.0, imag=np.imag(z_circle)/50.0,
            mode='lines', line=dict(color=C_VSWR, width=2, dash='dash'),
            name=f'VSWR={tgt_vswr}', hoverinfo='skip'
        ))

    # 2. 畫 Target X
    fig.add_trace(go.Scattersmith(
        real=[np.real(z_target)/50.0], imag=[np.imag(z_target)/50.0],
        mode='markers', marker=dict(color=C_TARGET, symbol='x', size=12),
        name="Target" if not conj_mode else "Target: Zs*",
        hovertemplate=f"<b>Target</b><br>Z: {np.real(z_target):.1f} {'+' if np.imag(z_target)>=0 else '-'} {abs(np.imag(z_target)):.1f}j Ω<extra></extra>"
    ))

    # 3. 畫 Start ZL
    z_curr = zl
    fig.add_trace(go.Scattersmith(
        real=[np.real(z_curr)/50.0], imag=[np.imag(z_curr)/50.0],
        mode='markers', marker=dict(color=C_START, size=8),
        name='Start ZL',
        hovertemplate=f"<b>Start ZL</b><br>Z: {np.real(z_curr):.1f} {'+' if np.imag(z_curr)>=0 else '-'} {abs(np.imag(z_curr)):.1f}j Ω<br>VSWR: {calc_vswr(z_curr):.2f}<extra></extra>"
    ))

    # 4. 畫動態匹配軌跡
    for item in st.session_state.netlist:
        z_start_node = z_curr
        path_g = []

        if "s2p" in item['type']:
            try:
                n = rf.Network(item['file'])
                n.interpolate_self(fc)
                if n.number_of_ports == 1: z_curr = n.z[0][0][0]
                else:
                    gl = z2g(z_start_node); s = n.s[0]
                    gin = s[0,0] + (s[0,1]*s[1,0]*gl)/(1-s[1,1]*gl)
                    z_curr = 50.0 * (1+gin)/(1-gin)
                path_g.extend([z2g(z_start_node), z2g(z_curr)])
                t_color = '#AAAAAA'; l_dash = 'dot'; l_width = 2
            except: pass
        elif "Trace" in item['type'] or "Stub" in item['type']:
            steps = 50
            z0_t, beta_t = calc_microstrip(item['w'], pcb_h, pcb_er, fc)
            if "Stub" in item['type']:
                is_open = "Open" in item['type']; y_s = 1.0/z_start_node
                for k in range(steps+1):
                    bl = beta_t * ((item['l'] * k / steps + 1e-6)/1000.0); tan_bl = np.tan(bl)
                    if not is_open and abs(tan_bl) < 1e-6: tan_bl = 1e-6
                    path_g.append(z2g(1.0/(y_s + (1j*(1/z0_t)*tan_bl if is_open else -1j*(1/z0_t)/tan_bl))))
                z_curr = 1.0/(y_s + (1j*(1/z0_t)*np.tan(beta_t*(item['l']/1000.0)) if is_open else -1j*(1/z0_t)/np.tan(beta_t*(item['l']/1000.0))))
            else:
                for k in range(steps+1):
                    t = np.tan(beta_t * ((item['l'] * k / steps)/1000.0)); den = z0_t + 1j*z_start_node*t 
                    if abs(den) < 1e-9: den = 1e-9
                    path_g.append(z2g(z0_t * (z_start_node + 1j*z0_t*t) / den))
                t_f = np.tan(beta_t * (item['l']/1000.0)); den_f = z0_t + 1j*z_start_node*t_f
                z_curr = z0_t * (z_start_node + 1j*z0_t*t_f) / den_f
            t_color = C_TRACE; l_dash = 'solid'; l_width = 3
        else:
            steps = 30
            z_actual = get_z_branch(fc, item['val'], item['esr'], item['par'], item['type'])
            if item['mode'] == 'ser':
                for k in range(steps+1): path_g.append(z2g(z_start_node + z_actual*(k/steps)))
                z_curr += z_actual
            else:
                y_s, y_c = 1.0/z_start_node, 1.0/z_actual
                for k in range(steps+1): path_g.append(z2g(1.0/(y_s + y_c*(k/steps))))
                z_curr = 1.0/(y_s + y_c)
            t_color = C_IND if item['comp'] == 'L' else C_CAP
            l_dash = 'solid'; l_width = 3

        if path_g:
            # 將 Gamma 轉回 Normalized Z 餵給 Plotly，同時生成 Hover 文字
            path_g = np.array(path_g)
            z_norm_path = (1 + path_g) / (1 - path_g + 1e-15)
            
            hover_texts = []
            for zn in z_norm_path:
                zr, zi = np.real(zn)*50.0, np.imag(zn)*50.0
                hover_texts.append(f"Z: {zr:.1f} {'+' if zi>=0 else '-'} {abs(zi):.1f}j Ω<br>VSWR: {calc_vswr(zr + 1j*zi):.2f}")

            fig.add_trace(go.Scattersmith(
                real=np.real(z_norm_path), imag=np.imag(z_norm_path),
                mode='lines', line=dict(color=t_color, width=l_width, dash=l_dash),
                name=item['type'], text=hover_texts, hoverinfo='text+name'
            ))

    st.session_state.z_curr = z_curr 

    # 5. 畫 Final Z
    fig.add_trace(go.Scattersmith(
        real=[np.real(z_curr)/50.0], imag=[np.imag(z_curr)/50.0],
        mode='markers', marker=dict(color=C_FINAL, size=10),
        name='Final Z',
        hovertemplate=f"<b>Final Z</b><br>Z: {np.real(z_curr):.1f} {'+' if np.imag(z_curr)>=0 else '-'} {abs(np.imag(z_curr)):.1f}j Ω<br>VSWR: {calc_vswr(z_curr):.2f}<extra></extra>"
    ))
    
    # 6. 畫蒙地卡羅
    if st.session_state.gamma_points:
        mc_g = np.array(st.session_state.gamma_points)
        mc_zn = (1 + mc_g) / (1 - mc_g + 1e-15)
        fig.add_trace(go.Scattersmith(
            real=np.real(mc_zn), imag=np.imag(mc_zn),
            mode='markers', marker=dict(color=C_FINAL, size=4, opacity=0.4),
            name='Yield (Monte Carlo)', hoverinfo='skip'
        ))

    # 🌟 設定 Plotly 專業儀器外觀 (修復 imaginaryaxis)
    final_vswr = calc_vswr(z_curr)
    fig.update_layout(
        smith=dict(
            bgcolor=BG_MAIN,
            realaxis=dict(gridcolor='#444444', linecolor='#555555'),
            imaginaryaxis=dict(gridcolor='#444444', linecolor='#555555')
        ),
        paper_bgcolor=BG_MAIN, plot_bgcolor=BG_MAIN,
        font=dict(color=FG_TEXT, size=14),
        margin=dict(l=20, r=20, t=80, b=20),
        height=800,
        title=dict(text=f"Final VSWR: {final_vswr:.3f}", font=dict(size=24), x=0.5, xanchor='center'),
        legend=dict(bgcolor='#252526', bordercolor='#555555', borderwidth=1, x=1.05, y=1)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not st.session_state.netlist:
        st.info("請先加入匹配元件以觀看頻率響應。")
    else:
        freqs = np.linspace(max(1, fc - span/2), fc + span/2, 201)
        gt_db_list, phases = [], []
        for f in freqs:
            abcd = np.matrix([[1,0],[0,1]], dtype=complex)
            for item in reversed(st.session_state.netlist):
                if "s2p" in item['type']: continue 
                elif "Trace" in item['type'] or "Stub" in item['type']:
                    z0_t, beta_t = calc_microstrip(item['w'], pcb_h, pcb_er, f); bl = beta_t * (item['l']/1000)
                    if 'Trace' in item['type']: m = np.matrix([[np.cos(bl), 1j*z0_t*np.sin(bl)], [1j/z0_t*np.sin(bl), np.cos(bl)]])
                    else: m = np.matrix([[1, 0], [1j/z0_t*np.tan(bl) if 'Open' in item['type'] else -1j/z0_t/np.tan(bl), 1]])
                else:
                    z_br = get_z_branch(f, item['val'], item['esr'], item['par'], item['type'])
                    m = np.matrix([[1, z_br], [0, 1]]) if item['mode'] == 'ser' else np.matrix([[1, 0], [1/z_br, 1]])
                abcd = abcd * m
            A, B, C, D = abcd[0,0], abcd[0,1], abcd[1,0], abcd[1,1]
            den = A*zl + B + C*zs*zl + D*zs
            gt = (4 * zs.real * zl.real) / (abs(den)**2 + 1e-12)
            gt_db_list.append(10 * np.log10(gt + 1e-12)); phases.append(np.angle(zl / den))
        
        gd = -np.diff(np.unwrap(phases)) / (2*np.pi*(freqs[1]-freqs[0])) * 1e9
        
        # 🌟 升級為 Plotly 雙軸互動式圖表
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Transducer Power Gain S21 (dB)", "Group Delay (ns)"),
                             vertical_spacing=0.1)

        # 繪製 S21 (包含 Hover 互動)
        fig2.add_trace(go.Scatter(x=freqs/1e9, y=gt_db_list, mode='lines',
                                  line=dict(color=C_CAP, width=2.5), name="S21",
                                  hovertemplate="Freq: %{x:.3f} GHz<br>S21: %{y:.2f} dB<extra></extra>"),
                       row=1, col=1)

        # 繪製 Group Delay (包含 Hover 互動)
        fig2.add_trace(go.Scatter(x=freqs[1:]/1e9, y=gd, mode='lines',
                                  line=dict(color=C_FINAL, width=2.5), name="GD",
                                  hovertemplate="Freq: %{x:.3f} GHz<br>Delay: %{y:.2f} ns<extra></extra>"),
                       row=2, col=1)

        # 標示中心頻率虛線
        fig2.add_vline(x=fc/1e9, line_width=2, line_dash="dash", line_color=C_TARGET)

        # 專業暗黑模式版面設定
        fig2.update_layout(
            paper_bgcolor=BG_MAIN, plot_bgcolor=BG_MAIN,
            font=dict(color=FG_TEXT, size=14),
            height=800, showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig2.update_xaxes(gridcolor='#444444', zerolinecolor='#555555')
        fig2.update_yaxes(gridcolor='#444444', zerolinecolor='#555555')
        fig2.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)

        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### 🎲 蒙地卡羅良率分析 (Monte Carlo Yield Analysis)")
    mc_tol = st.number_input("Component Tolerance (%)", value=5.0, step=1.0)
    if st.button("🚀 執行良率分析 (生成 300 個分佈點)", type="primary"):
        tol = mc_tol / 100.0 / 3.0
        pts = []
        for _ in range(300):
            z_c = zl
            for item in st.session_state.netlist:
                if 'val' in item:
                    r_val = max(np.random.normal(item['val'], item['val']*tol), 0.01)
                    z_br = get_z_branch(fc, r_val, item['esr'], item['par'], item['comp'])
                    z_c = (z_c + z_br) if item['mode'] == 'ser' else (1 / (1/z_c + 1/z_br))
            pts.append(z2g(z_c))
        st.session_state.gamma_points = pts
        st.rerun() 
    st.info("💡 執行完成後，請切換回 **『🎯 Smith Chart (Plotly)』** 分頁，即可在圖表上看到良率散佈點。")