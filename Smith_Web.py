import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile

# ==========================================
# 🌟 Streamlit 頁面與全域暗黑設定
# ==========================================
st.set_page_config(layout="wide", page_title="RF Matching Master - Web Edition v8.11")
plt.style.use('dark_background')

BG_MAIN = '#1E1E1E'
FG_TEXT = '#CCCCCC'
C_START = '#1E90FF'  
C_TARGET = '#FF3333' 
C_FINAL = '#00FF00'  
C_IND = '#FFD700'    
C_CAP = '#00BFFF'    
C_TRACE = '#B266FF'  # 傳輸線專屬：柔和紫
C_VSWR = '#CC7722'   
C_BW = '#FFFFFF'     

def get_plm_data():
    return [
        {"vendor": "Murata", "pn": "LQP03HQ1N0W02", "type": "L", "val": 1.0, "desc": "0201, High Q"},
        {"vendor": "Murata", "pn": "LQP03HQ5N6H02", "type": "L", "val": 5.6, "desc": "0201, High Q"},
        {"vendor": "TDK",    "pn": "MLG0603P2N0S",  "type": "L", "val": 2.0, "desc": "0201, Multilayer"},
        {"vendor": "Murata", "pn": "GJM0335C1E1R0B", "type": "C", "val": 1.0, "desc": "0201, C0G"},
        {"vendor": "Samsung","pn": "CL03C1R5BA3GNN", "type": "C", "val": 1.5, "desc": "0201, C0G"}
    ]

if 'netlist' not in st.session_state: st.session_state.netlist = []
if 'gamma_points' not in st.session_state: st.session_state.gamma_points = [] 

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

with st.sidebar:
    st.header("1. System & Load")
    fc = st.number_input("Center Freq (Hz)", value=10e9, format="%e")
    span = st.number_input("Span (Hz)", value=1e9, format="%e")
    colA, colB = st.columns(2)
    pcb_er = colA.number_input("Substrate Dk", value=4.4, step=0.1)
    pcb_h = colB.number_input("Height (mm)", value=0.1, step=0.01)
    colC, colD = st.columns(2)
    zl_real = colC.number_input("ZL Real (Ω)", value=10.0)
    zl_imag = colD.number_input("ZL Imag (Ω)", value=90.0)
    
    st.markdown("___")
    st.header("2. Source & Target")
    colE, colF = st.columns(2)
    zs_real = colE.number_input("Zs Real (Ω)", value=50.0)
    zs_imag = colF.number_input("Zs Imag (Ω)", value=0.0)
    conj_mode = st.checkbox("🔥 Conjugate Match (Target = Zs*)", value=False)
    show_vswr = st.checkbox("⭕ Target VSWR Circle", value=True)
    tgt_vswr = st.number_input("VSWR Value", value=1.2, step=0.1) if show_vswr else 1.5

    st.markdown("___")
    st.header("3. Smart Match & Optimization")
    col_p1, col_p2, col_p3 = st.columns(3)
    g_esr = col_p1.number_input("ESR", value=0.0, step=0.1)
    g_cp = col_p2.number_input("L Cp(pF)", value=0.0, step=0.01)
    g_ls = col_p3.number_input("C Ls(nH)", value=0.0, step=0.1)

    if st.button("🚀 Run Smart Match (Shortest Path)", use_container_width=True, type="primary"):
        target_gamma = (tgt_vswr - 1) / (tgt_vswr + 1)
        sweep_grid = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 3.0, 0.1), np.arange(3.0, 10.0, 0.5), np.arange(10.0, 31.0, 1.0)])
        zl_cmplx = complex(zl_real, zl_imag)
        zt = complex(zs_real, zs_imag).conjugate() if conj_mode else 50.0 + 0j
        
        def obj_smart(vals, topology):
            z_c = zl_cmplx; w = 2 * np.pi * fc
            for i, t in enumerate(topology):
                v = vals[i]
                if t == 'shn_C': z_c = 1 / (1/z_c + 1j*w*v*1e-12)
                elif t == 'ser_L': z_c = z_c + 1j*w*v*1e-9
                elif t == 'ser_C': z_c = z_c + 1/(1j*w*v*1e-12)
                elif t == 'shn_L': z_c = 1 / (1/z_c + 1/(1j*w*v*1e-9))
            return abs((z_c - zt) / (z_c + zt.conjugate()))

        double_topologies = [
            ('shn_C', 'ser_L'), ('ser_L', 'shn_C'), 
            ('ser_C', 'shn_L'), ('shn_L', 'ser_C'),
            ('shn_C', 'ser_C'), ('ser_C', 'shn_C'), 
            ('shn_L', 'ser_L'), ('ser_L', 'shn_L')
        ]
        
        valid_solutions = []
        for top in double_topologies:
            for v1 in sweep_grid:
                for v2 in sweep_grid:
                    g = obj_smart([v1, v2], top)
                    if g <= target_gamma:
                        z_mid = zl_cmplx; w = 2 * np.pi * fc
                        if top[0] == 'shn_C': z_mid = 1 / (1/z_mid + 1j * w * v1 * 1e-12)
                        elif top[0] == 'ser_L': z_mid = z_mid + 1j * w * v1 * 1e-9
                        elif top[0] == 'ser_C': z_mid = z_mid + 1 / (1j * w * v1 * 1e-12)
                        elif top[0] == 'shn_L': z_mid = 1 / (1/z_mid + 1 / (1j * w * v1 * 1e-9))
                        
                        g_start = z2g(zl_cmplx); g_mid = z2g(z_mid); g_target = z2g(zt)
                        path_length = abs(g_mid - g_start) + abs(g_target - g_mid)
                        valid_solutions.append({'topology': top, 'vals': [v1, v2], 'gamma': g, 'cost': path_length})
                        break
                if any(s['topology'] == top for s in valid_solutions): break
                
        if valid_solutions:
            best = min(valid_solutions, key=lambda x: x['cost'])
            st.session_state.netlist = [
                {'type': f"{'Shunt' if 'shn' in best['topology'][0] else 'Series'} {best['topology'][0][-1]}", 'comp': best['topology'][0][-1], 'mode': best['topology'][0][:3], 'val': best['vals'][0], 'esr': 0.0, 'par': 0.0},
                {'type': f"{'Shunt' if 'shn' in best['topology'][1] else 'Series'} {best['topology'][1][-1]}", 'comp': best['topology'][1][-1], 'mode': best['topology'][1][:3], 'val': best['vals'][1], 'esr': 0.0, 'par': 0.0}
            ]
        st.rerun()

    st.markdown("##### 🔧 Fine-Tuning")
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("👁️ Apply Parasitics", use_container_width=True):
        new_netlist = []
        for it in st.session_state.netlist:
            new_it = it.copy()
            if 'val' in new_it:
                new_it['esr'] = g_esr; new_it['par'] = g_cp if new_it['comp'] == 'L' else g_ls
            new_netlist.append(new_it)
        st.session_state.netlist = new_netlist
        st.rerun()
        
    if col_btn2.button("✨ Optimize Values", use_container_width=True):
        new_netlist = [it.copy() for it in st.session_state.netlist]
        for it in new_netlist:
            if 'val' in it: it['esr'] = g_esr; it['par'] = g_cp if it['comp'] == 'L' else g_ls
            
        def obj_opt(vals):
            z_c = complex(zl_real, zl_imag)
            zt_opt = complex(zs_real, zs_imag).conjugate() if conj_mode else 50.0 + 0j
            v_idx = 0
            for item in new_netlist:
                if 'val' in item:
                    z_br = get_z_branch(fc, vals[v_idx], item['esr'], item['par'], item['comp'])
                    z_c = (z_c + z_br) if item['mode'] == 'ser' else (1 / (1/z_c + 1/z_br))
                    v_idx += 1
                elif 'w' in item: # 讓微帶線在優化中能正確傳遞阻抗
                    z0_t, beta_t = calc_microstrip(item['w'], pcb_h, pcb_er, fc)
                    bl = beta_t * (item['l']/1000.0)
                    if 'Trace' in item['type']:
                        t = np.tan(bl)
                        den = z0_t + 1j*z_c*t
                        z_c = z0_t * (z_c + 1j*z0_t*t) / den if abs(den) > 1e-9 else z_c
                    else:
                        is_open = "Open" in item['type']
                        y_stub = 1j*(1/z0_t)*np.tan(bl) if is_open else -1j*(1/z0_t)/np.tan(bl)
                        z_c = 1.0/(1.0/z_c + y_stub)
            return abs((z_c - zt_opt)/(z_c + zt_opt.conjugate()))
            
        opt_vals = [it['val'] for it in new_netlist if 'val' in it]
        if opt_vals:
            res = minimize(obj_opt, opt_vals, bounds=[(0.01, 100)]*len(opt_vals), method='Nelder-Mead')
            v_idx = 0
            for item in new_netlist:
                if 'val' in item: item['val'] = round(res.x[v_idx], 3); v_idx += 1
        st.session_state.netlist = new_netlist
        st.rerun()

    st.markdown("___")
    st.header("4. Manual Add Component")
    
    # 🌟 解封所有微波傳輸線與殘樁選項
    comp_types = ["Series L", "Series C", "Shunt L", "Shunt C", "Trace", "Shunt Open Stub", "Shunt Short Stub"]
    ctype = st.selectbox("Component Type", comp_types)
    
    # 🌟 動態 UI：選擇傳輸線時顯示 W/L 參數
    v_val, v_w, v_l = 1.0, 0.4, 5.0
    if "Trace" in ctype or "Stub" in ctype:
        col_w, col_l = st.columns(2)
        v_w = col_w.number_input("Width W (mm)", value=0.4, step=0.1)
        v_l = col_l.number_input("Length L (mm)", value=5.0, step=0.1)
    else:
        v_val = st.number_input("Value", value=1.0, step=0.1)

    if st.button("⬇️ Add to Netlist", use_container_width=True):
        if "Trace" in ctype or "Stub" in ctype:
            new_item = {'type': ctype, 'w': v_w, 'l': v_l}
        else:
            new_item = {'type': ctype, 'comp': ctype[-1], 'mode': ctype[:3].lower(), 'val': v_val, 'esr': g_esr, 'par': g_cp if ctype[-1]=='L' else g_ls}
        st.session_state.netlist = st.session_state.netlist + [new_item] 
        st.rerun()

    st.header("Netlist")
    for i, item in enumerate(st.session_state.netlist):
        if 'val' in item: st.markdown(f"{i+1}. `{item['type']}`: {item['val']:.3f} | ESR: {item.get('esr', 0)}Ω | Par: {item.get('par', 0)}{'pF' if item.get('comp')=='L' else 'nH'}")
        elif 'w' in item: st.markdown(f"{i+1}. `{item['type']}`: W={item['w']:.2f}mm, L={item['l']:.2f}mm")
        else: st.markdown(f"{i+1}. `{item['type']}`: {item.get('file', 'File')}")
        
    col_d1, col_d2 = st.columns(2)
    if col_d1.button("🗑️ Delete Last", use_container_width=True) and st.session_state.netlist:
        st.session_state.netlist = st.session_state.netlist[:-1]; st.rerun()
    if col_d2.button("❌ Clear All", use_container_width=True):
        st.session_state.netlist = []; st.session_state.gamma_points = []; st.rerun()

zl = complex(zl_real, zl_imag)
zt = complex(zs_real, zs_imag).conjugate() if conj_mode else 50.0 + 0j

tab1, tab2, tab3 = st.tabs(["🎯 Smith Chart (BW & Q)", "📈 Bandwidth Analysis (S21/GD)", "🎲 Yield Analysis"])

with tab1:
    def calc_metrics(z_val):
        g_val = (z_val - zt) / (z_val + zt.conjugate())
        vswr = (1 + abs(g_val)) / (1 - min(abs(g_val), 0.999))
        q_factor = abs(np.imag(z_val)) / np.real(z_val) if np.real(z_val) > 1e-6 else 999
        return vswr, q_factor

    fig = go.Figure()

    if show_vswr and tgt_vswr > 1.0:
        rho = (tgt_vswr - 1) / (tgt_vswr + 1)
        theta = np.linspace(0, 2*np.pi, 150)
        gamma_circle = rho * np.exp(1j * theta)
        z_circle = (zt + gamma_circle * zt.conjugate()) / (1 - gamma_circle)
        fig.add_trace(go.Scattersmith(real=np.real(z_circle)/50.0, imag=np.imag(z_circle)/50.0, mode='lines', line=dict(color=C_VSWR, dash='dash'), name=f'VSWR={tgt_vswr}', hoverinfo='skip'))

    z_curr = zl
    v, q = calc_metrics(z_curr)
    fig.add_trace(go.Scattersmith(real=[np.real(z_curr)/50.0], imag=[np.imag(z_curr)/50.0], mode='markers', marker=dict(color=C_START, size=8), name='Start ZL',
                                  hovertemplate=f"<b>Start</b><br>Z: {np.real(z_curr):.1f} {'+' if np.imag(z_curr)>=0 else ''} {np.imag(z_curr):.1f}j Ω<br>VSWR: {v:.2f}<br><b>Q: {q:.2f}</b><extra></extra>"))

    for item in st.session_state.netlist:
        z_start_node = z_curr
        path_z = []
        steps = 40
        
        # 🌟 重新啟用傳輸線在史密斯圖上的旋轉計算
        if 'w' in item:
            z0_t, beta_t = calc_microstrip(item['w'], pcb_h, pcb_er, fc)
            if "Stub" in item['type']:
                is_open = "Open" in item['type']
                y_s = 1.0/z_start_node
                for k in range(steps+1):
                    bl = beta_t * ((item['l'] * k / steps + 1e-6)/1000.0)
                    tan_bl = np.tan(bl)
                    if not is_open and abs(tan_bl) < 1e-6: tan_bl = 1e-6
                    y_stub = 1j*(1/z0_t)*tan_bl if is_open else -1j*(1/z0_t)/tan_bl
                    path_z.append(1.0/(y_s + y_stub))
                bl_f = beta_t*(item['l']/1000.0)
                y_stub_f = 1j*(1/z0_t)*np.tan(bl_f) if is_open else -1j*(1/z0_t)/np.tan(bl_f)
                z_curr = 1.0/(y_s + y_stub_f)
            else: # Trace
                for k in range(steps+1):
                    bl = beta_t * ((item['l'] * k / steps)/1000.0)
                    t = np.tan(bl)
                    den = z0_t + 1j*z_start_node*t 
                    if abs(den) < 1e-9: den = 1e-9
                    path_z.append(z0_t * (z_start_node + 1j*z0_t*t) / den)
                bl_f = beta_t * (item['l']/1000.0)
                t_f = np.tan(bl_f)
                den_f = z0_t + 1j*z_start_node*t_f
                z_curr = z0_t * (z_start_node + 1j*z0_t*t_f) / den_f
            t_color = C_TRACE
        else:
            z_br = get_z_branch(fc, item['val'], item['esr'], item['par'], item['comp'])
            for k in range(steps+1):
                if item['mode'] == 'ser': temp_z = z_start_node + z_br*(k/steps)
                else: temp_z = 1 / (1/z_start_node + (1/z_br)*(k/steps))
                path_z.append(temp_z)
            z_curr = path_z[-1]
            t_color = C_IND if item['comp']=='L' else C_CAP
            
        hover_texts = []
        for pz in path_z:
            v_p, q_p = calc_metrics(pz)
            hover_texts.append(f"Z: {np.real(pz):.1f}{'+' if np.imag(pz)>=0 else ''}{np.imag(pz):.1f}j<br>VSWR: {v_p:.2f}<br><b>Q: {q_p:.2f}</b>")
        fig.add_trace(go.Scattersmith(real=np.real(path_z)/50.0, imag=np.imag(path_z)/50.0, mode='lines', line=dict(color=t_color, width=3), name=item['type'], text=hover_texts, hoverinfo='text+name'))

    if st.session_state.netlist:
        f_axis_smith = np.linspace(max(1e6, fc - span/2), fc + span/2, 201)
        zin_list = []
        for f_sweep in f_axis_smith:
            abcd = np.matrix([[1,0],[0,1]], dtype=complex)
            for item in reversed(st.session_state.netlist):
                # 🌟 寬頻 locus 的 ABCD 矩陣傳輸線支援
                if 'w' in item:
                    z0_t, beta_t = calc_microstrip(item['w'], pcb_h, pcb_er, f_sweep)
                    bl = beta_t * (item['l']/1000.0)
                    if 'Trace' in item['type']:
                        m = np.matrix([[np.cos(bl), 1j*z0_t*np.sin(bl)], [1j/z0_t*np.sin(bl), np.cos(bl)]])
                    else:
                        y_stub = 1j/z0_t*np.tan(bl) if 'Open' in item['type'] else -1j/z0_t/np.tan(bl)
                        m = np.matrix([[1, 0], [y_stub, 1]])
                else:
                    z_b = get_z_branch(f_sweep, item['val'], item['esr'], item['par'], item['comp'])
                    m = np.matrix([[1, z_b], [0, 1]]) if item['mode'] == 'ser' else np.matrix([[1, 0], [1/z_b, 1]])
                abcd = abcd * m
            zin = (abcd[0,0]*zl + abcd[0,1]) / (abcd[1,0]*zl + abcd[1,1])
            zin_list.append(zin)
        zin_list = np.array(zin_list)
        hover_bw = []
        for f_val, z_val in zip(f_axis_smith, zin_list):
            v_p, q_p = calc_metrics(z_val)
            hover_bw.append(f"Freq: {f_val/1e9:.3f} GHz<br>Z: {np.real(z_val):.1f}{'+' if np.imag(z_val)>=0 else ''}{np.imag(z_val):.1f}j<br>VSWR: {v_p:.2f}")
        fig.add_trace(go.Scattersmith(real=np.real(zin_list)/50.0, imag=np.imag(zin_list)/50.0, mode='lines', line=dict(color=C_BW, width=2, dash='solid'), name='Bandwidth Locus', text=hover_bw, hoverinfo='text+name'))

    if st.session_state.gamma_points:
        mc_g = np.array(st.session_state.gamma_points)
        mc_zn = (1 + mc_g) / (1 - mc_g + 1e-15)
        fig.add_trace(go.Scattersmith(real=np.real(mc_zn), imag=np.imag(mc_zn), mode='markers', marker=dict(color=C_FINAL, size=4, opacity=0.4), name='Yield (Monte Carlo)', hoverinfo='skip'))

    v_f, q_f = calc_metrics(z_curr)
    fig.add_trace(go.Scattersmith(real=[np.real(z_curr)/50.0], imag=[np.imag(z_curr)/50.0], mode='markers', marker=dict(color=C_FINAL, size=10), name='Final Z (Fc)', hovertemplate=f"<b>Final Z (Fc)</b><br>Z: {np.real(z_curr):.1f} {'+' if np.imag(z_curr)>=0 else ''} {np.imag(z_curr):.1f}j Ω<br>VSWR: {v_f:.2f}<br><b>Q: {q_f:.2f}</b><extra></extra>"))

    fig.update_layout(smith=dict(bgcolor=BG_MAIN, realaxis=dict(gridcolor='#444444'), imaginaryaxis=dict(gridcolor='#444444')), paper_bgcolor=BG_MAIN, height=800, title=dict(text=f"Center Freq VSWR: {v_f:.3f} | Nodal Q: {q_f:.2f}", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not st.session_state.netlist:
        st.info("Please add components to analyze bandwidth.")
    else:
        f_axis = np.linspace(max(1e6, fc-span/2), fc+span/2, 401)
        gt_list, phases = [], []
        zs_f_ref = complex(zs_real, zs_imag) if conj_mode else 50.0 + 0j
        
        for f in f_axis:
            abcd = np.matrix([[1,0],[0,1]], dtype=complex)
            for item in reversed(st.session_state.netlist):
                # 🌟 S21 的 ABCD 矩陣傳輸線支援
                if 'w' in item:
                    z0_t, beta_t = calc_microstrip(item['w'], pcb_h, pcb_er, f)
                    bl = beta_t * (item['l']/1000.0)
                    if 'Trace' in item['type']:
                        m = np.matrix([[np.cos(bl), 1j*z0_t*np.sin(bl)], [1j/z0_t*np.sin(bl), np.cos(bl)]])
                    else:
                        y_stub = 1j/z0_t*np.tan(bl) if 'Open' in item['type'] else -1j/z0_t/np.tan(bl)
                        m = np.matrix([[1, 0], [y_stub, 1]])
                else:
                    z_b = get_z_branch(f, item['val'], item['esr'], item['par'], item['comp'])
                    m = np.matrix([[1, z_b], [0, 1]]) if item['mode'] == 'ser' else np.matrix([[1, 0], [1/z_b, 1]])
                abcd = abcd * m
            A, B, C, D = abcd[0,0], abcd[0,1], abcd[1,0], abcd[1,1]
            den = A*zl + B + C*zs_f_ref*zl + D*zs_f_ref
            gt = (4 * zs_f_ref.real * zl.real) / (abs(den)**2 + 1e-12)
            gt_list.append(10*np.log10(gt)); phases.append(np.angle(zl/den))
        
        max_gain = np.max(gt_list)
        target_3db = max_gain - 3
        idx_above_3db = np.where(gt_list >= target_3db)[0]
        if len(idx_above_3db) > 1:
            bw_3db = (f_axis[idx_above_3db[-1]] - f_axis[idx_above_3db[0]]) / 1e6
            st.metric("Detected -3dB Bandwidth", f"{bw_3db:.2f} MHz")
        
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("S21 - Transducer Gain (dB)", "Group Delay (ns)"))
        fig2.add_trace(go.Scatter(x=f_axis/1e9, y=gt_list, mode='lines', line=dict(color=C_CAP), name="S21", hovertemplate="Freq: %{x:.3f}G<br>S21: %{y:.2f}dB"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=f_axis[1:]/1e9, y=-np.diff(np.unwrap(phases))/(2*np.pi*(f_axis[1]-f_axis[0]))*1e9, mode='lines', line=dict(color=C_FINAL), name="GD"), row=2, col=1)
        fig2.add_vline(x=fc/1e9, line_dash="dash", line_color=C_TARGET)
        fig2.update_layout(paper_bgcolor=BG_MAIN, plot_bgcolor=BG_MAIN, height=700, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### 🎲 蒙地卡羅良率分析 (Monte Carlo Yield Analysis)")
    mc_tol = st.number_input("Component Tolerance (%)", value=5.0, step=1.0)
    
    if st.button("🎲 Run Monte Carlo (300pts)", type="primary", use_container_width=True):
        st.session_state.gamma_points = []
        for _ in range(300):
            z_c = zl
            for it in st.session_state.netlist:
                # 🌟 蒙地卡羅的傳輸線誤差支援 (套用在 PCB 長度變異)
                if 'w' in it:
                    r_l = max(0.01, np.random.normal(it['l'], it['l']*(mc_tol/100/3)))
                    z0_t, beta_t = calc_microstrip(it['w'], pcb_h, pcb_er, fc)
                    bl = beta_t * (r_l/1000.0)
                    if "Trace" in it['type']:
                        t = np.tan(bl)
                        den = z0_t + 1j*z_c*t 
                        z_c = z0_t * (z_c + 1j*z0_t*t) / den if abs(den)>1e-9 else z_c
                    else:
                        is_open = "Open" in it['type']
                        y_stub = 1j*(1/z0_t)*np.tan(bl) if is_open else -1j*(1/z0_t)/np.tan(bl)
                        z_c = 1.0/(1.0/z_c + y_stub)
                else:
                    rv = max(0.01, np.random.normal(it['val'], it['val']*(mc_tol/100/3)))
                    z_b = get_z_branch(fc, rv, it['esr'], it['par'], it['comp'])
                    z_c = (z_c + z_b) if it['mode'] == 'ser' else 1/(1/z_c + 1/z_b)
            st.session_state.gamma_points.append(z2g(z_c))
        st.rerun() 

    if st.session_state.gamma_points:
        st.success("✅ 蒙地卡羅分析計算完成！ (已生成 300 個變異點)")
        st.info("💡 請點擊上方切換回 **『🎯 Smith Chart (BW & Q)』** 分頁，即可在史密斯圖上觀看良率分佈的散佈狀況。")
        if st.button("🗑️ 清除良率分析資料"):
            st.session_state.gamma_points = []
            st.rerun()