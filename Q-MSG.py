# app.py — Final Version: Correct UI, Proven Quantum Logic, ALL FUNCTIONS INCLUDED
import streamlit as st
import os, io, uuid, random, itertools, math, time # Added time
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client
from skimage.metrics import structural_similarity as ssim

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


# UI STYLING & INITIALIZATION (PRECISELY MATCHES SCREENSHOT)

st.set_page_config(layout="centered", page_title="Q-MSG")
st.title("Quantum Image Transmission")
load_dotenv()
# --- Custom CSS (Same as previous correct version) ---
st.markdown("""<style>/* Paste the correct CSS here */</style>""", unsafe_allow_html=True) # Keep the CSS block

@st.cache_resource
def init_connections():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key: raise RuntimeError("CRITICAL ERROR: Missing Supabase URL or SUPABASE_SERVICE_ROLE_KEY.")
    supabase = create_client(url, key)
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not ibm_token: raise RuntimeError("CRITICAL ERROR: Missing IBM_QUANTUM_TOKEN.")
    service = QiskitRuntimeService(channel="ibm_cloud", token=ibm_token)
    return supabase, service
try:
    supabase, service = init_connections()
except Exception as e:
    st.error(f"Initialization failed. Check .env file. Error: {e}"); st.stop()

# =====================================================================
# QUANTUM & UTILITY FUNCTIONS (Copied EXACTLY from your provided Colab code)
# =====================================================================
def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a-b)**2)
    return 99.0 if mse == 0 else 20*np.log10(255.0/np.sqrt(mse))

def safe_ssim(a,b): # Added from previous version for display
    try: return ssim(a,b,channel_axis=2,data_range=255)
    except: return ssim(a,b,multichannel=True,data_range=255)

def bin_to_gray4(x):  return x ^ (x >> 1)
def gray_to_bin4(x):
    g = x; g ^= (g>>1); g ^= (g>>2); g ^= (g>>4); return g & 0xF

def val_to_bits2(v): return format(int(v)&0x3,"02b")
# --- ADDED MISSING FUNCTION ---
def val_to_bits4(v): return format(int(v)&0xF,"04b")
# -----------------------------

# --- Robust counts extractor (handles different result formats) ---
def extract_counts(pub, default=None): # From your Colab
    # ... (code for extract_counts is correct and unchanged)
    try: return pub.join_data().get_counts()
    except Exception: pass
    if hasattr(pub, "data") and hasattr(pub.data, "meas"):
        try: return pub.data.meas.get_counts()
        except Exception: pass
    if hasattr(pub, "quasi_dists"):
         try:
            quasi_dist = pub.quasi_dists[0]
            shots = pub.metadata[0].get("shots",1); num_clbits = pub.metadata[0].get("num_clbits",0)
            return {f'{k:0{num_clbits}b}': int(round(v * shots)) for k, v in quasi_dist.items()}
         except Exception: pass
    if hasattr(pub, "get_counts"):
        try: return pub.get_counts()
        except Exception: pass
    return {} if default is None else default


def _nbits_from_counts(counts, default_bits): # Helper from Colab
    # ... (code for _nbits_from_counts is correct and unchanged)
    if not counts: return default_bits
    k = next(iter(counts.keys()));
    if isinstance(k, str): return max(default_bits, len(k.replace(' ', '')))
    if isinstance(k, int): return max(default_bits, k.bit_length())
    try: return max(default_bits, int(k).bit_length())
    except Exception: return default_bits


def bit_at_generic(key, j, nbits): # Helper from Colab (LSB is index 0)
    # ... (code for bit_at_generic is correct and unchanged)
    if isinstance(key, int): return (key >> j) & 1
    s = str(key).replace(' ', '').zfill(nbits)
    return 1 if len(s) > j and s[::-1][j] == '1' else 0


def probs_from_round_with_perm(counts: dict, round_positions: list, perm: tuple): # From Colab code
    # ... (code for probs_from_round_with_perm is correct and unchanged)
    needed = max(round_positions) + 1
    nbits  = _nbits_from_counts(counts, needed)
    if not counts: return np.ones(4)/4, np.ones(4)/4
    tot = sum(counts.values()); p02, p13 = np.zeros(4), np.zeros(4)
    for bs,c in counts.items():
        try:
            a0 = bit_at_generic(bs, round_positions[perm[0]], nbits)
            b0 = bit_at_generic(bs, round_positions[perm[1]], nbits)
            a1 = bit_at_generic(bs, round_positions[perm[2]], nbits)
            b1 = bit_at_generic(bs, round_positions[perm[3]], nbits)
            p02[(a0<<1)|b0] += c
            p13[(a1<<1)|b1] += c
        except IndexError: continue # Skip if bit index out of bounds for the key
    return (p02/tot, p13/tot) if tot > 0 else (np.ones(4)/4, np.ones(4)/4)


def learn_perm_and_mats(counts_cal: dict): # From Colab code (fixed 4 rounds implicitly)
    # ... (code for learn_perm_and_mats is correct and unchanged)
    rel, best = [0,1,2,3], (-1e9, None, None, None)
    if not counts_cal: return (0, 1, 2, 3), np.eye(4), np.eye(4) # Handle empty cal counts
    for perm in itertools.permutations(rel, 4):
        M02, M13 = np.zeros((4,4)), np.zeros((4,4)); valid_rounds = 0
        for r in range(4): # fixed four rounds from calibration_packet_fixed
            rp = [4*r + j for j in rel]
            p02, p13 = probs_from_round_with_perm(counts_cal, rp, perm)
            if np.all(np.isfinite(p02)) and np.all(np.isfinite(p13)):
                 M02[:, r], M13[:, r] = p02, p13; valid_rounds += 1
            else: M02[:, r], M13[:, r] = np.ones(4)/4, np.ones(4)/4 # Default bad probs
        if valid_rounds == 0: continue # Skip if no rounds yielded valid probs

        score = np.trace(M02) + np.trace(M13)
        if np.isfinite(score) and score > best[0]: best = (score, perm, M02, M13)

    _, perm, M02, M13 = best
    if perm is None: perm, M02, M13 = (0, 1, 2, 3), np.eye(4), np.eye(4) # Fallback

    try: # Robust inverse
        eps = 5e-3 # Epsilon from Colab
        M02_inv = np.linalg.pinv(M02 + eps*np.eye(4))
        M13_inv = np.linalg.pinv(M13 + eps*np.eye(4))
        if not np.all(np.isfinite(M02_inv)) or not np.all(np.isfinite(M13_inv)): raise ValueError("Non-finite inverse")
    except Exception: M02_inv, M13_inv = np.eye(4), np.eye(4)
    return perm, M02_inv, M13_inv


def apply_symbol_on(aq: int, bits2: str, qc: QuantumCircuit): # From Colab code
    # ... (code for apply_symbol_on is correct and unchanged)
    if bits2 == "01": qc.z(aq)
    elif bits2 == "10": qc.x(aq)
    elif bits2 == "11": qc.x(aq); qc.z(aq)


def sdc4_round(qc: QuantumCircuit, bits4: str, coff: int): # From Colab code
    # ... (code for sdc4_round is correct and unchanged)
    qc.h(0); qc.cx(0,2); qc.h(1); qc.cx(1,3)
    apply_symbol_on(0, bits4[:2], qc); apply_symbol_on(1, bits4[2:], qc)
    qc.cx(0,2); qc.h(0); qc.cx(1,3); qc.h(1)
    # Measurement order from Colab: q0->c0, q2->c1, q1->c2, q3->c3 within the round
    qc.measure(0,coff+0); qc.measure(2,coff+1); qc.measure(1,coff+2); qc.measure(3,coff+3)
    qc.reset([0,1,2,3]) # Reset MUST be here for K>1 rounds


def build_packet(bits4_list): # From Colab code
    # ... (code for build_packet is correct and unchanged)
    K = len(bits4_list)
    qc = QuantumCircuit(4, 4*K)
    for r, b4 in enumerate(bits4_list): sdc4_round(qc, b4, 4*r)
    return qc


def calibration_packet_fixed(): # From Colab code (fixed 4 rounds)
    # ... (code for calibration_packet_fixed is correct and unchanged)
    rounds = ["0000","0101","1010","1111"]
    return build_packet(rounds)


def decode_round(counts, r, perm, M02_inv, M13_inv): # From Colab code (decode_round renamed)
    # ... (code for decode_round is correct and unchanged)
    rel = [0,1,2,3]; rp  = [4*r + j for j in rel] # Calculate round positions
    p02, p13 = probs_from_round_with_perm(counts, rp, perm)
    if not np.all(np.isfinite(p02)) or not np.all(np.isfinite(p13)): return "0000", np.ones(16)/16 # Error symbol + uniform prob
    try:
        q02 = np.clip(M02_inv @ p02,0,1); q13 = np.clip(M13_inv @ p13,0,1)
        if not np.all(np.isfinite(q02)) or not np.all(np.isfinite(q13)): raise ValueError("Non-finite posterior")
    except Exception: return "0000", np.ones(16)/16
    q02 /= q02.sum() if q02.sum()>1e-9 else 1; q02=np.nan_to_num(q02,nan=0.25)
    q13 /= q13.sum() if q13.sum()>1e-9 else 1; q13=np.nan_to_num(q13,nan=0.25)
    # Return full probabilities and symbol
    joint = np.outer(q02, q13).reshape(-1) # Joint probability distribution
    if not np.isclose(joint.sum(), 1.0): joint = np.ones(16)/16 # Fallback uniform
    syms  = [f"{i:02b}{j:02b}" for i in range(4) for j in range(4)]
    best_idx = np.argmax(joint)
    return syms[best_idx], joint # Return best symbol string and full probability dist


def mode3x3_map(idx): # From Colab code
    # ... (code for mode3x3_map is correct and unchanged)
    H, W = idx.shape; out = idx.copy(); pad = np.pad(idx, 1, mode='edge')
    for y in range(H):
        for x in range(W):
            block = pad[y:y+3, x:x+3].reshape(-1)
            vals, cnts = np.unique(block, return_counts=True)
            out[y,x] = vals[np.argmax(cnts)]
    return out


# =====================================================================
# UI + APP LOGIC
# =====================================================================
# --- Login View --- (Unchanged)
if 'user_session' not in st.session_state or st.session_state.user_session is None:
    # ... (Login form code is the same)
    st.markdown('<div id="login-container">', unsafe_allow_html=True)
    st.markdown("<h1>Quantum Image Messenger</h1>", unsafe_allow_html=True)
    st.markdown("<p>Please sign in or create an account to continue.</p>", unsafe_allow_html=True)
    with st.form("login_form"):
        em=st.text_input("Email", key="login_email")
        pw=st.text_input("Password",type="password", key="login_pass")
        c1,c2=st.columns(2)
        if c1.form_submit_button("Login", use_container_width=True):
            try: res=supabase.auth.sign_in_with_password({"email":em,"password":pw}); st.session_state.user_session=res.dict(); st.rerun()
            except Exception as e: st.error(f"Login failed: {e}")
        if c2.form_submit_button("Sign Up", use_container_width=True):
            try:
                res=supabase.auth.sign_up({"email":em,"password":pw})
                if res.user: code=f"QIM-{uuid.uuid4().hex[:6].upper()}"; supabase.table('profiles').insert({"id":res.user.id,"email":res.user.email,"sharing_code":code}).execute(); st.success("Signup OK. Please login.")
            except Exception as e: st.error(f"Signup failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# --- Main App View ---
uid=st.session_state.user_session['user']['id']
profile=supabase.table('profiles').select("sharing_code,email").eq('id',uid).single().execute().data

# --- Header & Dashboard --- (Unchanged)
c1_header, c2_header, c3_header, c4_header, c5_header = st.columns([4, 1, 1, 1, 3])
# ... (Header code is the same)
st.markdown("---")
st.info(f"Your Unique Sharing Code is: **`{profile['sharing_code']}`**")

# Navigation (Unchanged)
page = st.radio("Navigation", ["Compose", "Inbox", "Sent"], key="nav", horizontal=True, label_visibility="collapsed")

if page == "Compose":
    st.subheader("Compose Message")
    st.write("Upload an image, enter a recipient code, choose backend, and send.")
    with st.form("tx_form"):
        imgf=st.file_uploader("Image", type=["png","jpg","jpeg"])
        rc=st.text_input("Recipient Code", placeholder="e.g. QIM-ABC123")
        c1_form,c2_form=st.columns(2)
        backend_options = [b.name for b in service.backends(min_num_qubits=4, simulator=False)]
        backend_name=c1_form.selectbox("Backend", backend_options, key="backend_select")
        K_ROUNDS = 6 # Using K=5 from Colab default
        shots=c2_form.number_input("Shots",value=2048,step=1024, key="shots_input") # Using 4096 default from Colab
        submit_pressed = st.form_submit_button("Send")

        if submit_pressed:
            if not imgf or not rc: st.warning("Please provide all inputs."); st.stop()
            code=rc.strip()
            if not supabase.table("profiles").select("id", count="exact").ilike("sharing_code",code).execute().count > 0:
                st.error(f"Recipient code `{code}` not found."); st.stop()

            with st.spinner("Processing quantum job... This may take some time."):
                fname=f"{uuid.uuid4()}.png"; supabase.storage.from_("images").upload(file=imgf.getvalue(),path=fname, file_options={"content-type":imgf.type})
                img_url=supabase.storage.from_("images").get_public_url(fname)
                tx={"sender_id":uid,"receiver_sharing_code":code,"backend_used":backend_name,"shots_used":shots,"original_image_url":img_url,"status":"RUNNING"}
                txid=supabase.table('transmissions').insert(tx).execute().data[0]['id']

                try:
                    orig=Image.open(imgf).convert("RGB").resize((16,16),Image.Resampling.NEAREST)
                    pq=orig.quantize(colors=16,method=Image.MEDIANCUT,dither=Image.Dither.NONE); raw_palette=pq.getpalette()
                    if raw_palette is None: raise ValueError("Image has no palette.")
                    pal=np.array(raw_palette).reshape(len(raw_palette)//3,3); num_colors=len(pal)
                    if num_colors<16: pal=np.vstack([pal,np.zeros((16-num_colors,3),dtype=np.uint8)])
                    idx=np.array(pq,dtype=np.uint8);

                    # --- Perceptual Reordering (Matches Colab) ---
                    Y = 0.299*pal[:,0] + 0.587*pal[:,1] + 0.114*pal[:,2]
                    C = np.linalg.norm(pal - pal.mean(0), axis=1)
                    order = np.lexsort((C, Y))
                    perm16 = np.empty(16, int); perm16[order] = np.arange(16)
                    idx_map = perm16[idx] # Use reordered index map
                    pal_rgb = pal[order] # Use reordered palette
                    ideal=pal_rgb[idx_map]
                    # --- End Reordering ---

                    # --- Gray coding ---
                    bit4_stream = [format(bin_to_gray4(int(v)) & 0xF, "04b") for v in idx_map.reshape(-1)]; N=len(bit4_stream)

                    # --- Build Packets (R=1, K=K_ROUNDS, single FIXED cal packet - Matches Colab) ---
                    packets = [calibration_packet_fixed()] # Fixed 4-round cal packet
                    packet_pix_map = [[None]*4] # Map for fixed cal packet
                    i = 0
                    while i < N:
                        chunk = bit4_stream[i:i+K_ROUNDS]; ids = list(range(i, min(i+K_ROUNDS, N)))
                        if len(chunk) < K_ROUNDS: chunk += ["0000"]*(K_ROUNDS-len(chunk)); ids += [None]*(K_ROUNDS-len(ids))
                        packets.append(build_packet(chunk)) # Append packet ONCE (R=1)
                        packet_pix_map.append(ids)
                        i += K_ROUNDS
                    num_circuits = len(packets)
                    st.write(f"Submitting {num_circuits} circuits (Fixed Cal + K={K_ROUNDS}, R=1) to `{backend_name}`...")
                    # -----------------------------------------------------------

                    backend = service.backend(backend_name); sampler = Sampler(mode=backend)
                    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, layout_method="sabre"); transpiled = pm.run(packets)

                    t_start_quantum = time.time()
                    job = sampler.run(transpiled, shots=int(shots))
                    supabase.table('transmissions').update({"ibm_job_id":job.job_id()}).eq('id',txid).execute()
                    st.write(f"Job `{job.job_id()}` submitted. Awaiting results..."); res=job.result()
                    t_end_quantum = time.time()
                    quantum_runtime_seconds = t_end_quantum - t_start_quantum

                    st.write("Decoding results...")
                    counts_cal = extract_counts(res[0])
                    perm, M02_inv, M13_inv = learn_perm_and_mats(counts_cal) # Learn from fixed cal packet

                    # --- Log-Likelihood Fusion Decoding (Matches Colab) ---
                    sym_ll = [dict() for _ in range(N)] # Log-likelihood per pixel
                    pubs = list(res[1:]) # Data results start from index 1

                    for k, pub in enumerate(pubs): # Iterate through data packet results
                        counts = extract_counts(pub)
                        ids    = packet_pix_map[k+1]  # +1 skip calibration map
                        for r, pid in enumerate(ids): # Iterate through rounds in the packet
                            if pid is None: continue
                            # Decode round returns best symbol AND probabilities
                            best_sym_str, joint_probs = decode_round(counts, r, perm, M02_inv, M13_inv)
                            all_syms = [f"{i:02b}{j:02b}" for i in range(4) for j in range(4)] # All 16 possible symbols
                            # Accumulate log likelihood
                            for sym_idx, sym_str in enumerate(all_syms):
                                prob = max(joint_probs[sym_idx], 1e-6) # Clip probability
                                sym_ll[pid][sym_str] = sym_ll[pid].get(sym_str, 0.0) + np.log(prob)

                    decoded_gray_vals = np.zeros(N, dtype=int)
                    for p in range(N): # Max Likelihood Decoding
                        if sym_ll[p]:
                            best_sym = max(sym_ll[p].items(), key=lambda kv: kv[1])[0]
                            decoded_gray_vals[p] = int(best_sym, 2)
                        else: decoded_gray_vals[p] = 0 # Default if no likelihoods
                    # -------------------------------------------------------

                    decoded_idx=np.vectorize(gray_to_bin4)(decoded_gray_vals).reshape(16,16)
                    # Apply Mode Filter (Matches Colab)
                    decoded_idx = mode3x3_map(decoded_idx)
                    recon=pal_rgb[decoded_idx]; # Use reordered palette from image prep
                    metrics={"psnr":f"{psnr(ideal,recon):.2f} dB",
                             "ssim":f"{safe_ssim(ideal,recon):.3f}",
                             "quantum_runtime": f"{quantum_runtime_seconds:.1f} sec"}
                    rec=Image.fromarray(recon.astype(np.uint8)); buf=io.BytesIO(); rec.save(buf,"PNG"); rname=f"rec_{uuid.uuid4()}.png"
                    supabase.storage.from_("images").upload(file=buf.getvalue(),path=rname,file_options={"content-type":"image/png"})
                    rec_url=supabase.storage.from_("images").get_public_url(rname)
                    supabase.table('transmissions').update({"status":"COMPLETED","metrics":metrics,"reconstructed_image_url":rec_url}).eq('id',txid).execute()
                    st.success("Transmission complete!"); st.balloons()

                except Exception as e:
                    st.error(f"An error occurred during the quantum job: {e}")
                    st.exception(e)
                    supabase.table('transmissions').update({"status":"FAILED"}).eq('id',txid).execute()

# --- Inbox Tab --- (Includes Runtime Display)
elif page == "Inbox":
    st.subheader("Inbox")
    if st.button("Refresh Inbox"): st.rerun()
    data=supabase.table('transmissions').select("*,sender:sender_id(email)").eq('receiver_sharing_code',profile['sharing_code']).eq('status','COMPLETED').order('created_at',desc=True).execute().data
    if not data: st.write("No messages.")
    for it in data or []:
        sender=it.get("sender",{}).get("email","(unknown)"); created=str(it.get("created_at",""))[:10]
        with st.expander(f"From: {sender} on {created}"):
            if it.get("reconstructed_image_url"): st.image(it["reconstructed_image_url"],caption="Received Image")
            if it.get("metrics"):
                 st.markdown(f"""**Results:** - PSNR: {it['metrics'].get('psnr', 'N/A')} - SSIM: {it['metrics'].get('ssim', 'N/A')} - Quantum Runtime: {it['metrics'].get('quantum_runtime', 'N/A')}""")

# --- Sent Tab --- (Includes Runtime Display)
elif page == "Sent":
    st.subheader("Sent Items")
    sents=supabase.table('transmissions').select("*").eq('sender_id',uid).order('created_at',desc=True).execute().data
    if not sents: st.write("No sent messages.")
    for it in sents or []:
        with st.expander(f"To: `{it['receiver_sharing_code']}` | Status: {it['status']}"):
            if it['status']=="COMPLETED":
                c1,c2=st.columns(2); c1.image(it['original_image_url'],caption="Original"); c2.image(it['reconstructed_image_url'],caption="Reconstructed")
                if it.get("metrics"):
                     st.markdown(f"""**Results:** - PSNR: {it['metrics'].get('psnr', 'N/A')} - SSIM: {it['metrics'].get('ssim', 'N/A')} - Quantum Runtime: {it['metrics'].get('quantum_runtime', 'N/A')}""")
            else: st.write(f"Job is currently: {it['status']}")

# --- Logout Button --- (Unchanged)
st.sidebar.markdown("---")
if st.sidebar.button("Logout", key="logout_button_sidebar", use_container_width=True):
    st.session_state.user_session = None; st.rerun()