# Q-MSG: Quantum Image Messenger via Superdense Coding ⚛️🖼️

**A full-stack web application demonstrating resource-efficient quantum image transmission using high-dimensional superdense coding, classical pre-processing, and error mitigation on real IBM Quantum hardware.**

---

## 📜 Overview

Q-MSG explores a hybrid classical-quantum approach to transmitting digital images, addressing the significant entanglement resource challenge posed by standard quantum communication protocols like superdense coding (SDC). [cite_start]By combining **Quantum Palette Multiplexing (QPM)** with a **4-Dimensional Superdense Coding (4D-SDC)** scheme realized on superconducting qubits, this project demonstrates a ~33% reduction in entanglement cost compared to baseline 6-bit RGB SDC, while maintaining perceptually acceptable image quality on noisy intermediate-scale quantum (NISQ) hardware[cite: 10, 36, 37, 276].

This prototype allows users to sign up, log in, send 16x16 images to other users via unique sharing codes, and view the results processed through real IBM Quantum backends. [cite_start]It incorporates advanced techniques like **Gray coding**, **K-Round circuit packing**, and **pairwise readout error mitigation** for enhanced robustness and efficiency, as detailed in the associated research paper[cite: 1, 2].

---

## 🎯 Problem Addressed

[cite_start]Standard qubit-based Superdense Coding is impractical for multimedia like images due to the "resource explosion" – transmitting a single 24-bit RGB pixel requires 12 Bell pairs[cite: 8, 27, 53, 54]. [cite_start]Even compressed representations demand significant entanglement (e.g., 3 pairs for 6 bits) [cite: 27, 55][cite_start], exceeding the capabilities of current NISQ devices[cite: 28, 95].

---

## ✨ Solution: QPM + 4D-SDC

[cite_start]Q-MSG implements the protocol described in *"Enhanced Superdense Coding for Efficient Image Transmission..."*[cite: 1, 2]:

1.  [cite_start]**Classical QPM:** Images are compressed to a 16-color palette (4 bits/pixel), achieving a 6x reduction[cite: 10, 103, 108, 179]. [cite_start]Palette indices are **Gray-coded** to minimize perceptual errors from bit-flips[cite: 10, 103, 111, 129, 199].
2.  [cite_start]**Quantum 4D-SDC:** Each 4-bit Gray symbol is encoded onto **two Bell pairs** (emulating a ququart), transmitting 4 classical bits per two pairs, saving 33% entanglement vs. 6-bit RGB SDC[cite: 9, 36, 104, 113, 181, 240, 341].
3.  **Efficiency & Robustness:**
    * [cite_start]**K-Round Packing:** Multiple pixels (K=6 in the reference implementation) are transmitted within a single dynamic circuit, drastically reducing circuit count and runtime[cite: 11, 38, 74, 130, 151, 247].
    * [cite_start]**Pairwise Mitigation:** Custom $4\times4$ readout calibration corrects correlated errors on Bell pairs[cite: 12, 105, 132, 134, 147, 219].
    * **(Optional) Redundancy/Fusion:** The logic from `run.py` (R=2, two calibration packets, `decode_fused`) offers potentially higher fidelity than the simpler R=1 approach from `Results.ipynb`. *(Note: The final Streamlit `app.py` uses the R=1 logic from `Results.ipynb`)*.

---

## 🚀 Key Features

* **User Authentication:** Secure signup/login via Supabase Auth.
* **Unique Sharing Codes:** Automatically generated codes for users to receive images.
* **Quantum Image Transmission:** Send 16x16 images processed via the QPM+4D-SDC protocol.
* **Real Hardware Execution:** Jobs run on actual IBM Quantum backends via Qiskit Runtime.
* **Backend Selection:** User can choose the target IBM Quantum device (e.g., `ibm_torino`).
* **Shot Selection:** User can configure the number of measurement shots.
* **Error Mitigation:** Implements pairwise readout calibration.
* **Result Visualization:**
    * *Receiver:* Sees only the final reconstructed image.
    * *Sender:* Sees a comparison of Original vs. Reconstructed images, plus PSNR/SSIM/Runtime metrics.
* **Inbox/Sent Views:** Track message history and status.

---

## 🛠️ Technology Stack

* **Frontend:** Streamlit (Python library for UI)
* **Backend Logic:** Python (integrated within the Streamlit script)
    * Quantum Processing: Qiskit, Qiskit Runtime
    * Image Handling: Pillow, NumPy, Scikit-Image
    * Code Generation: Coolname
* **Database/Auth/Storage:** Supabase (PostgreSQL, GoTrue Auth, Storage)
* **Quantum Backend:** IBM Quantum Platform

*(Note: The alternative architecture discussed uses Flask/RQ/Redis for the backend and React/Next.js for the frontend)*

---

## 🖼️ UI Preview

*(This uses the Streamlit version with custom CSS to match your design)*





---

## 🔧 Getting Started (Local Testing - Streamlit Version)

Follow these steps to run the Streamlit prototype locally in VS Code.

### Prerequisites

* **Python:** 3.10 or later recommended.
* **pip:** Python package installer.
* **Git:** For cloning (optional, if downloading).
* **Supabase Account:** A free account at [supabase.com](https://supabase.com).
    * Create a new project.
    * Run the SQL script (provided previously) in the SQL Editor to create `profiles` and `transmissions` tables.
    * Create a public Storage bucket named `images`.
    * Set up Storage Policies (Allow Authenticated Uploads `INSERT`, Allow Public Views `SELECT`).
    * Enable RLS on `profiles` and `transmissions` tables and add the necessary `SELECT` and `INSERT` policies (using `auth.role() = 'authenticated'` or `auth.uid() = id` where appropriate).
    * Copy your **Project URL**, **Public `anon` Key**, and **Secret `service_role_key`**.
* **IBM Quantum Account:** An account with API access enabled ([quantum.ibm.com](https://quantum.ibm.com)).
    * Copy your **IBM Quantum API Token**.

### Installation & Setup

1.  **Get the Code:** Clone the repository or download the `app.py` file.
    ```bash
    # If cloning:
    # git clone <your-repo-url>
    # cd <your-repo-folder>
    ```

2.  **Create Python Environment:**
    ```bash
    python -m venv venv
    # Activate (Windows PowerShell):
    .\venv\Scripts\Activate.ps1
    # Activate (macOS/Linux):
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install streamlit coolname qiskit qiskit-ibm-runtime numpy scikit-image Pillow requests supabase python-dotenv
    # Or if you have requirements.txt:
    # pip install -r requirements.txt
    ```

4.  **Create `.env` File:** Create a file named `.env` in the same directory as `app.py`. Add your secret keys:
    ```.env
    SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL"
    SUPABASE_SERVICE_ROLE_KEY="YOUR_SUPABASE_SECRET_SERVICE_ROLE_KEY"
    IBM_QUANTUM_TOKEN="YOUR_IBM_QUANTUM_API_TOKEN"
    ```
    *(Replace placeholders with your actual keys)*

### Running the Application

1.  **Open Terminal:** Make sure your virtual environment is active.
2.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```
3.  **Access:** Your browser should automatically open to the application (usually `http://localhost:8501`).
4.  **Test:** Sign up two users, log in, copy the recipient code, send an image, and check the results in the Inbox/Sent tabs. Use two browser windows (one incognito) to simulate sender and receiver simultaneously.

---

## ☁️ Deployment (Streamlit Version)

This Streamlit application can be easily deployed to **Streamlit Community Cloud** (free tier available):

1.  Push your `app.py` and `requirements.txt` to a **private** GitHub repository. **Do NOT push `.env`**.
2.  Sign up/Log in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3.  Click "New app", connect your repository, select `main` branch, and `app.py` as the main file.
4.  Go to "Advanced settings..." and paste the contents of your `.env` file into the "Secrets" section.
5.  Click "Deploy!".

---

## 🔬 Methodology Implemented (Based on Paper & Notebook)

The `app.py` file implements the following pipeline, directly based on your `Results.ipynb`/`Untitled3.ipynb` logic:

1.  **Image Preparation:** Loads image, resizes to 16x16, quantizes to 16 colors (no perceptual reordering in the final version to match `Results.ipynb`).
2.  **Gray Coding:** Converts palette indices to 4-bit Gray codes.
3.  **Circuit Generation:** Creates one fixed 4-round `calibration_packet_fixed` and multiple `packet_circuit_Krounds` data packets (K=5 or 6, R=1 redundancy). Total circuits: ~44-52.
4.  **Job Submission:** Transpiles circuits (`layout_method="sabre"`) and submits them as a single job to the selected IBM backend using `Sampler(mode=backend)`.
5.  **Calibration:** Learns permutation and inverse mitigation matrices (`M02_inv`, `M13_inv`) from the single calibration packet result using `learn_perm_and_mats`.
6.  **Decoding:** For each data packet round, calculates posterior probabilities using `decode_round` (which internally uses `probs_from_round_with_perm` and mitigation matrices).
7.  **Fusion:** Accumulates log-likelihoods (`sym_ll`) for each pixel across packets (if R>1 was used, which it isn't in the final version). Performs a **Max-Likelihood (Majority Vote)** on the results for each pixel.
8.  **Post-Processing:** Applies `gray_to_bin4` and the `mode3x3_map` spatial filter.
9.  **Reconstruction & Metrics:** Generates the final image and calculates PSNR/SSIM/Runtime.

---

## 📈 Expected Results

* **Entanglement Saving:** ~33% compared to 6-bit RGB SDC (2 pairs/pixel vs 3).
* **Circuit Count:** ~44-52 circuits for a 16x16 image (depending on K).
* [cite_start]**Image Quality:** Should achieve PSNR ~15-17 dB and SSIM ~0.6-0.7 on successful runs with moderate noise, consistent with your notebook results[cite: 253]. Results will vary based on hardware noise.

---

## ⚠️ Limitations

* [cite_start]Performance is highly dependent on the noise level of the specific IBM Quantum backend at the time of execution[cite: 380, 381].
* The free tier of Streamlit Community Cloud causes apps to sleep after inactivity.

---

## 🙏 Acknowledgements

* [cite_start]IBM Quantum for providing access to quantum hardware[cite: 406].
* [cite_start]Organizers and mentors of the Amaravati Quantum Valley Hackathon 2025[cite: 406].

---

## 📄 License

*(Add your chosen license here, e.g., MIT)*
