# Q-MSG: Quantum Image Messenger via Superdense Coding

**A web application demonstrating resource-efficient quantum image transmission using high-dimensional superdense coding, classical pre-processing, and error mitigation on real IBM Quantum hardware.**

---
## 📜 Overview

Q-MSG implements a hybrid classical-quantum protocol for transmitting images more efficiently than standard quantum methods. By combining **Quantum Palette Multiplexing (QPM)** with **4-Dimensional Superdense Coding (4D-SDC)**, it achieves a ~33% reduction in entanglement cost versus baseline 6-bit RGB SDC, while aiming for acceptable visual quality on NISQ hardware.

This prototype allows users to sign up/log in, send 16x16 images using unique sharing codes, and view results processed on IBM Quantum backends. It incorporates techniques like **Gray coding**, **K-Round packing** (K=5 or 6), and **pairwise readout mitigation** for robustness.

---

## ✨ Solution: QPM + 4D-SDC

1.  **Classical QPM:** Compresses images to a 16-color palette (4 bits/pixel) and applies **Gray coding** to minimize perceptual errors.
2.  **Quantum 4D-SDC:** Encodes each 4-bit symbol onto **two Bell pairs**, transmitting 4 bits per two pairs (33% entanglement saving).
3.  **Efficiency & Robustness:** Uses **K-Round Packing** and **Pairwise Mitigation** based on calibration runs. The final Streamlit version uses R=1 redundancy (no fusion) and a single calibration packet.

---

## 🚀 Key Features

* Secure user signup/login (Supabase Auth).
* Unique sharing codes for receiving images.
* Quantum transmission of 16x16 images via QPM+4D-SDC on IBM Quantum hardware.
* Backend selection and configurable shot count.
* Pairwise readout error mitigation.
* Sender views Original vs. Reconstructed images + Metrics (PSNR, SSIM, Runtime).
* Receiver views only the Reconstructed image.
* Inbox/Sent message tracking.

---

## 🛠️ Technology Stack

* **Frontend/Backend:** **Streamlit** (Python library for UI & logic integration).
* **Quantum:** **Qiskit**, **Qiskit Runtime**.
* **Database/Auth/Storage:** **Supabase**.
* **Image Processing:** **Pillow**, **NumPy**, **Scikit-Image**.
* **Code Generation:** **Coolname**.
* **Secrets:** **python-dotenv**.

---

## 🖼️ UI Preview

---

## 🔧 Getting Started (Local Testing - Streamlit Version)

### Prerequisites

* Python 3.10+, pip, Git.
* **Supabase Account:** Project created, `profiles` & `transmissions` tables set up (SQL script provided previously), `images` storage bucket created (public), RLS enabled with correct policies, **Project URL**, **Public `anon` Key**, and **Secret `service_role_key`** copied.
* **IBM Quantum Account:** API token copied.

### Setup

1.  **Clone/Download:** Get the `app.py` code.
2.  **Environment:** Create and activate a Python virtual environment (`venv`).
3.  **Install:** `pip install streamlit coolname qiskit qiskit-ibm-runtime numpy scikit-image Pillow requests supabase python-dotenv` (or use `requirements.txt`).
4.  **Configure `.env`:** Create `.env` file with `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `IBM_QUANTUM_TOKEN`.

### Run

1.  **Activate Environment.**
2.  **Run:** `streamlit run app.py`.
3.  **Access:** Open the provided `localhost` URL in your browser.
4.  **Test:** Sign up two users, log in (use two browser windows - one incognito), copy recipient code, send image, check results.

---

## ☁️ Deployment (Streamlit Version)

1.  Create `requirements.txt`: `pip freeze > requirements.txt`.
2.  Push `app.py` and `requirements.txt` to a **private** GitHub repository.
3.  Deploy via [share.streamlit.io](https://share.streamlit.io):
    * Connect GitHub repo.
    * Select `main` branch, `app.py`.
    * In "Advanced settings...", paste the contents of your `.env` file into "Secrets".
    * Click "Deploy!".

---

## 🙏 Acknowledgements

* We thank **IBM Quantum** for providing access to real quantum hardware via the cloud, which enabled the experimental validation of this protocol.

---
