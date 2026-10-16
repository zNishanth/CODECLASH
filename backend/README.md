# SPACEIND Backend // Cloud Deployment Guide

This directory contains the Python FastAPI backend with YOLOv8 Computer Vision, Homography Calibration, and WebSocket Telemetry Synchronization.

---

## 1. Option A: Deploy to Render.com (Recommended Free Tier)

1. Create a GitHub repo (e.g. `your-username/spaceind-backend`) and push the contents of this `backend/` folder to it.
2. Sign in to [Render.com](https://render.com) and click **New +** &rarr; **Web Service**.
3. Connect your `spaceind-backend` repository.
4. Set the following settings:
   - **Environment:** `Python 3` (or choose `Docker`)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python server.py`
   - **Plan:** Free
5. Click **Deploy Web Service**.
6. Render will generate your backend URL:
   `https://spaceind-backend.onrender.com`
7. Copy this URL and paste it into your Vercel Frontend under **`BACKEND`**!

---

## 2. Option B: Deploy to Railway.app

1. Sign in to [Railway.app](https://railway.app).
2. Click **New Project** &rarr; **Deploy from GitHub repo**.
3. Select your `spaceind-backend` repository.
4. Railway will automatically detect the `Dockerfile` or `requirements.txt` and launch the service.
5. In your Railway service settings, click **Generate Domain** to get your public URL (e.g. `https://spaceind-production.up.railway.app`).
6. Paste into your Vercel Frontend!

---

## 3. Option C: Run Locally (Local Pair with Vercel Frontend)

If you want your frontend hosted on Vercel while your backend runs on your local machine with your real webcam:
1. Double-click `run.bat` or run:
   ```bash
   python server.py
   ```
2. Open your Vercel website in the browser.
3. Click **`BACKEND`** in the top bar.
4. Enter: `http://127.0.0.1:8000`
5. Click **Save & Connect**!
   Your Vercel frontend will now communicate directly with your local machine's OpenCV and YOLO model over WebSockets!
