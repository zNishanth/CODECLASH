# SPACEIND Frontend // Vercel Deployment Guide

This directory contains the client-side 3D WebGL Digital Twin platform ready to deploy instantly to [Vercel](https://vercel.com).

---

## What is in this directory:
- `index.html`: The 6-workspace Digital Twin interface (Overview, 3D Twin, Vision Ingest, BMS Relays, Facility AI, ESG Analytics).
- `DigitalTwinScene.js`: Three.js WebGL engine featuring articulated mannequin doll avatars, cutaway architectural glass walls, and dynamic ground heatmaps.
- `vercel.json`: Clean URL routing and CORS cache headers.

---

## How to Deploy to Vercel (2 Methods)

### Method 1: Deploy with Vercel CLI (Fastest, 30 seconds)
1. Open PowerShell or Terminal in this `frontend` folder:
   ```bash
   cd "frontend"
   ```
2. Run the Vercel CLI:
   ```bash
   npx vercel
   ```
3. Follow the on-screen prompts:
   - Set up and deploy: **`Y`**
   - Which scope: Select your personal account
   - Link to existing project: **`N`**
   - Project name: `spaceind-digital-twin`
   - In which directory is your code located: **`./`**
   - Want to modify settings: **`N`**
4. Vercel will output your live URL: `https://spaceind-digital-twin.vercel.app`!

---

### Method 2: Deploy via GitHub
1. Push this `frontend` directory to a GitHub repository (e.g. `your-username/spaceind-frontend`).
2. Go to [vercel.com/new](https://vercel.com/new).
3. Import your GitHub repository.
4. Keep the default settings (Framework Preset: *Other*) and click **Deploy**.
5. Your dashboard is live globally on Vercel's Edge Network!

---

## Connecting Vercel Frontend to Cloud Backend
1. When your Vercel website loads, click **`BACKEND`** (or `CONNECT BACKEND`) in the top navigation bar.
2. Enter your cloud backend URL:
   - Example: `https://spaceind-backend.onrender.com` or `https://your-app.up.railway.app`
   - For local testing: `http://127.0.0.1:8000`
3. Click **Save & Connect**.
4. The dashboard will instantly bind to the live WebSocket feed and animate real room telemetry!
