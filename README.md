# SPACEIND // Green Building Digital Twin

Production-grade, responsive Green Building Digital Twin platform connecting live Computer Vision (webcam & multi-image uploads), dynamic 3D BIM spatial reconstruction (Three.js), transient building thermodynamics, and an autonomous Facility AI Copilot.

---

## Architecture & Features

1. **Visual System & Industrial Brutalist Aesthetic (Nullpoint Labs / Stencil Style)**
   - Strict 1328px grid with 4-column 1px-ruled layout.
   - Dual theme system with smooth CSS variable transitions:
     - **Dark Mode (Default):** Pitch background `#08090B`, slate cards `#10131A`, hairline grid borders `rgba(255, 255, 255, 0.08)`, text `#EDEDED`.
     - **Light Mode:** Crisp technical chalk background `#F4F4F0`, white surface cards `#FFFFFF`, borders `#E2E2DC`, text `#111111`.
     - **Universal Accent:** Safety Orange `#FF5500` / `#F15D1E`.
   - Halftone luminance dot-matrix canvas in team section.
   - Real-time 3D Three.js scene featuring animated HVAC ceiling fans, glowing volumetric zone lights, and dynamic occupant markers.

2. **Computer Vision & Hardware Ingestion Engine**
   - High-throughput OpenCV pipeline with `cv2.CAP_DSHOW` backend on Windows to prevent camera lock and freeze.
   - Multi-class YOLOv8 inference: occupants, chairs, laptops, screens, fans, bottles, phones.
   - Atomic replacement file writes for `cv_output.json` and immediate `.flush()` on `cv_output.jsonl`.
   - Real-time webcam toggle (`/api/cv/toggle_camera`), single-shot snapshot, and multi-image room extrusion reconstruction.

3. **Core Digital Twin & Physics Engine**
   - Transient sensible heat model:
     $$Q = N \times 100\text{ W} + U \cdot A \cdot \Delta T$$
   - HVAC chilling power calculation with coefficient of performance $\text{COP} = 3.2$.
   - Energy rate calculation at ₹8.50/kWh, grid carbon factor of $0.82\text{ kg CO}_2/\text{kWh}$.
   - Transient indoor room temperature dynamics via Newton's Law of Cooling.
   - Procedural room creation via polygon vertices and height extrusion (`POST /api/rooms/create`).
   - Hardware relay matrix with comfort safety guards ($18^\circ\text{C} - 26^\circ\text{C}$).

4. **Facility AI Copilot & Deterministic Energy Rules**
   - Autonomous Facility AI Copilot answering building telemetry queries and parsing actuation commands into discrete relay actions.
   - Deterministic energy optimization engine calculating real-time line-item wattage savings.
   - State-hash TTL cache preventing redundant API calls and rate-limit exhaustion.
   - Strict unbranded naming: **Facility AI Copilot**.

---

## Quick Start

### 1. Requirements
- Python 3.10+
- Installed packages:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Launch
- **Windows**: Double-click `run.bat` or run:
  ```powershell
  python server.py
  ```
- **Linux / macOS**:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### 3. Open in Browser
- Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Route `/`: Marketing Overview & System Specs
- Route `/app`: 3D Spatial Twin, Live Vision Ingest & Facility AI Matrix
