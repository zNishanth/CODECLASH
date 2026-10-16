/**
 * DigitalTwinScene.js
 * 
 * High-Performance Three.js Digital Twin Engine & BIM Spatial Visualizer.
 * 
 * Architectural & Visual Upgrades:
 * 1. Strict Asset Inventory: Completely removes ceiling fans and rotating blade animations.
 *    Replaces them with sleek architectural linear ceiling/wall diffusers featuring subtle
 *    translucent cyan airflow particle ribbons and gentle pulsation when HVAC is energized.
 * 2. Miniature Mannequin Avatars ("Chameleon Doll" Aesthetic):
 *    - Realistically proportioned (~1.7m equivalent height against standard 3m ceiling).
 *    - Articulated ceramic/holographic mannequin: head sphere, neck, segmented chest,
 *      waist ball joint, articulated shoulder/elbow/hip/knee joints, segmented limbs.
 *    - Strict Y=0 floor anchoring via homography-projected (X, Z) coordinates.
 *    - Dynamic holographic presence ring under feet with pulsing radar rings & Person ID badges.
 * 3. Industrial Digital Twin Aesthetic:
 *    - Dark sci-fi slate/charcoal background (#080B11).
 *    - Semi-transparent architectural glass cutaway walls with glowing wireframe edges.
 *    - Glowing neon zone boundary ribbons matching active energy/thermal telemetry.
 *    - Dynamic ground-plane thermal & occupancy heatmap texture rendered via offscreen canvas.
 * 4. WebGL Performance & Object Pooling:
 *    - Avatar object pool reusing pre-allocated meshes to eliminate GC pauses and GPU re-allocation.
 *    - Lerp (linear interpolation) position smoothing (anti-jitter) for smooth occupant gliding.
 *    - Full lifecycle disposal method (geometry, material, texture, context loss).
 */

class DigitalTwinScene {
  constructor(options = {}) {
    this.container = options.container || document.body;
    this.onZoneClick = options.onZoneClick || null;
    this.onAvatarClick = options.onAvatarClick || null;

    // Viewport dimensions
    this.width = this.container.clientWidth || 800;
    this.height = this.container.clientHeight || 500;

    // Active state cache
    this.currentRoomId = null;
    this.roomDim = [6.0, 5.0, 3.0]; // [Length, Width, Height]
    this.zonesData = {};
    this.staticAssets = {};
    this.telemetryKpis = {};

    // Heatmap dynamic canvas
    this.heatmapCanvas = document.createElement('canvas');
    this.heatmapCanvas.width = 512;
    this.heatmapCanvas.height = 512;
    this.heatmapCtx = this.heatmapCanvas.getContext('2d');
    this.heatmapTexture = new THREE.CanvasTexture(this.heatmapCanvas);
    this.heatmapTexture.minFilter = THREE.LinearFilter;
    this.heatmapTexture.magFilter = THREE.LinearFilter;

    // Avatar Object Pool (max 24 concurrent tracked occupants)
    this.MAX_AVATARS = 24;
    this.avatarPool = [];
    this.activeAvatars = new Map(); // track_id -> pooled avatar object

    // Three.js Core
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.animFrameId = null;
    this.clock = new THREE.Clock();

    // Scene Graph Groups
    this.roomGroup = new THREE.Group();
    this.zonesGroup = new THREE.Group();
    this.furnitureGroup = new THREE.Group();
    this.diffusersGroup = new THREE.Group();
    this.avatarsGroup = new THREE.Group();
    this.lightingGroup = new THREE.Group();
    this.hudGroup = new THREE.Group();

    // Materials Library
    this.materials = this._initMaterials();

    // Raycaster for interactivity
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this._initThree();
    this._initFloorAndHeatmap();
    this._initAvatarPool();
    this._setupEventListeners();
    this._animate();
  }

  // =========================================================================
  // INITIALIZATION & MATERIALS LIBRARY
  // =========================================================================
  _initMaterials() {
    return {
      // Semi-transparent architectural glass cutaway walls
      glassWall: new THREE.MeshPhysicalMaterial({
        color: 0x2A384C,
        transparent: true,
        opacity: 0.18,
        roughness: 0.1,
        metalness: 0.1,
        transmission: 0.6,
        ior: 1.4,
        depthWrite: false,
        side: THREE.DoubleSide
      }),
      // Wireframe outlines for cutaway glass
      glassEdge: new THREE.LineBasicMaterial({
        color: 0x4A6B8A,
        transparent: true,
        opacity: 0.45,
        linewidth: 1
      }),
      // Dark technical floor slab
      floorBase: new THREE.MeshStandardMaterial({
        color: 0x0B0E14,
        roughness: 0.75,
        metalness: 0.2
      }),
      // Ceramic Matte Mannequin Skin
      mannequinMatte: new THREE.MeshStandardMaterial({
        color: 0xEDEDED,
        roughness: 0.35,
        metalness: 0.15
      }),
      // Mannequin Joint Spheres (Metallic Titanium)
      mannequinJoint: new THREE.MeshStandardMaterial({
        color: 0x717D8A,
        roughness: 0.2,
        metalness: 0.8
      }),
      // Presence Ring Hologram (Cyan Glow)
      presenceRing: new THREE.MeshBasicMaterial({
        color: 0x00FFC2,
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide
      }),
      // Linear Diffuser Slit (Technical Graphite)
      diffuserHousing: new THREE.MeshStandardMaterial({
        color: 0x1E222B,
        roughness: 0.5,
        metalness: 0.7
      }),
      // Airflow Vector Glow
      diffuserAirflow: new THREE.MeshBasicMaterial({
        color: 0x06B6D4,
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide
      }),
      // Desks & static workstation furniture
      deskSurface: new THREE.MeshStandardMaterial({
        color: 0x161A23,
        roughness: 0.6,
        metalness: 0.3
      }),
      deskEdge: new THREE.LineBasicMaterial({
        color: 0x2E3846,
        transparent: true,
        opacity: 0.6
      }),
      // Screens / Monitors
      screenBezel: new THREE.MeshStandardMaterial({
        color: 0x0E1015,
        roughness: 0.4,
        metalness: 0.8
      }),
      screenGlow: new THREE.MeshBasicMaterial({
        color: 0x06B6D4
      })
    };
  }

  _initThree() {
    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x08090D);
    this.scene.fog = new THREE.FogExp2(0x08090D, 0.025);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 100);
    this.camera.position.set(10.5, 9.5, 12.0);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.innerHTML = '';
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    if (window.THREE && window.THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.06;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.05; // do not go below ground
      this.controls.minDistance = 3.0;
      this.controls.maxDistance = 35.0;
      this.controls.target.set(3.0, 1.0, 2.5);
    }

    // 5. Lighting Setup
    const ambientLight = new THREE.AmbientLight(0x3B485A, 1.2);
    this.lightingGroup.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0x7688A1, 0x111620, 0.8);
    hemiLight.position.set(0, 20, 0);
    this.lightingGroup.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xE0E8F5, 1.4);
    keyLight.position.set(8.0, 14.0, 10.0);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.bias = -0.0005;
    this.lightingGroup.add(keyLight);

    // 6. Assemble Groups
    this.scene.add(this.lightingGroup);
    this.scene.add(this.roomGroup);
    this.scene.add(this.zonesGroup);
    this.scene.add(this.furnitureGroup);
    this.scene.add(this.diffusersGroup);
    this.scene.add(this.avatarsGroup);
    this.scene.add(this.hudGroup);
  }

  // =========================================================================
  // FLOOR & THERMAL HEATMAP GROUND SHADER
  // =========================================================================
  _initFloorAndHeatmap() {
    // 1. Technical Grid Helper
    this.gridHelper = new THREE.GridHelper(30, 30, 0x222B38, 0x131922);
    this.gridHelper.position.y = 0.001;
    this.scene.add(this.gridHelper);

    // 2. Base Floor Mesh with Thermal Heatmap Texture Blending
    const floorGeo = new THREE.PlaneGeometry(1, 1);
    floorGeo.rotateX(-Math.PI / 2);

    this.floorMesh = new THREE.Mesh(floorGeo, this.materials.floorBase);
    this.floorMesh.receiveShadow = true;
    this.floorMesh.position.set(0, 0, 0);
    this.roomGroup.add(this.floorMesh);

    // 3. Thermal Heatmap Overlay Plane (additive glow blend)
    const heatmapMat = new THREE.MeshBasicMaterial({
      map: this.heatmapTexture,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.heatmapMesh = new THREE.Mesh(floorGeo, heatmapMat);
    this.heatmapMesh.position.y = 0.003;
    this.roomGroup.add(this.heatmapMesh);
  }

  _drawThermalHeatmap() {
    const ctx = this.heatmapCtx;
    const cw = this.heatmapCanvas.width;
    const ch = this.heatmapCanvas.height;

    // Clear with dark ambient thermal baseline
    ctx.clearRect(0, 0, cw, ch);

    // Draw active zone temperatures as soft ambient radial gradients
    Object.values(this.zonesData).forEach(zone => {
      const poly = zone.poly || [];
      if (poly.length >= 3) {
        // Find centroid of zone
        let cx = 0, cy = 0;
        poly.forEach(p => { cx += p[0]; cy += p[1]; });
        cx = (cx / poly.length) * cw;
        cy = (cy / poly.length) * ch;

        const temp = zone.measured_temp_c || 23.0;
        const setpoint = zone.setpoint_c || 22.0;
        const delta = temp - setpoint;

        // Color based on temperature delta: cool blue (<22) -> emerald (22-24) -> amber (>25)
        let heatColor = 'rgba(16, 185, 129, 0.08)'; // default green
        if (delta > 2.0) {
          heatColor = 'rgba(239, 68, 68, 0.15)'; // warm red
        } else if (delta > 0.8) {
          heatColor = 'rgba(245, 158, 11, 0.12)'; // amber
        } else if (delta < -1.0) {
          heatColor = 'rgba(6, 182, 212, 0.10)'; // cool cyan
        }

        const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 160);
        grad.addColorStop(0, heatColor);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 160, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw intense localized heat signatures under detected occupants
    this.activeAvatars.forEach(avatar => {
      if (avatar.visible) {
        // Map metric position [X, Z] back to canvas pixel space
        const normX = Math.max(0, Math.min(1, avatar.group.position.x / this.roomDim[0]));
        const normZ = Math.max(0, Math.min(1, avatar.group.position.z / this.roomDim[1]));
        const px = normX * cw;
        const py = normZ * ch;

        const pGrad = ctx.createRadialGradient(px, py, 4, px, py, 45);
        pGrad.addColorStop(0, 'rgba(255, 85, 0, 0.55)');   // safety orange core
        pGrad.addColorStop(0.4, 'rgba(245, 158, 11, 0.28)'); // warm amber halo
        pGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = pGrad;
        ctx.beginPath();
        ctx.arc(px, py, 45, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    this.heatmapTexture.needsUpdate = true;
  }

  // =========================================================================
  // MINIATURE ARTICULATED MANNEQUIN AVATARS (THE "DOLL / CHAMELEON" AESTHETIC)
  // =========================================================================
  _createMannequinMesh() {
    /**
     * Builds a realistically proportioned, articulated artist's mannequin doll:
     * - Height: 1.70m scaled exactly to room units.
     * - Minimalist matte ceramic body with metallic ball joints.
     * - Head sphere, neck, segmented chest, waist joint, pelvis, articulated limbs.
     */
    const doll = new THREE.Group();
    const matMatte = this.materials.mannequinMatte;
    const matJoint = this.materials.mannequinJoint;

    // 1. Holographic Floor Presence Ring (Base at Y=0)
    const ringGeo = new THREE.RingGeometry(0.24, 0.28, 32);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMesh = new THREE.Mesh(ringGeo, this.materials.presenceRing.clone());
    ringMesh.position.y = 0.01;
    doll.add(ringMesh);

    // Inner pulsing radar disc
    const discGeo = new THREE.CircleGeometry(0.23, 32);
    discGeo.rotateX(-Math.PI / 2);
    const discMat = new THREE.MeshBasicMaterial({
      color: 0x00FFC2,
      transparent: true,
      opacity: 0.12,
      depthWrite: false
    });
    const discMesh = new THREE.Mesh(discGeo, discMat);
    discMesh.position.y = 0.008;
    doll.add(discMesh);

    // 2. Feet (Y = 0.04)
    const footGeo = new THREE.BoxGeometry(0.08, 0.04, 0.16);
    const leftFoot = new THREE.Mesh(footGeo, matMatte);
    leftFoot.position.set(-0.10, 0.03, 0.02);
    doll.add(leftFoot);

    const rightFoot = new THREE.Mesh(footGeo, matMatte);
    rightFoot.position.set(0.10, 0.03, 0.02);
    doll.add(rightFoot);

    // 3. Lower Legs / Calves (Y = 0.25)
    const calfGeo = new THREE.CylinderGeometry(0.04, 0.035, 0.38, 16);
    const leftCalf = new THREE.Mesh(calfGeo, matMatte);
    leftCalf.position.set(-0.10, 0.24, 0);
    doll.add(leftCalf);

    const rightCalf = new THREE.Mesh(calfGeo, matMatte);
    rightCalf.position.set(0.10, 0.24, 0);
    doll.add(rightCalf);

    // 4. Knee Ball Joints (Y = 0.45)
    const kneeGeo = new THREE.SphereGeometry(0.048, 16, 16);
    const leftKnee = new THREE.Mesh(kneeGeo, matJoint);
    leftKnee.position.set(-0.10, 0.45, 0);
    doll.add(leftKnee);

    const rightKnee = new THREE.Mesh(kneeGeo, matJoint);
    rightKnee.position.set(0.10, 0.45, 0);
    doll.add(rightKnee);

    // 5. Thighs (Y = 0.68)
    const thighGeo = new THREE.CylinderGeometry(0.052, 0.044, 0.42, 16);
    const leftThigh = new THREE.Mesh(thighGeo, matMatte);
    leftThigh.position.set(-0.10, 0.68, 0);
    doll.add(leftThigh);

    const rightThigh = new THREE.Mesh(thighGeo, matMatte);
    rightThigh.position.set(0.10, 0.68, 0);
    doll.add(rightThigh);

    // 6. Hip Joints & Pelvis Base (Y = 0.92)
    const pelvisGeo = new THREE.CylinderGeometry(0.13, 0.11, 0.14, 16);
    const pelvis = new THREE.Mesh(pelvisGeo, matMatte);
    pelvis.position.set(0, 0.93, 0);
    doll.add(pelvis);

    const hipJointGeo = new THREE.SphereGeometry(0.055, 16, 16);
    const leftHip = new THREE.Mesh(hipJointGeo, matJoint);
    leftHip.position.set(-0.09, 0.90, 0);
    doll.add(leftHip);

    const rightHip = new THREE.Mesh(hipJointGeo, matJoint);
    rightHip.position.set(0.09, 0.90, 0);
    doll.add(rightHip);

    // 7. Waist Ball Joint (Y = 1.05)
    const waistJoint = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 16), matJoint);
    waistJoint.position.set(0, 1.05, 0);
    doll.add(waistJoint);

    // 8. Articulated Torso / Chest (Y = 1.25)
    const chestGeo = new THREE.CylinderGeometry(0.16, 0.12, 0.32, 16);
    const chest = new THREE.Mesh(chestGeo, matMatte);
    chest.position.set(0, 1.24, 0);
    doll.add(chest);

    // 9. Shoulder Joints (Y = 1.36)
    const shoulderGeo = new THREE.SphereGeometry(0.052, 16, 16);
    const leftShoulder = new THREE.Mesh(shoulderGeo, matJoint);
    leftShoulder.position.set(-0.19, 1.36, 0);
    doll.add(leftShoulder);

    const rightShoulder = new THREE.Mesh(shoulderGeo, matJoint);
    rightShoulder.position.set(0.19, 1.36, 0);
    doll.add(rightShoulder);

    // 10. Upper Arms (Y = 1.20)
    const armGeo = new THREE.CylinderGeometry(0.038, 0.032, 0.28, 16);
    const leftArm = new THREE.Mesh(armGeo, matMatte);
    leftArm.position.set(-0.21, 1.20, 0);
    doll.add(leftArm);

    const rightArm = new THREE.Mesh(armGeo, matMatte);
    rightArm.position.set(0.21, 1.20, 0);
    doll.add(rightArm);

    // 11. Elbow Joints & Forearms (Y = 0.95)
    const elbowGeo = new THREE.SphereGeometry(0.038, 16, 16);
    const leftElbow = new THREE.Mesh(elbowGeo, matJoint);
    leftElbow.position.set(-0.21, 1.05, 0);
    doll.add(leftElbow);

    const rightElbow = new THREE.Mesh(elbowGeo, matJoint);
    rightElbow.position.set(0.21, 1.05, 0);
    doll.add(rightElbow);

    const forearmGeo = new THREE.CylinderGeometry(0.032, 0.026, 0.26, 16);
    const leftForearm = new THREE.Mesh(forearmGeo, matMatte);
    leftForearm.position.set(-0.21, 0.91, 0.04);
    leftForearm.rotation.x = 0.2;
    doll.add(leftForearm);

    const rightForearm = new THREE.Mesh(forearmGeo, matMatte);
    rightForearm.position.set(0.21, 0.91, 0.04);
    rightForearm.rotation.x = 0.2;
    doll.add(rightForearm);

    // 12. Neck (Y = 1.44)
    const neckGeo = new THREE.CylinderGeometry(0.045, 0.055, 0.10, 16);
    const neck = new THREE.Mesh(neckGeo, matJoint);
    neck.position.set(0, 1.44, 0);
    doll.add(neck);

    // 13. Minimalist Ceramic Head Sphere (Y = 1.60)
    const headGeo = new THREE.SphereGeometry(0.115, 24, 24);
    headGeo.scale(1.0, 1.15, 1.05); // subtly stylized oval head
    const head = new THREE.Mesh(headGeo, matMatte);
    head.position.set(0, 1.58, 0);
    doll.add(head);

    // 14. Persona ID Badge Sprite (Y = 1.88)
    const labelCanvas = document.createElement('canvas');
    labelCanvas.width = 128;
    labelCanvas.height = 48;
    const lctx = labelCanvas.getContext('2d');
    const labelTexture = new THREE.CanvasTexture(labelCanvas);
    const spriteMat = new THREE.SpriteMaterial({ map: labelTexture, depthTest: false });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(0, 1.88, 0);
    sprite.scale.set(0.65, 0.24, 1.0);
    doll.add(sprite);

    // Return encapsulated object pool item
    return {
      group: doll,
      ringMesh: ringMesh,
      discMesh: discMesh,
      labelSprite: sprite,
      labelCanvas: labelCanvas,
      labelCtx: lctx,
      labelTexture: labelTexture,
      trackId: null,
      visible: false,
      targetX: 0,
      targetZ: 0,
      confidence: 1.0
    };
  }

  _initAvatarPool() {
    for (let i = 0; i < this.MAX_AVATARS; i++) {
      const avatarItem = this._createMannequinMesh();
      avatarItem.group.visible = false;
      this.avatarsGroup.add(avatarItem.group);
      this.avatarPool.push(avatarItem);
    }
  }

  _updateAvatarLabel(avatarItem, labelText, confidence) {
    const ctx = avatarItem.labelCtx;
    ctx.clearRect(0, 0, 128, 48);

    // Background pill
    ctx.fillStyle = 'rgba(10, 14, 22, 0.85)';
    ctx.strokeStyle = 'rgba(0, 255, 194, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, 120, 40, 8);
    ctx.fill();
    ctx.stroke();

    // Text
    ctx.font = 'bold 18px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00FFC2';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${labelText} [${Math.round(confidence * 100)}%]`, 64, 24);

    avatarItem.labelTexture.needsUpdate = true;
  }

  // =========================================================================
  // ARCHITECTURAL AIRFLOW LINEAR DIFFUSERS (REPLACING CEILING FANS)
  // =========================================================================
  _buildLinearDiffusers(diffusersList) {
    this.diffusersGroup.clear();

    diffusersList.forEach((diffuser, idx) => {
      const coords = diffuser.coordinates || [2.0, 2.0, 2.9];
      const posX = coords[0];
      const posZ = coords[1];
      const posY = coords[2] || 2.92;

      const diffGroup = new THREE.Group();
      diffGroup.position.set(posX, posY, posZ);

      // Sleek architectural linear slot fixture
      const housingGeo = new THREE.BoxGeometry(1.2, 0.06, 0.22);
      const housing = new THREE.Mesh(housingGeo, this.materials.diffuserHousing);
      diffGroup.add(housing);

      // Inner glowing diffuser vent slot
      const ventGeo = new THREE.BoxGeometry(1.1, 0.02, 0.08);
      const ventMesh = new THREE.Mesh(ventGeo, this.materials.diffuserAirflow);
      ventMesh.position.y = -0.025;
      diffGroup.add(ventMesh);

      // Translucent airflow particle ribbon downward vectors
      const ribbonGeo = new THREE.PlaneGeometry(0.9, 0.6);
      ribbonGeo.rotateX(Math.PI / 2);
      const ribbonMat = new THREE.MeshBasicMaterial({
        color: 0x06B6D4,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
        depthWrite: false
      });
      const ribbon = new THREE.Mesh(ribbonGeo, ribbonMat);
      ribbon.position.set(0, -0.32, 0);
      diffGroup.add(ribbon);

      diffGroup.userData = {
        id: diffuser.id || `diffuser_${idx}`,
        ribbonMesh: ribbon,
        isActive: diffuser.state === 'ON'
      };

      this.diffusersGroup.add(diffGroup);
    });
  }

  // =========================================================================
  // ROOM GEOMETRY & ZONE BOUNDARY RECONSTRUCTION
  // =========================================================================
  reconstructRoom(roomData) {
    if (!roomData) return;

    this.currentRoomId = roomData.room_id || 'room_default';
    this.roomDim = roomData.dim || [6.0, 5.0, 3.0];
    const [L, W, H] = this.roomDim;

    // 1. Resize Floor & Grid
    this.floorMesh.geometry.dispose();
    this.floorMesh.geometry = new THREE.PlaneGeometry(L, W);
    this.floorMesh.geometry.rotateX(-Math.PI / 2);
    this.floorMesh.position.set(L / 2, 0, W / 2);

    this.heatmapMesh.geometry.dispose();
    this.heatmapMesh.geometry = new THREE.PlaneGeometry(L, W);
    this.heatmapMesh.geometry.rotateX(-Math.PI / 2);
    this.heatmapMesh.position.set(L / 2, 0.003, W / 2);

    // Adjust Camera Target
    if (this.controls) {
      this.controls.target.set(L / 2, 1.0, W / 2);
    }

    // 2. Build Semi-Transparent Architectural Cutaway Glass Walls
    this._buildGlassWalls(L, W, H);

    // 3. Rebuild Zones & Neon Boundaries
    this.zonesData = roomData.zones || {};
    this._buildZones(L, W, H);

    // 4. Build Static Workstation Furniture & Diffusers
    const staticAssets = roomData.static_assets || {};
    this._buildFurniture(staticAssets.desks || [], L, W);
    this._buildLinearDiffusers(staticAssets.diffusers || staticAssets.fans || []);

    // Redraw initial heatmap
    this._drawThermalHeatmap();
  }

  _buildGlassWalls(L, W, H) {
    // Clear existing walls
    while (this.roomGroup.children.length > 2) {
      const child = this.roomGroup.children[2];
      if (child.geometry) child.geometry.dispose();
      this.roomGroup.remove(child);
    }

    const wallSpecs = [
      // Back wall (Z = 0)
      { size: [L, H, 0.08], pos: [L / 2, H / 2, 0] },
      // Left wall (X = 0)
      { size: [0.08, H, W], pos: [0, H / 2, W / 2] },
      // Right cutaway wall (X = L, lower height for visibility)
      { size: [0.08, H * 0.45, W], pos: [L, (H * 0.45) / 2, W / 2] },
      // Front cutaway wall (Z = W, lower height for camera ingress)
      { size: [L, H * 0.45, 0.08], pos: [L / 2, (H * 0.45) / 2, W] }
    ];

    wallSpecs.forEach(spec => {
      const geo = new THREE.BoxGeometry(...spec.size);
      const wallMesh = new THREE.Mesh(geo, this.materials.glassWall);
      wallMesh.position.set(...spec.pos);
      this.roomGroup.add(wallMesh);

      // Wireframe edge lines
      const edges = new THREE.EdgesGeometry(geo);
      const wireframe = new THREE.LineSegments(edges, this.materials.glassEdge);
      wireframe.position.set(...spec.pos);
      this.roomGroup.add(wireframe);
    });
  }

  _buildZones(L, W, H) {
    this.zonesGroup.clear();

    Object.entries(this.zonesData).forEach(([zoneId, zone]) => {
      const poly = zone.poly || [];
      if (poly.length < 3) return;

      // Build glowing neon border lines for zone delineation
      const points = [];
      poly.forEach(pt => {
        const x = pt[0] * L;
        const z = pt[1] * W;
        points.push(new THREE.Vector3(x, 0.02, z));
      });
      // Close loop
      points.push(points[0].clone());

      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: zone.occupancy > 0 ? 0x00FFC2 : 0x06B6D4,
        linewidth: 2,
        transparent: true,
        opacity: 0.85
      });
      const zoneLine = new THREE.Line(lineGeo, lineMat);
      zoneLine.userData = { zoneId: zoneId };
      this.zonesGroup.add(zoneLine);

      // Zone Floating Label Sprite
      let cx = 0, cz = 0;
      poly.forEach(pt => { cx += pt[0]; cz += pt[1]; });
      cx = (cx / poly.length) * L;
      cz = (cz / poly.length) * W;

      const zLabel = this._createZoneLabelSprite(zoneId, zone);
      zLabel.position.set(cx, 0.25, cz);
      this.zonesGroup.add(zLabel);
    });
  }

  _createZoneLabelSprite(zoneId, zone) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const displayName = zoneId.replace(/_/g, ' ').toUpperCase();
    const occ = zone.occupancy || 0;
    const temp = zone.measured_temp_c ? `${zone.measured_temp_c}°C` : '23°C';

    ctx.fillStyle = 'rgba(10, 14, 22, 0.85)';
    ctx.strokeStyle = occ > 0 ? 'rgba(0, 255, 194, 0.7)' : 'rgba(74, 107, 138, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 56, 10);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 20px "JetBrains Mono", monospace';
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(displayName, 16, 32);

    ctx.font = '16px "JetBrains Mono", monospace';
    ctx.fillStyle = occ > 0 ? '#00FFC2' : '#8899A6';
    ctx.textAlign = 'right';
    ctx.fillText(`${occ} OCC | ${temp}`, 240, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(1.4, 0.35, 1.0);
    sprite.userData = { zoneId: zoneId };
    return sprite;
  }

  _buildFurniture(desks, L, W) {
    this.furnitureGroup.clear();

    desks.forEach((desk, idx) => {
      const poly = desk.polygon || [];
      if (poly.length >= 4) {
        const minX = Math.min(...poly.map(p => p[0]));
        const maxX = Math.max(...poly.map(p => p[0]));
        const minZ = Math.min(...poly.map(p => p[1]));
        const maxZ = Math.max(...poly.map(p => p[1]));

        const deskW = maxX - minX;
        const deskD = maxZ - minZ;
        const deskH = desk.elevation || 0.75;

        const deskGeo = new THREE.BoxGeometry(deskW, 0.05, deskD);
        const deskMesh = new THREE.Mesh(deskGeo, this.materials.deskSurface);
        deskMesh.position.set(minX + deskW / 2, deskH, minZ + deskD / 2);
        deskMesh.castShadow = true;
        deskMesh.receiveShadow = true;
        this.furnitureGroup.add(deskMesh);

        // Desk wireframe edge
        const edges = new THREE.EdgesGeometry(deskGeo);
        const wire = new THREE.LineSegments(edges, this.materials.deskEdge);
        wire.position.copy(deskMesh.position);
        this.furnitureGroup.add(wire);

        // Legs
        const legGeo = new THREE.CylinderGeometry(0.025, 0.025, deskH, 8);
        const legPositions = [
          [minX + 0.05, deskH / 2, minZ + 0.05],
          [maxX - 0.05, deskH / 2, minZ + 0.05],
          [minX + 0.05, deskH / 2, maxZ - 0.05],
          [maxX - 0.05, deskH / 2, maxZ - 0.05]
        ];
        legPositions.forEach(pos => {
          const leg = new THREE.Mesh(legGeo, this.materials.deskSurface);
          leg.position.set(...pos);
          this.furnitureGroup.add(leg);
        });

        // Stylized Laptop on Desk
        const laptopBase = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.015, 0.22), this.materials.screenBezel);
        laptopBase.position.set(minX + deskW / 2, deskH + 0.03, minZ + deskD / 2);
        this.furnitureGroup.add(laptopBase);

        const screen = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.20, 0.015), this.materials.screenGlow);
        screen.position.set(minX + deskW / 2, deskH + 0.13, minZ + deskD / 2 - 0.10);
        screen.rotation.x = -0.25;
        this.furnitureGroup.add(screen);
      }
    });
  }

  // =========================================================================
  // TELEMETRY UPDATE & SMOOTH ANTI-JITTER INTERPOLATION
  // =========================================================================
  updateState(telemetryData) {
    if (!telemetryData) return;

    // 1. Detect Room Switch
    if (telemetryData.room_id && telemetryData.room_id !== this.currentRoomId) {
      this.reconstructRoom({
        room_id: telemetryData.room_id,
        name: telemetryData.room_name,
        dim: telemetryData.room_dim || this.roomDim,
        zones: telemetryData.zones,
        static_assets: telemetryData.static_assets
      });
    }

    // 2. Synchronize Zones & Temperatures
    if (telemetryData.zones) {
      this.zonesData = telemetryData.zones;
    }

    // 3. Synchronize Avatars with Object Pooling & Homography Coordinates
    const persons = (telemetryData.cv && telemetryData.cv.persons) || [];
    const activeTrackIds = new Set();
    const [L, W] = this.roomDim;

    persons.forEach(p => {
      const trackId = p.track_id || 1;
      activeTrackIds.add(trackId);

      let avatarItem = this.activeAvatars.get(trackId);

      // If not yet active, allocate from pool
      if (!avatarItem) {
        avatarItem = this.avatarPool.find(a => !a.visible);
        if (avatarItem) {
          avatarItem.visible = true;
          avatarItem.trackId = trackId;
          avatarItem.group.visible = true;
          this.activeAvatars.set(trackId, avatarItem);

          // Initial snap to position
          const startX = p.metric_pos_2d ? p.metric_pos_2d[0] : (p.floor_pos_2d[0] * L);
          const startZ = p.metric_pos_2d ? p.metric_pos_2d[1] : (p.floor_pos_2d[1] * W);
          avatarItem.group.position.set(startX, 0, startZ);
          avatarItem.targetX = startX;
          avatarItem.targetZ = startZ;
        }
      }

      if (avatarItem) {
        // Set target coordinates for smooth anti-jitter lerp interpolation
        const targetX = p.metric_pos_2d ? p.metric_pos_2d[0] : (p.floor_pos_2d[0] * L);
        const targetZ = p.metric_pos_2d ? p.metric_pos_2d[1] : (p.floor_pos_2d[1] * W);
        avatarItem.targetX = Math.max(0.1, Math.min(L - 0.1, targetX));
        avatarItem.targetZ = Math.max(0.1, Math.min(W - 0.1, targetZ));
        avatarItem.confidence = p.confidence || 0.95;

        this._updateAvatarLabel(avatarItem, `ID #${trackId}`, avatarItem.confidence);
      }
    });

    // Recycle avatars that are no longer detected
    this.activeAvatars.forEach((avatarItem, trackId) => {
      if (!activeTrackIds.has(trackId)) {
        avatarItem.visible = false;
        avatarItem.group.visible = false;
        avatarItem.trackId = null;
        this.activeAvatars.delete(trackId);
      }
    });

    // 4. Update Linear Diffuser Airflow States
    this.diffusersGroup.children.forEach(diffGroup => {
      const dData = diffGroup.userData;
      if (dData && dData.ribbonMesh) {
        // Find if zone relay is open or closed
        let isRelayActive = true;
        Object.values(this.zonesData).forEach(z => {
          if (z.relays && (z.relays.hvac === 'OPEN' || z.relays.diffuser === 'OPEN' || z.relays.fan === 'OPEN')) {
            isRelayActive = false;
          }
        });
        dData.ribbonMesh.visible = isRelayActive;
      }
    });

    // 5. Redraw Thermal Ground Heatmap
    this._drawThermalHeatmap();
  }

  // =========================================================================
  // CAMERA PRESETS & VIEW CONTROLS
  // =========================================================================
  setCameraView(viewName) {
    if (!this.camera || !this.controls) return;

    const [L, W, H] = this.roomDim;
    const target = new THREE.Vector3(L / 2, 0.8, W / 2);

    switch (viewName.toLowerCase()) {
      case 'perspective':
        this._tweenCamera(new THREE.Vector3(L * 1.5, H * 2.8, W * 1.8), target);
        break;
      case 'isometric':
        this._tweenCamera(new THREE.Vector3(L * 1.6, H * 2.5, W * 1.6), target);
        break;
      case 'topdown':
        this._tweenCamera(new THREE.Vector3(L / 2, H * 3.5, W / 2 + 0.01), target);
        break;
      case 'zone_a':
        this._tweenCamera(new THREE.Vector3(L * 0.35, H * 1.5, W * 0.9), new THREE.Vector3(L * 0.25, 0.5, W * 0.35));
        break;
      case 'zone_b':
        this._tweenCamera(new THREE.Vector3(L * 0.85, H * 1.5, W * 0.9), new THREE.Vector3(L * 0.75, 0.5, W * 0.35));
        break;
    }
  }

  _tweenCamera(toPos, toTarget) {
    // Smooth lerp transition
    const startPos = this.camera.position.clone();
    const startTarget = this.controls.target.clone();
    let progress = 0;

    const animateTransition = () => {
      progress += 0.05;
      if (progress <= 1.0) {
        this.camera.position.lerpVectors(startPos, toPos, progress);
        this.controls.target.lerpVectors(startTarget, toTarget, progress);
        requestAnimationFrame(animateTransition);
      } else {
        this.camera.position.copy(toPos);
        this.controls.target.copy(toTarget);
      }
    };
    animateTransition();
  }

  // =========================================================================
  // RENDER LOOP & LIFECYCLE MANAGEMENT
  // =========================================================================
  _animate() {
    this.animFrameId = requestAnimationFrame(() => this._animate());

    const delta = this.clock.getDelta();
    const elapsed = this.clock.getElapsedTime();

    // 1. Smooth Anti-Jitter Lerp Interpolation for Avatars
    const LERP_FACTOR = 0.12;
    this.activeAvatars.forEach(avatarItem => {
      if (avatarItem.visible) {
        // Interpolate (X, Z) on floor plane
        avatarItem.group.position.x += (avatarItem.targetX - avatarItem.group.position.x) * LERP_FACTOR;
        avatarItem.group.position.z += (avatarItem.targetZ - avatarItem.group.position.z) * LERP_FACTOR;
        avatarItem.group.position.y = 0; // strictly anchored to floor

        // Pulse Holographic Presence Ring
        const pulse = 1.0 + 0.08 * Math.sin(elapsed * 4.0);
        avatarItem.ringMesh.scale.set(pulse, pulse, pulse);
        avatarItem.discMesh.material.opacity = 0.10 + 0.05 * Math.sin(elapsed * 4.0);
      }
    });

    // 2. Airflow Diffuser Subtle Wave Animation
    this.diffusersGroup.children.forEach(diffGroup => {
      const ribbon = diffGroup.userData.ribbonMesh;
      if (ribbon && ribbon.visible) {
        ribbon.material.opacity = 0.20 + 0.12 * Math.sin(elapsed * 5.0);
      }
    });

    // 3. Update Controls & Render
    if (this.controls) {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
  }

  _setupEventListeners() {
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);

    this._onClick = (e) => this._handlePointerClick(e);
    this.renderer.domElement.addEventListener('click', this._onClick);
  }

  _handlePointerClick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Check click on zones
    const zoneIntersects = this.raycaster.intersectObjects(this.zonesGroup.children, true);
    if (zoneIntersects.length > 0) {
      const obj = zoneIntersects[0].object;
      const zoneId = obj.userData ? obj.userData.zoneId : null;
      if (zoneId && this.onZoneClick) {
        this.onZoneClick(zoneId);
        return;
      }
    }

    // Check click on avatars
    const avatarIntersects = this.raycaster.intersectObjects(this.avatarsGroup.children, true);
    if (avatarIntersects.length > 0) {
      let curr = avatarIntersects[0].object;
      while (curr && curr !== this.avatarsGroup) {
        if (curr.userData && curr.userData.trackId) {
          if (this.onAvatarClick) this.onAvatarClick(curr.userData.trackId);
          return;
        }
        curr = curr.parent;
      }
    }
  }

  resize() {
    if (!this.container || !this.renderer || !this.camera) return;
    this.width = this.container.clientWidth || 800;
    this.height = this.container.clientHeight || 500;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
  }

  dispose() {
    // Cancel render loop
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    // Remove listeners
    window.removeEventListener('resize', this._onResize);
    if (this.renderer && this.renderer.domElement) {
      this.renderer.domElement.removeEventListener('click', this._onClick);
    }

    // Dispose Geometries, Materials & Textures
    const disposeObj = (obj) => {
      if (!obj) return;
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
      if (obj.children) {
        obj.children.forEach(disposeObj);
      }
    };

    disposeObj(this.scene);

    if (this.heatmapTexture) this.heatmapTexture.dispose();
    if (this.controls) this.controls.dispose();

    // Force WebGL Context Loss to prevent GPU memory leaks
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.forceContextLoss();
      if (this.renderer.domElement && this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }

    this.scene = null;
    this.camera = null;
    this.renderer = null;
  }
}

// Global Export
if (typeof window !== 'undefined') {
  window.DigitalTwinScene = DigitalTwinScene;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DigitalTwinScene;
}
