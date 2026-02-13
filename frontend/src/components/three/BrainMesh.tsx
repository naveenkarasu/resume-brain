import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { useTexture } from '@react-three/drei';
import * as THREE from 'three';
import type { AppPhase } from '../../store/analysisStore';
import { ringState } from './ScoreRing';

interface Props {
  phase: AppPhase;
  score: number;
}

// --- Shape config + random selection (module level) ---
interface ShapeConfig {
  name: string;
  create: () => THREE.BufferGeometry;
  textured: boolean;
  texturePath?: string;
}

const SHAPES: ShapeConfig[] = [
  // Textured planets
  { name: 'earth',   create: () => new THREE.SphereGeometry(1.5, 64, 32), textured: true, texturePath: '2k_earth_daymap.jpg' },
  { name: 'mars',    create: () => new THREE.SphereGeometry(1.5, 64, 32), textured: true, texturePath: '2k_mars.jpg' },
  { name: 'jupiter', create: () => new THREE.SphereGeometry(1.5, 64, 32), textured: true, texturePath: '2k_jupiter.jpg' },
  { name: 'moon',    create: () => new THREE.SphereGeometry(1.5, 64, 32), textured: true, texturePath: '2k_moon.jpg' },
  // Procedural shapes
  { name: 'knot',      create: () => new THREE.TorusKnotGeometry(1.1, 0.4, 64, 16, 2, 3), textured: false },
  { name: 'crystal',   create: () => new THREE.DodecahedronGeometry(1.5, 2), textured: false },
  { name: 'icosphere', create: () => new THREE.IcosahedronGeometry(1.5, 3), textured: false },
];
const SHAPE_INDEX = Math.floor(Math.random() * SHAPES.length);
const selectedShape = SHAPES[SHAPE_INDEX];

// 1x1 white pixel PNG as data URL fallback for non-textured shapes
const FALLBACK_TEX = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg==';

// --- Dual grid geometry (works for any shape) ---
function createDualGridGeometry(sourceGeo: THREE.BufferGeometry): THREE.BufferGeometry {
  const posAttr = sourceGeo.attributes.position;

  // Build face list — works for both indexed and non-indexed geometry
  const faceCount = sourceGeo.index
    ? sourceGeo.index.count / 3
    : posAttr.count / 3;

  const getIdx = (f: number, v: number) =>
    sourceGeo.index ? sourceGeo.index.getX(f * 3 + v) : f * 3 + v;

  const faces: [number, number, number][] = [];
  for (let i = 0; i < faceCount; i++) {
    faces.push([getIdx(i, 0), getIdx(i, 1), getIdx(i, 2)]);
  }

  // Non-indexed geometry duplicates vertices per face.
  // Merge vertices at the same position so we can find shared faces.
  const EPS = 1e-4;
  const hashPos = (x: number, y: number, z: number) =>
    `${Math.round(x / EPS)},${Math.round(y / EPS)},${Math.round(z / EPS)}`;

  const vertexMap = new Map<string, number>(); // posHash → canonical index
  const canonical = new Int32Array(posAttr.count);
  for (let i = 0; i < posAttr.count; i++) {
    const key = hashPos(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
    if (!vertexMap.has(key)) vertexMap.set(key, i);
    canonical[i] = vertexMap.get(key)!;
  }

  // Remap faces to canonical vertex indices
  const cFaces: [number, number, number][] = faces.map(([a, b, c]) => [
    canonical[a], canonical[b], canonical[c],
  ]);

  // Compute face centroids (raw — works for any shape)
  const centroids: THREE.Vector3[] = [];
  for (const [a, b, c] of faces) {
    const cx = (posAttr.getX(a) + posAttr.getX(b) + posAttr.getX(c)) / 3;
    const cy = (posAttr.getY(a) + posAttr.getY(b) + posAttr.getY(c)) / 3;
    const cz = (posAttr.getZ(a) + posAttr.getZ(b) + posAttr.getZ(c)) / 3;
    centroids.push(new THREE.Vector3(cx, cy, cz));
  }

  // Map each canonical vertex → surrounding face indices
  const vertexFaces = new Map<number, number[]>();
  for (let fi = 0; fi < cFaces.length; fi++) {
    for (const vi of cFaces[fi]) {
      if (!vertexFaces.has(vi)) vertexFaces.set(vi, []);
      vertexFaces.get(vi)!.push(fi);
    }
  }

  // For each vertex, sort surrounding centroids by angle, connect consecutive
  const linePositions: number[] = [];
  const addedEdges = new Set<string>();

  for (const [vi, faceIndices] of vertexFaces) {
    const vx = posAttr.getX(vi);
    const vy = posAttr.getY(vi);
    const vz = posAttr.getZ(vi);
    const vertexVec = new THREE.Vector3(vx, vy, vz);

    // Compute vertex normal by averaging face normals of surrounding faces
    const normal = new THREE.Vector3();
    const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3();
    for (const fi of faceIndices) {
      const [fa, fb, fc] = faces[fi];
      _a.set(posAttr.getX(fa), posAttr.getY(fa), posAttr.getZ(fa));
      _b.set(posAttr.getX(fb), posAttr.getY(fb), posAttr.getZ(fb));
      _c.set(posAttr.getX(fc), posAttr.getY(fc), posAttr.getZ(fc));
      _b.sub(_a);
      _c.sub(_a);
      normal.add(new THREE.Vector3().crossVectors(_b, _c));
    }
    normal.normalize();

    // Local tangent-plane basis
    const up = Math.abs(normal.y) < 0.99
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
    const t1 = new THREE.Vector3().crossVectors(normal, up).normalize();
    const t2 = new THREE.Vector3().crossVectors(normal, t1).normalize();

    // Sort centroids by angle in tangent plane
    const sorted = faceIndices
      .map((fi) => {
        const diff = centroids[fi].clone().sub(vertexVec);
        return { fi, angle: Math.atan2(diff.dot(t2), diff.dot(t1)) };
      })
      .sort((a, b) => a.angle - b.angle);

    // Connect consecutive centroids with line segments
    for (let i = 0; i < sorted.length; i++) {
      const a = sorted[i].fi;
      const b = sorted[(i + 1) % sorted.length].fi;
      const edgeKey = Math.min(a, b) + '-' + Math.max(a, b);
      if (addedEdges.has(edgeKey)) continue;
      addedEdges.add(edgeKey);

      const ca = centroids[a];
      const cb = centroids[b];
      linePositions.push(ca.x, ca.y, ca.z, cb.x, cb.y, cb.z);
    }
  }

  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(linePositions);
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  // Compute normals: average face normals for each centroid
  const faceNormals: THREE.Vector3[] = [];
  const _fa = new THREE.Vector3(), _fb = new THREE.Vector3(), _fc = new THREE.Vector3();
  for (const [a, b, c] of faces) {
    _fa.set(posAttr.getX(a), posAttr.getY(a), posAttr.getZ(a));
    _fb.set(posAttr.getX(b), posAttr.getY(b), posAttr.getZ(b));
    _fc.set(posAttr.getX(c), posAttr.getY(c), posAttr.getZ(c));
    _fb.sub(_fa);
    _fc.sub(_fa);
    faceNormals.push(new THREE.Vector3().crossVectors(_fb, _fc).normalize());
  }

  // Each line segment connects two centroids; use face normal as vertex normal
  const normals = new Float32Array(positions.length);
  const centroidToNormal = new Map<string, THREE.Vector3>();
  for (let fi = 0; fi < centroids.length; fi++) {
    const c = centroids[fi];
    const key = `${Math.round(c.x * 1e4)},${Math.round(c.y * 1e4)},${Math.round(c.z * 1e4)}`;
    centroidToNormal.set(key, faceNormals[fi]);
  }

  for (let i = 0; i < positions.length; i += 3) {
    const key = `${Math.round(positions[i] * 1e4)},${Math.round(positions[i + 1] * 1e4)},${Math.round(positions[i + 2] * 1e4)}`;
    const n = centroidToNormal.get(key);
    if (n) {
      normals[i] = n.x;
      normals[i + 1] = n.y;
      normals[i + 2] = n.z;
    } else {
      // Fallback: normalized position
      const v = new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]).normalize();
      normals[i] = v.x;
      normals[i + 1] = v.y;
      normals[i + 2] = v.z;
    }
  }
  geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return geo;
}

// --- Shaders with Fresnel rim glow + scan lines + planet texture ---
const gradientVertexShader = `
  varying vec3 vWorldPos;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec2 vUv;

  void main() {
    vUv = uv;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize(cameraPosition - vWorldPos);
    gl_Position = projectionMatrix * viewMatrix * vec4(vWorldPos, 1.0);
  }
`;

const gradientFragmentShader = `
  uniform float greenArc;
  uniform float rotationY;
  uniform float uOpacity;
  uniform vec3 greenColor;
  uniform vec3 redColor;
  uniform float uFresnelPower;
  uniform float uFresnelIntensity;
  uniform float uTime;
  uniform float uScanLineSpeed;
  uniform float uScanLineCount;
  uniform sampler2D uPlanetTexture;
  uniform float uHasTexture;
  uniform float uTextureIntensity;

  varying vec3 vWorldPos;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec2 vUv;

  void main() {
    // Angle of fragment in XZ plane, synced with ring arc direction
    float angle = atan(vWorldPos.z, vWorldPos.x) + rotationY;
    angle = mod(angle + 6.28318530, 6.28318530);

    // Smooth blend at green/red boundary
    float blend = smoothstep(greenArc - 0.15, greenArc + 0.15, angle);
    vec3 gradientColor = mix(greenColor, redColor, blend);

    // Sample planet texture, blend with gradient
    vec3 texColor = texture2D(uPlanetTexture, vUv).rgb;
    float texMix = uTextureIntensity * uHasTexture;
    vec3 color = mix(gradientColor, texColor, texMix);

    // Subtle gradient tint even when texture dominates (keeps color coding visible)
    color = mix(color, color * (gradientColor * 0.4 + 0.6), texMix * 0.3);

    // Fresnel rim glow
    float fresnel = 1.0 - max(dot(normalize(vNormal), normalize(vViewDir)), 0.0);
    fresnel = pow(fresnel, uFresnelPower);
    color += color * uFresnelIntensity * fresnel;

    // Static scan lines — very faint horizontal stripes
    float scanLine = sin(vWorldPos.y * uScanLineCount) * 0.5 + 0.5;
    color *= 1.0 - scanLine * 0.08;

    // Moving sweep line — wide soft glow sweeping up/down
    float sweepY = sin(uTime * uScanLineSpeed) * 1.5;
    float sweep = 1.0 - smoothstep(0.0, 0.5, abs(vWorldPos.y - sweepY));
    color += color * sweep * 0.25;

    gl_FragColor = vec4(color, uOpacity);
  }
`;

// Dummy 1x1 texture for wireframe material (never displayed but needed for uniform)
const dummyTexture = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1, THREE.RGBAFormat);
dummyTexture.needsUpdate = true;

export default function BrainMesh({ phase, score }: Props) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const wireRef = useRef<THREE.LineSegments>(null!);

  // Animated refs for smooth phase transitions
  const currentTextureIntensity = useRef(selectedShape.textured ? 0.65 : 0);
  const currentSolidOpacity = useRef(selectedShape.textured ? 0.30 : 0.08);
  const currentWireOpacity = useRef(selectedShape.textured ? 0.15 : 0.50);
  const currentFresnelIntensity = useRef(selectedShape.textured ? 0.3 : 0.8);
  const currentScanLineCount = useRef(selectedShape.textured ? 40 : 80);
  const currentDisplacementAmp = useRef(selectedShape.textured ? 0.02 : 0.06);
  const currentDisplacementSpeed = useRef(selectedShape.textured ? 0.4 : 0.8);

  // Create shape geometry once (random per page load)
  const solidGeo = useMemo(() => {
    const geo = selectedShape.create();
    geo.computeVertexNormals();
    return geo;
  }, []);

  const originalPositions = useMemo(
    () => solidGeo.attributes.position.array.slice() as Float32Array,
    [solidGeo]
  );

  const originalNormals = useMemo(
    () => solidGeo.attributes.normal.array.slice() as Float32Array,
    [solidGeo]
  );

  // Use IcosahedronGeometry for wireframe on textured shapes (SphereGeo is too dense)
  const wireframeSourceGeo = useMemo(() => {
    if (selectedShape.textured) {
      return new THREE.IcosahedronGeometry(1.5, 3);
    }
    return solidGeo;
  }, [solidGeo]);

  // Pre-compute dual grid wireframe geometry
  const hexGridGeo = useMemo(() => createDualGridGeometry(wireframeSourceGeo), [wireframeSourceGeo]);

  // Load planet texture (or fallback for non-textured shapes)
  const texturePath = selectedShape.textured
    ? `${import.meta.env.BASE_URL}textures/planets/${selectedShape.texturePath}`
    : FALLBACK_TEX;
  const planetTexture = useTexture(texturePath);

  const GREEN = useMemo(() => new THREE.Color('#22c55e'), []);
  const RED = useMemo(() => new THREE.Color('#ef4444'), []);
  const BLUE = useMemo(() => new THREE.Color('#3b82f6'), []);

  const isTextured = selectedShape.textured;

  // Shader material for solid globe fill
  const solidMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: gradientVertexShader,
        fragmentShader: gradientFragmentShader,
        uniforms: {
          greenArc: { value: 0 },
          rotationY: { value: 0 },
          uOpacity: { value: isTextured ? 0.30 : 0.08 },
          greenColor: { value: GREEN },
          redColor: { value: RED },
          uFresnelPower: { value: 2.0 },
          uFresnelIntensity: { value: isTextured ? 0.3 : 0.8 },
          uTime: { value: 0 },
          uScanLineSpeed: { value: 1.5 },
          uScanLineCount: { value: isTextured ? 40.0 : 80.0 },
          uPlanetTexture: { value: planetTexture },
          uHasTexture: { value: isTextured ? 1.0 : 0.0 },
          uTextureIntensity: { value: isTextured ? 0.65 : 0.0 },
        },
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [GREEN, RED, isTextured, planetTexture]
  );

  // Shader material for hex-grid wireframe (never textured)
  const wireMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: gradientVertexShader,
        fragmentShader: gradientFragmentShader,
        uniforms: {
          greenArc: { value: 0 },
          rotationY: { value: 0 },
          uOpacity: { value: isTextured ? 0.15 : 0.5 },
          greenColor: { value: GREEN },
          redColor: { value: RED },
          uFresnelPower: { value: 2.0 },
          uFresnelIntensity: { value: 0.5 },
          uTime: { value: 0 },
          uScanLineSpeed: { value: 1.5 },
          uScanLineCount: { value: isTextured ? 40.0 : 80.0 },
          uPlanetTexture: { value: dummyTexture },
          uHasTexture: { value: 0.0 },
          uTextureIntensity: { value: 0.0 },
        },
        transparent: true,
        depthWrite: false,
      }),
    [GREEN, RED, isTextured]
  );

  // Dispose GPU resources on unmount
  useEffect(() => {
    return () => {
      solidGeo.dispose();
      hexGridGeo.dispose();
      if (wireframeSourceGeo !== solidGeo) wireframeSourceGeo.dispose();
      solidMat.dispose();
      wireMat.dispose();
    };
  }, [solidGeo, hexGridGeo, wireframeSourceGeo, solidMat, wireMat]);

  useFrame(({ clock }, delta) => {
    const t = clock.elapsedTime;
    const mesh = meshRef.current;
    const geo = mesh.geometry;
    const pos = geo.attributes.position;

    // --- Phase transition targets ---
    let targetTextureIntensity: number;
    let targetSolidOpacity: number;
    let targetWireOpacity: number;
    let targetFresnelIntensity: number;
    let targetScanLineCount: number;
    let targetDisplacementAmp: number;
    let targetDisplacementSpeed: number;

    if (isTextured) {
      if (phase === 'idle') {
        targetTextureIntensity = 0.65;
        targetSolidOpacity = 0.30;
        targetWireOpacity = 0.15;
        targetFresnelIntensity = 0.3;
        targetScanLineCount = 40;
        targetDisplacementAmp = 0.02;
        targetDisplacementSpeed = 0.4;
      } else if (phase === 'loading') {
        targetTextureIntensity = 0.15;
        targetSolidOpacity = 0.15;
        targetWireOpacity = 0.60;
        targetFresnelIntensity = 0.8;
        targetScanLineCount = 80;
        targetDisplacementAmp = 0.15;
        targetDisplacementSpeed = 3.0;
      } else {
        // results
        targetTextureIntensity = 0.15;
        targetSolidOpacity = 0.08;
        targetWireOpacity = 0.50;
        targetFresnelIntensity = 0.8;
        targetScanLineCount = 80;
        targetDisplacementAmp = 0.06;
        targetDisplacementSpeed = 0.8;
      }
    } else {
      // Procedural shapes — only displacement changes during loading
      targetTextureIntensity = 0;
      targetSolidOpacity = 0.08;
      targetWireOpacity = 0.50;
      targetFresnelIntensity = 0.8;
      targetScanLineCount = 80;
      targetDisplacementAmp = phase === 'loading' ? 0.15 : 0.06;
      targetDisplacementSpeed = phase === 'loading' ? 3.0 : 0.8;
    }

    // Smooth lerp all animated values
    const lerpSpeed = 2.0;
    const lerpFactor = 1 - Math.exp(-lerpSpeed * delta);

    currentTextureIntensity.current += (targetTextureIntensity - currentTextureIntensity.current) * lerpFactor;
    currentSolidOpacity.current += (targetSolidOpacity - currentSolidOpacity.current) * lerpFactor;
    currentWireOpacity.current += (targetWireOpacity - currentWireOpacity.current) * lerpFactor;
    currentFresnelIntensity.current += (targetFresnelIntensity - currentFresnelIntensity.current) * lerpFactor;
    currentScanLineCount.current += (targetScanLineCount - currentScanLineCount.current) * lerpFactor;
    currentDisplacementAmp.current += (targetDisplacementAmp - currentDisplacementAmp.current) * lerpFactor;
    currentDisplacementSpeed.current += (targetDisplacementSpeed - currentDisplacementSpeed.current) * lerpFactor;

    // Apply animated values to solid material
    solidMat.uniforms.uOpacity.value = currentSolidOpacity.current;
    solidMat.uniforms.uFresnelIntensity.value = currentFresnelIntensity.current;
    solidMat.uniforms.uScanLineCount.value = currentScanLineCount.current;
    solidMat.uniforms.uTextureIntensity.value = currentTextureIntensity.current;

    // Apply animated values to wire material
    wireMat.uniforms.uOpacity.value = currentWireOpacity.current;
    wireMat.uniforms.uScanLineCount.value = currentScanLineCount.current;

    // Vertex displacement animation (solid mesh only)
    const speed = currentDisplacementSpeed.current;
    const amplitude = currentDisplacementAmp.current;

    for (let i = 0; i < pos.count; i++) {
      const ox = originalPositions[i * 3];
      const oy = originalPositions[i * 3 + 1];
      const oz = originalPositions[i * 3 + 2];
      const nx = originalNormals[i * 3];
      const ny = originalNormals[i * 3 + 1];
      const nz = originalNormals[i * 3 + 2];
      const offset =
        Math.sin(t * speed + ox * 2) * amplitude +
        Math.cos(t * speed * 0.7 + oy * 3) * amplitude * 0.5;
      pos.setXYZ(i, ox + nx * offset, oy + ny * offset, oz + nz * offset);
    }
    pos.needsUpdate = true;

    // Update shader uniforms — sync green arc + rotation with ring
    const greenArc =
      phase === 'results'
        ? (score / 100) * Math.PI * 2
        : Math.PI * 2; // Full blue-ish before results
    const rotY = ringState.rotationY;

    solidMat.uniforms.greenArc.value = greenArc;
    solidMat.uniforms.rotationY.value = rotY;
    solidMat.uniforms.uTime.value = t;
    wireMat.uniforms.greenArc.value = greenArc;
    wireMat.uniforms.rotationY.value = rotY;
    wireMat.uniforms.uTime.value = t;

    // Before results: use blue for both
    if (phase !== 'results') {
      solidMat.uniforms.greenColor.value = BLUE;
      solidMat.uniforms.redColor.value = BLUE;
      wireMat.uniforms.greenColor.value = BLUE;
      wireMat.uniforms.redColor.value = BLUE;
    } else {
      solidMat.uniforms.greenColor.value = GREEN;
      solidMat.uniforms.redColor.value = RED;
      wireMat.uniforms.greenColor.value = GREEN;
      wireMat.uniforms.redColor.value = RED;
    }

    // Globe does NOT rotate in Y — the shader handles the gradient rotation
    // via ringState.rotationY. This keeps globe, ring, and keywords perfectly aligned.
    mesh.rotation.y = 0;
    mesh.rotation.x = Math.sin(t * 0.2) * 0.1;
    wireRef.current.rotation.copy(mesh.rotation);

    // Pulse during loading
    if (phase === 'loading') {
      const pulse = 1 + Math.sin(t * 4) * 0.05;
      mesh.scale.setScalar(pulse);
      wireRef.current.scale.setScalar(pulse);
    } else {
      mesh.scale.setScalar(1);
      wireRef.current.scale.setScalar(1);
    }
  });

  return (
    <group>
      <mesh ref={meshRef} geometry={solidGeo}>
        <primitive object={solidMat} attach="material" />
      </mesh>
      <lineSegments ref={wireRef} geometry={hexGridGeo}>
        <primitive object={wireMat} attach="material" />
      </lineSegments>
    </group>
  );
}
