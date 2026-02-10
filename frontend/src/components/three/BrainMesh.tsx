import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { AppPhase } from '../../store/analysisStore';
import { ringState } from './ScoreRing';

interface Props {
  phase: AppPhase;
  score: number;
}

// Custom shader: green/red gradient split synced with ring arc
const gradientVertexShader = `
  varying vec3 vWorldPos;
  void main() {
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * viewMatrix * vec4(vWorldPos, 1.0);
  }
`;

const gradientFragmentShader = `
  uniform float greenArc;
  uniform float rotationY;
  uniform float uOpacity;
  uniform vec3 greenColor;
  uniform vec3 redColor;
  varying vec3 vWorldPos;

  void main() {
    // Angle of fragment in XZ plane, synced with ring arc direction
    // Ring green arc starts at world angle -rotationY, extends CCW by greenArc
    // So: normalized = worldAngle + rotationY maps ring start to 0
    float angle = atan(vWorldPos.z, vWorldPos.x) + rotationY;
    angle = mod(angle + 6.28318530, 6.28318530); // normalize to 0..2PI

    // Smooth blend at green/red boundary
    float blend = smoothstep(greenArc - 0.15, greenArc + 0.15, angle);
    vec3 color = mix(greenColor, redColor, blend);

    gl_FragColor = vec4(color, uOpacity);
  }
`;

export default function BrainMesh({ phase, score }: Props) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const wireRef = useRef<THREE.LineSegments>(null!);

  const originalPositions = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(1.5, 3);
    return geo.attributes.position.array.slice() as Float32Array;
  }, []);

  const GREEN = useMemo(() => new THREE.Color('#22c55e'), []);
  const RED = useMemo(() => new THREE.Color('#ef4444'), []);
  const BLUE = useMemo(() => new THREE.Color('#3b82f6'), []);

  // Shader material for solid globe fill
  const solidMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: gradientVertexShader,
        fragmentShader: gradientFragmentShader,
        uniforms: {
          greenArc: { value: 0 },
          rotationY: { value: 0 },
          uOpacity: { value: 0.08 },
          greenColor: { value: GREEN },
          redColor: { value: RED },
        },
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [GREEN, RED]
  );

  // Shader material for wireframe
  const wireMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: gradientVertexShader,
        fragmentShader: gradientFragmentShader,
        uniforms: {
          greenArc: { value: 0 },
          rotationY: { value: 0 },
          uOpacity: { value: 0.5 },
          greenColor: { value: GREEN },
          redColor: { value: RED },
        },
        transparent: true,
        depthWrite: false,
      }),
    [GREEN, RED]
  );

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const mesh = meshRef.current;
    const geo = mesh.geometry;
    const pos = geo.attributes.position;

    // Vertex displacement animation
    const speed = phase === 'loading' ? 3 : 0.8;
    const amplitude = phase === 'loading' ? 0.15 : 0.06;

    for (let i = 0; i < pos.count; i++) {
      const ox = originalPositions[i * 3];
      const oy = originalPositions[i * 3 + 1];
      const oz = originalPositions[i * 3 + 2];
      const offset =
        Math.sin(t * speed + ox * 2) * amplitude +
        Math.cos(t * speed * 0.7 + oy * 3) * amplitude * 0.5;
      pos.setXYZ(i, ox + ox * offset, oy + oy * offset, oz + oz * offset);
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
    wireMat.uniforms.greenArc.value = greenArc;
    wireMat.uniforms.rotationY.value = rotY;

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
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[1.5, 3]} />
        <primitive object={solidMat} attach="material" />
      </mesh>
      <lineSegments ref={wireRef}>
        <wireframeGeometry args={[new THREE.IcosahedronGeometry(1.5, 3)]} />
        <primitive object={wireMat} attach="material" />
      </lineSegments>
    </group>
  );
}
