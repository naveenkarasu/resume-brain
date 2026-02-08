import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { AppPhase } from '../../store/analysisStore';

interface Props {
  phase: AppPhase;
  score: number;
}

function getScoreColor(score: number): THREE.Color {
  // Smooth gradient: red(0) -> orange(40) -> yellow(55) -> lime(70) -> green(85+)
  const red = new THREE.Color('#ef4444');
  const orange = new THREE.Color('#f97316');
  const yellow = new THREE.Color('#eab308');
  const lime = new THREE.Color('#84cc16');
  const green = new THREE.Color('#22c55e');

  if (score >= 85) return green;
  if (score >= 70) return new THREE.Color().lerpColors(lime, green, (score - 70) / 15);
  if (score >= 55) return new THREE.Color().lerpColors(yellow, lime, (score - 55) / 15);
  if (score >= 40) return new THREE.Color().lerpColors(orange, yellow, (score - 40) / 15);
  return new THREE.Color().lerpColors(red, orange, Math.max(0, score) / 40);
}

function getTargetColor(phase: AppPhase, score: number): THREE.Color {
  if (phase !== 'results') return new THREE.Color('#3b82f6');
  return getScoreColor(score);
}

export default function BrainMesh({ phase, score }: Props) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const wireRef = useRef<THREE.LineSegments>(null!);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null!);
  const wireMaterialRef = useRef<THREE.LineBasicMaterial>(null!);

  const originalPositions = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(1.5, 3);
    return geo.attributes.position.array.slice() as Float32Array;
  }, []);

  const currentColor = useRef(new THREE.Color('#3b82f6'));

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    const mesh = meshRef.current;
    const geo = mesh.geometry;
    const pos = geo.attributes.position;

    // Vertex displacement
    const speed = phase === 'loading' ? 3 : 0.8;
    const amplitude = phase === 'loading' ? 0.15 : 0.06;

    for (let i = 0; i < pos.count; i++) {
      const ox = originalPositions[i * 3];
      const oy = originalPositions[i * 3 + 1];
      const oz = originalPositions[i * 3 + 2];
      const offset = Math.sin(t * speed + ox * 2) * amplitude +
        Math.cos(t * speed * 0.7 + oy * 3) * amplitude * 0.5;
      pos.setXYZ(i, ox + ox * offset, oy + oy * offset, oz + oz * offset);
    }
    pos.needsUpdate = true;

    // Smooth color transition
    const target = getTargetColor(phase, score);
    currentColor.current.lerp(target, 0.02);
    materialRef.current.color.copy(currentColor.current);
    wireMaterialRef.current.color.copy(currentColor.current);

    // Rotation
    mesh.rotation.y += 0.003;
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
        <meshBasicMaterial
          ref={materialRef}
          transparent
          opacity={0.08}
          side={THREE.DoubleSide}
        />
      </mesh>
      <lineSegments ref={wireRef}>
        <wireframeGeometry args={[new THREE.IcosahedronGeometry(1.5, 3)]} />
        <lineBasicMaterial
          ref={wireMaterialRef}
          transparent
          opacity={0.5}
        />
      </lineSegments>
    </group>
  );
}
