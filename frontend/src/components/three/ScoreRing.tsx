import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function getScoreColor(score: number): string {
  if (score >= 85) return '#22c55e';
  if (score >= 70) return '#84cc16';
  if (score >= 55) return '#eab308';
  if (score >= 40) return '#f97316';
  return '#ef4444';
}

interface Props {
  score: number;
  visible: boolean;
}

export default function ScoreRing({ score, visible }: Props) {
  const ref = useRef<THREE.Mesh>(null!);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null!);
  const currentArc = useRef(0);
  const currentColor = useRef(new THREE.Color('#3b82f6'));

  useFrame((_, delta) => {
    if (!ref.current) return;

    const targetArc = visible ? (score / 100) * Math.PI * 2 : 0;
    currentArc.current += (targetArc - currentArc.current) * delta * 2;

    // Recreate geometry with current arc
    const newGeo = new THREE.TorusGeometry(2.2, 0.04, 8, 64, currentArc.current);
    ref.current.geometry.dispose();
    ref.current.geometry = newGeo;

    // Smooth color transition
    if (materialRef.current) {
      const targetColor = new THREE.Color(getScoreColor(score));
      currentColor.current.lerp(targetColor, 0.03);
      materialRef.current.color.copy(currentColor.current);
    }

    ref.current.rotation.z += delta * 0.3;
  });

  return (
    <mesh ref={ref} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[2.2, 0.04, 8, 64, 0.01]} />
      <meshBasicMaterial ref={materialRef} color="#3b82f6" transparent opacity={0.7} />
    </mesh>
  );
}
