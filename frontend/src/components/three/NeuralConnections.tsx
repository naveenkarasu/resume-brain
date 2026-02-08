import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface Props {
  active: boolean;
  score?: number;
}

const CONNECTION_COUNT = 12;

function createConnectionGeometries() {
  return Array.from({ length: CONNECTION_COUNT }, (_, i) => {
    const angle = (i / CONNECTION_COUNT) * Math.PI * 2;
    const points = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(
        Math.cos(angle) * 1.5,
        (Math.random() - 0.5) * 1,
        Math.sin(angle) * 1.5
      ),
      new THREE.Vector3(
        Math.cos(angle) * 3,
        (Math.random() - 0.5) * 1.5,
        Math.sin(angle) * 3
      ),
    ];
    return new THREE.BufferGeometry().setFromPoints(
      new THREE.CatmullRomCurve3(points).getPoints(20)
    );
  });
}

function getLineColor(score: number): string {
  if (score >= 85) return '#22c55e';
  if (score >= 70) return '#84cc16';
  if (score >= 55) return '#eab308';
  if (score >= 40) return '#f97316';
  if (score > 0) return '#ef4444';
  return '#6366f1'; // Default indigo when no score
}

export default function NeuralConnections({ active, score = 0 }: Props) {
  const groupRef = useRef<THREE.Group>(null!);
  const materialRef = useRef<THREE.LineBasicMaterial>(null!);
  const [lines] = useState(createConnectionGeometries);

  useFrame(({ clock }) => {
    if (!materialRef.current) return;
    const targetOpacity = active ? 0.3 : 0;
    materialRef.current.opacity +=
      (targetOpacity - materialRef.current.opacity) * 0.05;

    // Update color based on score
    const targetColor = new THREE.Color(getLineColor(score));
    materialRef.current.color.lerp(targetColor, 0.03);

    if (groupRef.current) {
      groupRef.current.rotation.y = clock.elapsedTime * 0.15;
    }
  });

  return (
    <group ref={groupRef}>
      {lines.map((geo, i) => (
        <line key={i}>
          <primitive object={geo} attach="geometry" />
          <lineBasicMaterial
            ref={i === 0 ? materialRef : undefined}
            color="#6366f1"
            transparent
            opacity={0}
          />
        </line>
      ))}
    </group>
  );
}
