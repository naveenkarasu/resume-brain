import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const COUNT = 600;

function createParticleGeometry() {
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(COUNT * 3);
  for (let i = 0; i < COUNT; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 20;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  return geo;
}

export default function BackgroundParticles() {
  const ref = useRef<THREE.Points>(null!);
  const [geometry] = useState(createParticleGeometry);

  useFrame((_, delta) => {
    const pos = ref.current.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < COUNT; i++) {
      const y = pos.getY(i) + delta * 0.15;
      pos.setY(i, y > 10 ? -10 : y);
    }
    pos.needsUpdate = true;
    ref.current.rotation.y += delta * 0.01;
  });

  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial
        color="#4488ff"
        size={0.03}
        transparent
        opacity={0.4}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}
