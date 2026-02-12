import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useTexture } from '@react-three/drei';
import * as THREE from 'three';

export default function SpaceSkybox() {
  const ref = useRef<THREE.Mesh>(null!);
  const texture = useTexture('/textures/2k_stars_milky_way.jpg');

  useFrame((_, delta) => {
    ref.current.rotation.y += delta * 0.002;
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[50, 32, 32]} />
      <meshBasicMaterial map={texture} side={THREE.BackSide} />
    </mesh>
  );
}
