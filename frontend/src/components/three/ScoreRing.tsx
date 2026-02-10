import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface Props {
  score: number;
  visible: boolean;
}

// Shared state so ParticleField can sync with the ring
export const ringState = { rotationY: 0, greenArc: 0, score: 0 };

/**
 * Solid gradient score ring around the globe.
 * Green portion = match score, Red portion = remaining.
 * Ring sits horizontal (XZ plane) like Saturn's ring.
 * Rotates around Y axis to stay flat.
 */
export default function ScoreRing({ score, visible }: Props) {
  const greenRef = useRef<THREE.Mesh>(null!);
  const redRef = useRef<THREE.Mesh>(null!);
  const groupRef = useRef<THREE.Group>(null!);

  const currentGreenArc = useRef(0);
  const currentOpacity = useRef(0);

  const RADIUS = 2.3;
  const TUBE = 0.07;
  const RADIAL_SEGMENTS = 24;
  const ARC_SEGMENTS = 128;

  const greenMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#22c55e',
        emissive: '#22c55e',
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0,
        roughness: 0.3,
        metalness: 0.4,
      }),
    []
  );

  const redMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#ef4444',
        emissive: '#ef4444',
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0,
        roughness: 0.3,
        metalness: 0.4,
      }),
    []
  );

  useFrame((_, delta) => {
    if (!groupRef.current) return;

    // Animate opacity
    const targetOpacity = visible ? 1.0 : 0;
    currentOpacity.current += (targetOpacity - currentOpacity.current) * delta * 3;

    // Animate green arc
    const targetGreenArc = visible ? (score / 100) * Math.PI * 2 : 0;
    currentGreenArc.current += (targetGreenArc - currentGreenArc.current) * delta * 2;

    const greenArc = Math.max(currentGreenArc.current, 0.01);
    const redArc = Math.max(Math.PI * 2 - greenArc, 0.01);

    // Update green arc
    if (greenRef.current) {
      greenRef.current.geometry.dispose();
      greenRef.current.geometry = new THREE.TorusGeometry(
        RADIUS, TUBE, RADIAL_SEGMENTS, ARC_SEGMENTS, greenArc
      );
      greenMat.opacity = currentOpacity.current;
    }

    // Update red arc - starts where green ends
    if (redRef.current) {
      redRef.current.geometry.dispose();
      redRef.current.geometry = new THREE.TorusGeometry(
        RADIUS, TUBE, RADIAL_SEGMENTS, ARC_SEGMENTS, redArc
      );
      // The torus arc starts at angle 0 in its local XY plane.
      // After the mesh rotation [PI/2, 0, 0], the torus local Z-rotation
      // becomes a rotation in the XZ world plane. So we offset the red arc
      // by greenArc in the mesh's local Z rotation.
      redRef.current.rotation.set(Math.PI / 2, 0, greenArc);
      redMat.opacity = currentOpacity.current * 0.7;
    }

    // Rotate around Y to keep the ring horizontal (like Saturn's ring spinning)
    groupRef.current.rotation.y += delta * 0.15;

    // Export state for ParticleField sync
    ringState.rotationY = groupRef.current.rotation.y;
    ringState.greenArc = currentGreenArc.current;
    ringState.score = score;
  });

  return (
    <group ref={groupRef}>
      {/* Green arc (score portion) */}
      <mesh ref={greenRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[RADIUS, TUBE, RADIAL_SEGMENTS, ARC_SEGMENTS, 0.01]} />
        <primitive object={greenMat} attach="material" />
      </mesh>

      {/* Red arc (remaining portion) */}
      <mesh ref={redRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[RADIUS, TUBE, RADIAL_SEGMENTS, ARC_SEGMENTS, 0.01]} />
        <primitive object={redMat} attach="material" />
      </mesh>
    </group>
  );
}
