import { useRef, useMemo, useState, useCallback } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { ringState } from './ScoreRing';

interface Keyword {
  text: string;
  matched: boolean;
}

interface Props {
  keywords: Keyword[];
  visible: boolean;
  score: number;
}

interface KeywordOrbProps extends Keyword {
  baseAngle: number;
  orbitRadius: number;
  index: number;
}

// Shared mutable state for ConstellationLines to read positions each frame
export const keywordPositions: { x: number; y: number; z: number; matched: boolean }[] = [];

/**
 * Single keyword floating OUTSIDE the ring, in the same horizontal plane.
 * Orbits at a slightly larger radius than the ring, synced with ring rotation.
 * Uses Billboard so text always faces the camera and stays readable.
 */
function KeywordOrb({ text, matched, baseAngle, orbitRadius, index }: KeywordOrbProps) {
  const ref = useRef<THREE.Group>(null!);
  const scaleRef = useRef(1);
  const [hovered, setHovered] = useState(false);

  const onOver = useCallback(() => {
    setHovered(true);
    document.body.style.cursor = 'pointer';
  }, []);
  const onOut = useCallback(() => {
    setHovered(false);
    document.body.style.cursor = 'auto';
  }, []);

  useFrame((_, delta) => {
    if (!ref.current) return;

    const worldAngle = baseAngle - ringState.rotationY;
    ref.current.position.x = Math.cos(worldAngle) * orbitRadius;
    ref.current.position.z = Math.sin(worldAngle) * orbitRadius;
    ref.current.position.y = 0;

    // Smooth scale on hover
    const target = hovered ? 1.6 : 1;
    scaleRef.current += (target - scaleRef.current) * Math.min(delta * 10, 1);
    ref.current.scale.setScalar(scaleRef.current);

    keywordPositions[index] = {
      x: ref.current.position.x,
      y: ref.current.position.y,
      z: ref.current.position.z,
      matched,
    };
  });

  const color = matched ? '#22c55e' : '#ef4444';
  const emissiveIntensity = hovered ? 1.5 : 0.8;

  return (
    <group ref={ref}>
      {/* Visible glowing dot */}
      <mesh>
        <sphereGeometry args={[0.05, 12, 12]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
        />
      </mesh>
      {/* Larger invisible hit area for easier hover */}
      <mesh visible={false} onPointerOver={onOver} onPointerOut={onOut}>
        <sphereGeometry args={[0.2, 8, 8]} />
        <meshBasicMaterial />
      </mesh>
      <Billboard follow lockX={false} lockY={false} lockZ={false}>
        <Text
          position={[0, 0.18, 0]}
          fontSize={0.13}
          color={color}
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.012}
          outlineColor="#000000"
          fontWeight={600}
        >
          {text}
        </Text>
      </Billboard>
    </group>
  );
}

/**
 * Keywords orbit OUTSIDE the ring, parallel to it (same Y=0 plane).
 * - Green keywords on the green arc side
 * - Red keywords on the red arc side
 * - All orbit at a larger radius so they don't overlap with the ring
 * - Billboard ensures text is always readable from any angle
 */
export default function ParticleField({ keywords, visible, score }: Props) {
  const { matchedKeywords, missingKeywords } = useMemo(() => {
    const matched = keywords.filter((k) => k.matched).slice(0, 10);
    const missing = keywords.filter((k) => !k.matched).slice(0, 10);
    return { matchedKeywords: matched, missingKeywords: missing };
  }, [keywords]);

  const keywordAngles = useMemo(() => {
    const angles: { text: string; matched: boolean; baseAngle: number; orbitRadius: number }[] = [];
    const greenArc = (score / 100) * Math.PI * 2;
    const redArc = Math.PI * 2 - greenArc;

    const INNER_ORBIT = 2.9;
    const OUTER_ORBIT = 3.5;

    // Distribute green keywords across the green arc
    if (matchedKeywords.length > 0) {
      const padding = greenArc * 0.06;
      const usableArc = greenArc - padding * 2;
      const step =
        matchedKeywords.length > 1
          ? usableArc / (matchedKeywords.length - 1)
          : 0;
      matchedKeywords.forEach((kw, i) => {
        const a =
          padding +
          (matchedKeywords.length > 1 ? step * i : usableArc / 2);
        const radius = i % 2 === 0 ? INNER_ORBIT : OUTER_ORBIT;
        angles.push({ ...kw, baseAngle: a, orbitRadius: radius });
      });
    }

    // Distribute red keywords across the red arc
    if (missingKeywords.length > 0) {
      const padding = redArc * 0.06;
      const usableArc = redArc - padding * 2;
      const step =
        missingKeywords.length > 1
          ? usableArc / (missingKeywords.length - 1)
          : 0;
      missingKeywords.forEach((kw, i) => {
        const a =
          greenArc +
          padding +
          (missingKeywords.length > 1 ? step * i : usableArc / 2);
        const radius = i % 2 === 0 ? INNER_ORBIT : OUTER_ORBIT;
        angles.push({ ...kw, baseAngle: a, orbitRadius: radius });
      });
    }

    return angles;
  }, [matchedKeywords, missingKeywords, score]);

  if (!visible || keywordAngles.length === 0) {
    keywordPositions.length = 0;
    return null;
  }

  keywordPositions.length = keywordAngles.length;

  return (
    <group>
      {keywordAngles.map((kw, i) => (
        <KeywordOrb
          key={kw.text}
          text={kw.text}
          matched={kw.matched}
          baseAngle={kw.baseAngle}
          orbitRadius={kw.orbitRadius}
          index={i}
        />
      ))}
    </group>
  );
}
