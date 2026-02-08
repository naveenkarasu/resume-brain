import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface Keyword {
  text: string;
  matched: boolean;
}

interface Props {
  keywords: Keyword[];
  visible: boolean;
}

function KeywordOrb({ text, matched, index, total }: Keyword & { index: number; total: number }) {
  const ref = useRef<THREE.Group>(null!);
  const angle = (index / total) * Math.PI * 2;
  const radius = 3.2;

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.elapsedTime;
    const speed = 0.2;
    const a = angle + t * speed;
    ref.current.position.x = Math.cos(a) * radius;
    ref.current.position.z = Math.sin(a) * radius;
    ref.current.position.y = Math.sin(t * 0.5 + index) * 0.5;

    // Always face camera
    ref.current.lookAt(0, ref.current.position.y, 0);
    ref.current.rotateY(Math.PI);
  });

  const color = matched ? '#22c55e' : '#ef4444';

  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[0.06, 8, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <Text
        position={[0, 0.15, 0]}
        fontSize={0.12}
        color={color}
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.01}
        outlineColor="#000000"
      >
        {text}
      </Text>
    </group>
  );
}

export default function ParticleField({ keywords, visible }: Props) {
  const displayKeywords = useMemo(
    () => keywords.slice(0, 16),
    [keywords]
  );

  if (!visible || displayKeywords.length === 0) return null;

  return (
    <group>
      {displayKeywords.map((kw, i) => (
        <KeywordOrb
          key={kw.text}
          text={kw.text}
          matched={kw.matched}
          index={i}
          total={displayKeywords.length}
        />
      ))}
    </group>
  );
}
