import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { keywordPositions } from './ParticleField';

interface Props {
  visible: boolean;
}

const MAX_LINES = 100;
const MAX_DISTANCE = 2.5;

// Colors matching the keyword dot palette
const GREEN = { r: 0.133, g: 0.773, b: 0.369 }; // #22c55e
const RED = { r: 0.937, g: 0.267, b: 0.267 };    // #ef4444

export default function ConstellationLines({ visible }: Props) {
  const ref = useRef<THREE.LineSegments>(null!);
  const opacityRef = useRef(0);

  const { geometry, material } = useMemo(() => {
    // Pre-allocate for MAX_LINES line segments (2 vertices each, 3 components)
    const positions = new Float32Array(MAX_LINES * 2 * 3);
    const colors = new Float32Array(MAX_LINES * 2 * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setDrawRange(0, 0);

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    return { geometry: geo, material: mat };
  }, []);

  useFrame((_, delta) => {
    if (!ref.current) return;

    // Fade in/out
    const targetOpacity = visible ? 0.3 : 0;
    opacityRef.current += (targetOpacity - opacityRef.current) * delta * 3;
    material.opacity = opacityRef.current;

    if (opacityRef.current < 0.01) {
      geometry.setDrawRange(0, 0);
      return;
    }

    const posAttr = geometry.attributes.position as THREE.BufferAttribute;
    const colorAttr = geometry.attributes.color as THREE.BufferAttribute;
    const positions = posAttr.array as Float32Array;
    const colors = colorAttr.array as Float32Array;

    let lineCount = 0;
    const n = keywordPositions.length;

    for (let i = 0; i < n && lineCount < MAX_LINES; i++) {
      for (let j = i + 1; j < n && lineCount < MAX_LINES; j++) {
        const a = keywordPositions[i];
        const b = keywordPositions[j];
        if (!a || !b) continue;

        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = a.z - b.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < MAX_DISTANCE) {
          const idx = lineCount * 6;
          positions[idx] = a.x;
          positions[idx + 1] = a.y;
          positions[idx + 2] = a.z;
          positions[idx + 3] = b.x;
          positions[idx + 4] = b.y;
          positions[idx + 5] = b.z;

          // Per-vertex color: each endpoint gets its own keyword's color.
          // Same-type lines are solid green or red.
          // Mixed lines get a green→red gradient (GPU interpolates automatically).
          const colorA = a.matched ? GREEN : RED;
          const colorB = b.matched ? GREEN : RED;

          const cidx = lineCount * 6;
          colors[cidx]     = colorA.r;
          colors[cidx + 1] = colorA.g;
          colors[cidx + 2] = colorA.b;
          colors[cidx + 3] = colorB.r;
          colors[cidx + 4] = colorB.g;
          colors[cidx + 5] = colorB.b;

          lineCount++;
        }
      }
    }

    posAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    geometry.setDrawRange(0, lineCount * 2);
  });

  return <lineSegments ref={ref} geometry={geometry} material={material} />;
}
