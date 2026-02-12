import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { Suspense, useMemo } from 'react';
import * as THREE from 'three';
import BrainMesh from './BrainMesh';
import BackgroundParticles from './BackgroundParticles';
import SpaceSkybox from './SpaceSkybox';
import ScoreRing from './ScoreRing';
import ParticleField from './ParticleField';
import NeuralConnections from './NeuralConnections';
import ConstellationLines from './ConstellationLines';
import { useAnalysisStore } from '../../store/analysisStore';

/** Smoothly moves the globe upward and scales it down as the user scrolls. */
function ScrollGroup({ children }: { children: React.ReactNode }) {
  const ref = useRef<THREE.Group>(null!);
  const currentY = useRef(0);
  const currentScale = useRef(1);

  useFrame((_, delta) => {
    const progress = Math.min(window.scrollY / (window.innerHeight * 0.3), 1);
    const targetY = progress * 1.4;
    const targetScale = 1 - progress * 0.5;
    const lerp = 1 - Math.exp(-5 * delta);

    currentY.current += (targetY - currentY.current) * lerp;
    currentScale.current += (targetScale - currentScale.current) * lerp;

    ref.current.position.y = currentY.current;
    const s = currentScale.current;
    ref.current.scale.set(s, s, s);
  });

  return <group ref={ref}>{children}</group>;
}

export default function BrainScene() {
  const { phase, result } = useAnalysisStore();

  const keywords = useMemo(() => {
    if (!result) return [];
    const matched = result.matched_keywords.map((kw) => ({
      text: kw,
      matched: true,
    }));
    const missing = result.missing_keywords.map((kw) => ({
      text: kw,
      matched: false,
    }));
    return [...matched, ...missing];
  }, [result]);

  return (
    <div className="fixed inset-0" style={{ zIndex: 0 }}>
      <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
        <Suspense fallback={null}>
          <SpaceSkybox />
          <BackgroundParticles />

          <ScrollGroup>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={0.3} />

            <BrainMesh
              phase={phase}
              score={result?.overall_score ?? 0}
            />
            <ScoreRing
              score={result?.overall_score ?? 0}
              visible={phase === 'results'}
            />
            <ParticleField
              keywords={keywords}
              visible={phase === 'results'}
              score={result?.overall_score ?? 0}
            />
            <ConstellationLines visible={phase === 'results'} />
            <NeuralConnections active={phase === 'loading'} score={result?.overall_score ?? 0} />
          </ScrollGroup>

          <OrbitControls
            enableZoom={false}
            enablePan={false}
            autoRotate={phase === 'idle'}
            autoRotateSpeed={0.5}
          />

          <EffectComposer>
            <Bloom
              intensity={0.8}
              luminanceThreshold={0.2}
              luminanceSmoothing={0.9}
              mipmapBlur
              radius={0.4}
            />
          </EffectComposer>
        </Suspense>
      </Canvas>
    </div>
  );
}
