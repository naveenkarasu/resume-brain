import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import BrainMesh from './BrainMesh';
import BackgroundParticles from './BackgroundParticles';
import ScoreRing from './ScoreRing';
import ParticleField from './ParticleField';
import NeuralConnections from './NeuralConnections';
import { useAnalysisStore } from '../../store/analysisStore';

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
    <div className="w-full h-[400px] md:h-[500px]">
      <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.3} />

        <BrainMesh
          phase={phase}
          score={result?.overall_score ?? 0}
        />
        <BackgroundParticles />
        <ScoreRing
          score={result?.overall_score ?? 0}
          visible={phase === 'results'}
        />
        <ParticleField
          keywords={keywords}
          visible={phase === 'results'}
        />
        <NeuralConnections active={phase === 'loading'} score={result?.overall_score ?? 0} />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate={phase === 'idle'}
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
