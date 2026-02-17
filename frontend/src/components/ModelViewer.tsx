import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, useGLTF, Environment } from '@react-three/drei'
import { Suspense, useEffect, useRef } from 'react'
import * as THREE from 'three'

function CityModel({ url }: { url: string }) {
  const { scene } = useGLTF(url)
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)  // eslint-disable-line @typescript-eslint/no-explicit-any

  useEffect(() => {
    // Auto-fit camera to model bounds
    const box = new THREE.Box3().setFromObject(scene)
    const size = box.getSize(new THREE.Vector3())
    const center = box.getCenter(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z)
    const dist = maxDim * 1.5

    // Position camera to the SOUTH looking north so the model's
    // left-right matches the map (east = right, west = left).
    // Model uses +Z = south (Three.js right-handed convention),
    // so camera at +Z looks -Z = north.
    camera.position.set(
      center.x + dist * 0.3,
      center.y + dist * 0.8,
      center.z + dist * 0.6
    )
    camera.lookAt(center)
    camera.updateProjectionMatrix()

    if (controlsRef.current) {
      controlsRef.current.target.copy(center)
      controlsRef.current.update()
    }

    // Fix z-fighting: enable polygonOffset on flat overlay layers
    // (road, paved, railway) so the GPU biases their depth slightly
    // toward the camera, preventing flicker against the terrain.
    const overlayNames = new Set(['road', 'paved', 'railway', 'pitch', 'track'])
    scene.traverse((child: THREE.Object3D) => {
      if (child instanceof THREE.Mesh && overlayNames.has(child.name)) {
        const mat = child.material as THREE.MeshStandardMaterial
        if (mat) {
          mat.polygonOffset = true
          mat.polygonOffsetFactor = -1
          mat.polygonOffsetUnits = -1
        }
      }
    })
  }, [scene, camera])

  return (
    <>
      <primitive object={scene} />
      <OrbitControls ref={controlsRef} makeDefault />
    </>
  )
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#89b4fa" wireframe />
    </mesh>
  )
}

export default function ModelViewer({ url }: { url: string }) {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ fov: 50, near: 1, far: 50000 }}
        gl={{ antialias: true }}
      >
        {/* Sky-like gradient background */}
        <color attach="background" args={['#d4e6f1']} />

        {/* Lighting: ambient fill + strong directional sun + secondary fill */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[300, 500, 200]}
          intensity={1.2}
          castShadow={false}
        />
        <directionalLight
          position={[-200, 300, -100]}
          intensity={0.3}
        />
        <hemisphereLight
          args={['#87ceeb', '#8b7355', 0.4]}
        />

        <Suspense fallback={<LoadingFallback />}>
          <CityModel url={url} />
          <Environment preset="city" background={false} />
        </Suspense>

        <gridHelper args={[2000, 100, '#a0a0a0', '#c8c8c8']} />
      </Canvas>
    </div>
  )
}
