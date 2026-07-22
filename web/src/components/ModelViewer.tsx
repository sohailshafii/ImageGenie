import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

// The single reusable three.js viewer (web.md): an interactive, orbit-controlled
// 3D view of a model's normalized mesh.
//
// The mesh is the pipeline's normalized PLY (server.md#serving-artifacts) — it is
// already centered on the origin and scaled so its largest extent is 1, so the
// camera framing below is fixed and needs no per-model fitting. That is the
// normalize stage paying off in the UI.
//
// One download per model opened, not per view: once the geometry is loaded,
// orbiting is entirely client-side.
//
// Everything created here — renderer, geometry, material, controls, the
// animation frame, the resize listener — is disposed on unmount so remounting
// doesn't leak GPU memory (web.md: "Dispose of GPU resources on unmount").

type ViewerStatus = 'loading' | 'ready' | 'unavailable';

export function ModelViewer({ src }: { src?: string | null }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<ViewerStatus>(src ? 'loading' : 'unavailable');

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    setStatus(src ? 'loading' : 'unavailable');

    let width = mount.clientWidth;
    let height = mount.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(2.4, 1.5, 2.4);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(3, 4, 2);
    scene.add(keyLight);

    // Matches the offscreen renders' material so the viewer and the thumbnails
    // read as the same object (server/app/workers/render.py).
    const material = new THREE.MeshStandardMaterial({
      color: 0xb4b4bf,
      roughness: 0.75,
      metalness: 0.0,
    });

    // Tracked so the cleanup below can dispose whatever actually got created —
    // a load that resolves after unmount must not leave GPU memory behind.
    let geometry: THREE.BufferGeometry | null = null;
    let mesh: THREE.Mesh | null = null;
    let disposed = false;

    if (src) {
      new PLYLoader().load(
        src,
        (loaded) => {
          if (disposed) {
            loaded.dispose(); // arrived too late to be shown; don't leak it
            return;
          }
          // Pipeline PLYs carry no normals, so lighting would be flat without
          // this — computing them is what makes the shape legible.
          loaded.computeVertexNormals();
          geometry = loaded;
          mesh = new THREE.Mesh(loaded, material);
          scene.add(mesh);
          setStatus('ready');
        },
        undefined,
        () => {
          // Expected for a model the pipeline hasn't normalized yet.
          if (!disposed) setStatus('unavailable');
        },
      );
    }

    let frameId = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      width = mount.clientWidth;
      height = mount.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', onResize);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      if (mesh) scene.remove(mesh);
      geometry?.dispose();
      material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [src]);

  return (
    <div className="model-viewer-wrap">
      <div ref={mountRef} className="model-viewer" />
      {status !== 'ready' && (
        <p className="model-viewer-status" role="status">
          {status === 'loading' ? 'Loading mesh…' : 'No 3D mesh for this model yet'}
        </p>
      )}
    </div>
  );
}
