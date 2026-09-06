import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

import type { MeshFormat } from '../api/types';

// The single reusable three.js viewer (web.md): an interactive, orbit-controlled
// 3D view of a model's normalized mesh.
//
// The mesh is the pipeline's normalized mesh (server.md#serving-artifacts) — it is
// already centered on the origin and scaled so its largest extent is 1, so the
// camera framing below is fixed and needs no per-model fitting. That is the
// normalize stage paying off in the UI.
//
// Two formats, because there are two arms. The default arm's PLY carries geometry
// only, so it is drawn with the neutral material the offscreen renders use. The
// texture A/B's arm stores GLB, which carries the model's own materials — those
// are kept as-authored, since showing them is the entire point of that preview.
//
// One download per model opened, not per view: once the geometry is loaded,
// orbiting is entirely client-side.
//
// Everything created here — renderer, geometry, material, controls, the
// animation frame, the resize listener — is disposed on unmount so remounting
// doesn't leak GPU memory (web.md: "Dispose of GPU resources on unmount").

// 'unavailable' = no mesh exists (src is null); 'failed' = a mesh was offered but
// the fetch/parse failed. Kept distinct so a genuine load error doesn't read as
// "the pipeline hasn't produced this yet".
type ViewerStatus = 'loading' | 'ready' | 'unavailable' | 'failed';

/** Dispose every geometry, material and texture under `root` (GLB scenes are trees). */
function disposeTree(root: THREE.Object3D): void {
  root.traverse((child) => {
    const asMesh = child as THREE.Mesh;
    if (!asMesh.isMesh) return;
    asMesh.geometry?.dispose();
    const materials = Array.isArray(asMesh.material) ? asMesh.material : [asMesh.material];
    for (const material of materials) {
      if (!material) continue;
      // A GLB's materials own their textures, and disposing the material does not
      // free those — the atlas can be 16384px wide, so leaking one is expensive.
      for (const value of Object.values(material)) {
        if ((value as THREE.Texture | null)?.isTexture) (value as THREE.Texture).dispose();
      }
      material.dispose();
    }
  });
}

export function ModelViewer({
  src,
  format = 'ply',
}: {
  src?: string | null;
  format?: MeshFormat;
}) {
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
    let object: THREE.Object3D | null = null;
    let disposed = false;

    // A URL was offered but the mesh couldn't be fetched/parsed (e.g. a network
    // error) — distinct from "no mesh exists", handled by !src above.
    const onLoadError = () => {
      if (!disposed) setStatus('failed');
    };

    if (src && format === 'glb') {
      new GLTFLoader().load(
        src,
        (gltf) => {
          if (disposed) {
            disposeTree(gltf.scene); // arrived too late to be shown; don't leak it
            return;
          }
          // Keep the GLB's own materials: this arm exists to show them.
          object = gltf.scene;
          scene.add(object);
          setStatus('ready');
        },
        undefined,
        onLoadError,
      );
    } else if (src) {
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
          object = new THREE.Mesh(loaded, material);
          scene.add(object);
          setStatus('ready');
        },
        undefined,
        onLoadError,
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
      if (object) {
        scene.remove(object);
        // The PLY path's geometry is disposed below and its material is shared,
        // so only a loaded GLB tree owns anything this has to walk.
        if (format === 'glb') disposeTree(object);
      }
      geometry?.dispose();
      material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [src, format]);

  return (
    <div className="model-viewer-wrap">
      <div ref={mountRef} className="model-viewer" />
      {status !== 'ready' && (
        <p className="model-viewer-status" role="status">
          {status === 'loading'
            ? 'Loading mesh…'
            : status === 'failed'
              ? 'Couldn’t load the 3D mesh'
              : 'No 3D mesh for this model yet'}
        </p>
      )}
    </div>
  );
}
