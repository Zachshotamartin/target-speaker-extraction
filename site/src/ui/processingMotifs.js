import {Group, TorusGeometry, CircleGeometry, BoxGeometry, CapsuleGeometry, MeshBasicMaterial, Mesh, DoubleSide} from 'three';

function motif(color, radius, spin, build) {
  const group = new Group(), content = new Group();
  const geometries = new Set(), materials = new Set();
  group.add(content);
  function mesh(geometry, opacity = .85) {
    const material = new MeshBasicMaterial({color, transparent: true, opacity, side: DoubleSide});
    const object = new Mesh(geometry, material);
    geometries.add(geometry); materials.add(material);
    content.add(object);
    return object;
  }
  const animate = build(content, mesh);
  return {group, radius, spin, update: animate, dispose() {
    geometries.forEach(geometry => geometry.dispose());
    materials.forEach(material => material.dispose());
  }};
}

export function createProcessingMotifs(color) {
  const sun = motif(color, 1, .48, (content, mesh) => {
    const center = mesh(new CircleGeometry(.14, 24));
    const halo = mesh(new TorusGeometry(.32, .026, 6, 64));
    const rayGeometry = new BoxGeometry(.05, .19, .035);
    const rays = Array.from({length: 12}, (_, index) => {
      const ray = mesh(rayGeometry);
      const angle = index * Math.PI / 6;
      ray.position.set(Math.sin(angle) * .61, Math.cos(angle) * .61, 0);
      ray.rotation.z = -angle;
      return ray;
    });
    return (time, spin, weight) => {
      content.rotation.z = spin;
      center.scale.setScalar(1 + Math.sin(time * 2) * .08);
      center.material.opacity = halo.material.opacity = .85 * weight;
      rays.forEach((ray, index) => {
        ray.scale.y = 1 + Math.sin(time * 2.8 + index * .7) * .28;
        ray.material.opacity = (.75 + .15 * Math.sin(time * 1.8 + index)) * weight;
      });
    };
  });

  const ripples = motif(color, .9, .2, (content, mesh) => {
    const center = mesh(new CircleGeometry(.085, 24));
    const rings = Array.from({length: 3}, () => mesh(new TorusGeometry(.78, .025, 6, 64)));
    return (time, spin, weight) => {
      content.rotation.z = spin * .2;
      center.material.opacity = .9 * weight;
      rings.forEach((ring, index) => {
        const age = (time * .36 + index / rings.length) % 1;
        ring.scale.setScalar(.2 + age * .8);
        // Each wave grows out from the voice and fades before restarting at the center.
        ring.material.opacity = Math.sin(Math.PI * age) * .8 * weight;
      });
    };
  });

  const waveform = motif(color, 1.06, .12, (content, mesh) => {
    const geometry = new CapsuleGeometry(.045, 1, 4, 8);
    const bars = Array.from({length: 7}, (_, index) => {
      const bar = mesh(geometry);
      bar.position.x = (index - 3) * .2;
      return bar;
    });
    return (time, _spin, weight) => {
      content.rotation.z = Math.sin(time * .8) * .06;
      bars.forEach((bar, index) => {
        const envelope = 1 - Math.abs(index - 3) * .17;
        bar.scale.y = .18 + envelope * (.38 + .5 * Math.sin(time * 3.2 + index * .8) ** 2);
        bar.material.opacity = .85 * weight;
      });
    };
  });

  const orbit = motif(color, .86, .78, (content, mesh) => {
    const core = mesh(new CircleGeometry(.1, 24));
    const geometry = new TorusGeometry(.69, .025, 6, 64);
    const rings = Array.from({length: 3}, () => mesh(geometry));
    return (time, spin, weight) => {
      content.rotation.set(.2, spin * .45, spin);
      core.material.opacity = .9 * weight;
      rings.forEach((ring, index) => {
        ring.rotation.set(Math.PI / 3 + Math.sin(time * .7 + index * 2) * .25, index * Math.PI / 3, index * Math.PI / 4);
        ring.material.opacity = .75 * weight;
      });
    };
  });
  return [sun, ripples, waveform, orbit];
}
