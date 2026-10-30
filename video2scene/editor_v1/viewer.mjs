import * as THREE from 'three';
import {OrbitControls} from './vendor/three/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from './vendor/three/examples/jsm/loaders/GLTFLoader.js';
import {artifactURL} from './state.mjs';

export class SceneViewer {
  constructor(element,onSelect,onMessage) {
    this.element=element;this.onSelect=onSelect;this.onMessage=onMessage;this.generation=0;this.roots=new Map();this.pendingAssets=0;this.assetErrors=[];this.selectedId=null;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#eaf0e9');
    this.camera=new THREE.PerspectiveCamera(40,1,.001,1000);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;element.appendChild(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x667c6a,2.6));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(2,5,3);this.scene.add(light);
    this.group=new THREE.Group();this.scene.add(this.group);this.loader=new GLTFLoader();this.selection=null;
    this.resize=new ResizeObserver(()=>{const r=element.getBoundingClientRect();this.renderer.setSize(r.width,r.height);this.camera.aspect=r.width/r.height;this.camera.updateProjectionMatrix()});this.resize.observe(element);
    let start=null;this.renderer.domElement.addEventListener('pointerdown',e=>start=[e.clientX,e.clientY]);
    this.renderer.domElement.addEventListener('pointerup',e=>{if(!start||Math.hypot(e.clientX-start[0],e.clientY-start[1])>5)return;const r=element.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),this.camera);const hit=ray.intersectObjects(this.group.children,true)[0];if(hit){let root=hit.object;while(root&&!root.userData.instanceId)root=root.parent;if(root)this.onSelect(root.userData.instanceId)}});
    this.renderer.setAnimationLoop(()=>{this.controls.update();this.renderer.render(this.scene,this.camera)});
  }
  disposeTree(root){root.traverse(o=>{o.geometry?.dispose();if(o.material){for(const material of (Array.isArray(o.material)?o.material:[o.material])) material.dispose()}})}
  async update(snapshot,selected,showAssets=true){
    const generation=++this.generation;this.pendingAssets=0;this.assetErrors=[];this.selectedId=selected;for(const child of [...this.group.children]){this.group.remove(child);this.disposeTree(child)}this.roots.clear();
    if(this.selection){this.scene.remove(this.selection);this.selection.geometry.dispose();this.selection.material.dispose();this.selection=null}
    let fallback=0;
    const assetNote=()=>this.onMessage(!showAssets?'尺寸代理视图 · 始终使用正式版本的变换与边界':this.pendingAssets?`正在载入 ${this.pendingAssets} 个实际资产…`:this.assetErrors.length?'部分资产未能载入，保留带尺寸的代理；正式场景未改变。':`${fallback} 个对象使用尺寸代理；可用 GLB 显示实际资产。`);
    for(const obj of snapshot.objects){
      const root=new THREE.Group();root.userData.instanceId=obj.instance_id;root.userData.renderMode='proxy';root.position.fromArray(obj.transform.translation_m);root.quaternion.fromArray(obj.transform.rotation_xyzw);root.scale.fromArray(obj.transform.scale);this.group.add(root);this.roots.set(obj.instance_id,root);
      const lo=obj.local_bounds_m.min,hi=obj.local_bounds_m.max;const shape=new THREE.BoxGeometry(...hi.map((v,i)=>Math.max(v-lo[i],.001)));
      const proxy=new THREE.Mesh(shape,new THREE.MeshStandardMaterial({color:obj.kind==='support'?0xb6c6b7:0xe1e8d9,roughness:.85,metalness:0}));proxy.position.fromArray(hi.map((v,i)=>(v+lo[i])/2));root.add(proxy);
      const edge=new THREE.LineSegments(new THREE.EdgesGeometry(shape),new THREE.LineBasicMaterial({color:0x76907c,transparent:true,opacity:.35}));edge.position.copy(proxy.position);root.add(edge);
      const ref=obj.asset.artifact;
      if(showAssets && (ref.media_type==='model/gltf-binary'||ref.uri.toLowerCase().endsWith('.glb'))){
        this.pendingAssets++;this.loader.load(artifactURL(ref),gltf=>{if(generation!==this.generation){this.disposeTree(gltf.scene);return}this.pendingAssets--;root.remove(proxy,edge);this.disposeTree(proxy);this.disposeTree(edge);const native=new THREE.Group();native.matrix.fromArray(obj.asset.native_to_object.flat()).transpose();native.matrixAutoUpdate=false;native.add(gltf.scene);root.add(native);root.userData.renderMode='asset';this.select(this.selectedId);this.renderer.render(this.scene,this.camera);assetNote()},undefined,()=>{if(generation===this.generation){this.pendingAssets--;this.assetErrors.push(obj.instance_id);assetNote()}});
      }else fallback++;
    }
    this.group.updateMatrixWorld(true);this.select(selected);if(!this.fitted){this.reset();this.fitted=true}
    assetNote();
  }
  select(identity){this.selectedId=identity;if(this.selection){this.scene.remove(this.selection);this.selection.geometry.dispose();this.selection.material.dispose()}const root=this.roots.get(identity);this.selection=root?new THREE.BoxHelper(root,0x2b7858):null;if(this.selection)this.scene.add(this.selection)}
  addWhiteboxEdges(snapshot){
    for(const obj of snapshot.objects){
      if(!obj.asset.asset_version.startsWith('whitebox-'))continue;
      const root=this.roots.get(obj.instance_id);if(!root||root.userData.renderMode!=='asset')continue;
      if(!root.userData.whiteboxEdges){
        const meshes=[];root.traverse(node=>{if(node.isMesh)meshes.push(node)});
        for(const mesh of meshes){const edges=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:0x779185,transparent:true,opacity:.45}));edges.userData.whiteboxEdge=true;mesh.add(edges)}
        root.userData.whiteboxEdges=true;
      }
      root.traverse(node=>{if(node.userData.whiteboxEdge)node.material.color.setHex(obj.instance_id===this.selectedId?0x2b7858:0x779185)});
    }
    if(this.selection&&snapshot.objects.some(obj=>obj.instance_id===this.selectedId&&obj.asset.asset_version.startsWith('whitebox-')))this.selection.visible=false;
  }
  reset(){const box=new THREE.Box3().setFromObject(this.group),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),d=Math.max(size.x,size.y,size.z,.25);this.camera.position.copy(center).add(new THREE.Vector3(d*1.35,d*1.15,d*1.5));this.camera.near=Math.max(.001,d/1000);this.camera.far=Math.max(100,d*100);this.camera.updateProjectionMatrix();this.controls.target.copy(center);this.controls.update()}
}
