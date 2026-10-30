"""Bake procedural base colors for GLB only; restore the editable native materials."""
import argparse
import json
import sys
from pathlib import Path
import bpy


def needs_bake(material):
    if not material or not material.use_nodes:return False
    bsdf=next((n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
    return bool(bsdf and bsdf.inputs['Base Color'].is_linked and bsdf.inputs['Base Color'].links[0].from_node.type!='TEX_IMAGE')


def export_web(directory):
    directory=Path(directory);textures=directory/'web_textures';textures.mkdir(exist_ok=True)
    scene=bpy.context.scene;previous_samples=scene.cycles.samples
    objects=[o for o in scene.objects if o.type=='MESH' and not o.hide_render]
    originals=[];records=[]
    try:
        for ob in objects:
            if not any(needs_bake(m) for m in ob.data.materials):continue
            original=ob.data;ob.data=original.copy();originals.append((ob,original))
            indices=[p.material_index for p in ob.data.polygons]
            materials=[m.copy() for m in original.materials]
            ob.data.materials.clear()
            for m in materials:ob.data.materials.append(m)
            for polygon,index in zip(ob.data.polygons,indices):polygon.material_index=index
            # Keep original render UVs for any image nodes feeding procedural mixes.
            previous_render=next((uv.name for uv in ob.data.uv_layers if uv.active_render),None)
            bake_name=ob.data.uv_layers.new(name='Web_albedo_atlas').name
            ob.data.uv_layers.active_index=len(ob.data.uv_layers)-1
            if previous_render:ob.data.uv_layers[previous_render].active_render=True
            bpy.ops.object.select_all(action='DESELECT');ob.hide_set(False);ob.select_set(True);bpy.context.view_layer.objects.active=ob
            bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=1.151917,island_margin=.025)
            bpy.ops.object.mode_set(mode='OBJECT')
            images=[];settings=[]
            for i,m in enumerate(materials):
                bsdf=next((n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
                settings.append((bsdf.inputs['Roughness'].default_value,bsdf.inputs['Metallic'].default_value))
                image=bpy.data.images.new(f'Web albedo {ob.name} {i}',width=1024,height=1024,alpha=False)
                image.colorspace_settings.name='sRGB'
                tex=m.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image
                for node in m.node_tree.nodes:node.select=False
                tex.select=True;m.node_tree.nodes.active=tex;images.append(image)
            scene.cycles.samples=4
            bpy.ops.object.bake(type='DIFFUSE',pass_filter={'COLOR'},margin=8,uv_layer=bake_name)
            for name in [uv.name for uv in ob.data.uv_layers]:
                if name!=bake_name:ob.data.uv_layers.remove(ob.data.uv_layers[name])
            ob.data.uv_layers.active_index=0;ob.data.uv_layers[0].active_render=True
            for i,(m,image,settings_row) in enumerate(zip(materials,images,settings)):
                path=textures/f'{len(records):02d}_{i}_base_color.png'
                image.filepath_raw=str(path);image.file_format='PNG';image.save();image.pack()
                m.node_tree.nodes.clear();out=m.node_tree.nodes.new('ShaderNodeOutputMaterial');bsdf=m.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
                tex=m.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image
                bsdf.inputs['Roughness'].default_value,bsdf.inputs['Metallic'].default_value=settings_row
                m.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color']);m.node_tree.links.new(bsdf.outputs[0],out.inputs[0])
            records.append(dict(object=ob.name,materials=len(materials),resolution=1024))
        bpy.ops.object.select_all(action='DESELECT')
        for ob in objects:
            ob.select_set(True);parent=ob.parent
            while parent:parent.select_set(True);parent=parent.parent
        bpy.ops.export_scene.gltf(filepath=str(directory/'model_web.glb'),export_format='GLB',use_selection=True,export_apply=True,export_extras=True)
    finally:
        for ob,original in originals:ob.data=original
        scene.cycles.samples=previous_samples
        bpy.data.orphans_purge(do_recursive=True)
    (directory/'web_materials.json').write_text(json.dumps(dict(baked=records,native_materials_preserved=True,
        note='Base color baked for browser preview; native procedural graphs preserved. Fine bump may still be simplified.'),indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);export_web(a.output)
