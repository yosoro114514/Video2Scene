"""Instance-aware editable Blender geometry. Imported only inside Blender.

All inputs use the project's right-handed Y-up scene coordinates. Local flat
assets use X=width, Y=height, Z=front. No camera-facing billboards are used.
"""
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector

C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
I=np.eye(3)


def material(name, color, rough=.5, metal=0, texture=None):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes=True
    p=m.node_tree.nodes.get("Principled BSDF")
    rgb=np.array(color[:3],float)/255
    rgb=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    p.inputs["Base Color"].default_value=(*rgb,1)
    p.inputs["Roughness"].default_value=rough
    p.inputs["Metallic"].default_value=metal
    p.inputs["Specular IOR Level"].default_value=.18 if metal==0 else .5
    m.diffuse_color=(*rgb,1)
    if texture:
        t=m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image=bpy.data.images.load(str(texture),check_existing=True)
        m.node_tree.links.new(t.outputs["Color"],p.inputs["Base Color"])
    return m


class Asset:
    def __init__(self,e,output):
        self.e=e;self.name=e["id"];self.c=np.array(e["center"]);self.r=np.array(e["rotation"]);self.d=np.array(e["dimensions"]);self.output=output
        self.collection=bpy.data.collections.new("REFINED · "+self.name)
        bpy.context.scene.collection.children.link(self.collection)
        self.parent=bpy.data.objects.new(self.name+"__asset",None)
        self.collection.objects.link(self.parent)
        self.parent["evidence_frames"]=str(e.get("evidence_frames",[]))
        self.parent["source"]=e.get("source","Video and geometric estimate")
        self.m=material("V2 "+self.name,e.get("color",[100,100,100]))
        self.dark=material("V2 black ABS",[17,19,21],.46)
        self.rubber=material("V2 rubber",[12,13,14],.78)
        self.steel=material("V2 brushed metal",[139,145,148],.28,.8)

    def finish(self,o,suffix,offset=(0,0,0),rot=None,mat=None):
        o.name=self.name+"__"+suffix
        for col in list(o.users_collection):col.objects.unlink(o)
        self.collection.objects.link(o);o.parent=self.parent
        matrix=np.eye(4);matrix[:3,:3]=C@self.r@(I if rot is None else rot);matrix[:3,3]=C@(self.c+self.r@np.asarray(offset))
        o.matrix_world=Matrix(matrix.tolist())
        o["source_instance"]=self.name;o["instance"]=self.name
        o["source"]=self.e.get("source","Source video")
        o["geometry_provenance"]="visible shape interpreted from video; hidden cross-sections inferred"
        o.data.materials.clear();o.data.materials.append(mat or self.m)
        return o

    def mesh(self,suffix,vertices,faces,mat=None,offset=(0,0,0),rot=None,smooth=True):
        mesh=bpy.data.meshes.new(self.name+" "+suffix);mesh.from_pydata(vertices,[],faces);mesh.update()
        o=bpy.data.objects.new(suffix,mesh);self.collection.objects.link(o)
        if smooth:
            for p in mesh.polygons:p.use_smooth=True
        return self.finish(o,suffix,offset,rot,mat)

    def box(self,suffix,offset,dims,mat=None,bevel=.001,rot=None):
        bpy.ops.mesh.primitive_cube_add(size=1)
        o=bpy.context.object
        for v in o.data.vertices:v.co=Vector(np.asarray(v.co)*dims)
        self.finish(o,suffix,offset,rot,mat)
        if bevel:
            b=o.modifiers.new("Edge radius", "BEVEL");b.width=min(bevel,min(dims)*.35);b.segments=3
            n=o.modifiers.new("Face normals","WEIGHTED_NORMAL");n.keep_sharp=True
        return o

    def ellipsoid(self,suffix,offset,dims,mat=None,rot=None):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=40,ring_count=24,radius=.5)
        o=bpy.context.object
        for v in o.data.vertices:v.co=Vector(np.array(v.co)*dims)
        for p in o.data.polygons:p.use_smooth=True
        return self.finish(o,suffix,offset,rot,mat)

    def cylinder(self,suffix,offset,dims,mat=None,rot=None):
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=.5,depth=1)
        o=bpy.context.object
        for v in o.data.vertices:
            x,y,z=v.co;v.co=(x*dims[0],z*dims[1],-y*dims[2])
        for p in o.data.polygons:p.use_smooth=len(p.vertices)==4
        self.finish(o,suffix,offset,rot,mat)
        b=o.modifiers.new("Rim radius","BEVEL");b.width=min(.001,min(dims)*.2);b.segments=3
        return o

    def loft(self,suffix,rings,mat=None,exponent=4,offset=(0,0,0),rot=None,slope=0):
        """Rounded XZ cross-sections (height,width,length,z_offset), closed shell."""
        count=64;vertices=[]
        low=min(r[0] for r in rings);high=max(r[0] for r in rings)
        for y,w,l,zc in rings:
            for j in range(count):
                a=2*math.pi*j/count;ca,sa=math.cos(a),math.sin(a)
                x=np.sign(ca)*abs(ca)**(2/exponent)*w/2
                z=np.sign(sa)*abs(sa)**(2/exponent)*l/2+zc
                # A sloped upper shell must not tilt its support-contact plane.
                taper=(y-low)/max(high-low,1e-9)
                vertices.append([x,y+slope*z*taper,z])
        faces=[tuple(reversed(range(count)))]
        for k in range(len(rings)-1):
            for j in range(count):faces.append((k*count+j,k*count+(j+1)%count,(k+1)*count+(j+1)%count,(k+1)*count+j))
        faces.append(tuple((len(rings)-1)*count+j for j in range(count)))
        return self.mesh(suffix,vertices,faces,mat,offset,rot)

    def curve(self,suffix,points,radius=.0015,mat=None,world=False,poly=False):
        data=bpy.data.curves.new(suffix,"CURVE");data.dimensions="3D";data.resolution_u=16;data.bevel_depth=radius;data.bevel_resolution=3;data.use_fill_caps=True
        s=data.splines.new("POLY" if poly else "BEZIER")
        if poly:
            s.points.add(len(points)-1)
            for p,co in zip(s.points,points):p.co=(*co,1)
        else:
            s.bezier_points.add(len(points)-1)
            for p,co in zip(s.bezier_points,points):p.co=co;p.handle_left_type="AUTO";p.handle_right_type="AUTO"
        o=bpy.data.objects.new(suffix,data);self.collection.objects.link(o)
        self.finish(o,suffix,mat=mat or self.rubber)
        if world:
            matrix=np.eye(4);matrix[:3,:3]=C;o.matrix_world=Matrix(matrix.tolist())
        return o

    def decal(self,suffix,offset,dims,texture,axis="top"):
        w,h=dims
        vertices=([[-w/2,0,h/2],[w/2,0,h/2],[w/2,0,-h/2],[-w/2,0,-h/2]] if axis=="top" else [[-w/2,-h/2,0],[w/2,-h/2,0],[w/2,h/2,0],[-w/2,h/2,0]])
        mat=material("V2 "+self.name+" "+suffix,[255,255,255],.7,texture=texture)
        o=self.mesh(suffix,vertices,[(0,1,2,3)],mat,offset,smooth=False)
        uv=o.data.uv_layers.new()
        for loop,xy in zip(uv.data,[(0,0),(1,0),(1,1),(0,1)]):loop.uv=xy
        return o

    def keyboard(self,laptop=False):
        w,h,l=self.d
        shell=self.m
        self.loft("tapered_lower_case",[(-h/2,w*.96,l*.95,0),(-h*.3,w,l,0),(h*.28,w*.995,l*.985,0)],shell,exponent=12,slope=-.025)
        self.loft("upper_deck",[(h*.15,w*.985,l*.975,0),(h*.38,w*.97,l*.94,0)],shell,exponent=14,slope=-.025)
        keys=material("V2 "+self.name+" keycaps",[9,10,11] if laptop or np.mean(self.e["color"][:3])<180 else [173,171,161],.65)
        if laptop:
            span=w*.91;unit=span/15.2;left=-span/2;pitch=l*.102;back=-l*.39
        else:
            unit=w/25.3;left=-w*.465;pitch=l*.147;back=-l*.36
        def key(label,x,z,units=1):
            top=h*.42-.025*z
            self.box("key_"+label,[left+(x+units/2)*unit,top+.0025,z],[unit*(units-.12),.005,pitch*.77],keys,bevel=.0012)
        # Distinct function row, alphanumeric stagger, long modifier keys.
        for n in range(12):key("F"+str(n+1),2+n+(n//4)*.32,back,.76)
        key("escape",0,back,.85)
        rows=[([1]*13+[2],0),([1.5]+[1]*12+[1.5],.05),([1.8]+[1]*11+[2.2],.08),([2.3]+[1]*10+[2.7],.1)]
        for row,(widths,_) in enumerate(rows):
            x=0;z=back+(row+1)*pitch
            for j,u in enumerate(widths):key(f"{row}_{j}",x,z,u);x+=u
        x=0;z=back+5*pitch
        for j,u in enumerate([1.4,1.2,1.2,6.2,1.2,1.2,1.2,1.4]):key("bottom_"+str(j),x,z,u);x+=u
        if not laptop:
            for row in range(2):
                for col in range(3):key(f"nav_{row}_{col}",16+col,back+(row+1)*pitch)
            for col in range(3):key("arrow_"+str(col),16+col,back+5*pitch)
            key("arrow_up",17,back+4*pitch)
            for row in range(5):
                for col in range(4):
                    if col==3 and row in [2,4]:continue
                    if row==4 and col==1:continue
                    key(f"num_{row}_{col}",19.5+col,back+(row+1)*pitch,2 if row==4 and col==0 else 1)
            for i in range(3):self.ellipsoid("status_"+str(i),[w*(.34+i*.04),h*.48,-l*.40],[.003,.002,.003],material("V2 muted indicator",[52,88,34],.5))
        else:
            # On laptops keyboard occupies rear half; move generated key geometry back.
            for o in self.collection.objects:
                if "__key_" in o.name:
                    for v in o.data.vertices:v.co.z*=.70
                    local=np.linalg.inv(C@self.r)@(np.array(o.matrix_world.translation)-C@self.c)
                    local[2]=local[2]*.64-l*.15
                    o.matrix_world.translation=Vector(C@(self.c+self.r@local))
            self.box("touchpad_recess",[0,h*.48,l*.25],[w*.27,.001,l*.23],self.dark,.001)
            self.box("touchpad",[0,h*.50,l*.25],[w*.25,.001,l*.21],material("V2 touchpad black",[29,30,31],.8),.001)
            for side in [-1,1]:self.box("touchpad_click_"+str(side),[side*w*.065,h*.52,l*.39],[w*.12,.002,l*.045],self.dark,.001)
            self.ellipsoid("trackpoint",[0,h*.7,-l*.06],[.006,.003,.006],material("V2 trackpoint",[148,26,25],.75))
            for side in [-1,1]:self.cylinder("hinge_"+str(side),[side*w*.35,h*.8,-l*.45],[.014,w*.13,.014],self.dark,rot=np.array([[0,1,0],[1,0,0],[0,0,-1]]))
        if self.e.get("cable_points"):self.curve("observed_cable",self.e["cable_points"],world=True)

    def phone(self):
        w,h,l=self.d
        # Rear-high wedge with softer shoulders and a narrower lower footprint.
        self.loft("wedge_case",[(-h/2,w*.88,l*.91,0),(-h*.32,w*.98,l*.99,0),(h*.19,w,l,0),(h*.32,w*.94,l*.95,0)],exponent=7,slope=-.07)
        x=-w*.30
        receiver=material("V2 handset warm ABS",[190,187,167],.46)
        for sign in [-1,1]:
            self.box("receiver_cradle_"+str(sign),[x,h*.49,sign*l*.30],[w*.32,.013,l*.15],self.m,.004)
        # Continuous receiver whose ends broaden into the earpieces.
        n=48;verts=[]
        for t in np.linspace(-1,1,21):
            end=abs(t)**4
            z=t*l*.36;hw=w*(.065+.075*end);hy=.009+.010*end
            cy=h*.40+.009*(1-t*t)-.05*z
            for j in range(n):
                a=j/n*2*math.pi;verts.append([x+hw*math.cos(a),cy+hy*math.sin(a),z])
        faces=[tuple(reversed(range(n)))]
        for k in range(20):
            for j in range(n):faces.append((k*n+j,k*n+(j+1)%n,(k+1)*n+(j+1)%n,(k+1)*n+j))
        faces.append(tuple(20*n+j for j in range(n)))
        self.mesh("continuous_handset",verts,faces,receiver)
        keymat=material("V2 phone grey keys",[132,132,116],.65)
        for row in range(4):
            for col in range(3):
                z=l*(.04+row*.092)
                self.box(f"dial_{row}_{col}",[w*(.05+col*.135),h*.35-.07*z,z],[w*.105,.005,l*.068],keymat,.0015)
        self.box("directory_panel",[w*.16,h*.35+.07*l*.20,-l*.20],[w*.46,.002,l*.31],material("V2 directory ivory",[198,195,174],.85),.001)
        for j in range(7):
            z=-l*.30+j*l*.036
            self.box("directory_line_"+str(j),[w*.16,h*.38-.07*z,z],[w*.37,.0008,.0006],keymat,.0001)
        # Helical receiver cord on observed front/side of telephone; route is approximate.
        points=[]
        for t in np.linspace(0,1,300):
            a=t*24*2*math.pi
            points.append([x+w*.16*t+.006*math.cos(a),-h*.30+.006*math.sin(a),l*.48+l*.12*math.sin(t*math.pi)])
        self.curve("coiled_handset_cord",points,.0012,receiver,poly=True)

    def mouse(self):
        w,h,l=self.d
        self.loft("flat_base",[(-h*.5,w*.75,l*.86,0),(-h*.40,w*.9,l*.98,0)],self.rubber,exponent=2.7)
        self.loft("arched_shell",[(-h*.38,w*.92,l*.98,0),(-h*.1,w,l,0),(h*.22,w*.87,l*.78,l*.03),(h*.43,w*.48,l*.45,l*.07),(h*.49,w*.10,l*.10,l*.08)],exponent=2.5)
        for side in [-1,1]:
            self.curve("button_edge_"+str(side),[[side*w*.015,h*.38,-l*.07],[side*w*.017,h*.25,-l*.29],[side*w*.02,h*.02,-l*.46]],.00065)
        self.ellipsoid("wheel_socket",[0,h*.34,-l*.18],[w*.21,.004,l*.23],self.rubber)
        self.ellipsoid("scroll_wheel",[0,h*.41,-l*.18],[w*.13,.006,l*.16],material("V2 scroll wheel",[85,88,91],.7))
        if self.e.get("cable_points"):self.curve("observed_cable",self.e["cable_points"],world=True)

    def monitor(self):
        w,h,t=self.d;front=self.e.get("front",1)
        # Cross-section loft is rotated so its height axis becomes panel depth.
        rotate=np.array([[1,0,0],[0,0,-front],[0,front,0]])
        self.loft("contoured_back",[(-t*.8,w*.72,h*.70,0),(-t*.5,w*.93,h*.91,0),(t*.1,w,h,0),(t*.45,w*.99,h*.99,0)],self.m,exponent=12,rot=rotate)
        bezel=.017 if "silver" in self.name or self.name=="monitor" else .014
        screen=material("V2 dark LCD",[3,4,5],.65)
        self.box("screen_recess",[0,.003,front*(t*.48)],[w-2*bezel,h-bezel*2.2,.002],screen,.001)
        for side in [-1,1]:self.box("bezel_side_"+str(side),[side*(w-bezel)/2,0,front*t*.53],[bezel,h,.006],self.m,.0015)
        self.box("bezel_upper",[0,(h-bezel)/2,front*t*.53],[w-2*bezel,bezel,.006],self.m,.0015)
        self.box("bezel_lower",[0,-h/2+bezel*.7,front*t*.53],[w-2*bezel,bezel*1.4,.006],self.m,.0015)
        for j in range(4):self.box("control_"+str(j),[w*(.23+j*.055),-h*.474,front*(t*.55+.002)],[.007,.0025,.002],self.dark,.0006)
        support=self.e.get("support",self.e.get("table_height",self.c[1]-h*.68))
        baseworld=np.array([self.c[0],support+.012,self.c[2]-.015])
        baselocal=self.r.T@(baseworld-self.c)
        self.loft("elliptic_base",[(-.009,w*.42,w*.33,0),(-.004,w*.46,w*.37,0),(.005,w*.44,w*.35,0),(.010,w*.34,w*.26,0)],self.dark,exponent=2,offset=baselocal,rot=self.r.T)
        endlocal=self.r.T@(np.array([self.c[0],self.c[1]-h*.34,self.c[2]-.035])-self.c)
        self.curve("support_neck",[baselocal.tolist(),((baselocal+endlocal)/2+[0,0,-.015]).tolist(),endlocal.tolist()],.019,self.dark)
        self.box("rear_mount",[0,-h*.28,-front*t*.80],[w*.16,h*.22,t*.5],self.dark,.005)
        for j in range(12):self.box("rear_vent_"+str(j),[(j-5.5)*w*.027,h*.25,-front*t*.60],[w*.013,h*.08,.001],self.dark,.0005)

    def gamepad(self):
        w,h,l=self.d
        self.m=material("V2 controller black",[22,23,24],.52)
        self.loft("central_shell",[(-h*.43,w*.85,l*.6,-l*.10),(0,w,l*.71,-l*.10),(h*.42,w*.83,l*.55,-l*.11)],self.m,exponent=3)
        for side in [-1,1]:
            self.loft("continuous_grip_"+str(side),[(-h*.4,w*.29,l*.65,0),(0,w*.38,l*.76,0),(h*.3,w*.28,l*.66,0)],self.m,exponent=2.5,offset=[side*w*.32,0,l*.19])
            self.cylinder("analog_socket_"+str(side),[side*w*.16,h*.45,l*.03],[.024,.004,.024],self.rubber)
            self.cylinder("analog_stick_"+str(side),[side*w*.16,h*.6,l*.03],[.018,.009,.018],self.rubber)
        for dims in [[.020,.005,.007],[.007,.005,.020]]:self.box("dpad_"+str(dims[0]),[-w*.30,h*.43,-l*.19],dims,self.rubber,.001)
        for j,(dx,dz) in enumerate([(0,-1),(1,0),(0,1),(-1,0)]):self.cylinder("action_"+str(j),[w*.30+dx*.007,h*.43,-l*.19+dz*.007],[.006,.004,.006],material("V2 action "+str(j),[[58,99,57],[133,45,43],[53,70,123],[154,125,45]][j],.4))

    def apparatus(self):
        self.cylinder('vertical_rod',[.025,.34,0],[.009,.68,.009],self.steel)
        for sign in [-1,1]:
            self.curve('cast_foot_'+str(sign),[[.025,.023,0],[.025+sign*.075,.014,.035],[.025+sign*.12,.012,.075]],.012,self.steel)
        self.curve('rear_foot',[[.025,.023,0],[.025,.012,-.10]],.012,self.steel)
        self.cylinder('clamp',[.025,.21,0],[.023,.033,.023],self.dark)
        self.curve('ring_arm',[[.025,.21,0],[.025,.21,.04]],.007,self.steel)
        # Thick observed black circular housing with an open aperture.
        points=[[.025+.045*math.cos(t),.21+.045*math.sin(t),.043] for t in np.linspace(0,2*math.pi,100)]
        self.curve('black_ring',points,.012,self.dark,poly=True)
        # Bell-like spun metal body with a neck, shoulder and rounded lower skirt.
        radii=[(.07,.022),(.085,.047),(.12,.058),(.16,.055),(.19,.037),(.205,.012)]
        vertices=[];n=64
        for y,rad in radii:
            for j in range(n):
                theta=j/n*math.tau
                rr=rad*(1+.025*math.cos(theta*12))
                vertices.append([-.075+rr*math.cos(theta),y,rr*math.sin(theta)])
        faces=[tuple(reversed(range(n)))]
        for k in range(len(radii)-1):
            for j in range(n):faces.append((k*n+j,k*n+(j+1)%n,(k+1)*n+(j+1)%n,(k+1)*n+j))
        faces.append(tuple((len(radii)-1)*n+j for j in range(n)))
        self.mesh('fluted_metal_bell',vertices,faces,self.steel)
        self.curve('bell_attachment',[[-.075,.205,0],[-.075,.235,0],[.025,.235,0]],.004,self.steel)

    def build(self):
        fn={"keyboard":self.keyboard,"laptop":lambda:self.keyboard(True),"telephone":self.phone,"mouse":self.mouse,"monitor":self.monitor,"gamepad":self.gamepad,"apparatus":self.apparatus}.get(self.e["template"])
        if fn:fn()
        return bool(fn)
