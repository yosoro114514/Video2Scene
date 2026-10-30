"""Data-only mesh compiler in normalized right-handed object Y-up coordinates.

This file is copied and hash-pinned beside the Blender worker. It never imports
model code, reads a supplied path, or executes a string from a recipe.
"""
import math

MAX_OBJECT_VERTICES = 50000
MAX_OBJECT_FACES = 50000


def add(a, b):
    return [x+y for x,y in zip(a,b)]


def subtract(a, b):
    return [x-y for x,y in zip(a,b)]


def multiply(a, s):
    return [x*s for x in a]


def dot(a, b):
    return sum(x*y for x,y in zip(a,b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def unit(v):
    length = math.sqrt(dot(v,v))
    if length < 1e-9:
        raise ValueError('Recipe contains a zero-length direction')
    return multiply(v,1/length)


def transform(vertices, center, quaternion):
    x,y,z,w = quaternion
    norm=math.sqrt(x*x+y*y+z*z+w*w)
    if abs(norm*norm-1)>1e-3:
        raise ValueError('Part rotation is too far from a unit quaternion')
    x,y,z,w=[v/norm for v in quaternion]
    rotation = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return [[dot(row,point)+center[i] for i,row in enumerate(rotation)] for point in vertices]


def compile_part(part):
    kind=part['kind']; vertices=[]; faces=[]
    if kind=='box':
        x,y,z=[v/2 for v in part['dimensions']]
        vertices=[[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]]
        faces=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]]
    elif kind=='lathe':
        profile,n=part['profile'],part['segments']
        for radius,height in profile:
            vertices.extend([[radius*math.cos(i*2*math.pi/n),height,radius*math.sin(i*2*math.pi/n)] for i in range(n)])
        for j in range(len(profile)-1):
            faces.extend([[j*n+i,(j+1)*n+i,(j+1)*n+(i+1)%n,j*n+(i+1)%n] for i in range(n)])
    elif kind=='torus':
        major,minor,n,m=part['major_radius'],part['tube_radius'],part['major_segments'],part['minor_segments']
        for i in range(n):
            a=i*2*math.pi/n
            for j in range(m):
                b=j*2*math.pi/m
                vertices.append([(major+minor*math.cos(b))*math.cos(a),minor*math.sin(b),(major+minor*math.cos(b))*math.sin(a)])
        for i in range(n):
            for j in range(m):
                faces.append([i*m+j,i*m+(j+1)%m,((i+1)%n)*m+(j+1)%m,((i+1)%n)*m+j])
    elif kind=='tube':
        points,radius,n=part['path'],part['radius'],part['sides']
        previous=None
        for i,point in enumerate(points):
            tangent=unit(subtract(points[min(i+1,len(points)-1)],points[max(0,i-1)]))
            if previous is None:
                reference=[0,1,0] if abs(tangent[1])<.9 else [1,0,0]
                normal=unit(cross(tangent,reference))
            else:
                normal_candidate=subtract(previous,multiply(tangent,dot(previous,tangent)))
                normal=unit(normal_candidate) if dot(normal_candidate,normal_candidate)>1e-12 else unit(cross(tangent,[1,0,0] if abs(tangent[0])<.9 else [0,0,1]))
            binormal=unit(cross(tangent,normal));previous=normal
            for j in range(n):
                angle=j*2*math.pi/n
                vertices.append(add(point,add(multiply(normal,radius*math.cos(angle)),multiply(binormal,radius*math.sin(angle)))))
        for i in range(len(points)-1):
            faces.extend([[i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j] for j in range(n)])
        if part['caps']:
            faces.extend([list(reversed(range(n))),[(len(points)-1)*n+j for j in range(n)]])
    elif kind=='mesh':
        vertices=[list(v) for v in part['vertices']];faces=[list(f) for f in part['faces']]
    else:
        raise ValueError('Unsupported recipe primitive: '+str(kind))
    if not vertices or not faces or len(vertices)>MAX_OBJECT_VERTICES or len(faces)>MAX_OBJECT_FACES:
        raise ValueError('Empty or over-budget recipe geometry')
    if any(not math.isfinite(v) or abs(v)>8 for point in vertices for v in point):
        raise ValueError('Recipe vertex is nonfinite or outside normalized range')
    if any(len(face)<3 or len(set(face))<3 or any(type(i) is not int or i<0 or i>=len(vertices) for i in face) for face in faces):
        raise ValueError('Invalid recipe face index')
    vertices=transform(vertices,part['center'],part['rotation_xyzw'])
    return vertices,faces


def compile_object(recipe, bounds):
    """Compile every model-authored part; fit their union to the locked bounds."""
    parts=[];count=faces_count=0
    for part in recipe['parts']:
        vertices,faces=compile_part(part);count+=len(vertices);faces_count+=len(faces)
        if count>MAX_OBJECT_VERTICES or faces_count>MAX_OBJECT_FACES:
            raise ValueError('Object recipe exceeds total mesh budget')
        parts.append({'name':part['name'],'vertices':vertices,'faces':faces,'material':part['material'],'smooth':part['smooth']})
    lower=[min(v[i] for part in parts for v in part['vertices']) for i in range(3)]
    upper=[max(v[i] for part in parts for v in part['vertices']) for i in range(3)]
    if any(upper[i]-lower[i]<1e-6 for i in range(3)):
        raise ValueError('Recipe union must have positive 3D extent')
    for part in parts:
        part['vertices']=[[bounds['min'][i]+(point[i]-lower[i])/(upper[i]-lower[i])*(bounds['max'][i]-bounds['min'][i]) for i in range(3)] for point in part['vertices']]
    normalizations=[{'part':p['name'],'input':p['rotation_xyzw'],'normalized':multiply(p['rotation_xyzw'],1/math.sqrt(dot(p['rotation_xyzw'],p['rotation_xyzw'])))}
                    for p in recipe['parts'] if abs(dot(p['rotation_xyzw'],p['rotation_xyzw'])-1)>1e-12]
    return parts,{'input_normalized_bounds':{'min':lower,'max':upper},'fit_mode':'per-axis union fit to locked object_y_up bounds',
                  'vertex_count':count,'face_count':faces_count,'part_count':len(parts),'part_rotation_numeric_normalization':normalizations}
