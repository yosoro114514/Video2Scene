"""Conservative multipart priors for whole-object detector bounds.

These structures are estimates, not measurements of hidden parts. Kept separate
from the explicitly reviewed, component-level refinement assets.
"""
import math
import json
import numpy as np
from .blender_assets import Asset, material


class AutomaticAsset(Asset):
    def __init__(self,e,output):
        super().__init__(e,output)
        self.parent['assumptions']=json.dumps(e.get('assumptions',[]),ensure_ascii=False)
        self.parent['review_history']=json.dumps(e.get('review_history',[]),ensure_ascii=False)
        self.parent['fitted_parts']=json.dumps(e.get('fitted_parts',{}),ensure_ascii=False)

    def part_frame(self,part):
        self.c=np.array(part['center']);self.r=np.array(part['rotation']);self.d=np.array(part['dimensions'])

    def fitted_monitor(self,part):
        self.part_frame(part);w,h,_=self.d;bezel=.010
        t=min(.045,max(.018,float(self.e['dimensions'][2])))
        self.c=self.c-self.r[:,2]*(t*.48+.001);self.d=np.array([w+2*bezel,h+2*bezel,t])
        self.monitor()
        if part.get('texture'):self.decal('video_screen',[0,0,t*.48+.002],[w,h],part['texture'],axis='front')

    def fitted_chair(self,part):
        self.part_frame(part);w,h,_=self.d;backcenter=self.c.copy();backrotation=self.r.copy()
        self.box('back_frame',[0,0,0],[w,h,.027],self.m,.022)
        inset=material('Reviewed chair inset '+self.name,[32,35,39],.8)
        count=int(self.e.get('parameters',{}).get('back_slats',8));count=max(3,min(count,16))
        for i in range(count):
            self.box('back_inset_'+str(i),[0,h*(-.37+i*.74/(count-1)),.016],[w*.82,h*.028,.003],inset,.002)
        front=backrotation[:,2].copy();front[1]=0;front/=max(np.linalg.norm(front),1e-8)
        up=np.array([0.,1.,0.]);right=np.cross(up,front);self.r=np.column_stack([right,up,front])
        depth=min(.65,max(.30,w*.93));bottom=backcenter-backrotation[:,1]*h*.5
        self.c=bottom+front*depth*.42-up*.025
        seat=material('Reviewed seat '+self.name,self.e.get('parameters',{}).get('seat_color',[24,27,32]),.8)
        self.loft('seat_cushion',[(-.04,w*.93,depth*.92,0),(-.015,w*1.01,depth,0),(.018,w*.97,depth*.96,0)],seat,exponent=5)
        arm=material('Reviewed chair arms '+self.name,self.e.get('parameters',{}).get('arm_color',[185,188,189]),.58)
        for side in [-1,1]:
            self.curve('arm_'+str(side),[[side*w*.48,-.03,-depth*.34],[side*w*.55,.15,-depth*.30],[side*w*.56,.17,depth*.24],[side*w*.50,.02,depth*.38]],.018,arm)

    def desk(self):
        part=self.e.get('fitted_parts',{}).get('desk_top')
        if part:self.part_frame(part)
        w,h,t=self.d;thickness=float(self.e.get('parameters',{}).get('desk_thickness',.025))
        if part:
            self.box('tabletop',[0,0,-thickness/2],[w,h,thickness],self.m,.012)
        else:
            self.box('tabletop',[0,0,0],self.d,self.m,.01)
        legs=float(self.e.get('parameters',{}).get('desk_leg_height',0))
        if legs>0:
            # World vertical legs only when the review explicitly observes them.
            if part:
                for x in [-.43,.43]:
                    for y in [-.43,.43]:
                        top=self.c+self.r@np.array([w*x,h*y,0]);local=self.r.T@(top-np.array([0,legs/2,0])-self.c)
                        self.box('leg_'+str(x)+'_'+str(y),local,[.035,legs,.035],self.dark,.002,self.r.T)

    def fitted_panel(self,part):
        self.part_frame(part);w,h,t=self.d
        self.box('observed_panel',[0,0,-t/2],[w,h,t],self.m,.002)
        if part.get('texture'):self.decal('video_panel',[0,0,.001],[w,h],part['texture'],axis='front')

    def laptop_assembly(self):
        w,h,l=self.d.copy();center=self.c.copy();dims=self.d.copy()
        parameters=self.e.get('parameters',{});part=self.e.get('fitted_parts',{}).get('screen')
        thickness=float(parameters.get('base_thickness',min(.022,h*.13)))
        if part:
            screen_r=np.array(part['rotation']);screen_c=np.array(part['center']);sw,sh,_=part['dimensions']
            front=screen_r[:,2].copy();front[1]=0;front/=max(np.linalg.norm(front),1e-8)
            self.r=np.column_stack([np.cross([0,1,0],front),[0,1,0],front])
            hinge=screen_c-screen_r[:,1]*sh/2
            depth=min(.38,max(.20,l*.86));self.c=hinge+front*depth*.43-np.array([0,thickness*.55,0]);self.d=np.array([sw*1.08,thickness,depth])
            self.m=material('Reviewed laptop case '+self.name,self.e.get('color',[29,32,37]),.55);self.keyboard(True)
            self.part_frame(part);self.box('display_back',[0,0,-.004],[sw*1.065,sh*1.10,.008],self.m,.003)
            self.box('display_glass',[0,0,.001],[sw,sh,.0015],self.dark,.001)
            if part.get('texture'):self.decal('video_screen',[0,0,.002],[sw,sh],part['texture'],axis='front')
            return
        # A detection includes both the display and the deck; do not use its
        # entire vertical extent as the keyboard chassis thickness.
        self.c=center+self.r@np.array([0,-h/2+thickness/2,l*.04])
        self.d=np.array([w,thickness,l*.86])
        self.m=material('Automatic laptop case '+self.name,self.e.get('color',[29,32,37]),.55)
        self.keyboard(True)
        self.c=center;self.d=dims
        lean=math.radians(parameters.get('screen_lean_deg',12));panel_h=(h-thickness)/math.cos(lean)
        rotation=np.array([[1,0,0],[0,math.cos(lean),math.sin(lean)],[0,-math.sin(lean),math.cos(lean)]])
        panel_center=np.array([0,-h/2+thickness+(h-thickness)/2,-l*.36-panel_h*math.sin(lean)/2])
        t=min(.009,w*.025)
        self.box('display_back',panel_center,[w,panel_h,t],self.m,.003,rotation)
        self.box('display_glass',panel_center+rotation@np.array([0,0,t*.55]),[w*.92,panel_h*.88,.0015],material('Automatic laptop LCD',[27,30,40],.35),.001,rotation)
        self.ellipsoid('webcam',panel_center+rotation@np.array([0,panel_h*.465,t*.65]),[.003,.003,.001],self.dark,rotation)

    def chair(self):
        w,h,l=self.d
        # Only the observed bounds are filled. Feet below an occluded seat are
        # deliberately left for later evidence-backed completion.
        seat_y=-h*.40;seat_depth=l*.83
        self.loft('seat_cushion',[(seat_y-h*.06,w*.83,seat_depth*.9,l*.04),(seat_y,w*.88,seat_depth,l*.04),(seat_y+h*.055,w*.84,seat_depth*.96,l*.04)],self.m,exponent=5)
        back_y=seat_y+h*.50;back_h=h*.80
        self.box('back_frame',[0,back_y,-l*.36],[w*.84,back_h,l*.075],self.m,.018)
        slat=material('Automatic chair inset '+self.name,[44,48,53],.8)
        count=max(3,min(16,int(self.e.get('parameters',{}).get('back_slats',7))))
        for i in range(count):
            self.box('back_inset_'+str(i),[0,back_y-back_h*.35+i*back_h*.70/(count-1),-l*.31],[w*.70,back_h*.045,.008],slat,.004)
        arm=material('Automatic chair arm '+self.name,self.e.get('parameters',{}).get('arm_color',[170,173,173]),.6)
        for sign in [-1,1]:
            self.curve('arm_'+str(sign),[[sign*w*.43,seat_y,-l*.25],[sign*w*.48,seat_y+h*.29,-l*.20],[sign*w*.48,seat_y+h*.31,l*.18],[sign*w*.43,seat_y+h*.05,l*.29]],min(w*.035,.018),arm)

    def bottle(self):
        w,h,l=self.d;diam=min(w,l)
        self.loft('bottle_body',[(-h*.5,diam*.84,diam*.84,0),(-h*.46,diam,diam,0),(h*.20,diam,diam,0),(h*.32,diam*.44,diam*.44,0),(h*.43,diam*.42,diam*.42,0)],self.m,exponent=2)
        cap=material('Bottle cap '+self.name,self.e.get('parameters',{}).get('bottle_cap_color',[17,19,21]))
        self.cylinder('cap',[0,h*.465,0],[diam*.46,h*.07,diam*.46],cap)

    def cup(self):
        w,h,l=self.d;diam=min(w,l)
        self.loft('cup_body',[(-h*.5,diam*.70,diam*.70,0),(h*.47,diam,diam,0)],self.m,exponent=2)
        self.cylinder('cup_rim',[0,h*.47,0],[diam,h*.035,diam],self.m)

    def backpack(self):
        w,h,l=self.d
        self.loft('fabric_body',[(-h*.5,w*.8,l*.8,0),(-h*.35,w,l,0),(h*.25,w*.91,l*.82,-l*.05),(h*.48,w*.5,l*.55,-l*.07)],self.m,exponent=3)
        for side in [-1,1]:
            self.curve('strap_'+str(side),[[side*w*.24,h*.39,-l*.30],[side*w*.32,0,-l*.60],[side*w*.25,-h*.37,-l*.31]],min(.012,w*.03),self.dark)

    def build(self):
        fitted=self.e.get('fitted_parts',{});kind=self.e['template']
        if kind=='monitor' and 'screen' in fitted:self.fitted_monitor(fitted['screen']);return True
        if kind=='chair' and 'chair_back' in fitted:self.fitted_chair(fitted['chair_back']);return True
        if kind=='proxy' and 'panel' in fitted:self.fitted_panel(fitted['panel']);return True
        fn={'laptop_assembly':self.laptop_assembly,'chair':self.chair,'bottle':self.bottle,'cup':self.cup,'backpack':self.backpack,'desk':self.desk}.get(self.e['template'])
        if fn:
            fn();return True
        return super().build()
