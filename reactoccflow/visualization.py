"""Offline oracle-label QA; XY transposition adapted from copied IR-WM plotter.

Source snapshot/checksums: docs/provenance/visualization_sources.json.
Matplotlib is optional and imported only when rendering. Future paths are
label overlays for inspection; this module is never an inference input.
"""
from __future__ import annotations
import numpy as np


# Occ3D18 colors; unknown is independent from free space.
PALETTE=np.array([
    [90,90,90],[180,110,70],[220,30,30],[230,160,20],[0,115,210],
    [170,90,50],[200,20,160],[230,40,65],[240,130,30],[110,70,170],
    [60,70,210],[150,150,150],[190,170,140],[210,180,190],[140,170,90],
    [90,110,130],[50,130,60],[245,245,245]],dtype=np.uint8)


def semantic_bev(semantics):
    """Topmost nonfree known voxel, otherwise known free or unknown255."""
    source=np.asarray(semantics)
    if source.ndim!=3 or source.shape[2]<1: raise ValueError('Expected nonempty XYZ semantic grid')
    if not np.isin(source,[*range(18),255]).all(): raise ValueError('Invalid Occ3D label')
    occupied=(source!=17)&(source!=255)
    top=source.shape[2]-1-np.argmax(occupied[...,::-1],axis=-1)
    out=np.take_along_axis(source,top[...,None],axis=2)[...,0].copy()
    out[~occupied.any(axis=2)]=np.where((source==17).any(axis=2),17,255)[~occupied.any(axis=2)]
    return out


def actor_footprint(state,size):
    """Four current body corners, size=(length,width,height), state x/y/sin/cos."""
    state,size=np.asarray(state),np.asarray(size)
    if state.shape!=(6,) or size.shape!=(3,) or not np.isfinite(state).all() or (size<=0).any():
        raise ValueError('Expected finite actor state and positive LWH dimensions')
    yaw=np.arctan2(state[2],state[3])
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*size[:2]/2
    return corners@rotation.T+state[:2]


def render_label_panel(axis,semantics,observation,target=None,*,title='',extent=(-40,40,-40,40)):
    from matplotlib.patches import Polygon
    bev=semantic_bev(semantics)
    image=np.full((*bev.shape,3),35,dtype=np.uint8)
    known=bev!=255; image[known]=PALETTE[bev[known]]
    axis.imshow(image.transpose(1,0,2),origin='lower',extent=extent,interpolation='nearest')
    history=observation['actor_history']; masks=observation['actor_observed_mask']
    for actor in range(len(history)):
        if not masks[actor,-1]: continue
        valid=masks[actor]
        # NaNs split missing observations rather than draw a fictitious bridge.
        line=np.where(valid[:,None],history[actor,:,:2],np.nan)
        axis.plot(line[:,0],line[:,1],color='#F7E463',linewidth=1)
        axis.add_patch(Polygon(actor_footprint(history[actor,-1],observation['actor_size'][actor]),
                              fill=False,edgecolor='#FF3344',linewidth=1))
        if target is not None:
            valid=target['actor_target_mask'][actor]
            line=np.where(valid[:,None],target['actor_future'][actor,:,:2],np.nan)
            axis.plot(line[:,0],line[:,1],'--',color='#00FFFF',linewidth=.8)
    if target is not None:
        line=np.where(target['ego_target_mask'][:,None],target['ego_future'][:,:2],np.nan)
        axis.plot(np.r_[0,line[:,0]],np.r_[0,line[:,1]],color='#FF8C00',linewidth=1.5)
    axis.arrow(0,0,3,0,width=.25,color='#111111',length_includes_head=True)
    axis.set(xlim=extent[:2],ylim=extent[2:],xlabel='E0 x / m',ylabel='E0 y / m',title=title)
    axis.set_aspect('equal')
    return axis
