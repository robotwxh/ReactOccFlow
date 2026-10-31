"""Official nuScenes groups and train-only observation normalization.

Official train/val selection is adapted after copying the OpenMMLab converter
(Copyright (c) OpenMMLab. All rights reserved. Modified by Zhiqi Li).
The 80/10/10 hash grouping and masked streaming statistics are new after a
search of both references; see docs/provenance/split_sources.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from reactoccflow.provenance import sha256_file


def grouped_scene_split(official_train, official_val, *, seed=2026):
    """Stable whole-scene groups; never use frame counts or future events."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('Split seed must be an integer, excluding bool')
    seed = int(seed)
    train, val = list(official_train), list(official_val)
    if len(train) < 10 or not val:
        raise ValueError('Need >=10 official training scenes and nonempty official validation')
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError('Duplicate official scene')
    if set(train) & set(val):
        raise ValueError('Official train/val scenes overlap')
    if not all(isinstance(scene,str) and scene for scene in train+val):
        raise ValueError('Scene identities must be nonempty strings')
    ordered = sorted(train, key=lambda scene: (hashlib.sha256(f'{seed}:{scene}'.encode()).hexdigest(), scene))
    n_train, n_tune = len(train)*8//10, len(train)//10
    return {'version':1, 'seed':int(seed), 'group_key':'official_scene_name',
            'hash_scheme':'sha256(str(seed) + ":" + scene), ascending hex, scene tie-break',
            'groups':{'train':sorted(ordered[:n_train]),
                      'tune':sorted(ordered[n_train:n_train+n_tune]),
                      'calibration':sorted(ordered[n_train+n_tune:]),
                      'evaluation':sorted(val)},
            'rules':{'normalization_fit_group':'train','evaluation_usage':'final_evaluation_only',
                     'counterfactual_branches':'all actions and random seeds retain parent scene group'}}


def validate_scene_membership(split, scenes):
    membership = {}
    if set(split['groups']) != {'train','tune','calibration','evaluation'}:
        raise ValueError('Expected four fixed groups')
    for group, values in split['groups'].items():
        for scene in values:
            if scene in membership: raise ValueError(f'Duplicate scene group: {scene}')
            membership[scene] = group
    requested=list(scenes)
    missing=set(requested)-membership.keys()
    if missing: raise ValueError(f'Unknown scenes: {sorted(missing)}')
    return {scene:membership[scene] for scene in requested}


class _Moments:
    def __init__(self, width):
        self.n=np.zeros(width,np.int64)
        self.mean=np.zeros(width,np.float64)
        self.m2=np.zeros(width,np.float64)

    def update(self, values, mask):
        values=np.asarray(values,np.float64).reshape(-1,len(self.n))
        mask=np.broadcast_to(np.asarray(mask,bool),values.shape)
        for dim in range(len(self.n)):
            x=values[mask[:,dim],dim]
            if not len(x): continue
            if not np.isfinite(x).all(): raise ValueError('Non-finite valid observation')
            n=len(x); mean=x.mean(); m2=np.square(x-mean).sum()
            delta=mean-self.mean[dim]; total=self.n[dim]+n
            self.m2[dim] += m2 + delta*delta*self.n[dim]*n/total
            self.mean[dim] += delta*n/total
            self.n[dim]=total

    def result(self):
        if (self.n==0).any(): raise ValueError('No training observations for normalization dimension')
        std=np.sqrt(self.m2/self.n)
        return {'count':self.n.tolist(),'mean':self.mean.tolist(),
                'std':np.where(std>1e-8,std,1.).tolist(),
                'constant_dimension':(std<=1e-8).tolist(),'variance':'population'}


def fit_observation_normalization(cache_root, split):
    """Only read train observation shards; neither targets nor held-out arrays."""
    root=Path(cache_root)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('coordinate_frame')!='E0': raise ValueError('Expected E0 research labels')
    scenes, shard_names = set(), set()
    for shard in manifest['shards']:
        scene, name = shard['scene'], shard['file']
        if not isinstance(scene, str) or not scene or not isinstance(name, str) or not name:
            raise ValueError('Shard scene and file names must be nonempty strings')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe shard path')
        if scene in scenes:
            raise ValueError(f'Duplicate shard scene: {scene}')
        if relative in shard_names:
            raise ValueError(f'Duplicate shard file: {name}')
        scenes.add(scene)
        shard_names.add(relative)
    membership=validate_scene_membership(split,[s['scene'] for s in manifest['shards']])
    xy, velocity, ego = _Moments(2), _Moments(2), _Moments(4)
    files=[]
    for shard in manifest['shards']:
        if membership[shard['scene']]!='train': continue
        relative=Path(shard['file'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe shard path')
        path=root/relative
        with np.load(path,allow_pickle=False) as data:
            states=data['observation.actor_history']
            observed=data['observation.actor_observed_mask']
            velocity_valid=data['observation.actor_velocity_mask']
            ego_states=data['observation.ego_state']
            ego_valid=data['observation.ego_state_mask']
            if states.ndim != 4 or states.shape[-1] != 6 or states.dtype.kind not in 'fiu':
                raise ValueError('actor_history must be a numeric rank4 array with feature width6')
            for name, mask in (('actor_observed_mask', observed), ('actor_velocity_mask', velocity_valid)):
                if mask.shape != states.shape[:-1] or mask.dtype.kind != 'b':
                    raise ValueError(f'{name} must be boolean with exact actor state shape excluding features')
            if (velocity_valid & ~observed).any():
                raise ValueError('actor velocity mask must be a subset of actor observed mask')
            if ego_states.ndim != 2 or ego_states.shape != (states.shape[0], 4) or ego_states.dtype.kind not in 'fiu':
                raise ValueError('ego_state must be a numeric rank2 array with matching sample count and width4')
            if ego_valid.shape != ego_states.shape or ego_valid.dtype.kind != 'b':
                raise ValueError('ego_state_mask must be boolean and exactly match ego_state shape')
            xy.update(states[...,:2].reshape(-1,2),observed.reshape(-1,1))
            velocity.update(states[...,4:].reshape(-1,2),velocity_valid.reshape(-1,1))
            ego.update(ego_states,ego_valid)
        files.append({'scene':shard['scene'],'file':str(relative),'sha256':sha256_file(path)})
    canonical=json.dumps(split,sort_keys=True,separators=(',',':')).encode()
    return {'version':1,'fit_group':'train','coordinate_frame':'E0','heading':'sin/cos unchanged',
            'source_manifest_sha256':sha256_file(root/'manifest.json'),
            'split_sha256':hashlib.sha256(canonical).hexdigest(),'files':files,
            'actor_xy':xy.result(),'actor_velocity':velocity.result(),'ego_state':ego.result()}
