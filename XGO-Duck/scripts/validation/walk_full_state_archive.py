"""Store a live walking snapshot as a data-only archive and read it back."""
from dataclasses import fields, is_dataclass, replace
import json

import numpy as np
import torch
import warp as wp


def encode(value, arrays):
    if isinstance(value,(torch.Tensor,wp.array,np.ndarray,np.generic)):
        index = f'a{len(arrays):04d}'
        if isinstance(value,torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            kind = 'torch'
        elif isinstance(value,wp.array):
            array,kind = value.numpy(),'warp'
        else:
            array = np.asarray(value)
            kind = 'scalar' if isinstance(value,np.generic) else 'numpy'
        assert not array.dtype.hasobject
        arrays[index] = array
        return [kind,index]
    if isinstance(value,dict):
        assert all(type(key) in (str,int) for key in value)
        return ['dict',[[key,encode(item,arrays)] for key,item in value.items()]]
    if isinstance(value,(tuple,list)):
        return [type(value).__name__,[encode(item,arrays) for item in value]]
    if is_dataclass(value):
        return ['record',[[field.name,encode(getattr(value,field.name),arrays)] for field in fields(value)]]
    assert type(value) in (str,int,float,bool) or value is None,type(value)
    return ['value',value]


def decode(node, template, arrays):
    kind,value = node
    if kind in ('warp','torch','numpy','scalar'):
        source = arrays[value]
        if kind=='warp':
            assert isinstance(template,wp.array)
            if template.size:
                cpu = wp.array(source,dtype=template.dtype,device='cpu')
                assert cpu.shape==template.shape
                wp.copy(template,cpu)
                if template.device.is_cuda:
                    wp.synchronize_device(template.device)
            else:
                assert source.size==0
            return template
        if kind=='torch':
            assert isinstance(template,torch.Tensor) and tuple(template.shape)==source.shape
            tensor = torch.from_numpy(source)
            assert tensor.dtype==template.dtype
            template.copy_(tensor)
            return template
        assert isinstance(template,(np.ndarray,np.generic)) and source.dtype==template.dtype
        assert source.shape==template.shape
        return source[()] if kind=='scalar' else source
    if kind=='dict':
        assert isinstance(template,dict) and [key for key,_ in value]==list(template)
        return {key:decode(item,template[key],arrays) for key,item in value}
    if kind in ('tuple','list'):
        assert type(template).__name__==kind and len(value)==len(template)
        return type(template)(decode(item,old,arrays) for item,old in zip(value,template,strict=True))
    if kind=='record':
        assert is_dataclass(template) and [key for key,_ in value]==[field.name for field in fields(template)]
        return replace(template,**{key:decode(item,getattr(template,key),arrays) for key,item in value})
    assert kind=='value' and type(template) is type(value)
    return value


def save(path, payload):
    assert not path.exists()
    arrays = {}
    layout = encode(payload,arrays)
    arrays['layout'] = np.frombuffer(json.dumps(layout,separators=(',',':')).encode(),dtype=np.uint8)
    temporary = path.with_suffix('.tmp.npz')
    assert not temporary.exists()
    try:
        np.savez_compressed(temporary,**arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(arrays)-1


def load(path, template):
    with np.load(path,allow_pickle=False) as archive:
        layout = json.loads(archive['layout'].tobytes())
        return decode(layout,template,archive)
