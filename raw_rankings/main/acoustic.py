"""Read only the fields used in the official VGGish TFRecord SequenceExamples."""
import struct
from pathlib import Path
import numpy as np

def varint(buf,i):
    value=shift=0
    while True:
        b=buf[i]
        i+=1
        value|=(b&127)<<shift
        if b<128:
            return value,i
        shift+=7

def fields(buf):
    i=0
    while i<len(buf):
        key,i=varint(buf,i)
        number,wire=key>>3,key&7
        if wire==0:
            data,i=varint(buf,i)
        elif wire==2:
            n,i=varint(buf,i)
            data=buf[i:i+n]
            i+=n
        elif wire in (1,5):
            n={1:8,5:4}[wire]
            data=buf[i:i+n]
            i+=n
        else:
            raise ValueError(wire)
        yield number,data

def map_fields(buf):
    return {dict(fields(pair))[1].decode():dict(fields(pair))[2] for number,pair in fields(buf) if number==1}

def record(path):
    with path.open("rb") as f:
        while raw:=f.read(8):
            size,=struct.unpack("<Q",raw)
            f.read(4)
            yield f.read(size)
            assert len(f.read(4))==4

def embed(root,weak):
    vectors={}
    # The official metadata has distinct video IDs per split.
    keys={k[0]:k for k in weak}
    assert len(keys)==len(weak)
    for path in sorted(root.glob("*.tfrecord")):
        for buf in record(path):
            message=dict(fields(buf))
            metadata=map_fields(message[1])
            video=dict(fields(dict(fields(metadata["video_id"]))[1]))[1].decode()
            if video not in keys:
                continue
            embedding=map_fields(message[2])["audio_embedding"]
            chunks=[dict(fields(dict(fields(frame))[1]))[1] for _,frame in fields(embedding)]
            v=np.frombuffer(b"".join(chunks),dtype=np.uint8).reshape(-1,128)
            assert len(v)>0
            key=keys[video]
            assert key not in vectors
            if "start_time_seconds" in metadata:
                raw=dict(fields(metadata["start_time_seconds"]))
                if 2 in raw:
                    binary=dict(fields(raw[2]))[1]
                    assert abs(struct.unpack("<f",binary)[0]*1000-key[1])<=1
            vectors[key]=np.r_[v.mean(axis=0),v.max(axis=0)]
    return vectors

def predict(features,train,test,index):
    train_embed=embed(features/"bal_train",train)
    test_embed=embed(features/"eval",test)
    assert len(train_embed)>=21000 and len(test_embed)>=20000
    keys=sorted(train_embed)
    x=np.stack([train_embed[k] for k in keys])
    y=np.zeros((len(x),len(index)),dtype=float)
    for j,k in enumerate(keys):
        y[j,[index[mid] for mid in train[k][1]]]=1.
    center=x.mean(0)
    scale=x.std(0).clip(1e-5)
    x=(x-center)/scale
    bias=y.mean(0)
    w=np.linalg.solve(x.T@x+100*np.eye(x.shape[1]),x.T@(y-bias))
    test_keys=sorted(test_embed)
    predicted=(np.stack([test_embed[k] for k in test_keys])-center)/scale@w+bias
    return dict(zip(test_keys,predicted)),{"acoustic_train":len(keys),"acoustic_eval":len(test_keys)}
