from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from tapestry_m1.reranker.distributed_groups import epoch_order,local_batch,checkpoint_steps,lr_factor,accumulate_groups


def test_exact_epoch_and_tail():
    order=epoch_order(42509,20261111);seen=[]
    assert checkpoint_steps(42509,8,[.1,.25,.5,.75,1.])==[532,1329,2657,3986,5314]
    for step in range(1,5315):
        for rank in range(4):
            local,n=local_batch(order,step,8,rank,4);seen.extend(local)
            assert n==(5 if step==5314 else 8)
    assert len(seen)==len(set(seen))==42509
    assert sorted(seen)==list(range(42509))
    assert [len(local_batch(order,5314,8,r,4)[0]) for r in range(4)]==[2,1,1,1]
    assert lr_factor(1,5314,10)==.1
    assert lr_factor(10,5314,10)==1.
    assert lr_factor(5314,5314,10)==0.


def worker(rank,port):
    torch.set_num_threads(1)
    from datetime import timedelta
    import os
    os.environ['GLOO_SOCKET_IFNAME']='lo'
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,timeout=timedelta(seconds=60))
    torch.manual_seed(17)
    model=torch.nn.Linear(3,5,bias=False)
    reference=torch.nn.Linear(3,5,bias=False);reference.load_state_dict(model.state_dict())
    ddp=DistributedDataParallel(model)
    optimizer=torch.optim.SGD(ddp.parameters(),lr=.1)
    reference_optimizer=torch.optim.SGD(reference.parameters(),lr=.1)
    x=torch.arange(39,dtype=torch.float32).reshape(13,3)/39
    y=torch.arange(13)%5
    order=list(range(13))
    for step in [1,2]:
        local,n=local_batch(order,step,8,rank,4)
        optimizer.zero_grad();reference_optimizer.zero_grad()
        def loss_fn(i):return torch.nn.functional.cross_entropy(ddp(x[i:i+1]),y[i:i+1])
        accumulate_groups(ddp,local,n,4,loss_fn)
        ids=order[(step-1)*8:step*8]
        torch.nn.functional.cross_entropy(reference(x[ids]),y[ids]).backward()
        torch.testing.assert_close(model.weight.grad,reference.weight.grad,rtol=1e-5,atol=1e-7)
        optimizer.step();reference_optimizer.step()
        torch.testing.assert_close(model.weight,reference.weight,rtol=1e-5,atol=1e-7)
    dist.destroy_process_group()


def test_real_four_process_gradients_match_serial_full_and_tail(tmp_path):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,),nprocs=4,join=True)
