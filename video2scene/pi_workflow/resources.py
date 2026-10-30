"""Process-wide admission for model calls and Blender execution groups.

An author may launch Blender from its CLI tools, so reserve a Blender slot for
the entire author invocation. This is deliberately conservative, not a claim
that we can count arbitrary subprocesses launched inside an author workspace.
"""
from contextlib import contextmanager
import threading
import time

from video2scene.contracts_v1 import ContractError


class ResourceBudget:
    def __init__(self, model=3, blender=1):
        if type(model) is not int or not 1<=model<=4 or type(blender) is not int or not 1<=blender<=2:
            raise ValueError('Model concurrency must be 1–4 and Blender group concurrency 1–2')
        self.limits={'model':model,'blender':blender}
        self.used=dict.fromkeys(self.limits,0);self.peak=dict(self.used)
        self.condition=threading.Condition()

    @contextmanager
    def hold(self,*names,context=None,notify=None):
        names=set(names);start=time.monotonic();announced=False
        with self.condition:
            while any(self.used[n]>=self.limits[n] for n in names):
                if context:context.check_cancelled()
                # Never call a state callback while holding the resource lock.
                if notify and not announced:
                    self.condition.release()
                    try:notify('waiting_resource',resources=sorted(names))
                    finally:self.condition.acquire()
                    announced=True
                self.condition.wait(.1)
            if context:context.check_cancelled()
            for n in names:
                self.used[n]+=1;self.peak[n]=max(self.peak[n],self.used[n])
        try:
            if notify:notify('resource_acquired',resources=sorted(names),wait_seconds=time.monotonic()-start)
            yield
        finally:
            with self.condition:
                for n in names:self.used[n]-=1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:return {'limits':dict(self.limits),'used':dict(self.used),'peak':dict(self.peak),
            'blender_scope':'host Blender processes and entire author invocation groups'}
