"""A retained OS process record must not prevent a completed host restart."""
import os
import subprocess
import sys
import pytest

from video2scene.pi_workflow.process_owner import process_birth,owner_live


def test_current_process_identity_is_live():
    birth=process_birth(os.getpid())
    assert birth not in (None,'unknown')
    assert owner_live({'pid':os.getpid(),'process_birth':birth})


@pytest.mark.skipif(os.name!='nt',reason='Windows retained process handle regression')
def test_exited_process_with_retained_handle_is_not_live():
    child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE)
    try:
        birth=process_birth(child.pid)
        assert birth not in (None,'unknown')
        child.stdin.close()
        child.wait(timeout=10)
        # Popen still holds the process handle here; OpenProcess can succeed.
        assert process_birth(child.pid) is None
        assert not owner_live({'pid':child.pid,'process_birth':birth})
    finally:
        if child.poll() is None:
            child.terminate();child.wait(timeout=10)
