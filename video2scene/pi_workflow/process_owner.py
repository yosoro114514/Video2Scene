"""Read-only process identity checks; never terminate or adopt a foreign process."""
import ctypes
import json
import os
from pathlib import Path


def process_birth(pid):
    if os.name=='nt':
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        # A terminated process can still be opened while another process retains
        # a handle to it. Birth time alone does not establish that it is alive.
        handle=kernel.OpenProcess(0x1000|0x100000,False,int(pid))
        if not handle:return 'unknown' if ctypes.get_last_error()==5 else None
        try:
            state=kernel.WaitForSingleObject(handle,0)
            if state==0:return None  # Signalled process handle: process exited.
            if state!=258:return 'unknown'  # WAIT_TIMEOUT is the only live result.
            times=[wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*(ctypes.byref(t) for t in times)):return 'unknown'
            return str((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)
        finally:kernel.CloseHandle(handle)
    try:
        fields=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] in ('Z','X') else fields[19]
    except FileNotFoundError:return None
    except OSError:return 'unknown'


def owner_record():return {'pid':os.getpid(),'process_birth':process_birth(os.getpid())}
def owner_live(record):
    if 'pid' not in record or 'process_birth' not in record:return True  # legacy/unknown is not safe to steal
    current=process_birth(record['pid'])
    return current=='unknown' or current is not None and current==record['process_birth']


def archive_dead_claim(path):
    path=Path(path)
    if not path.exists():return True
    try:record=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,OSError):return False
    if owner_live(record):return False
    # A file rename inside its own verified directory, never a recursive move.
    target=path.with_name(path.name+'.orphaned.'+str(record['pid']))
    if target.parent.resolve()!=path.parent.resolve():raise ValueError('Invalid claim archive')
    path.replace(target);return True
