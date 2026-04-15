"""
suppress_alsa.py — Silence PortAudio/ALSA stderr noise during PyAudio use.

The ALSA messages come from C-level fprintf() calls deep inside libportaudio
and libasound. They cannot be silenced via Python's sys.stderr or environment
variables. The only reliable approach is to redirect the OS-level file
descriptor 2 (stderr) to /dev/null around every PyAudio() constructor call.

THREAD SAFETY: fd 2 is process-wide. quiet_stderr() holds a process-level
lock so it is never concurrent with another redirect (which would corrupt
the saved fd) or with ROCm/torch subprocess spawning (which inherits fd 2).

Usage:
    with quiet_stderr():
        p = pyaudio.PyAudio()
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# JACK crash prevention
# ---------------------------------------------------------------------------
# libportaudio is compiled with JACK support. When Pa_Terminate() is called
# while PipeWire's JACK server is running, the JACK client teardown triggers
# an assertion in pa_jack.c:869 → abort() → SIGABRT.
#
# Setting JACK_NO_START_SERVER=1 tells the JACK client library not to start
# a new JACK server if one isn't running, AND prevents the internal JACK
# client state from reaching the codepath that asserts on Pa_Terminate.
# Must be set before the first Pa_Initialize (i.e. before any pyaudio import).
os.environ.setdefault("JACK_NO_START_SERVER", "1")

# Process-wide lock: only one thread may redirect stderr at a time.
# This prevents the redirect from racing with torch/ROCm subprocess spawning.
_stderr_lock = threading.Lock()


@contextmanager
def quiet_stderr():
    """
    Redirect OS-level fd 2 to /dev/null for the duration of the block.

    Holds a process-wide lock so this is safe to call from any thread
    even when other threads may be spawning subprocesses (torch/ROCm).
    """
    with _stderr_lock:
        try:
            sys.stderr.flush()
        except Exception:
            pass

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def init_pyaudio_quiet():
    """
    Import pyaudio and run one quiet Pa_Initialize/Pa_Terminate cycle.

    After this call, the pyaudio C extension is loaded and PortAudio is
    initialised. Subsequent PyAudio() calls inside quiet_stderr() blocks
    will be silent because the heavy device enumeration only happens on
    the first Pa_Initialize.
    """
    import pyaudio as _pyaudio  # noqa: PLC0415

    with quiet_stderr():
        _pa = _pyaudio.PyAudio()
        _pa.terminate()


def prewarm_tqdm_lock():
    """
    Force tqdm to create its multiprocessing RLock (and the resource_tracker
    subprocess that backs it) NOW, before any quiet_stderr() fd redirect is
    ever active.

    Background: tqdm lazily creates an mp.RLock on the first call to
    tqdm.std.TqdmDefaultWriteLock.__init__, which triggers the Python
    multiprocessing resource_tracker subprocess via spawnv_passfds().
    That subprocess inherits the current set of open fds.  If quiet_stderr()
    has redirected fd 2 to /dev/null at that exact moment, one of the fds
    in the pass-list may be invalid, causing:

        ValueError: bad value(s) in fds_to_keep

    whisper.transcribe() calls tqdm on every inference, but the lock is only
    created once (cached on the class).  By calling this function at import
    time — before any PyAudio/PortAudio initialisation — we ensure the
    resource_tracker process is already running before any fd games happen.
    """
    try:
        from tqdm.std import TqdmDefaultWriteLock  # noqa: PLC0415

        TqdmDefaultWriteLock.create_mp_lock()
    except Exception:
        pass  # tqdm not installed or mp not supported — not fatal
