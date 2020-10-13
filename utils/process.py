__all__ = ["Process", "ProcessError"]

import os
import sys
import shlex
import queue
import select
import logging
import subprocess
import collections
import threading
from itertools import chain
from dataclasses import dataclass
from typing import Sequence, Mapping


def poll(fd: int, stop_event: threading.Event,
         ret_queue: queue.Queue, out_stream=None):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    ret = None
    stop = False
    while not stop:
        events = poller.poll(100)
        if not events and stop_event.is_set():
            break
        for poll_fd, poll_event in events:
            if poll_fd != fd:
                continue
            data = os.read(poll_fd, 65536).decode()
            if not data:
                if stop_event.is_set():
                    stop = True
                    break
                else:
                    continue
            if out_stream is not None:
                out_stream.write(data)
            if ret is None:
                ret = data
            else:
                ret += data
    ret_queue.put(ret)


Pipe = collections.namedtuple("Pipe", ("read", "write"))


class ProcessError(Exception):
    pass


@dataclass
class Process:
    args: Sequence[str]
    cwd: str = "/"
    inherit_env: bool = True
    aux_env: Mapping[str, str] = None
    aux_paths: Sequence[str] = None
    print_stdout: bool = False
    print_stderr: bool = False
    _logger: logging.Logger = None

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.args = list(map(str, self.args))

    def create_env(self) -> Mapping[str, str]:
        env = dict()
        if self.inherit_env:
            env.update(os.environ.copy())
            env["PATH"] = ":".join((":".join(sys.path), env["PATH"]))
        if self.aux_env is not None:
            env.update(self.aux_env)
        if self.aux_paths is not None:
            env["PATH"] = ":".join(chain(self.aux_paths, [env["PATH"]]))
        return env

    def run(self, timeout=None):
        self._logger.info(f"running '{' '.join(map(shlex.quote, self.args))}'")
        stdout, stderr = Pipe(*os.pipe()), Pipe(*os.pipe())
        process = subprocess.Popen(
            self.args,
            bufsize=0,
            env=self.create_env(),
            cwd=self.cwd,
            stdout=os.fdopen(stdout.write, "w"),
            stderr=os.fdopen(stderr.write, "w")
        )
        stop_event = threading.Event()
        stdout_queue, stderr_queue = queue.Queue(), queue.Queue()
        stdout_thread = threading.Thread(
            name="stdout_thread",
            target=poll,
            kwargs=dict(
                fd=stdout.read,
                stop_event=stop_event,
                ret_queue=stdout_queue,
                out_stream=None if not self.print_stdout else sys.stdout
            )
        )
        stdout_thread.start()
        stderr_thread = threading.Thread(
            name="stdout_thread",
            target=poll,
            kwargs=dict(
                fd=stderr.read,
                stop_event=stop_event,
                ret_queue=stderr_queue,
                out_stream=None if not self.print_stderr else sys.stderr
            )
        )
        stderr_thread.start()
        ret = process.wait(timeout)
        stop_event.set()
        stdout_thread.join()
        stderr_thread.join()
        return ret, stdout_queue.get(), stderr_queue.get()
