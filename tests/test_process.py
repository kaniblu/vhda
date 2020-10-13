import os
import sys
import select
import subprocess

from utils import Process


def poll_read_fd(fd, timeout=1.0, buffer=65535):
    events, _, _ = select.select([fd], [], [], timeout)
    if not events:
        return
    ret = b''
    while True:
        events, _, _ = select.select([fd], [], [], timeout)
        if not events:
            return ret
        assert events[0] == fd
        data = os.read(fd, buffer)
        if not data:
            raise ValueError(f"empty data detected!")
        ret += data


def test_pipe_select():
    r, w = os.pipe()
    assert poll_read_fd(r) is None
    os.write(w, b"hello there")
    assert poll_read_fd(r) == b"hello there"
    assert poll_read_fd(r) is None


def test_subprocess_pipe_select():
    r, w = os.pipe()
    process = subprocess.Popen(
        args=(sys.executable, "-c", "print('hello')"),
        stdout=os.fdopen(w, "wb")
    )
    events, _, _ = select.select([r], [], [], 1.0)
    if events:
        ret = b''
        while True:
            events, _, _ = select.select([r], [], [], 1.0)
            if not events:
                return ret
            assert events[0] == r
            data = os.read(r, 65536)
            if not data:
                if process.poll() is not None:
                    break
                else:
                    continue
            ret += data
            sys.stdout.write(data.decode())
        assert ret == b"hello\n"


def test_large_output():
    # There used to be a bug in subprocess.PIPE that causes subprocess.Process
    # to hang indefinitely if the child process produces more than or equal
    # to 2^16 bytes of data
    # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
    process = Process((sys.executable, "-c",
                       "import time; print('x ' * (2 ** 16))"),
                      print_stdout=True)
    print(process.run())


def test2():
    process = Process((sys.executable, "-c",
                       "import time\nfor i in range(10):\n\t"
                       "time.sleep(0.3)\n\tprint(i)"), print_stdout=True)
    print(process.run())


if __name__ == "__main__":
    test2()
