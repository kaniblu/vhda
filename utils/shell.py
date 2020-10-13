__all__ = ["download", "create_cache", "get_file", "ShellUtils"]

import os
import pathlib
import logging
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Union, Sequence

import tqdm
import requests

from . import gdrive


def download(url, path, progress=False, buffer_size=4096):
    resp = requests.get(url, stream=True)
    path = os.path.realpath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if not os.path.isdir(dirname):
        raise FileExistsError(f"a file exists at the path {dirname}")
    with open(path, "wb") as f:
        with tqdm.tqdm(
                total=(int(resp.headers["content-length"])
                if "content-length" in resp.headers else None),
                dynamic_ncols=True,
                unit_scale=True,
                unit="bytes",
                disable=not progress
        ) as t:
            for chunk in resp.iter_content(buffer_size):
                t.update(len(chunk))
                f.write(chunk)


def create_cache(relpath, local_dir="~/.local/share/tyche") -> pathlib.Path:
    local_dir = pathlib.Path(local_dir).expanduser()
    return pathlib.Path(local_dir).joinpath(relpath)


def get_file(relpath, fallback_url=None, local_dir="~/.local/share/tyche",
             no_cache=False, **download_kwargs) -> pathlib.Path:
    """Try getting the path of a local path. If the path does not exist, the
    file will be downloaded from the fallback url (if provided). Or else, it
    will throw FileNotFoundError.

    Arguments:
        relpath (str): Path of a target file relative to the local dir.
        fallback_url (str): The url to download the file from if the file
            does not exist. (default: None)
        local_dir (str): The local directory file where files will be cached.
            (default: $HOME/.local/share/temp)
        no_cache (bool): Whether to disable caching.
    Return:
        The full path of the target file.
    """
    path = create_cache(relpath, local_dir)
    if not path.exists() or no_cache:
        if fallback_url is None:
            raise FileNotFoundError(f"no file found at {path}")
        download(fallback_url, path, **download_kwargs)
    return path


@dataclass
class ShellUtils:
    _logger: logging.Logger = field(init=False, default=None)

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _prepare_path(path):
        return pathlib.Path(path).absolute()

    def copy(self,
             src: Union[str, pathlib.Path],
             dst: Union[str, pathlib.Path],
             overwrite: bool = False,
             force_dir: bool = False):
        src, dst = self._prepare_path(src), self._prepare_path(dst)
        if not src.exists():
            raise FileNotFoundError(f"no source file detected at {str(src)}")
        if src.is_dir():
            raise IsADirectoryError(f"can't copy source directory {str(src)}; "
                                    f"use `self.copy_dir` or `sync` instead.")
        self._logger.info(f"copying file from {str(src)} to {str(dst)}...")
        if dst.is_dir():
            if not force_dir:
                raise IsADirectoryError(f"directory exists at "
                                        f"copy dest. {str(dst)}")
            shutil.rmtree(str(dst))
        elif dst.exists():
            if not overwrite:
                raise FileExistsError(f"dst file exists at {str(dst)}")
            os.remove(str(dst))
        shutil.copy(str(src), str(dst))

    def copy_dir(self,
                 src: Union[str, pathlib.Path],
                 dst: Union[str, pathlib.Path],
                 exclude: Sequence[Union[str, pathlib.Path]] = tuple(),
                 overwrite: bool = False):
        src, dst = self._prepare_path(src), self._prepare_path(dst)
        exclude = set(path.absolute() for path in
                      map(self._prepare_path, exclude))
        if not src.exists():
            raise FileNotFoundError(f"no source dir detected at {str(src)}")
        if not src.is_dir():
            raise FileExistsError(f"can't copy source file {str(src)}; "
                                  f"use `self.copy` to copy file instead.")
        self._logger.info(f"copying dir from {str(src)} to {str(dst)}...")
        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"file or directory exists at {str(dst)}")
            shutil.rmtree(str(dst)) if dst.is_dir() else os.remove(str(dst))
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.glob("*"):
            item = item.absolute()
            if item in exclude:
                continue
            if item.is_dir():
                self.copy_dir(
                    src=item,
                    dst=dst.joinpath(item.name),
                    exclude=tuple(p for p in exclude if item in p.parents),
                    overwrite=overwrite
                )
            else:
                self.copy(
                    src=item,
                    dst=dst.joinpath(item.name),
                    overwrite=overwrite,
                    force_dir=overwrite
                )

    def sync(self,
             src: Union[str, pathlib.Path],
             dst: Union[str, pathlib.Path],
             all_files: bool = False,
             force_dir: bool = False, ):
        src, dst = self._prepare_path(src), self._prepare_path(dst)
        if not src.exists():
            raise FileNotFoundError(f"no source file or dir "
                                    f"detected at {str(src)}")
        self._logger.info(f"syncing file or dir from "
                          f"{str(src)} to {str(dst)}...")
        if not src.is_dir():
            return self.copy(src, dst, True, force_dir)
        if dst.exists():
            shutil.rmtree(str(dst)) if dst.is_dir() else os.remove(str(dst))
        dst.mkdir()
        for f in src.glob("*"):
            if not all_files and f.name.startswith("."):
                continue
            self.sync(f, dst.joinpath(f.name), all_files, force_dir)

    def mkdir(self, path: Union[str, pathlib.Path], silent=False):
        path = self._prepare_path(path)
        self._logger.info(f"creating directory at {str(path)}...")
        path.mkdir(parents=True, exist_ok=silent)

    def remove(self, path: Union[str, pathlib.Path],
               recursive=False, silent=False):
        path = self._prepare_path(path)
        if not path.exists():
            if not silent:
                raise FileNotFoundError(f"no file or directory found "
                                        f"at {str(path)}")
            return
        self._logger.info(f"removing file or directory at {str(path)}...")
        if path.is_dir():
            if not recursive:
                raise IsADirectoryError(f"can't remove directory "
                                        f"(recursive=False)")
            shutil.rmtree(str(path))
        else:
            os.remove(str(path))

    def extract_targz(self,
                      path: Union[str, pathlib.Path],
                      extract_dir: Union[str, pathlib.Path] = None):
        path = self._prepare_path(path)
        self._logger.info(f"extracting tar/gz file {str(path)}...")
        if extract_dir is None:
            extract_dir = pathlib.Path(".")
        ext = tuple(path.suffixes)
        if ext == (".tar", ".gz"):
            open_mode = "r:gz"
        else:
            open_mode = "r:"
        with tarfile.open(path, open_mode) as f:
            f.extractall(str(extract_dir))

    def extract_zip(self,
                    path: Union[str, pathlib.Path],
                    extract_dir: Union[str, pathlib.Path] = None):
        path = self._prepare_path(path)
        self._logger.info(f"extracting zip file {str(path)}...")
        if extract_dir is None:
            extract_dir = pathlib.Path(".")
        with zipfile.ZipFile(str(path), "r") as f:
            f.extractall(str(extract_dir))

    def extract(self,
                path: Union[str, pathlib.Path],
                extract_dir: Union[str, pathlib.Path] = None):
        """Automatically extract files according to the
        compression format inferred from the filename.
        """
        path = self._prepare_path(path)
        self._logger.info(f"extracting compressed file {str(path)}...")
        ext = tuple(path.suffixes)
        if ext == (".zip",):
            extractor = self.extract_zip
        elif ext == (".tar", ".gz") or ext == (".tar",):
            extractor = self.extract_targz
        else:
            raise TypeError(f"unsupported file type: {ext}")
        return extractor(path, extract_dir)

    def download(self,
                 url, path: Union[str, pathlib.Path],
                 overwrite: bool = False,
                 progress: bool = False) -> int:
        path = self._prepare_path(path)
        if path.exists():
            if path.is_dir():
                raise IsADirectoryError(f"a directory exists at {path}")
            if not overwrite:
                raise FileExistsError(f"a dst file exists at {path}")
            os.remove(path)
        self._logger.info(f"downloading from {url}...")
        return download(url, path, progress)

    def download_gdrive(self,
                        fid: str, path: Union[str, pathlib.Path],
                        overwrite: bool = False):
        path = self._prepare_path(path)
        if path.exists():
            if path.is_dir():
                raise IsADirectoryError(f"a directory exists at {path}")
            if not overwrite:
                raise FileExistsError(f"a dst file exists at {path}")
            os.remove(path)
        self._logger.info(f"downloading from google drive id={fid}...")
        with open(str(path), "wb") as f:
            gdrive.download(f"https://drive.google.com/uc?id={fid}", f)
