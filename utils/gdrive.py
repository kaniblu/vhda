__all__ = ["download"]

# https://github.com/wkentaro/gdown

import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import warnings
import logging

import requests
import six
import tqdm
from six.moves import urllib_parse

CHUNK_SIZE = 512 * 1024  # 512KB


def parse_url(url):
    """Parse URLs especially for Google Drive links.
    file_id: ID of file on Google Drive.
    is_download_link: Flag if it is download link of Google Drive.
    """
    parsed = urllib_parse.urlparse(url)
    query = urllib_parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname == 'drive.google.com'
    is_download_link = parsed.path.endswith('/uc')

    file_id = None
    if is_gdrive and 'id' in query:
        file_ids = query['id']
        if len(file_ids) == 1:
            file_id = file_ids[0]
    match = re.match(r'^/file/d/(.*?)/view$', parsed.path)
    if match:
        file_id = match.groups()[0]

    if is_gdrive and not is_download_link:
        warnings.warn(
            'You specified Google Drive Link but it is not the correct link '
            "to download the file. Maybe you should try: {url}".format(
                url='https://drive.google.com/uc?id={}'.format(file_id)
            )
        )

    return file_id, is_download_link


def get_url_from_gdrive_confirmation(contents):
    url = ''
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = 'https://docs.google.com' + m.groups()[0]
            url = url.replace('&amp;', '&')
            return url
        m = re.search('confirm=([^;&]+)', line)
        if m:
            confirm = m.groups()[0]
            url = re.sub(
                r'confirm=([^;&]+)', r'confirm={}'.format(confirm), url
            )
            return url
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace('\\u003d', '=')
            url = url.replace('\\u0026', '&')
            return url


def download(url, output):
    logger = logging.getLogger("gdrive.download")
    url_origin = url
    sess = requests.session()
    file_id, is_download_link = parse_url(url)

    while True:
        res = sess.get(url, stream=True)
        if 'Content-Disposition' in res.headers:
            # This is the file
            break
        if not (file_id and is_download_link):
            break

        # Need to redirect with confiramtion
        url = get_url_from_gdrive_confirmation(res.text)

        if url is None:
            logger.error('Permission denied: %s' % url_origin, file=sys.stderr)
            logger.error(
                "Maybe you need to change permission over "
                "'Anyone with the link'?",
                file=sys.stderr,
            )
            return

    if output is None:
        if file_id and is_download_link:
            m = re.search(
                'filename="(.*)"', res.headers['Content-Disposition']
            )
            output = m.groups()[0]
        else:
            output = osp.basename(url)

    output_is_path = isinstance(output, six.string_types)

    if output_is_path:
        tmp_file = tempfile.mktemp(
            suffix=tempfile.template,
            prefix=osp.basename(output),
            dir=osp.dirname(output),
        )
        f = open(tmp_file, 'wb')
    else:
        tmp_file = None
        f = output

    logger.info(f"download started")
    try:
        total = res.headers.get('Content-Length')
        if total is not None:
            total = int(total)
        pbar = tqdm.tqdm(total=total, unit='B', unit_scale=True)
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
        if tmp_file:
            f.close()
            shutil.move(tmp_file, output)
    except IOError as e:
        logger.exception(e, file=sys.stderr)
        return
    finally:
        try:
            if tmp_file:
                os.remove(tmp_file)
        except OSError:
            pass

    return output
