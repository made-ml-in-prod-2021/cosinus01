import requests
import logging
import urllib
import zipfile
import os
import json
import sys

logger = logging.getLogger()

API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'

def _get_real_direct_link(sharing_link, proxyDict):
    pk_request = requests.get(API_ENDPOINT.format(sharing_link), proxies=proxyDict, verify=False)
    return pk_request.json().get('href')

def _extract_filename(direct_link):
    for chunk in direct_link.strip().split('&'):
        if chunk.startswith('filename='):
            return chunk.split('=')[1]
    return None

def download_yadisk_link(sharing_link, proxyDict, path, unzip=False):
    filename = ""
    direct_link = _get_real_direct_link(sharing_link, proxyDict)
    if direct_link:
        filename = path + _extract_filename(direct_link)
        download = requests.get(direct_link, proxies=proxyDict, verify=False)
        with open(filename, 'wb') as out_file:
            out_file.write(download.content)
        if unzip:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(filename) 
            logger.info('Downloaded "{}" to "{}"'.format(sharing_link, filename[:-4]))
        else:
            logger.info('Downloaded "{}" to "{}"'.format(sharing_link, filename))
    else:
        logger.info('Failed to download "{}"'.format(sharing_link))
    return filename