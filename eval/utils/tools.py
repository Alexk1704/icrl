import os
import sys
import json
import glob
import gzip
import numpy
import pickle
import tarfile

from datetime import datetime


def resolve_root(dir_path):
    return glob.glob(f'{dir_path}**', recursive=True)

def resolve_pattern(dir_path, pattern):
    return glob.glob(os.path.join(dir_path, pattern), recursive=True)

def return_dirs(entries):
    return list(filter(lambda x: os.path.isdir(x), entries))

def return_files(entries):
    return list(filter(lambda x: os.path.isfile(x), entries))

def return_existing(entries):
    return list(filter(lambda x: os.path.exists(x), entries))

def resolve_dirlevels(entries):
    dirs = {}
    for entry in entries:
        path = os.path.dirname(entry)

        try: dirs[path].append(entry)
        except: dirs[path] = [entry]

    return dirs

def filter_dirlevels(entries, dir_levels):
    dirs = []
    for dir_level in dir_levels:
        if dir_level in entries:
            dirs.extend(entries[dir_level])

    return dirs

def resolve_filetypes(entries):
    files = {}
    for entry in entries:
        name = os.path.basename(entry)
        index = name.find('.')

        extension = ''
        if index != -1:
            extension = name[index:]

        try: files[extension].append(entry)
        except: files[extension] = [entry]

    return files

def filter_filetypes(entries, file_types):
    files = []
    for file_type in file_types:
        if file_type in entries:
            files.extend(entries[file_type])

    return files

def unpack_archive(file_path):
    def check_existence(tar, dest_path):
        for tarinfo in tar:
            file_path = os.path.join(dest_path, tarinfo.name)
            if os.path.exists(file_path):
                return True
        return False

    if file_path.find('tar.gz') != -1:
        tar = tarfile.open(file_path)
        parent_dir = os.path.dirname(file_path)
        exp_id = file_path[int(len(parent_dir)+1):-7]
        save_dir = os.path.join(parent_dir, exp_id)
        if not check_existence(tar, save_dir):
            tar.extractall(path=save_dir)
            tar.close()
    else:
        with open(file_path, 'rb') as fp:
            with tarfile.open(fileobj=fp, mode='r') as tar:
                if not check_existence(tar, os.path.dirname(file_path)):
                    tar.extractall(path=os.path.dirname(file_path))

def pack_archive(file_path, content_path):
    if file_path.find('gz') != -1:
        with gzip.open(file_path, 'wb') as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                tar.add(content_path)
    else:
        with open(file_path, 'wb') as fp:
            with tarfile.open(fileobj=fp, mode='w') as tar:
                tar.add(content_path)

def return_filepointer(file_path):
    if file_path.find('gz') != -1:
        return gzip.open(file_path, 'rb')
    else:
        return open(file_path, 'rb')

def load_file(file_path):
    if any(file_path.endswith(extension) for extension in ['.log', '.log.gz']):
        return load_log(file_path)

    if any(file_path.endswith(extension) for extension in ['.json', '.json.gz']):
        return load_json(file_path)

    if any(file_path.endswith(extension) for extension in ['.npy', '.npy.gz']):
        return load_numpy(file_path)

    if any(file_path.endswith(extension) for extension in ['.pkl', '.pkl.gz']):
        return load_pickle(file_path)

def load_log(file_path):
    if file_path.find('gz') != -1:
        with gzip.open(file_path, 'rt') as gz:
            obj = gz.read()
    else:
        with open(file_path, 'r') as fp:
            obj = fp.read()
    return obj

def load_json(file_path):
    if file_path.find('gz') != -1:
        with gzip.open(file_path, 'rt') as gz:
            obj = json.load(gz)
    else:
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
    return obj

def load_numpy(file_path):
    if file_path.find('gz') != -1:
        with gzip.open(file_path, 'rb') as gz:
            obj = numpy.load(gz)
    else:
        with open(file_path, 'rb') as fp:
            obj = numpy.load(fp)
    return obj

def load_pickle(file_path):
    if file_path.find('gz') != -1:
        with gzip.open(file_path, 'rb') as gz:
            obj = pickle.load(gz)
    else:
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
    return obj

def ensure_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def handle_file(file_path, pattern='%n_%c'):
    base_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    name, extension = os.path.splitext(file_name)
    counter = len(glob.glob(f'{base_path}/{file_name}'))
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    tokens = {
        '%n': name,
        '%c': counter,
        '%t': timestamp,
    }

    for token, expansion in tokens.items():
        pattern = pattern.replace(token, expansion)

    ensure_path(base_path)
    return os.path.join(base_path, f'{pattern}.{extension}')
