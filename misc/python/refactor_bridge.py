import os
import re
import sys
import glob
import shutil


assert len(sys.argv) > 3, 'Missing argument: <path> <pattern> <replacement>'


path = sys.argv[1]
pattern = sys.argv[2]
replacement = sys.argv[3]


entries = glob.glob(f'{path}/**', recursive=True)
pattern = re.compile(r'{}'.format(pattern))

lookups = set()


def refactor_name(root_path):
    for entry in os.listdir(root_path):
        entry_path = os.path.join(root_path, entry)

        if os.path.isdir(entry_path):
            refactor_name(entry_path)

        if os.path.exists(entry_path):
            if pattern.search(entry) != None:
                new_entry = pattern.sub(replacement, entry)
                new_entry_path = os.path.join(root_path, new_entry)

                entry_pattern = f'[{"][".join(entry)}]'
                lookups.add((entry_pattern, new_entry))

                shutil.move(entry_path, new_entry_path)
                print(f'Renamed: {entry_path} -> {new_entry_path}')

def refactor_content(root_path):
    for entry in os.listdir(root_path):
        entry_path = os.path.join(root_path, entry)

        if os.path.isdir(entry_path):
            refactor_content(entry_path)

        if os.path.isfile(entry_path):
            try:
                with open(entry_path, 'r') as fp:
                    temp = fp.read()

                for keyword, mapping in lookups:
                    temp = re.sub(keyword, mapping, temp)
                    temp = re.sub(keyword.lower(), mapping.lower(), temp)
                    temp = re.sub(keyword.upper(), mapping.upper(), temp)

                    if '[.]' in keyword and '.' in mapping:
                        temp = re.sub(keyword.split('[.]')[0], mapping.split('.')[0], temp)
                        temp = re.sub(keyword.split('[.]')[0].lower(), mapping.split('.')[0].lower(), temp)
                        temp = re.sub(keyword.split('[.]')[0].upper(), mapping.split('.')[0].upper(), temp)

                with open(entry_path, 'w') as fp:
                    fp.write(temp)
            except Exception as e:
                print(f'File {entry_path} raised {type(e)} exception: {e}')


refactor_name(path)
refactor_content(path)
