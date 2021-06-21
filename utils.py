import os
import sys
import json
import shutil
import datetime
import subprocess


def make_dirs(dir_list):
    for _dir in dir_list:
        if _dir is not None and not os.path.isdir(_dir):
            os.makedirs(_dir)


def chown_dirs(dir_list, uid, gid):
    for _dir in dir_list:
        if _dir is not None and os.path.isdir(_dir):
            subprocess.run(['chown', '-R', '{}:{}'.format(uid, gid), _dir])


def rm_dirs(dir_list):
    for _dir in dir_list:
        if _dir is not None and os.path.isdir(_dir):
            shutil.rmtree(_dir, ignore_errors=True)


def fix_dir(checkpoint_dir, params_file='params.json'):
    checkpoint_file = 'checkpoint'
    for path, dnames, fnames in os.walk(checkpoint_dir):
        if checkpoint_file not in fnames:
            continue
        filename = os.path.join(path, checkpoint_file)
        if not os.path.isfile(filename):
            continue
        new_lines = []
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                tag, path_and_file = line.split(':')
                path_and_file = path_and_file.strip().strip('"')
                path, file = os.path.split(path_and_file)
                if path:
                    new_line = '{}: "{}"'.format(tag, file)
                    new_lines.append(new_line)
        if new_lines:
            with open(filename, 'w') as file:
                file.write('\n'.join(new_lines))


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()



def get_closest_value(arr, target):
    def find_closest(val1, val2, target):
        return val2 if target - val1 >= val2 - target else val1
    n = len(arr)
    left = 0
    right = n - 1
    mid = 0

    # edge case - last or above all
    if target >= arr[n - 1]:
        return arr[n - 1]
    # edge case - first or below all
    if target <= arr[0]:
        return arr[0]
    # BSearch solution: Time & Space: Log(N)

    while left < right:
        mid = (left + right) // 2  # find the mid
        if target < arr[mid]:
            right = mid
        elif target > arr[mid]:
            left = mid + 1
        else:
            return arr[mid]

    if target < arr[mid]:
        return find_closest(arr[mid - 1], arr[mid], target)
    else:
        return find_closest(arr[mid], arr[mid + 1], target)


def time_to_string(seconds):
    return str(datetime.timedelta(seconds=round(seconds)))
