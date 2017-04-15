from fabric.api import local, env, put, cd, run
from fabric.api import sudo, warn_only
from fabric.contrib.project import rsync_project

env.path = '/home/carnd/CarND-Behavioral-Cloning-P3'
env.shell = "/bin/bash -c"
env.hosts = ['52.59.71.221']
env.user = 'carnd'


def sync():
    rsync_project(remote_dir=env.path, local_dir='.', delete=True,
                  exclude=['*.pyc', '*.DS_Store', '.git', '.cache/*', '*json', '*pkl'])



