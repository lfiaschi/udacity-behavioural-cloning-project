from fabric.api import local, env, put, cd, run
from fabric.api import sudo, warn_only
from fabric.contrib.project import rsync_project
from fabric.context_managers import cd, prefix
import os
WORKDIR = '/home/carnd/CarND-Behavioral-Cloning-P3'

env.shell = "/bin/bash -l -i -c" #load the environment
env.hosts = ['52.59.71.221']
env.user = 'carnd'


def sync():
    rsync_project(remote_dir=WORKDIR, local_dir='.', delete=True,
                  exclude=['*.pyc', '*.DS_Store', '.git', '.cache/*', '*json', '*pkl','models'])


def fetch_models():
    os.system('rsync -rtuv {}@{}:/home/carnd/CarND-Behavioral-Cloning-P3/models ./'.format(env.user, env.hosts[0]))


def deploy():

    sync()

    with prefix('cd {} && source activate carnd-term1'.format(WORKDIR)):

        run('python train.py')

    fetch_models()



