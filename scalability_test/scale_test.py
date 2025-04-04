#!/usr/bin/env python3

import os
import time

import parsl
from parsl import MonitoringHub
from parsl.config import Config
from parsl.providers import LSFProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import JsrunLauncher
from parsl.addresses import address_by_hostname
from parsl.app.app import bash_app
from parsl.data_provider.files import File
from parsl.data_provider.dynamic_files import DynamicFileList
from parsl.app.watcher import bash_watch

executor_name = "my_executor"
host = address_by_hostname()


def make_executor(walltime, nodes):
    """ make an executor
    """
    return HighThroughputExecutor(
        label=executor_name,
        address=address_by_hostname(),
        worker_debug=True,
        provider=LSFProvider(launcher=JsrunLauncher(overrides=''),
                             walltime=walltime,
                             nodes_per_block=nodes,
                             init_blocks=1,
                             max_blocks=1,
                             bsub_redirection=True,
                             queue='pdebug',
                             worker_init=(
                                 "module load spectrum-mpi\n"
                                 "source ../emirge/config/activate_env.sh\n"
                                 "source ../emirge/mirgecom/scripts/mirge-testing-env.sh\n"
                                 "export PYOPENCL_CTX=port:tesla\n"
                                 "export XDG_CACHE_HOME=/tmp/$USER/xdg-scratch\n"),
                             project='uiuc')
        )


execu = make_executor(os.environ["WALLTIME"], int(os.environ["NODES"]))

config = Config(executors=[execu],
                monitoring=MonitoringHub(hub_address=address_by_hostname(),
                                         hub_port=55055,
                                         monitoring_debug=False,
                                         resource_monitoring_interval=10,
                                         file_provenance=True,
                                         )
                )
parsl.load(config)


@bash_app(executors=[executor_name])
def run(path, outputs=None, n=None, p="../", s=None, N=None, t="scalability_test", stdout=None, stderr=None):
    """ run the code
    """
    import os
    cmd = os.path.join(path, "..", "scripts", "multi_scalability.sh")
    if n:
        cmd += f" -n {n}"
    if p:
        cmd += f" -p {p}"
    if s:
        cmd += f" -s {s}"
    if N:
        cmd += f" -N {N}"
    if t:
        cmd += f" -t {t}"
    return cmd


def main():
    outp = DynamicFileList()
    path = os.path.dirname(os.path.realpath(__file__))
    outp.append(File(os.path.join(path, "scal_p4.txt")))
    s1 = run(path, outputs=outp, n=int(os.environ["NODES"]) * 4, stdout=outp[0])
    #s1 = bash_watch(run, outputs=outp, n=int(os.environ["NODES"]) * 4, stdout=outp[0])
    s1.result()  # wait until it is done
    time.sleep(2)

if __name__ == "__main__":
    main()
