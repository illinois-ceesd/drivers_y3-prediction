#!/usr/bin/env python3

import parsl
from parsl.config import Config
from parsl.providers import LSFProvider, SlurmProvider
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import JsrunLauncher, SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.app.app import bash_app, python_app

executor_name = "my_executor"
host = address_by_hostname()
if 'lassen' in host.lower():
    executor = HighThroughputExecutor(label=executor_name,
                                      address=address_by_hostname(),
                                      worker_debug=True,
                                      provider=LSFProvider(launcher=JsrunLauncher(overrides=''),
                                                           walltime='01:00:00',
                                                           nodes_per_block=1,
                                                           init_blocks=1,
                                                           max_blocks=1,
                                                           bsub_redirection=True,
                                                           queue='pdebug',
                                                           worker_init=("module load spectrum-mpi\n"
                                                                        "source emirge/miniforge3/bin/activate mirgeDriver.Y3prediction\n"
                                                                        "export PYOPENCL_CTX=port:tesla\n"
                                                                        "export XDG_CACHE_HOME=/tmp/$USER/xdg-scratch\n"
                                                                        ),
                                                           project='uiuc'
                                                           )
                                      )
elif 'quartz' in host.lower():
    executor = HighThroughputExecutor(label=executor_name,
                                      address=address_by_hostname(),
                                      worker_debug=True,
                                      provider=SlurmProvider(launcher=SrunLauncher(overrides=''),
                                                             walltime='01:00:00',
                                                             nodes_per_block=1,
                                                             init_blocks=1,
                                                             max_blocks=1,
                                                             scheduler_options='#SBATCH -q pdebug',
                                                             worker_init=("module load spectrum-mpi\n"
                                                                          "source emirge/miniforge3/bin/activate mirgeDriver.Y3prediction\n"
                                                                          "export XDG_CACHE_HOME=/tmp/$USER/xdg-scratch\n"
                                                                          ),
                                                             )
                                      )
else:
    executor = ThreadPoolExecutor(label=executor_name)

config = Config(executors=[executor])
parsl.load(config)


@bash_app(executors=[executor_name])
def gen_mesh(ncpus, outputs=[]):
    """ Generate the mesh

    Parameters
    ----------
    ncpus: the number of cpus to use
    outputs: list of parsl.File objects of inout file(s)
    """
    return f"gmsh -setnumber size 12.8 -setnumber blratio 4 -setnumber cavityfac 6 -setnumber isofac 2 -setnumber injectorfac 4 -setnumber blratiocavity 2 -setnumber blratioinjector 2 -setnumber samplefac 4 -setnumber blratiosample 8 -setnumber blratiosurround 2 -setnumber shearfac 6 -o {outputs[0]} -nopopup -format msh2 ./actii_from_brep_2d.geo -2 -nt {ncpus}"


@bash_app(executors=[executor_name])
def _mpirun(module="driver.py", yml="run_params.yaml", c=None, t=None, r=None, lazy=True):
    """ Generate and return the mpirun command line based on inputs

        Parameters
        ----------
        module: the python module to run (default is "driver.py")
        yml: the name of the yaml comtrol file (default is "run_params.yaml")
        c: the file name to pass to the -c argument of the module (default is None, i.e. do not set the argument)
        t: the file name to pass to the -t argument of the module (default is None, i.e. do not set the argument)
        r: the file name to pass to the -r argument of the module (default is None, i.e. do not set the argument)
        lazy: boolean, whether to set the lazt flag for the module
    """
    cmd = f"mpirun -n 2 python -u -O -m mpi4py {module} -i {yml} --log"
    if lazy:
        cmd += " --lazy"
    if t:
        cmd += f" -t {t}"
    if r:
        cmd += f" -r {r}"
    if c:
        cmd += f" -c {c}"
    print(f"\n\n{cmd}\n\n")
    return cmd


@python_app(executors=[executor_name])
def make_mesh(ncpus):
    """ Set up and generate the mesh

        Parameters
        ----------
        ncpus: the number of cpus to use
    """
    import os
    import re
    from parsl.data_provider.files import File
    print(" ================================ START MESH ============================")
    meshfile = "actii_2d.msh"
    cwd = os.getcwd()
    os.chdir("mesh")

    mesh = gen_mesh(ncpus, outputs=[File(meshfile)])
    try:
        if os.path.isfile(mesh.outputs[0].result()):
            regex = re.compile(r'^[0-9]+\s4\b(?!.*")')
            matches = 0
            with open(mesh.outputs[0].result(), 'r') as fh:
                for line in fh.readlines():
                    if regex.match(line):
                        matches += 1
            print(f"Created mesh with {matches} tets.\n")
            return True
        print("Mesh creation failed.")
        return False
    finally:
        os.chdir(cwd)
        print("============================= FINISH MESH =============================")


@python_app(executors=[executor_name])
def mpirun(directory, filepat="", module="driver.py", yml="run_params.yaml", c=None, t=None, r=None, lazy=True, inputs=[], outputs=[]):
    """ Set up, generate, and run the mpirun command.

        Parameters
        ----------
        directory: the directory to cd into
        filepat: the file pattern to look for output files (e.g. 'prediction-000000100')
        module: the python module to run (default is "driver.py")
        yml: the name of the yaml comtrol file (default is "run_params.yaml")
        c: the file name to pass to the -c argument of the module (default is None, i.e. do not set the argument)
        t: the file name to pass to the -t argument of the module (default is None, i.e. do not set the argument)
        r: the file name to pass to the -r argument of the module (default is None, i.e. do not set the argument)
        lazy: boolean, whether to set the lazt flag for the module
        inputs: list of parsl.File objects giving any input files
        outputs: list of parsl.File objects generated by this operation (given as an empty list and populated by this function)

    """
    import os
    import shutil
    import glob
    from parsl.data_provider.files import File
    print(f"**********************   STARTING   {directory}    ******************")
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        os.makedirs('init_data', exist_ok=True)
        # copy input files
        for ifile in inputs:
            shutil.copy2(ifile.filepath, f"init_data/{ifile.filename}")
        run = _mpirun(module=module, yml=yml, c=c, t=t, r=r, lazy=lazy)
        try:
            if run.result() != 0:
                raise Exception(f"mpirun failed with exit code {run.result()}")
        except:
            raise
        if filepat:
            for fname in glob.glob(f"{os.getcwd()}/restart_data/{filepat}*"):
                outputs.append(File(fname))
    finally:
        os.chdir(cwd)
        print(f"*******************************   FINISHED {directory} ***************************")


def main():
    m = make_mesh(1)
    m.result()
    outp = []
    s1 = mpirun(directory='step1', filepat='prediction-000000100', outputs=outp)
    s1.result()  # wait until it is done

    outp2 = []   # to hold the output files
    s2 = mpirun(directory='step2', filepat='prediction-000000200', t='init_data/prediction-000000100', r='init_data/prediction-000000100',
                lazy=True, inputs=outp, outputs=outp2)

    # wait until it is done, would prefer to not have to call this directly, but since the output files (outp2)
    # is populated in s2, and not pre-populated, Parsl does not know to wait to run the next step, this still needs work.
    s2.result()
    outp3 = []   # to hold the output files
    s3i = mpirun(directory='step3', filepat='step3_init', yml='init_params.yaml', c='step3_init', lazy=True, inputs=outp2, outputs=outp3)
    s3i.result()  # wait until it is done

    s3t = mpirun(directory='step3', module='prediction_scalar_to_multispecies.py', yml='init_params.yaml',
                 r='init_data/prediction-000000200', c='step3_transfer_n0_to_n7', inputs=outp3)
    s3t.result()  # wait until it is done
    s3 = mpirun(directory='step3', t='init_data/step3_init-000000000', r='init_data/step3_transfer_n0_to_n7-000000200', lazy=True)
    s3.result()  # wait until it is done


if __name__ == "__main__":
    main()
